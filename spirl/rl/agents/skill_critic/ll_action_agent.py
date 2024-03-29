from email import policy
from posixpath import split
from spirl.utils.general_utils import ParamDict, map_dict, AttrDict
from spirl.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np, parse_one_hot

# this agent should be similar to the state-conditioned close-loop agent
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent

import matplotlib.pyplot as plt
import numpy as np
import torch
import copy

class LLActionAgent(ActionPriorSACAgent):
    def __init__(self, config):
        ActionPriorSACAgent.__init__(self, config)
        # critic is Qa(s,z,k,a)
        # policy is PIa(a|s,z,k)
        self._update_ll_policy_flag = True
        self._update_hl_q_flag = True
        self._update_ll_q_flag = True
        

    def fast_assign_flags(self, flags):
        self._update_ll_policy_flag = flags[0]
        self._update_hl_q_flag = flags[1]
        self._update_ll_q_flag = flags[2]

    def update_by_hl_agent(self, hl_agent):
        self.hl_agent = hl_agent
        self.hl_critics = hl_agent.critics
        self.hl_critic_targets = hl_agent.critic_targets
        self.hl_critic_opts = hl_agent.critic_opts

    def update(self, experience_batch=None, vis=False):
        # push experience batch into replay buffer

        if experience_batch is not None:
            self.add_experience(experience_batch)
            if vis:
                self.visualize_actions(experience_batch)

        # for _ in range(self._hp.update_iterations):
        for _ in range(1):
            # sample batch and normalize
            experience_batch = self._sample_experience()
            experience_batch = self._normalize_batch(experience_batch)
            experience_batch = map2torch(experience_batch, self._hp.device)
            experience_batch = self._preprocess_experience(experience_batch)


            # (1) LL policy loss
            policy_output = self._run_policy(experience_batch.observation)
            # update alpha
            alpha_loss = self._update_alpha(experience_batch, policy_output)

            if self._update_ll_policy_flag:
                policy_loss = self._compute_policy_loss(experience_batch, policy_output)
            else:
                with torch.no_grad():
                    policy_loss = self._compute_policy_loss(experience_batch, policy_output)

            # (2) Qz(s,z,k) loss, Qz_target
            if self._update_hl_q_flag:
                hl_q_target = self._compute_hl_q_target(experience_batch, policy_output)
                hl_critic_loss, hl_qs = self._compute_hl_critic_loss(experience_batch, hl_q_target)
            else:
                with torch.no_grad():
                    hl_q_target = self._compute_hl_q_target(experience_batch, policy_output)
                    hl_critic_loss, hl_qs = self._compute_hl_critic_loss(experience_batch, hl_q_target)

            # (3) Qa(s,z,k,a) loss, Qa_target
            if self._update_ll_q_flag:
                ll_q_target, v_next, q_next, u_next = self._compute_ll_q_target(experience_batch)
                ll_critic_loss, ll_qs = self._compute_ll_critic_loss(experience_batch, ll_q_target)
            else:
                with torch.no_grad():
                    ll_q_target, v_next, q_next, u_next = self._compute_ll_q_target(experience_batch)
                    ll_critic_loss, ll_qs = self._compute_ll_critic_loss(experience_batch, ll_q_target)
            

            # (4) update loss
            # policy
            if self._update_ll_policy_flag:
                self._perform_update(policy_loss, self.policy_opt, self.policy)
            # hl
            if self._update_hl_q_flag:
                [self._perform_update(critic_loss, critic_opt, critic)
                        for critic_loss, critic_opt, critic in zip(hl_critic_loss, self.hl_critic_opts, self.hl_critics)]
            # ll
            if self._update_ll_q_flag:
                [self._perform_update(critic_loss, critic_opt, critic)
                        for critic_loss, critic_opt, critic in zip(ll_critic_loss, self.critic_opts, self.critics)]


            # (5) soft update targets
            if self._update_hl_q_flag:
                [self._soft_update_target_network(critic_target, critic)
                        for critic_target, critic in zip(self.hl_critic_targets, self.hl_critics)]
            if self._update_ll_q_flag:
                [self._soft_update_target_network(critic_target, critic)
                        for critic_target, critic in zip(self.critic_targets, self.critics)]

            # logging
            info = AttrDict(    # losses
                ll_policy_loss=policy_loss,
                ll_alpha_loss=alpha_loss,
                qz_critic_loss_1=hl_critic_loss[0],
                qz_critic_loss_2=hl_critic_loss[1],
                qa_critic_loss_1=ll_critic_loss[0],
                qa_critic_loss_2=ll_critic_loss[1],
            )
            # if self._update_steps % 100 == 0:
            #     info.update(AttrDict(       # gradient norms
            #         policy_grad_norm=avg_grad_norm(self.policy),
            #         critic_1_grad_norm=avg_grad_norm(self.critics[0]),
            #         critic_2_grad_norm=avg_grad_norm(self.critics[1]),
            #     ))
            info.update(AttrDict(       # misc
                ll_alpha=self.alpha,
                ll_pi_KLD=policy_output.prior_divergence.mean(),
                ll_policy_entropy=policy_output.dist.entropy().mean(),
                ll_avg_sigma = policy_output.dist.sigma.mean(),
                hl_q_target=hl_q_target.mean(),
                hl_q_1=hl_qs[0].mean(),
                hl_q_2=hl_qs[1].mean(),
                ll_q_target=ll_q_target.mean(),
                ll_q_1=ll_qs[0].mean(),
                ll_q_2=ll_qs[1].mean(),
                v_next=v_next.mean(), 
                q_next=q_next.mean(), 
                u_next=u_next.mean(),
            ))
            info.update(self._aux_info(experience_batch, policy_output))
            info = map_dict(ten2ar, info)

            self._update_steps += 1

        if self._hp.visualize_values and vis:
            hl_q_target = self._compute_hl_q_target(experience_batch, policy_output, vis=True)
            ll_q_target, v_next, q_next, u_next = self._compute_ll_q_target(experience_batch, vis=True)
            self.visualize_gradients(experience_batch)

        return info

    # ================================ hl critic ================================
    def _compute_hl_q_target(self, experience_batch, policy_output, vis=False): 
        # Qz(s,z,k) = Qa(s,z,k,a) - alph * DKL(PIa)
        state = experience_batch.observation_next
        act = experience_batch.action

        with torch.no_grad():
            qa_target = torch.min(*[critic_target(state, act).q for critic_target in self.critic_targets])# this part is
            hl_q_target = qa_target - self.alpha * policy_output.prior_divergence[:, None]
            hl_q_target = hl_q_target.squeeze(-1)
            hl_q_target = hl_q_target.detach()
            check_shape(hl_q_target, [self._hp.batch_size])

        if vis:
            self.visualize_HL_Q(qa_target, hl_q_target)

        return hl_q_target

    def _compute_hl_critic_loss(self, experience_batch, hl_q_target): 
        # Qz(s,z,k), the input is only obs, not action(a) here
        split_obs = self._split_obs(experience_batch.observation)
        obs = split_obs.state
        act = torch.cat((split_obs.z, split_obs.time_index), dim=-1)

        hl_qs = [critic(obs, act).q.squeeze(-1) for critic in self.hl_critics]
        check_shape(hl_qs[0], [self._hp.batch_size])
        hl_critic_losses = [0.5 * (q - hl_q_target).pow(2).mean() for q in hl_qs] # mse loss
        return hl_critic_losses, hl_qs

    #  =============================== ll policy ================================
    '''
    def _compute_policy_loss(self, experience_batch, policy_output):
        q_est = torch.min(*[critic(experience_batch.observation, self._prep_action(policy_output.action)).q
                                      for critic in self.critics])
        policy_loss = -1 * q_est + self.alpha * policy_output.log_prob[:, None]
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean()
    '''

    # ================================ ll critic ================================
    def _compute_ll_q_target(self, experience_batch, vis=False):
        # Qa(s,z,k,a) = r + gamma * U_next
        # U_next = [1-beta] * Qz_next + [beta] * V_next
        # V_next = Qz_next - hl_alp * DKL(PIz_next)

        with torch.no_grad():
            # (0) use s to generate hl policy
            split_obs = self._split_obs(experience_batch.observation_next)
            hl_policy_output_next = self.hl_agent._run_policy(split_obs.state)

            # (1) V_next
            v_next, q_next = self._compute_next_value(experience_batch, hl_policy_output_next)

            # (2) U_next 
            beta_mask = self._compute_termination_mask(experience_batch.observation_next)
            u_next =  (1 - beta_mask) * q_next + beta_mask * v_next

            # (3) ll_q_target
            ll_q_target = experience_batch.reward * self._hp.reward_scale + (1 - experience_batch.done) * self._hp.discount_factor * u_next
            ll_q_target = ll_q_target.squeeze(-1)
            ll_q_target = ll_q_target.detach()
            check_shape(ll_q_target, [self._hp.batch_size])

            if vis:
                self.visualize_LL_Q(ll_q_target, v_next, q_next, u_next, experience_batch.reward)

        return ll_q_target, v_next, q_next, u_next

    def _compute_ll_critic_loss(self, experience_batch, ll_q_target):
        # Qa(s,z,k,a)
        ll_qs = [critic(experience_batch.observation, self._prep_action(experience_batch.action.detach())).q.squeeze(-1) \
                for critic in self.critics]     # no gradient propagation into policy here!

        check_shape(ll_qs[0], [self._hp.batch_size])
        ll_critic_losses = [0.5 * (q - ll_q_target).pow(2).mean() for q in ll_qs]
        return ll_critic_losses, ll_qs

    def _compute_next_value(self, experience_batch, hl_policy_output_next): 
        # V = Qz - alp_z * DKL(PI_z)
        split_obs = self._split_obs(experience_batch.observation_next)
        obs = split_obs.state
        act = torch.cat((split_obs.z, split_obs.time_index), dim=-1)
        q_next = torch.min(*[critic_target(obs, act).q for critic_target in self.hl_critic_targets])
        next_val = (q_next - self.hl_agent.alpha * hl_policy_output_next.prior_divergence[:, None])
        check_shape(next_val, [self._hp.batch_size, 1])
        return next_val.squeeze(-1), q_next.squeeze(-1)

    def _compute_termination_mask(self, obs):
        # [1,0,0] is the start of a new hl step, so return 1 in the mask
        split_obs  = self._split_obs(obs) # to s,z,k
        beta_onehot = split_obs.time_index
        beta = parse_one_hot(beta_onehot)
        beta_mask = torch.where(beta==torch.tensor(0), 1, 0)
        return beta_mask

    # =============================== utils ===========================
    def _split_obs(self, obs):
        assert obs.shape[1] == self.state_dim + self.latent_dim + self.onehot_dim
        return AttrDict(
            state=obs[:, :self.state_dim],   # condition decoding on state
            z=obs[:, self.state_dim:-self.onehot_dim],
            time_index=obs[:, -self.onehot_dim:],
        )
        
    # ====================================== property =======================================
    @property
    def state_dim(self):
        return self._hp.policy_params.policy_model_params.state_dim

    @property
    def latent_dim(self):
        return self._hp.policy_params.policy_model_params.nz_vae

    @property
    def onehot_dim(self):
        return self._hp.policy_params.policy_model_params.n_rollout_steps

    @property
    def action_dim(self):
        return self._hp.policy_params.policy_model_params.action_dim


    # ================================== visualize =============================
    def visualize_HL_Q(self, qa_target, hl_q_target):
        print('visualize_HL_Q, alpha', self.alpha)
        alp_KLD = hl_q_target - qa_target.squeeze()

        # print('qa_target {}, hl_q_target {}, alp_KLD {}'.format(qa_target.shape, hl_q_target.shape, alp_KLD.shape))


        plt.figure(figsize=(14, 8))

        plt.subplot(2,2,1)
        plt.plot(map2np(qa_target), 'b.')
        plt.grid()
        plt.title('qa_target')

        plt.subplot(2,2,2)
        plt.plot(map2np(alp_KLD), 'b.')
        plt.grid()
        plt.title('LL alp_KLD')

        plt.subplot(2,2,4)
        plt.plot(map2np(hl_q_target), 'b.')
        plt.grid()
        plt.title('hl_q_target')

        plt.show()


    def visualize_LL_Q(self, ll_q_target, v_next, q_next, u_next, reward):
        print('visualize_LL_Q, alpha', self.hl_agent.alpha)
        alp_KLD = v_next - q_next

        plt.figure(figsize=(14, 8))

        plt.subplot(2,2,1)
        plt.plot(map2np(q_next), 'b.')
        plt.grid()
        plt.title('q_next')

        plt.subplot(2,2,2)
        plt.plot(map2np(alp_KLD), 'b.')
        plt.grid()
        plt.title('HL alp_KLD')

        plt.subplot(2,2,3)
        plt.plot(map2np(reward), 'b.')
        plt.grid()
        plt.title('reward')

        plt.subplot(2,2,4)
        plt.plot(map2np(ll_q_target), 'b.')
        plt.grid()
        plt.title('ll_q_target')

        plt.show()


    def visualize_gradients(self, experience_batch):
        obs = copy.deepcopy(experience_batch.observation)
        assert obs.shape[1] == self.state_dim + self.latent_dim + self.onehot_dim
        obs.requires_grad = True
        policy_output = self._run_policy(obs) # output a MultiVarient Gaussian in pytorch

        mean = policy_output.dist.mean
        sigma = policy_output.dist.sigma

        chosen_dist_target = mean.mean() # we can change this
        chosen_dist_target.backward()
        grads = map2np(obs.grad)

        # 1. plot grad for states and all z
        # however, the dim of s is too high, so we only show the mean and its std for all dim
        state_grad = np.mean( np.abs(grads[:, :self.state_dim]), axis=1)
        latent_grad = np.mean( np.abs(grads[:, self.state_dim: self.state_dim + self.latent_dim]), axis=1)

        
        plt.figure(figsize=(10, 4))
        plt.plot(state_grad, 'b.', label='abs grad of states')
        plt.plot(latent_grad, 'r.', label='abs grad of latent z')
        plt.legend()
        plt.title('gradient of input w.r.t output mean')
        plt.grid()
        plt.show()
        
        
        state_grad_mean = np.mean(state_grad)
        latent_grad_mean = np.mean(latent_grad)
        print('state_grad', state_grad_mean, 'latent_grad', latent_grad_mean, 'ratio', state_grad_mean/latent_grad_mean)

        # 2. plot grad for latent variable z, just plot them all
        latent_grad_raw = grads[:, self.state_dim: self.state_dim + self.latent_dim]
        # latent_grad_raw = np.abs(latent_grad_raw)
        latent_dim = self.latent_dim
        plt_rows = int(latent_dim/2)

        
        plt.figure(figsize=(14, 2 * latent_dim))
        for idx in range(latent_dim):
            plt.subplot(plt_rows, 2, idx+1)
            plt.plot(latent_grad_raw[:, idx], 'b.')
            plt.title('HL z gradient: dim {}'.format(idx))
            plt.grid()
        plt.show()
        
        
        # 3. plot distribution, a better way
        skip_dim = []
        
        import seaborn as sns
        fs = 16
        plt.figure(figsize=(14, 6))
        sns.kdeplot(state_grad, fill=True, label='mean of all states', cut=0)
        for idx in range(latent_dim):
            if idx in skip_dim:
                continue
            sns.kdeplot(latent_grad_raw[:, idx], fill=True, label='dim_' + str(idx), cut=0)
        plt.legend(fontsize=fs)
        plt.ylabel('Density', fontsize=fs)
        plt.xlabel('Gradient', fontsize=fs)
        plt.title('gradient of states and latent variables', fontsize=fs)
        plt.grid()
        plt.ylim([0, 1e5])
        plt.xlim([-1e-4, 2e-4])
        plt.show()
        


    # ================================== offline ================================
    def offline(self):
        self.update(self, experience_batch=None, vis=True)

        '''
        from tqdm import tqdm
        q_target_store = []
        update_time = self._hp.update_iterations
        # update_time = int(1e4)
        for idx in tqdm(range(update_time)):
            # print('======= iter =====', idx )
            # sample batch and normalize
            experience_batch = self._sample_experience()
            experience_batch = self._normalize_batch(experience_batch)
            experience_batch = map2torch(experience_batch, self._hp.device)
            experience_batch = self._preprocess_experience(experience_batch)


            # (1) LL policy loss
            policy_output = self._run_policy(experience_batch.observation)
            # update alpha
            alpha_loss = self._update_alpha(experience_batch, policy_output)
            policy_loss = self._compute_policy_loss(experience_batch, policy_output)


            # (2) Qz(s,z,k) loss, Qz_target
            hl_q_target = self._compute_hl_q_target(experience_batch, policy_output)
            hl_critic_loss, hl_qs = self._compute_hl_critic_loss(experience_batch, hl_q_target)
            

            # (3) Qa(s,z,k,a) loss, Qa_target
            ll_q_target, v_next, q_next, u_next = self._compute_ll_q_target(experience_batch)
            ll_critic_loss, ll_qs = self._compute_ll_critic_loss(experience_batch, ll_q_target)
            
            q_target_store.append(map2np(ll_q_target.mean()))

            # (4) update loss
            # policy
            # self._perform_update(policy_loss, self.policy_opt, self.policy)
            # hl
            [self._perform_update(critic_loss, critic_opt, critic)
                    for critic_loss, critic_opt, critic in zip(hl_critic_loss, self.hl_critic_opts, self.hl_critics)]
            # ll
            [self._perform_update(critic_loss, critic_opt, critic)
                    for critic_loss, critic_opt, critic in zip(ll_critic_loss, self.critic_opts, self.critics)]


            # (5) soft update targets
            [self._soft_update_target_network(critic_target, critic)
                    for critic_target, critic in zip(self.hl_critic_targets, self.hl_critics)]

            [self._soft_update_target_network(critic_target, critic)
                    for critic_target, critic in zip(self.critic_targets, self.critics)]


        # finally plot the ll q target
        plt.figure(figsize=(10,7))
        plt.plot(q_target_store, 'b.')
        plt.show()

    '''