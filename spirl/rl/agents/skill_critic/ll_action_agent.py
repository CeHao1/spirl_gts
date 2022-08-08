from posixpath import split
from spirl.utils.general_utils import ParamDict, map_dict, AttrDict
from spirl.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np, parse_one_hot

import torch

# this agent should be similar to the state-conditioned close-loop agent
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent

import matplotlib.pyplot as plt

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

    def update(self, experience_batch):
        # push experience batch into replay buffer
        self.add_experience(experience_batch)

        for _ in range(self._hp.update_iterations):
            # sample batch and normalize
            experience_batch = self._sample_experience()
            experience_batch = self._normalize_batch(experience_batch)
            experience_batch = map2torch(experience_batch, self._hp.device)
            experience_batch = self._preprocess_experience(experience_batch)


            # (1) LL policy loss
            policy_output = self._run_policy(experience_batch.observation)
            # update alpha
            alpha_loss = self._update_alpha(experience_batch, policy_output)

            if self.self._update_ll_policy_flag:
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
            if self._update_ll_policy_flag:
                ll_q_target, v_next, q_next, u_next = self._compute_ll_q_target(experience_batch)
                ll_critic_loss, ll_qs = self._compute_ll_critic_loss(experience_batch, ll_q_target)
            else:
                with torch.no_grad():
                    ll_q_target, v_next, q_next, u_next = self._compute_ll_q_target(experience_batch)
                    ll_critic_loss, ll_qs = self._compute_ll_critic_loss(experience_batch, ll_q_target)
            

            # (4) update loss
            # policy
            if self.self._update_ll_policy_flag:
                self._perform_update(policy_loss, self.policy_opt, self.policy)
            # hl
            if self._update_hl_q_flag:
                [self._perform_update(critic_loss, critic_opt, critic)
                        for critic_loss, critic_opt, critic in zip(hl_critic_loss, self.hl_critic_opts, self.hl_critics)]
            # ll
            if self._update_ll_policy_flag:
                [self._perform_update(critic_loss, critic_opt, critic)
                        for critic_loss, critic_opt, critic in zip(ll_critic_loss, self.critic_opts, self.critics)]


            # (5) soft update targets
            if self._update_hl_q_flag:
                [self._soft_update_target_network(critic_target, critic)
                        for critic_target, critic in zip(self.hl_critic_targets, self.hl_critics)]
            if self._update_ll_policy_flag:
                [self._soft_update_target_network(critic_target, critic)
                        for critic_target, critic in zip(self.critic_targets, self.critics)]

            # logging
            info = AttrDict(    # losses
                policy_loss=policy_loss,
                alpha_loss=alpha_loss,
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
        return info

    # ================================ hl critic ================================
    def _compute_hl_q_target(self, experience_batch, policy_output): 
        # Qz(s,z,k) = Qa(s,z,k,a) - alph * DKL(PIa)
        # split_obs = self._split_obs(experience_batch.observation_next)
        # state = split_obs.state
        # act = torch.cat((split_obs.z, split_obs.time_index), dim=-1)

        state = experience_batch.observation_next
        act = experience_batch.action

        with torch.no_grad():
            qa_target = torch.min(*[critic_target(state, act).q for critic_target in self.critic_targets])# this part is
            hl_q_target = qa_target - self.alpha * policy_output.prior_divergence[:, None]
            hl_q_target = hl_q_target.squeeze(-1)
            hl_q_target = hl_q_target.detach()
            check_shape(hl_q_target, [self._hp.batch_size])

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

    # ================================ ll critic ================================
    def _compute_ll_q_target(self, experience_batch, if_off_plot=False):
        # Qa(s,z,k,a) = r + gamma * U_next
        # U_next = [1-beta] * Qz_next + [beta] * V_next
        # V_next = Qz_next - hl_alp * DKL(PIz_next)

        with torch.no_grad():
            # (0) use s to generate hl policy
            split_obs = self._split_obs(experience_batch.observation_next)
            hl_policy_output_next = self.hl_agent._run_policy(split_obs.state)
            # policy_output_next = self._run_policy(experience_batch.observation_next)

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


        if if_off_plot:
            figsize=(10,7)

            # plot values:
            plt.figure(figsize=figsize)
            plt.plot(map2np(q_next), 'b.', label='q_next')
            plt.plot(map2np(v_next), 'r.', label='v_next')
            plt.plot(map2np(v_next-q_next), 'k.', label='KLD')
            plt.title('values')
            plt.legend()
            plt.show()

            plt.figure(figsize=figsize)
            plt.plot(map2np(u_next), 'b.', label='u_next')
            plt.plot(map2np(experience_batch.reward), 'r.', label='reward')
            plt.plot(map2np(ll_q_target), 'k.', label='ll q target')
            plt.title('q target components')
            plt.legend()
            plt.show()

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
        next_val = (q_next - self.alpha * hl_policy_output_next.prior_divergence[:, None])
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


    # ================================== offline ================================
    def offline(self):
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
            ll_q_target, v_next, q_next, u_next = self._compute_ll_q_target(experience_batch, if_off_plot=True)
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

