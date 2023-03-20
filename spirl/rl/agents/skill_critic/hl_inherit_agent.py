
import torch
import numpy as np
import matplotlib.pyplot as plt

from spirl.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np, make_one_hot
from spirl.utils.general_utils import ParamDict, map_dict, AttrDict

from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent

## this is the high-level skill agent in the skill-critic

class HLInheritAgent(ActionPriorSACAgent):
    def __init__(self, config):
        ActionPriorSACAgent.__init__(self, config)
        self._update_hl_policy_flag = True

    def fast_assign_flags(self, flags):
        self._update_hl_policy_flag = flags[0]
        self._update_hl_q_flag = flags[1]

    def update(self, experience_batch=None):

        # we only update policy on HL

        # push experience batch into replay buffer
        if experience_batch is not None:
            self.add_experience(experience_batch)
        # obs = (s), action=(z)

        # for _ in range(self._hp.update_iterations):
        for _ in range(1):
            # sample batch and normalize
            experience_batch = self._sample_experience()
            experience_batch = self._normalize_batch(experience_batch)
            experience_batch = map2torch(experience_batch, self._hp.device)
            experience_batch = self._preprocess_experience(experience_batch)

            policy_output = self._run_policy(experience_batch.observation)

            # update alpha
            alpha_loss = self._update_alpha(experience_batch, policy_output)

            # compute policy loss
            if self._update_hl_policy_flag: # update only when the flag is on
                policy_loss, q_est = self._compute_policy_loss(experience_batch, policy_output)
            else:
                with torch.no_grad():
                    policy_loss, q_est = self._compute_policy_loss(experience_batch, policy_output)

            if self._update_hl_q_flag:
                hl_q_target = self._compute_hl_q_target(experience_batch, policy_output)
                hl_critic_loss, hl_qs = self._compute_hl_critic_loss(experience_batch, hl_q_target)
            else:
                with torch.no_grad():
                    hl_q_target = self._compute_hl_q_target(experience_batch, policy_output)
                    hl_critic_loss, hl_qs = self._compute_hl_critic_loss(experience_batch, hl_q_target)


            # update losses
            if self._update_hl_policy_flag:
                self._perform_update(policy_loss, self.policy_opt, self.policy)

            if self._update_hl_q_flag:
                [self._perform_update(critic_loss, critic_opt, critic)
                        for critic_loss, critic_opt, critic in zip(hl_critic_loss, self.critic_opts, self.critics)]

            if self._update_hl_q_flag:
                [self._soft_update_target_network(critic_target, critic)
                        for critic_target, critic in zip(self.critic_targets, self.critics)]

            # logging
            info = AttrDict(    # losses
                # hl_policy_loss=policy_loss,
                # hl_alpha_loss=alpha_loss,
            )

            # if self._update_steps % 100 == 0:
            #     info.update(AttrDict(       # gradient norms
            #         policy_grad_norm=avg_grad_norm(self.policy),
            #     ))

            info.update(AttrDict(       # misc
                hl_alpha=self.alpha,
                hl_pi_KLD=policy_output.prior_divergence.mean(),
                # hl_policy_entropy=policy_output.dist.entropy().mean(),
                hl_avg_sigma = policy_output.dist.sigma.mean(),
                # hl_target_divergence=self._target_divergence(self.schedule_steps),
                hl_avg_reward=experience_batch.reward.mean(),
                hl_pi_avg_q=q_est.mean(),
            ))
            info.update(self._aux_info(experience_batch, policy_output))
            info = map_dict(ten2ar, info)

            self._update_steps += 1

        return info

    def _compute_policy_loss(self, experience_batch, policy_output):
        # update the input as (s, z, k0), k0 =  one-hot method 
        # PIz(z|s) = argmax E[ Qz(s,z,k0) - alpz*DKL(PIz||Pa) ] 

        act = self._prep_action(policy_output.action) # QHL(s, z), no K
        q_est = torch.min(*[critic(experience_batch.observation, act).q for critic in self.critics])

        policy_loss = -1 * q_est + self.alpha * policy_output.prior_divergence[:, None]
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean(), q_est

    def _get_k0_onehot(self, obs):
        # generate one-hot, length=high-level steps, index=0
        idx = torch.tensor([0], device=self.device)
        k0 = make_one_hot(idx, self.n_rollout_steps).repeat(obs.shape[0], 1)
        return k0
    
    # ================================ hl critic ================================
    def _compute_hl_q_target(self, experience_batch, policy_output):
        with torch.no_grad():
            q_next = torch.min(*[critic_target(experience_batch.observation_next, self._prep_action(policy_output.action)).q
                                for critic_target in self.critic_targets])
            next_val = (q_next - self.alpha * policy_output.log_prob[:, None])
            check_shape(next_val, [self._hp.batch_size, 1])
            return next_val.squeeze(-1)
    
    def _compute_hl_critic_loss(self, experience_batch, q_target):
        qs = self._compute_q_estimates(experience_batch)
        check_shape(qs[0], [self._hp.batch_size])
        critic_losses = [0.5 * (q - q_target).pow(2).mean() for q in qs]
        return critic_losses, qs

    # ========== vis maze hl q value ==========
    def _vis_hl_q(self, logger, step):
        experience_batch = self.replay_buffer.get()
        size = self.replay_buffer.size
        states = experience_batch.observation[:size, :2]
        obs = experience_batch.observation[:size]
        rew = experience_batch.reward[:size]

        batch_size = 1024
        batch_num = int(np.ceil(size / batch_size))
        q_est_sum = []
        KLD_sum = []
        policy_v_sum = []

        for i in range(batch_num):
            obs_batch = obs[i*batch_size:(i+1)*batch_size]
            obs_batch = map2torch(obs_batch, self._hp.device)
            policy_output = self._run_policy(obs_batch)

            act = self._prep_action(policy_output.action) # QHL(s, z), no K
            q_est = torch.min(*[critic(obs_batch, act).q for critic in self.critics])
            policy_v = q_est - self.alpha * policy_output.prior_divergence[:, None]

            q_est_sum.append(q_est.detach().cpu().numpy())
            KLD_sum.append(policy_output.prior_divergence.detach().cpu().numpy())
            policy_v_sum.append(policy_v.detach().cpu().numpy())

        q_est = np.concatenate(q_est_sum, axis=0)
        KLD = np.concatenate(KLD_sum, axis=0)
        policy_v = np.concatenate(policy_v_sum, axis=0)
        
        from spirl.data.maze.src.maze_agents import plot_maze_value
        plot_maze_value(q_est, states, logger, step, size, fig_name='vis hl_q')
        plot_maze_value(KLD, states, logger, step, size, fig_name='vis hl_KLD')
        plot_maze_value(policy_v, states, logger, step, size, fig_name='vis hl_policy_v')
        plot_maze_value(rew, states, logger, step, size, fig_name='vis hl_rew')

    # =================== visualize =================
    def visualize_actions(self, experience_batch):
        # visualize the latent variables
        obs = np.array(experience_batch.observation)
        act = np.array(experience_batch.action)
        rew = np.array(experience_batch.reward)
        if len(obs.shape) == 3: # if the dim is (batch_size, 20cars, s and z )
            obs = obs[:,0,:]
            act = act[:,0,:]
            rew = rew[:,0]

        act_dim = act.shape[1] # dimension of the latent variable
        plt_rows = int(act_dim/2)

        print('visualize HL latent variances')

        # show the distributions of policy(latent variable)
        policy_output = self.act(obs) # input (1000, x)
        dist_batch = policy_output['dist'] # (1000, x)

        mean = np.array([dist.mu for dist in dist_batch])
        sigma = np.array([np.exp(dist.log_sigma) for dist in dist_batch])
        act = policy_output.action
        
        
        # show latent variables
        plt.figure(figsize=(14, 4 *plt_rows))
        for idx in range(act_dim):
            plt.subplot(plt_rows, 2, idx+1)
            plt.plot(act[:, idx], 'b.')
            plt.title('HL z value: dim {}'.format(idx))
            plt.grid()
        plt.show()
        

        print('mean of all dims, abs value \n', np.mean(np.abs(act), axis=0), np.abs(act).shape)
        
        
        plt.figure(figsize=(14, 4 *act_dim))
        for idx in range(act_dim):
            plt.subplot(act_dim, 2, 2*idx + 1)
            plt.plot(mean[:,idx], 'b.')
            plt.grid()
            plt.title('HL z mean, dim {} '.format(idx))

            plt.subplot(act_dim, 2, 2*idx + 1 + 1)
            plt.plot(sigma[:,idx], 'b.')
            plt.grid()
            plt.title('HL z sigma, dim {} '.format(idx))

        plt.show()    
        
        
        import seaborn as sns
        fs = 16
        plt.figure(figsize=(14, 6))
        for idx in range(act_dim):
            sns.kdeplot(act[:, idx], fill=True, label='dim_' + str(idx), cut=0)
        plt.legend(loc='upper left', fontsize=fs)
        plt.ylabel('Density', fontsize=fs)
        plt.title('distribution of latent variables', fontsize=fs)
        plt.grid()
        plt.show()

    # =================== offline ===================
    def offline(self):
        self.update()
        

    @property
    def n_rollout_steps(self):
        return self._hp.policy_params.prior_model_params.n_rollout_steps


    
