
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

        # return super().update(experience_batch)

        # logging
        info = AttrDict(    # losses
        )

        if not (self._update_hl_policy_flag or self._update_hl_q_flag):
            return info

        # push experience batch into replay buffer
        if experience_batch is not None:
            self.add_experience(experience_batch)
        # obs = (s), action=(z)


        # sample batch and normalize
        experience_batch = self._sample_experience()
        experience_batch = self._normalize_batch(experience_batch)
        experience_batch = map2torch(experience_batch, self._hp.device)
        experience_batch = self._preprocess_experience(experience_batch)

        policy_output = self._run_policy(experience_batch.observation)

        # update alpha
        alpha_loss = self._update_alpha(experience_batch, policy_output)
        # info.update(AttrDict(hl_alpha_loss=alpha_loss,))

        # compute policy loss
        if self._update_hl_policy_flag: # update only when the flag is on
            policy_loss, q_est = self._compute_policy_loss(experience_batch, policy_output)

        if self._update_hl_q_flag:
            q_target = self._compute_hl_q_target(experience_batch)
            critic_loss, qs = self._compute_critic_loss(experience_batch, q_target)

        # update losses
        if self._update_hl_policy_flag:
            self._perform_update(policy_loss, self.policy_opt, self.policy)

            info.update(AttrDict(
                hl_policy_loss=policy_loss,
                hl_pi_avg_q=q_est.mean(),
            ))

        if self._update_hl_q_flag:
            [self._perform_update(critic_loss, critic_opt, critic)
                    for critic_loss, critic_opt, critic in zip(critic_loss, self.critic_opts, self.critics)]

            info.update(AttrDict(
                hl_q_target=q_target.mean(),
                hl_q_1=qs[0].mean(),
                hl_q_2=qs[1].mean(),
                qz_critic_loss_1=critic_loss[0],
                qz_critic_loss_2=critic_loss[1],
            ))

        if self._update_hl_q_flag:
            [self._soft_update_target_network(critic_target, critic)
                    for critic_target, critic in zip(self.critic_targets, self.critics)]

        # if self._update_steps % 100 == 0:
        #     info.update(AttrDict(       # gradient norms
        #         policy_grad_norm=avg_grad_norm(self.policy),
        #     ))

        info.update(AttrDict(       # misc
            hl_alpha=self.alpha,
            hl_pi_KLD=policy_output.prior_divergence.mean(),
            # hl_policy_entropy=policy_output.dist.entropy().mean(),
            hl_avg_sigma = policy_output.dist.sigma.mean(),
            hl_target_divergence=self._target_divergence(self.schedule_steps),
            hl_avg_reward=experience_batch.reward.mean(),
        ))
        info.update(self._aux_info(experience_batch, policy_output))
        info = map_dict(ten2ar, info)

        self._update_steps += 1

        return info

    
    # ================================ hl policy ================================
    def _compute_policy_loss(self, experience_batch, policy_output):
        """Computes loss for policy update."""
        q_est = torch.min(*[critic(experience_batch.observation, self._prep_action(policy_output.action)).q
                                      for critic in self.critics])
        policy_loss = -1 * q_est + self.alpha * policy_output.prior_divergence[:, None]
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean(), q_est
    
    
    # ================================ hl critic ================================
    def _compute_hl_q_target(self, experience_batch):
        with torch.no_grad():
            policy_output_next = self._run_policy(experience_batch.observation_next)
            value_next = self._compute_next_value(experience_batch, policy_output_next)
            q_target = experience_batch.reward * self._hp.reward_scale + \
                            (1 - experience_batch.done) * self._hp.discount_factor * value_next
            if self._hp.clip_q_target:
                q_target = self._clip_q_target(q_target)
            q_target = q_target.detach()
            check_shape(q_target, [self._hp.batch_size])
        return q_target
    
    def _compute_hl_critic_loss(self, experience_batch, q_target):
        qs = self._compute_q_estimates(experience_batch)
        check_shape(qs[0], [self._hp.batch_size])
        critic_losses = [0.5 * (q - q_target).pow(2).mean() for q in qs]
        return critic_losses, qs


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


    
