
import torch
import numpy as np
import matplotlib.pyplot as plt

from spirl.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np, make_one_hot
from spirl.utils.general_utils import ParamDict, map_dict, AttrDict

from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent

## this should be a skill prior agent + special critic update

class HLSKillAgent(ActionPriorSACAgent):
    def __init__(self, config):
        ActionPriorSACAgent.__init__(self, config)
        # SAC agent will initialize policy, and double-Q, double-Q-target

        # critics is Qz(s, z, k)
        # policy is PIz(z|s) when k=0
        # Qa will generate the Q target
        self._update_hl_policy_flag = True

    def fast_assign_flags(self, flags):
        self._update_hl_policy_flag = flags[0]

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
                policy_loss = self._compute_policy_loss(experience_batch, policy_output)
        
                self._perform_update(policy_loss, self.policy_opt, self.policy)
            else:
                with torch.no_grad():
                    policy_loss = self._compute_policy_loss(experience_batch, policy_output)

            # logging
            info = AttrDict(    # losses
                hl_policy_loss=policy_loss,
                hl_alpha_loss=alpha_loss,
            )

            # if self._update_steps % 100 == 0:
            #     info.update(AttrDict(       # gradient norms
            #         policy_grad_norm=avg_grad_norm(self.policy),
            #     ))

            info.update(AttrDict(       # misc
                hl_alpha=self.alpha,
                hl_pi_KLD=policy_output.prior_divergence.mean(),
                hl_policy_entropy=policy_output.dist.entropy().mean(),
                hl_avg_sigma = policy_output.dist.sigma.mean(),
            ))
            info.update(self._aux_info(experience_batch, policy_output))
            info = map_dict(ten2ar, info)

            self._update_steps += 1

        return info

    def _compute_policy_loss(self, experience_batch, policy_output):
        # update the input as (s, z, k0), k0 =  one-hot method 
        # PIz(z|s) = argmax E[ Qz(s,z,k0) - alpz*DKL(PIz||Pa) ] 

        k0 = self._get_k0_onehot(experience_batch.observation)
        act = torch.cat((self._prep_action(policy_output.action), k0), dim=-1)

        # input = torch.cat((experience_batch.observation, self._prep_action(policy_output.action), k0), dim=-1)    
        q_est = torch.min(*[critic(experience_batch.observation, act).q for critic in self.critics])

        policy_loss = -1 * q_est + self.alpha * policy_output.prior_divergence[:, None]
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean()

    def _get_k0_onehot(self, obs):
        # generate one-hot, length=high-level steps, index=0
        idx = torch.tensor([0], device=self.device)
        k0 = make_one_hot(idx, self.n_rollout_steps).repeat(obs.shape[0], 1)
        return k0

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

        # show latent variables
        plt.figure(figsize=(14, 4 *plt_rows))
        for idx in range(act_dim):
            plt.subplot(plt_rows, 2, idx+1)
            plt.plot(act[:, idx], 'b.')
            plt.title('HL z value: dim {}'.format(idx))
            plt.grid()
        plt.show()

        # show the distributions of policy(latent variable)
        policy_output = self.act(obs) # input (1000, x)
        dist_batch = policy_output['dist'] # (1000, x)

        mean = np.array([dist.mu for dist in dist_batch])
        sigma = np.array([np.exp(dist.log_sigma) for dist in dist_batch])
        
        
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

    # =================== offline ===================
    def offline(self):
        self.update()
        '''
        for _ in range(self._hp.update_iterations):
            # sample batch and normalize
            experience_batch = self._sample_experience()
            experience_batch = self._normalize_batch(experience_batch)
            experience_batch = map2torch(experience_batch, self._hp.device)
            experience_batch = self._preprocess_experience(experience_batch)

            policy_output = self._run_policy(experience_batch.observation)

            # update alpha
            alpha_loss = self._update_alpha(experience_batch, policy_output)

            # compute policy loss
            policy_loss = self._compute_policy_loss(experience_batch, policy_output)
        
            self._perform_update(policy_loss, self.policy_opt, self.policy)

        '''

    @property
    def n_rollout_steps(self):
        return self._hp.policy_params.prior_model_params.n_rollout_steps


    