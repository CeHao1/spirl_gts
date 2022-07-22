
import torch

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
 
    # def update_by_ll_agent(self, ll_agent):
    #     self.ll_agent = ll_agent

    def update(self, experience_batch):

        # we only update policy on HL

        # push experience batch into replay buffer
        self.add_experience(experience_batch)
        # obs = (s), action=(z)

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


            # logging
            info = AttrDict(    # losses
                policy_loss=policy_loss,
                alpha_loss=alpha_loss,
            )

            if self._update_steps % 100 == 0:
                info.update(AttrDict(       # gradient norms
                    policy_grad_norm=avg_grad_norm(self.policy),
                ))

            info.update(AttrDict(       # misc
                alpha=self.alpha,
                pi_log_prob=policy_output.log_prob.mean(),
                policy_entropy=policy_output.dist.entropy().mean(),
                avg_sigma = policy_output.dist.sigma.mean(),
            ))
            info.update(self._aux_info(experience_batch, policy_output))
            info = map_dict(ten2ar, info)

            self._update_steps += 1

        return info

    def _compute_policy_loss(self, experience_batch, policy_output):
        # update the input as (s, z, k0), k0 =  one-hot method 
        # PIz(z|s) = argmax E[ Qz(s,z,k0) - alpz*DKL(PIz||Pa) ] 

        k0 = self._get_k0_onehot(experience_batch.observation)
        input = torch.cat((experience_batch.observation, self._prep_actionpolicy_output.action, k0), dim=-1)
        
        q_est = torch.min(*[critic(input).q for critic in self.critics])

        policy_loss = -1 * q_est + self.alpha * policy_output.prior_divergence[:, None]
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean()

    def _get_k0_onehot(self, obs):
        # generate one-hot, length=high-level steps, index=0
        idx = torch.tensor(torch.arange(1), device=self.device)
        k0 = make_one_hot(idx, self.n_rollout_steps).repeat(obs.shape[0], 1, 1)
        return k0


    @property
    def n_rollout_steps(self):
        return self._hp.hl_policy_params.prior_model_params.n_rollout_steps


    