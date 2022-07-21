from spirl.utils.general_utils import ParamDict, map_dict, AttrDict
from spirl.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np, parse_one_hot

import torch

# this agent should be similar to the state-conditioned close-loop agent
from spirl.rl.agents.ac_agent import SACAgent

class LLActionAgent(SACAgent):
    def __init__(self, config):
        LLActionAgent.__init__(self, config)
        # critic is Qa(s,z,k,a)
        # policy is PIa(a|s,z,k)

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
            policy_loss = self._compute_policy_loss(experience_batch, policy_output)


            # (2) Qz(s,z,k) loss, Qz_target
            hl_q_target = self._compute_hl_q_target(experience_batch, policy_output)
            hl_critic_loss, hl_qs = self._compute_hl_critic_loss(experience_batch, hl_q_target)
            

            # (3) Qa(s,z,k,a) loss, Qa_target
            ll_q_target = self._compute_ll_q_target(experience_batch)
            ll_critic_loss, ll_qs = self._compute_ll_critic_loss(experience_batch, ll_q_target)
            

            # (4) update loss
            # policy
            self._perform_update(policy_loss, self.policy_opt, self.policy)
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

            # logging
            info = AttrDict(    # losses
                policy_loss=policy_loss,
                alpha_loss=alpha_loss,
                qz_critic_loss_1=hl_critic_loss[0],
                qz_critic_loss_2=hl_critic_loss[1],
                qa_critic_loss_1=ll_critic_loss[0],
                qa_critic_loss_2=ll_critic_loss[1],
            )
            if self._update_steps % 100 == 0:
                info.update(AttrDict(       # gradient norms
                    policy_grad_norm=avg_grad_norm(self.policy),
                    critic_1_grad_norm=avg_grad_norm(self.critics[0]),
                    critic_2_grad_norm=avg_grad_norm(self.critics[1]),
                ))
            info.update(AttrDict(       # misc
                alpha=self.alpha,
                pi_log_prob=policy_output.log_prob.mean(),
                policy_entropy=policy_output.dist.entropy().mean(),
                avg_sigma = policy_output.dist.sigma.mean(),
                q_target=hl_q_target.mean(),
                q_1=hl_qs[0].mean(),
                q_2=hl_qs[1].mean(),
                q_target=ll_q_target.mean(),
                q_1=ll_qs[0].mean(),
                q_2=ll_qs[1].mean(),
            ))
            info.update(self._aux_info(experience_batch, policy_output))
            info = map_dict(ten2ar, info)

            self._update_steps += 1

        return info


    # ================================ policy ================================
    # LL policy update is identical to the parent prior sac agent.

    # def _compute_policy_loss(self, experience_batch, policy_output):
    #     q_est = torch.min(*[critic(experience_batch.observation, self._prep_action(policy_output.action)).q
    #                                   for critic in self.critics])
    #     policy_loss = -1 * q_est + self.alpha * policy_output.prior_divergence[:, None]
    #     check_shape(policy_loss, [self._hp.batch_size, 1])
    #     return policy_loss.mean()

    # ================================ hl critic ================================
    def _compute_hl_q_target(self, experience_batch, policy_output): 
        # Qz(s,z,k) = Qa(s,z,k,a) - alph * DKL(PIa)
        with torch.no_grad():
            qa_target = torch.min(*[critic_target(experience_batch.observation_next, self._prep_action(policy_output.action)).q
                                for critic_target in self.critic_targets])
            hl_q_target = qa_target - self.alpha * policy_output.prior_divergence[:, None]
            hl_q_target = hl_q_target.detach()
            check_shape(hl_q_target, [self._hp.batch_size])

        return hl_q_target

    def _compute_hl_critic_loss(self, experience_batch, hl_q_target): 
        # Qz(s,z,k), the input is only obs, not action(a) here
        hl_qs = [critic(experience_batch.observation).q.squeeze(-1) for critic in self.hl_critics]
        check_shape(hl_qs[0], [self._hp.batch_size])
        hl_critic_losses = [0.5 * (q - hl_q_target).pow(2).mean() for q in hl_qs] # mse loss
        return hl_critic_losses, hl_qs

    # ================================ ll critic ================================
    def _compute_ll_q_target(self, experience_batch):
        # Qa(s,z,k,a) = r + gamma * U_next
        # U_next = [1-beta] * Qz_next + [beta] * V_next
        # V_next = Qz_next - hl_alp * DKL(PIa_next)

        with torch.no_grad():
            policy_output_next = self._run_policy(experience_batch.observation_next)
            # (1) V_next
            v_next, q_next = self._compute_next_value(experience_batch, policy_output_next)

            # (2) U_next 
            beta_mask = self._compute_termination_mask(experience_batch.observation_next)
            u_next =  (1 - beta_mask) * q_next + beta_mask * v_next

            # (3) ll_q_target
            ll_q_target = experience_batch.reward * self._hp.reward_scale + (1 - experience_batch.done) * self._hp.discount_factor * u_next
            ll_q_target = ll_q_target.detach()
            check_shape(ll_q_target, [self._hp.batch_size])

        return ll_q_target

    def _compute_ll_critic_loss(self, experience_batch, ll_q_target):
        # Qa(s,z,k,a)
        ll_qs = [critic(experience_batch.observation, self._prep_action(experience_batch.action.detach())).q.squeeze(-1) for critic in self.critics]     # no gradient propagation into policy here!

        check_shape(ll_qs[0], [self._hp.batch_size])
        ll_critic_losses = [0.5 * (q - ll_q_target).pow(2).mean() for q in ll_qs]
        return ll_critic_losses, ll_qs

    def _compute_next_value(self, experience_batch, policy_output): 
        q_next = torch.min(*[critic_target(experience_batch.observation_next, self._prep_action(policy_output.action)).q
                             for critic_target in self.critic_targets])
        next_val = (q_next - self.alpha * policy_output.log_prob[:, None])
        check_shape(next_val, [self._hp.batch_size, 1])
        return next_val.squeeze(-1), q_next.squeeze(-1)

    def _compute_termination_mask(self, obs):
        # [1,0,0] is the start of a new hl step, so return 1 in the mask
        split_obs  = self._split_obs(obs) # to s,z,k
        beta_onehot = split_obs.k
        beta = parse_one_hot(beta_onehot)
        beta_mask = torch.where(beta==torch.tensor(0), 1, 0)
        return beta_mask

    # =============================== utils ===========================
    def _split_obs(self, obs):
        assert obs.shape[1] == self.state_dim + self.latent_dim + self.onehot_dim
        return AttrDict(
            s=obs[:, :self.state_dim],   # condition decoding on state
            z=obs[:, self.state_dim:-self.onehot_dim],
            k=obs[:, -self.onehot_dim:],
        )

    # ====================================== property =======================================
    @property
    def state_dim(self):
        return self._hp.ll_policy_params.ll_model_params.state_dim

    @property
    def latent_dim(self):
        return self._hp.ll_policy_params.ll_model_params.nz_vae

    @property
    def onehot_dim(self):
        return self._hp.ll_policy_params.ll_model_params.n_rollout_steps

    @property
    def action_dim(self):
        return self._hp.ll_policy_params.ll_model_params.action_dim