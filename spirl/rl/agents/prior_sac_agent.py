import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from spirl.rl.agents.ac_agent import SACAgent
from spirl.utils.general_utils import ParamDict, ConstantSchedule, AttrDict
from spirl.utils.pytorch_utils import check_shape, map2torch

class ActionPriorSACAgent(SACAgent):
    """Implements SAC with non-uniform, learned action / skill prior."""
    def __init__(self, config):
        SACAgent.__init__(self, config)
        self._target_divergence = self._hp.td_schedule(self._hp.td_schedule_params)

    def _default_hparams(self):
        default_dict = ParamDict({
            'alpha_min': None,                # minimum value alpha is clipped to, no clipping if None
            'td_schedule': ConstantSchedule,  # schedule used for target divergence param
            'td_schedule_params': AttrDict(   # parameters for target divergence schedule
                p = 1.,
            ),
        })
        return super()._default_hparams().overwrite(default_dict)

    def update(self, experience_batch):
        info = super().update(experience_batch)
        info.target_divergence = self._target_divergence(self.schedule_steps)
        return info

    def _compute_alpha_loss(self, policy_output):
        """Computes loss for alpha update based on target divergence."""
        return self.alpha * (self._target_divergence(self.schedule_steps) - policy_output.prior_divergence).detach().mean()

    def _compute_policy_loss(self, experience_batch, policy_output):
        """Computes loss for policy update."""
        q_est = torch.min(*[critic(experience_batch.observation, self._prep_action(policy_output.action)).q
                                      for critic in self.critics])
        policy_loss = -1 * q_est + self.alpha * policy_output.prior_divergence[:, None]
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean()

    def _compute_next_value(self, experience_batch, policy_output):
        """Computes value of next state for target value computation."""
        q_next = torch.min(*[critic_target(experience_batch.observation_next, self._prep_action(policy_output.action)).q
                             for critic_target in self.critic_targets])
        next_val = (q_next - self.alpha * policy_output.prior_divergence[:, None])
        check_shape(next_val, [self._hp.batch_size, 1])
        return next_val.squeeze(-1)

    def _aux_info(self, experience_batch, policy_output):
        """Stores any additional values that should get logged to WandB."""
        aux_info = super()._aux_info(experience_batch, policy_output)
        aux_info.prior_divergence = policy_output.prior_divergence.mean()
        if 'ensemble_divergence' in policy_output:      # when using ensemble thresholded prior divergence
            aux_info.ensemble_divergence = policy_output.ensemble_divergence.mean()
            aux_info.learned_prior_divergence = policy_output.learned_prior_divergence.mean()
            aux_info.below_ensemble_div_thresh = policy_output.below_ensemble_div_thresh.mean()
        return aux_info

    def state_dict(self, *args, **kwargs):
        d = super().state_dict(*args, **kwargs)
        d['update_steps'] = self._update_steps
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        self._update_steps = state_dict.pop('update_steps')
        super().load_state_dict(state_dict, *args, **kwargs)

    @property
    def alpha(self):
        if self._hp.alpha_min is not None:
            return torch.clamp(super().alpha, min=self._hp.alpha_min)
        return super().alpha
    
    # ========== vis maze hl q value ==========
    def _vis_hl_q(self, logger, step):
        self._vis_q(logger, step, prefix='hl')
    
    def _vis_q(self, logger, step, prefix='hl', plot_type='maze',
               content=['q', 'KLD', 'policy_v', 'rew', 
                        'action', 'action_nosquash', 'action_recent', 'action_nosquash_recent']):
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
        action_sum = []
        action_nosquash_sum = []

        for i in range(batch_num):
            obs_batch = obs[i*batch_size:(i+1)*batch_size]
            obs_batch = map2torch(obs_batch, self._hp.device)
            policy_output = self._run_policy(obs_batch)

            with self.no_squash_mode():
                policy_output_no_squash = self._run_policy(obs_batch)
                

            act = self._prep_action(policy_output.action) # QHL(s, z), no K
            q_est = torch.min(*[critic(obs_batch, act).q for critic in self.critics])
            policy_v = q_est - self.alpha * policy_output.prior_divergence[:, None]

            q_est_sum.append(q_est.detach().cpu().numpy())
            KLD_sum.append(policy_output.prior_divergence.detach().cpu().numpy())
            policy_v_sum.append(policy_v.detach().cpu().numpy())
            action_sum.append(policy_output.action.detach().cpu().numpy())
            action_nosquash_sum.append(policy_output_no_squash.action.detach().cpu().numpy())

        q_est = np.concatenate(q_est_sum, axis=0)
        KLD = np.concatenate(KLD_sum, axis=0)
        policy_v = np.concatenate(policy_v_sum, axis=0)
        action_sum = np.concatenate(action_sum, axis=0)
        action_nosquash_sum = np.concatenate(action_nosquash_sum, axis=0)
        
        
        if plot_type == 'maze':
            from spirl.data.maze.src.maze_agents import plot_maze_value
            if 'q' in content:
                plot_maze_value(q_est, states, logger, step, size, fig_name= prefix+'_q')
            if 'KLD' in content:
                plot_maze_value(KLD, states, logger, step, size, fig_name= prefix+'_policy KLD')
            if 'policy_v' in content:
                plot_maze_value(policy_v, states, logger, step, size, fig_name= prefix+'_policy_v')
            if 'rew' in content:
                plot_maze_value(rew, states, logger, step, size, fig_name= prefix+'_rew')
                
        elif plot_type == 'gts':
            pass
        
        if 'action' in content:
            plot_action_dist(action_sum, logger, step, size, 
                         fig_name=prefix+'_vis squash,action')
        if 'action_nosquash' in content:
            plot_action_dist(action_sum, logger, step, size=int(1e4), 
                         fig_name=prefix+'_vis squash, recent 10k,action')
        if 'action_recent' in content:
            plot_action_dist(action_nosquash_sum, logger, step, size, 
                         fig_name=prefix+'_vis nosquash, action', xlim=[-4.5, 4.5])
        if 'action_nosquash_recent' in content:
            plot_action_dist(action_nosquash_sum, logger, step, size=int(1e4), 
                         fig_name=prefix+'_vis nosquash, recent 10k, action', xlim=[-4.5, 4.5])


def plot_action_dist(action, logger, step, size, fig_name='vis action', bw=0.5, xlim=None):
    fs = 16
    fig = plt.figure(figsize=(10, 5))
    fig.tight_layout()
    for idx in range(len(action[0])):
        sns.kdeplot(action[-size:, idx], fill=True, label='dim_' + str(idx), cut=0, bw_adjust=bw)
    # plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=fs)
    plt.legend(loc='upper left', fontsize=14, framealpha=0.5)
    plt.ylabel('Density', fontsize=fs)
    plt.title('distribution of latent variables, size ' + str(size), fontsize=fs)
    plt.grid()
    if xlim is not None:
        plt.xlim(xlim)
    logger.log_plot(fig, name= fig_name, step=step)
    plt.close(fig)

class RandActScheduledActionPriorSACAgent(ActionPriorSACAgent):
    """Adds scheduled call to random action (aka prior execution) -> used if downstream policy trained from scratch."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._omega = self._hp.omega_schedule(self._hp.omega_schedule_params)

    def _default_hparams(self):
        default_dict = ParamDict({
            'omega_schedule': ConstantSchedule,  # schedule used for omega param
            'omega_schedule_params': AttrDict(   # parameters for omega schedule
                p = 0.1,
            ),
        })
        return super()._default_hparams().overwrite(default_dict)

    def _act(self, obs):
        """Call random action (aka prior policy) omega percent of times."""
        if np.random.rand() <= self._omega(self._update_steps):
            return super()._act_rand(obs)
        else:
            return super()._act(obs)

    def update(self, experience_batch):
        if 'delay' in self._hp.omega_schedule_params and self._update_steps < self._hp.omega_schedule_params.delay:
            # if schedule has warmup phase in which *only* prior is sampled, train policy to minimize divergence
            self.replay_buffer.append(experience_batch)
            experience_batch = self.replay_buffer.sample(n_samples=self._hp.batch_size)
            experience_batch = map2torch(experience_batch, self._hp.device)
            policy_output = self._run_policy(experience_batch.observation)
            policy_loss = policy_output.prior_divergence.mean()
            self._perform_update(policy_loss, self.policy_opt, self.policy)
            self._update_steps += 1
            info = AttrDict(prior_divergence=policy_output.prior_divergence.mean())
        else:
            info = super().update(experience_batch)
        info.omega = self._omega(self._update_steps)
        return info

