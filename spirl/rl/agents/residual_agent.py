from spirl.rl.agents.ac_agent import SACAgent

from collections import deque
import contextlib

import numpy as np
from spirl.utils.general_utils import ParamDict, split_along_axis, AttrDict
from spirl.utils.general_utils import ConstantSchedule
from spirl.utils.pytorch_utils import map2torch, map2np, no_batchnorm_update


# base is skill space agent, but combined with sac agent

class ResidualAgent(SACAgent):


    def __init__(self, config):
        super().__init__(config)
        self._update_model_params()     # transfer some parameters to model

        self._policy = self._hp.model(self._hp.model_params, logger=None)
        self.load_model_weights(self._policy, self._hp.model_checkpoint, self._hp.model_epoch)

        self.action_plan = deque()
        self._action_damp = self._hp.damp_schedule(self._hp.damp_schedule_params)

    def _default_hparams(self):
        default_dict = ParamDict({
            'model': None,              # policy class
            'model_params': None,       # parameters for the policy class
            'model_checkpoint': None,   # checkpoint path of the model
            'model_epoch': 'latest',    # epoch that checkpoint should be loaded for (defaults to latest)

            'damp_schedule': ConstantSchedule,      # dampening schedule for residual action
            'damp_schedule_params': ParamDict(p=1),    # parameters for dampening schedule

        })
        return super()._default_hparams().overwrite(default_dict)
    
    def update(self, experience_batch):
        info =  super().update(experience_batch)
        info.action_damp = self._action_damp(self.schedule_steps)
        return info

    def _base_policy_act(self, obs):
        assert len(obs.shape) == 2 and obs.shape[0] == 1  # assume single-observation batches with leading 1-dim
        if not self.action_plan:
            # generate action plan if the current one is empty
            split_obs = self._split_obs(obs)
            with no_batchnorm_update(self._policy) if obs.shape[0] == 1 else contextlib.suppress():
                actions = self._policy.decode(map2torch(split_obs.z, self._hp.device),
                                              map2torch(split_obs.cond_input, self._hp.device),
                                              self._policy.n_rollout_steps)
            self.action_plan = deque(split_along_axis(map2np(actions), axis=1))
        return AttrDict(action=self.action_plan.popleft())

    def _act(self, obs):

        # part 1, base policy
        base_policy_output = self._base_policy_act(obs)

        # part 2, sac policy
        sac_policy_output = super()._act(obs)

        # return combined action
        sac_policy_output.action = base_policy_output.action  + self._action_damp(self.schedule_steps) * sac_policy_output.action
        return sac_policy_output
    
    def reset(self):
        super().reset()
        self.action_plan = deque()      # reset action plan


    def _split_obs(self, obs):
        assert obs.shape[1] == self._policy.state_dim + self._policy.latent_dim
        return AttrDict(
            cond_input=obs[:, :-self._policy.latent_dim],   # condition decoding on state
            z=obs[:, -self._policy.latent_dim:],
        )

    def _update_model_params(self):
        self._hp.model_params.device = self._hp.device  # transfer device to low-level model
        self._hp.model_params.batch_size = 1            # run only single-element batches

    def _act_rand(self, obs):
        # part 1, base policy
        base_action = self._base_policy_act(obs)

        # part 2, sac policy
        policy_output = super()._act_rand(obs)
    
        # return combined action
        policy_output.action = base_action.action + self._action_damp(self.schedule_steps) *  policy_output.action
        return policy_output
    

    

class ACResidualAgent(ResidualAgent):
    """Unflattens prior input part of observation."""
    def _split_obs(self, obs):
        unflattened_obs = map2np(self._policy.unflatten_obs(
            map2torch(obs[:, :-self._policy.latent_dim], device=self.device)))
        return AttrDict(
            cond_input=unflattened_obs.prior_obs,
            z=obs[:, -self._policy.latent_dim:],
        )
