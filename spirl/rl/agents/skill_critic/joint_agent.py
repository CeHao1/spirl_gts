import imp
from spirl.rl.components.agent import HierarchicalAgent, FixedIntervalHierarchicalAgent, FixedIntervalTimeIndexedHierarchicalAgent

from spirl.utils.general_utils import ParamDict, get_clipped_optimizer, AttrDict, prefix_dict, map_dict

from enum import Enum

class skill_critic_stages(Enum):
    WARM_START = 0
    HL_TRAIN = 1
    LL_TRAIN = 2
    HYBRID = 3


class JointAgent(FixedIntervalTimeIndexedHierarchicalAgent):
    def __init__(self, config):
        super().__init__(config)
        self.ll_agent.update_by_hl_agent(self.hl_agent)
        self._train_stage = None
        '''
        We should define some different modes, such as \
        1) Full training,
        2) Only train Q, for something
        3) Use deterministic policy

        '''
        # update the trianing stage
        if self._train_stage is None:
            self._train_stage = self._hp.initial_train_stage

        self.train_stages_control(self._train_stage)

    def _default_hparams(self):
        default_dict = ParamDict({
            'initial_train_stage': skill_critic_stages.HYBRID,
        })
        return super()._default_hparams().overwrite(default_dict)

    def train_stages_control(self, stage=None):
        print('!! Change SC stage to ', stage)

        if stage == skill_critic_stages.WARM_START:
        # 1) warm-start stage
            # policy: HL var, LL var
            # update: HL Q, LL Q (to convergence)
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([False])
            self.ll_agent.fast_assign_flags([False, True, True])

        elif stage == skill_critic_stages.HL_TRAIN:
        # 2) HL training stage:
            # policy: HL var, LL determine
            # update: HL Q, LL Q, HL Pi
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_on_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True])
            self.ll_agent.fast_assign_flags([False, True, True])

        elif stage == skill_critic_stages.LL_TRAIN:
        # 3) LL training stage:
            # policy: HL var, LL var
            # update: HL Q, LL Q, LL Pi
            self.hl_agent.switch_on_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([False])
            self.ll_agent.fast_assign_flags([True, True, True])

        elif stage == skill_critic_stages.HYBRID:
        # 4) hybrid stage
            # policy: all var
            # update: all
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True])
            self.ll_agent.fast_assign_flags([True, True, True])

        else:
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True])
            self.ll_agent.fast_assign_flags([True, True, True])

    '''
    # 
    def act(self, obs): # obs is numpy array
        """Output dict contains is_hl_step in case high-level action was performed during this action."""
        obs_input = obs[None] if len(obs.shape) == 1 else obs    # need batch input for agents
        output = AttrDict()
        if self._perform_hl_step_now:
            # perform step with high-level policy
            self._last_hl_output = self.hl_agent.act(obs_input)
            output.is_hl_step = True
            if len(obs_input.shape) == 2 and len(self._last_hl_output.action.shape) == 1:
                self._last_hl_output.action = self._last_hl_output.action[None]  # add batch dim if necessary
                self._last_hl_output.log_prob = self._last_hl_output.log_prob[None]
        else:
            output.is_hl_step = False
        output.update(prefix_dict(self._last_hl_output, 'hl_'))

        # perform step with low-level policy
        assert self._last_hl_output is not None
        output.update(self.ll_agent.act(self.make_ll_obs(obs_input, self._last_hl_output.action)))

        return self._remove_batch(output) if len(obs.shape) == 1 else output
    '''


    # ====================== for update, we have some stages ====================

    def update(self, experience_batches):

        return super().update(experience_batches)

