import imp
from spirl.rl.components.agent import HierarchicalAgent, FixedIntervalHierarchicalAgent, FixedIntervalTimeIndexedHierarchicalAgent

from spirl.utils.general_utils import ParamDict, get_clipped_optimizer, AttrDict, prefix_dict, map_dict

from enum import Enum
from tqdm import tqdm

class skill_critic_stages(Enum):
    WARM_START = 0
    HL_TRAIN = 1
    LL_TRAIN = 2
    HYBRID = 3

    LL_TRAIN_PI = 4
    SC_WO_LLVAR = 5
    FIX_LL_PI = 6

class JointInheritAgent(FixedIntervalTimeIndexedHierarchicalAgent):
    def __init__(self, config):
        super().__init__(config)
        self.ll_agent.update_by_hl_agent(self.hl_agent)
        self._train_stage = None

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
        print('!! Change Skill-critic stage to ', stage)

        if stage == skill_critic_stages.WARM_START:
        # 1) warm-start stage
            # policy: HL var, LL var
            # update: HL Q, LL Q (to convergence)
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([False, True,])
            self.ll_agent.fast_assign_flags([False, True])

        elif stage == skill_critic_stages.HL_TRAIN:
        # 2) HL training stage:
            # policy: HL var, LL determine
            # update: HL Q, LL Q, HL Pi
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_on_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True, True])
            self.ll_agent.fast_assign_flags([False, False])

        elif stage == skill_critic_stages.LL_TRAIN:
        # 3) LL training stage:
            # policy: HL var, LL var
            # update: HL Q, LL Q, LL Pi
            self.hl_agent.switch_on_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([False, False])
            self.ll_agent.fast_assign_flags([True, True])

        elif stage == skill_critic_stages.HYBRID:
        # 4) hybrid stage
            # policy: all var
            # update: all
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True, True])
            self.ll_agent.fast_assign_flags([True, True])

        elif stage == skill_critic_stages.FIX_LL_PI:
        # 5) only train LL policy, without LL variance
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True, True])
            self.ll_agent.fast_assign_flags([False, True])
            
        elif stage == skill_critic_stages.SC_WO_LLVAR:
        # 6) only train LL policy, without LL variance
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_on_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True, True])
            self.ll_agent.fast_assign_flags([True, True])

        else:
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True, True])
            self.ll_agent.fast_assign_flags([True, True])

    '''
    # same as the parent class
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
        """Updates high-level and low-level agents depending on which parameters are set."""
        assert isinstance(experience_batches, AttrDict)  # update requires batches for both HL and LL
        update_outputs = AttrDict()

        # 1) add experience
        if self._hp.update_hl:
            self.hl_agent.add_experience(experience_batches.hl_batch)

        if self._hp.update_ll:
            self.ll_agent.add_experience(experience_batches.ll_batch)


        # 2) for and update HL, LL
        # for idx in tqdm(range(self._hp.update_iterations)):
        for idx in range(self._hp.update_iterations):
            vis = True if idx == self._hp.update_iterations -1 else False
            
            if self._hp.update_hl:
                hl_update_outputs = self.hl_agent.update()
                update_outputs.update(hl_update_outputs)

            if self._hp.update_ll:
                ll_update_outputs = self.ll_agent.update(vis=vis)
                update_outputs.update(ll_update_outputs)

        return update_outputs


    def offline(self):
        vis = True
        hl_experience_batch = self.hl_agent._sample_experience()
        ll_experience_batch = self.ll_agent._sample_experience()
        
        self.hl_agent.visualize_actions(hl_experience_batch)
        self.ll_agent.visualize_actions(ll_experience_batch)
        self.ll_agent.visualize_gradients(ll_experience_batch)
        
        hl_update_outputs = self.hl_agent.update()
        ll_update_outputs = self.ll_agent.update(vis=vis)
