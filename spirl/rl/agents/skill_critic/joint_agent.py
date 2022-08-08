import imp
from spirl.rl.components.agent import HierarchicalAgent, FixedIntervalHierarchicalAgent, FixedIntervalTimeIndexedHierarchicalAgent

from spirl.utils.general_utils import ParamDict, get_clipped_optimizer, AttrDict, prefix_dict, map_dict

class JointAgent(FixedIntervalTimeIndexedHierarchicalAgent):
    def __init__(self, config):
        super().__init__(config)
        self.ll_agent.update_by_hl_agent(self.hl_agent)

    '''
    We should define some different modes, such as \
    1) Full training,
    2) Only train Q, for something
    3) Use deterministic policy

    '''
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

    # def update(self, experience_batches):
    #     '''
    #     assert isinstance(experience_batches, AttrDict)  # update requires batches for both HL and LL
    #     update_outputs = AttrDict()
    #     if self._hp.update_hl:
    #         print('updating hl agent', type(self.hl_agent))
    #         hl_update_outputs = self.hl_agent.update(experience_batches.hl_batch)
    #         update_outputs.update(prefix_dict(hl_update_outputs, "hl_"))
    #     if self._hp.update_ll:
    #         print('updating ll agent', type(self.ll_agent))
    #         ll_update_outputs = self.ll_agent.update(experience_batches.ll_batch)
    #         update_outputs.update(ll_update_outputs)
    #     return update_outputs
    #     '''

    #     # 