import imp
from spirl.rl.components.agent import HierarchicalAgent, FixedIntervalHierarchicalAgent, FixedIntervalTimeIndexedHierarchicalAgent


class JointAgent(FixedIntervalTimeIndexedHierarchicalAgent):
    def __init__(self, config):
        super().__init__(config)
    #     # the HRL agent will initialize 

    # def post_process(self):
        # self.hl_agent.update_by_ll_agent(self.ll_agent)
        self.ll_agent.update_by_hl_agent(self.hl_agent)

    # def act(self, obs): # obs is numpy array
    #     pass

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