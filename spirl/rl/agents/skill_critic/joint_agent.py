import imp
from spirl.rl.components.agent import HierarchicalAgent, FixedIntervalHierarchicalAgent, FixedIntervalTimeIndexedHierarchicalAgent


class JointAgent(HierarchicalAgent):
    def __init__(self, config):
        super().__init__(config)
        # the HRL agent will initialize 

        self.hl_agent.update_by_ll_agent(self.ll_agent)
        self.ll_agent.update_by_hl_agent(self.hl_agent)

    def act(self, obs): # obs is numpy array
        pass

    def update(self, experience_batches):
        pass