

# this agent should be similar to the state-conditioned close-loop agent
from spirl.rl.agents.ac_agent import SACAgent

class LLActionAgent(SACAgent):
    

    def update_by_hl_agent(self, hl_agent):
        self.hl_agent = hl_agent

