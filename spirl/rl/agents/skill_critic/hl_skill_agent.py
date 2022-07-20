
import torch

from spirl.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np

# from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent

## this should be a skill prior agent + special critic update

class HLSKillAgent(ActionPriorSACAgent):
    def __init__(self, config):
        HLSKillAgent.__init__(self, config)

 
    def update_by_ll_agent(self, ll_agent):
        self.ll_agent = ll_agent







    