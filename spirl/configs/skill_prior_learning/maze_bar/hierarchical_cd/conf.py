
import os

from spirl.configs.skill_prior_learning.maze.hierarchical_cl.conf import * 
from spirl.models.cond_dec_spirl_mdl import ImageTimeIndexCDSPiRLMDL

from spirl.components.logger import Logger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.maze import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    # 'model': CDSPiRLMdl,
    'model': ImageTimeIndexCDSPiRLMDL,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'maze_bar'),
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',

    'batch_size':128,
}
configuration = AttrDict(configuration)

'''
model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=1e-2,
    prior_input_res=data_spec.res,
    n_input_frames=2,
    cond_decode=True,
)
'''
