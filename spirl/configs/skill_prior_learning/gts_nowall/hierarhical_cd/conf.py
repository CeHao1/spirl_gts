
import os

from spirl.configs.skill_prior_learning.gts.hierarchical_cl.conf import * 
from spirl.models.cond_dec_spirl_mdl import CDSPiRLMdl, TimeIndexCDSPiRLMDL

from spirl.components.logger import Logger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.gts import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    # 'model': CDSPiRLMdl,
    'model': TimeIndexCDSPiRLMDL,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'gts'),
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',

    'batch_size':128,
}
configuration = AttrDict(configuration)


weights_01 = [100, 1e-4] # -> err [2e-2, 2e-3, 1e-3] -> vis (prior too bad)
model_weight = weights_01

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    cond_decode=True,

    n_rollout_steps = 10,
    nz_vae = 10,

    action_dim_weights = [100.0, 1.0],
    
    reconstruction_mse_weight = model_weight[0],
    kl_div_weight=model_weight[1],

    # nll_prior_train = False,
)