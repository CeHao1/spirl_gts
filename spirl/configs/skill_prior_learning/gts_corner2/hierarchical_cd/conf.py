
import os

from spirl.models.cond_dec_spirl_mdl import TimeIndexCDSPiRLMDL

from spirl.components.logger import Logger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.gts import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    'model': TimeIndexCDSPiRLMDL,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'gts_corner2'),
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',

    'batch_size':128,
}
configuration = AttrDict(configuration)


weights_01 = [100, 5e-4, 1e-3] # -> err [2e-2, 2e-3, 1e-3] -> vis (prior too bad)
weights_02 = [100, 5e-4, 1e-2] 
weights_03 = [1, 5e-2, 2e-1] 
model_weight = weights_03

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
    learned_prior_weight = model_weight[2],

    nll_prior_train = False,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1