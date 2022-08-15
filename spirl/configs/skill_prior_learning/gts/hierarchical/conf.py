import os

from spirl.models.skill_prior_mdl import SkillPriorMdl
from spirl.components.logger import Logger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.gts import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    'model': SkillPriorMdl,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'gts'),
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',

    'batch_size':128,
    # 'batch_size':1024,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    # n_rollout_steps=10,
    nz_enc=128,
    nz_mid=128,

    n_processing_layers=5,
    n_rollout_steps = 4,
    nz_vae = 6,


    #============== for vae training ==========
    reconstruction_mse_weight = 100.,
    # kl_div_weight=5e-4,
    # learned_prior_weight = 1e-10,
    action_dim_weights = [100.0, 1.0],
    # action_dim_weights = [1.0, 1.0],

    # learned_prior_weight = 1e-3,

    # ============= for prior training =========
    # reconstruction_mse_weight = 1e-10,
    kl_div_weight=1e-3,
    learned_prior_weight = 1e-1,

    nll_prior_train = False,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
