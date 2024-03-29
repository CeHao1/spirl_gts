


import os
import copy

from spirl.utils.general_utils import AttrDict
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.configs.default_data_configs.gts import data_spec


from spirl.rl.envs.gts_multi import GTSEnv_Multi
from spirl.rl.components.sampler_batched import HierarchicalSamplerBached

from spirl.rl.components.critic import MLPCritic
from spirl.rl.components.agent import FixedIntervalTimeIndexedHierarchicalAgent
from spirl.models.cond_dec_spirl_mdl import TimeIndexCDSPiRLMDL
from spirl.rl.policies.cd_model_policy import TimeIndexedCDMdlPolicy

from spirl.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent
from spirl.rl.agents.ac_agent import SACAgent


# overall implementation of CD
# Environment
env_config = AttrDict(
    reward_norm=1.,
    do_init = False,
    action_standard = True,
)


configuration = AttrDict(    {
    'seed': 2,
    'agent': FixedIntervalTimeIndexedHierarchicalAgent,
    
    'data_dir': '.',
    'num_epochs': 300,
    'max_rollout_len': 10000,
    'n_steps_per_epoch': 200,
    'n_warmup_steps': 100,
    'use_update_after_sampling':True,

    'environment': GTSEnv_Multi,
    'sampler':HierarchicalSamplerBached,
} )

sampler_config = AttrDict(
    number_of_agents = 20,
)

# Replay Buffer
replay_params = AttrDict(
    dump_replay=False,
)

# Observation Normalization
obs_norm_params = AttrDict(
)

base_agent_params = AttrDict(
    # batch_size=256, #256,
    batch_size=64, 
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
)


# ================= low level ====================

# model
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    nz_vae = 6,
    n_rollout_steps=4,

    cond_decode = True,
)

# policy
ll_policy_params = AttrDict(
    policy_model = TimeIndexCDSPiRLMDL,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/gts/hierarchical_cd"),

    
    manual_log_sigma=[1e-10, 1e-12],
)
ll_policy_params.update(ll_model_params)

# critic
ll_critic_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim + ll_model_params.nz_vae + ll_model_params.n_rollout_steps,
    output_dim=1,
    n_layers=5,  # number of policy network layer
    nz_mid=256,
    action_input=True,
)

# agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    policy=TimeIndexedCDMdlPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,                
    critic_params=ll_critic_params
))

# ================= high level ====================

# model, same as ll

# policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=1.,        # prior is Gaussian with unit variance
    nz_mid=256,
    n_layers=5,

    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
)

# critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim + hl_policy_params.prior_model_params.n_rollout_steps,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=5,  # number of policy network layer
    nz_mid=256,
    action_input=True,
)

# agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=LearnedPriorAugmentedPIPolicy,
    policy_params=hl_policy_params,
    critic=MLPCritic,
    critic_params=hl_critic_params,

    td_schedule_params=AttrDict(p=5.),
))
# ================== joint agent ===================
agent_config = AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,

    update_ll=True,
    log_video_caption=False,

    update_iterations = 512,
    discount_factor = 0.98 ,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec


'''

configuration.agent = FixedIntervalTimeIndexedHierarchicalAgent

ll_model_params.cond_decode = True
# create LL conditioned decoder policy
ll_policy_params = AttrDict(
    # policy_model=CDSPiRLMdl,
    policy_model = TimeIndexCDSPiRLMDL,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/gts/hierarchical_cd"),

    manual_log_sigma=[1e-10, 1e-12],
)
ll_policy_params.update(ll_model_params)

ll_critic_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim + ll_model_params.nz_vae + ll_model_params.n_rollout_steps,
    output_dim=1,
    n_layers=5,  # number of policy network layer
    nz_mid=256,
    action_input=True,
    # unused_obs_size = ll_model_params.nz_vae + ll_model_params.n_rollout_steps, # whether remove latent variable, or add it
)

ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    policy=TimeIndexedCDMdlPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,                   
    critic_params=ll_critic_params
))


# update HL policy model params
hl_policy_params.update(AttrDict(
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=True,
))

'''