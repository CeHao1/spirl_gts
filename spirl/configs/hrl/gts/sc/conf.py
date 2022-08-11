

import os
import copy

from spirl.utils.general_utils import AttrDict
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.configs.default_data_configs.gts import data_spec


from spirl.rl.envs.gts_multi import GTSEnv_Multi
from spirl.rl.envs.gts_raw import GTSEnv_Raw
from spirl.rl.components.sampler_batched import HierarchicalSamplerBached


from spirl.rl.agents.skill_critic.joint_agent import JointAgent
from spirl.models.cond_dec_spirl_mdl import TimeIndexCDSPiRLMDL
from spirl.rl.policies.cd_model_policy import DecoderRegu_TimeIndexedCDMdlPolicy
from spirl.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from spirl.rl.components.critic import MLPCritic

from spirl.rl.agents.skill_critic.hl_skill_agent import HLSKillAgent
from spirl.rl.agents.skill_critic.ll_action_agent import LLActionAgent

# Environment
env_config = AttrDict(
    reward_norm=1.,
    do_init = False,
    # action_standard = True,

    # reward_function = eval_time_trial_reward_function,
    # done_function = eval_time_trial_done_function,
    
)


configuration = AttrDict(    {
    'seed': 2,
    'agent': JointAgent,
    
    'data_dir': '.',
    'num_epochs': 2000,
    'max_rollout_len': 10000,
    'n_steps_per_epoch': 10000,
    'n_warmup_steps': 160000,
    'use_update_after_sampling':True,

    'environment': GTSEnv_Raw,
    'sampler':HierarchicalSamplerBached,

    # 'n_steps_per_epoch': 200,
    # 'n_warmup_steps': 200,
} )

sampler_config = AttrDict(
    number_of_agents = 20,
)

# Replay Buffer
replay_params = AttrDict(
    dump_replay=True,
)

# Observation Normalization
obs_norm_params = AttrDict(
)

base_agent_params = AttrDict(
    batch_size=64, #256,
    # batch_size=4096, 
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

    
    manual_log_sigma=[-3, -2],
)
ll_policy_params.update(ll_model_params)

# critic
ll_critic_params = AttrDict(
    input_dim=data_spec.state_dim + ll_model_params.nz_vae + ll_model_params.n_rollout_steps,
    action_dim=data_spec.n_actions,
    output_dim=1,
    n_layers=5,  # number of policy network layer
    nz_mid=256,
    action_input=True,
)

# agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    policy=DecoderRegu_TimeIndexedCDMdlPolicy,
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

    squash_output_dist = False, # do not squash the tanh output
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
    hl_agent=HLSKillAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=LLActionAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,

    update_ll=True,
    log_video_caption=False,

    update_iterations = 1280,
    # update_iterations = 32,
    discount_factor = 0.98 ,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec
