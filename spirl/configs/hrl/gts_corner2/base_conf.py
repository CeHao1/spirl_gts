import os
import copy

from spirl.utils.general_utils import AttrDict
from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.policies.mlp_policies import MLPPolicy
from spirl.rl.components.critic import MLPCritic
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.skill_space_agent import SkillSpaceAgent
from spirl.models.skill_prior_mdl import SkillPriorMdl
from spirl.configs.default_data_configs.gts import data_spec

from spirl.rl.envs.gts_corner2.gts_corner2_single import GTSEnv_Corner2_Single
from spirl.rl.components.sampler_batched import HierarchicalSamplerBatched

from spirl.utils.gts_utils import  corner2_done_function, corner2_spare_reward_function

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'hierarchical RL on the gts env'

# Environment
env_config = AttrDict(
    reward_norm=1.,
    # do_init = False,
    # action_standard = True,

    reward_function = corner2_spare_reward_function,
    done_function = corner2_done_function,
    ip_address = '192.168.1.108',
)

configuration = AttrDict(    {
    'seed': 2,
    'agent': FixedIntervalHierarchicalAgent,
    
    'data_dir': '.',
    'num_epochs': 100,
    'max_rollout_len': 600,
    # 'n_steps_per_epoch': 10000,
    # 'n_steps_per_update': 1000,
    # 'n_warmup_steps': 160000,
    # 'use_update_after_sampling':True,

    'environment': GTSEnv_Corner2_Single,
    'sampler':HierarchicalSamplerBatched,
    
    'n_steps_per_epoch': 10000,
    'n_steps_per_update': 1200,
    'n_warmup_steps': 600,
    'log_output_interval': 600,
    'log_image_interval': 600,

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
    batch_size=256, #256,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
)


###### Low-Level ######
# LL Policy
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    nz_vae=10,
    n_rollout_steps=10,
)


# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    model=SkillPriorMdl,
    model_params=ll_model_params,
    model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                  "skill_prior_learning/gts_corner2/hierarchical"),
))

###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=1.,        # prior is Gaussian with unit variance
    nz_mid=256,
    n_layers=5,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=5,  # number of policy network layer
    nz_mid=256,
    action_input=True,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=MLPPolicy,
    policy_params=hl_policy_params,
    critic=MLPCritic,
    critic_params=hl_critic_params,
))


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=SACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SkillSpaceAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_video_caption=False,

    update_iterations = 1280,
    discount_factor = 0.98 ,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

