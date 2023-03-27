import os
import copy

from spirl.utils.general_utils import AttrDict
from spirl.rl.agents.skill_critic.joint_agent import JointAgent, skill_critic_stages
from spirl.rl.components.critic import SplitObsMLPCritic, MLPCritic
from spirl.rl.components.sampler import ACMultiImageAugmentedHierarchicalSampler, TrainAfter_ACMultiImageAugmentedHierarchicalSampler
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from spirl.rl.envs.maze import ACRandMaze0S40Env
from spirl.rl.agents.skill_critic.ll_action_agent import MazeLLActionAgent
from spirl.rl.policies.cd_model_policy import AC_DecoderRegu_TimeIndexedCDMdlPolicy
from spirl.data.maze.src.maze_agents import MazeHLSkillAgent
from spirl.models.cond_dec_spirl_mdl import ImageTimeIndexCDSPiRLMDL
from spirl.configs.default_data_configs.maze import data_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'skill critic on the maze env'

configuration = {
    'seed': 42,
    'agent': JointAgent,
    'environment': ACRandMaze0S40Env,
    # 'sampler': TrainAfter_ACMultiImageAugmentedHierarchicalSampler,
    'sampler': ACMultiImageAugmentedHierarchicalSampler,
    'data_dir': '.',
    # "use_update_after_sampling": True,
    'num_epochs': 500,
    'max_rollout_len': 2000,
    'n_steps_per_epoch': 1e5,
    'n_warmup_steps': 5e3,

    # 'n_steps_per_update': 2e3, # hl step
    # 'n_steps_per_epoch': 1e5,  # global step
    # 'log_output_per_epoch' : 1e5 / 2e3,

    # 'n_warmup_steps': 500,
    # 'log_image_interval' : 2000,


    # 'n_steps_per_update': 5e2, # hl step
    # 'n_steps_per_epoch': 1e4,  # global step
    # 'log_output_per_epoch' : 1e4 / 5e2,


    # 'n_warmup_steps': 500,
    # 'n_steps_per_update': 2000, # hl step
    # 'n_steps_per_epoch': 1e5,  # global step
    # 'log_output_per_epoch' : 1e4 / 200,
}
configuration = AttrDict(configuration)

# Replay Buffer
replay_params = AttrDict(
    capacity=1e6,
    dump_replay=False,
)

# Observation Normalization
obs_norm_params = AttrDict(
)

sampler_config = AttrDict(
    n_frames=2,
)

base_agent_params = AttrDict(
    batch_size=256,
    # batch_size=1024,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
)

######================= Low-Level ===============######
# LL Policy Model
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    nz_vae = 10,
    n_rollout_steps=10,
    kl_div_weight=1e-2,
    prior_input_res=data_spec.res,
    n_input_frames=2,
    cond_decode=True,
)

# LL Policy
ll_policy_params = AttrDict(
    policy_model=ImageTimeIndexCDSPiRLMDL, 
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/maze_h/hierarchical_cd"),
    # initial_log_sigma=-50.,

    manual_log_sigma=[0, 0],
)
ll_policy_params.update(ll_model_params)

# LL Critic
ll_critic_params = AttrDict(
    action_dim=data_spec.n_actions,
    # input_dim=data_spec.state_dim + ll_model_params.nz_vae + ll_model_params.n_rollout_steps,
    input_dim=data_spec.state_dim + ll_model_params.nz_vae + ll_model_params.n_rollout_steps,
    output_dim=1,
    action_input=True,
    
    discard_part = 'mid', # obs = (s+z+t) + a //remove image in the middle
    unused_obs_start = data_spec.state_dim,
    unused_obs_size=ll_model_params.prior_input_res **2 * 3 * ll_model_params.n_input_frames,

)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    policy=AC_DecoderRegu_TimeIndexedCDMdlPolicy, 
    policy_params=ll_policy_params,
    critic=SplitObsMLPCritic,
    # obs(s + z + t) + a = 4 + 10 + 10 + 2
    critic_params=ll_critic_params,

    # td_schedule_params=AttrDict(p=1.),
    fixed_alpha = 0.001,
    # discount_factor=1,

    # visualize_values = True,
))

######=============== High-Level ===============########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,        # prior is Gaussian with unit variance
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,

    # squash_output_dist = False,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim, # QHL(s, z), no K
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=2,  # number of policy network layers
    nz_mid=256,
    action_input=True,
    unused_obs_size=ll_model_params.prior_input_res **2 * 3 * ll_model_params.n_input_frames,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=ACLearnedPriorAugmentedPIPolicy,
    policy_params=hl_policy_params,
    critic=SplitObsMLPCritic,
    # critic=MLPCritic,
    critic_params=hl_critic_params,
    td_schedule_params=AttrDict(p=1.),

    # fixed_alpha = 0.1,

    # visualize_values = True,
))

#####========== Joint Agent =======#######
agent_config = AttrDict(
    hl_agent=MazeHLSkillAgent, 
    hl_agent_params=hl_agent_config,
    ll_agent=MazeLLActionAgent,  
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_videos=False,
    update_hl=True,
    update_ll=True,
    
    # update_iterations = 256,
    initial_train_stage = skill_critic_stages.HL_TRAIN
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=0.05,
    screen_height=ll_model_params.prior_input_res,
    screen_width=ll_model_params.prior_input_res,
)