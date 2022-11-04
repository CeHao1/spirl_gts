import os

from spirl.utils.general_utils import AttrDict
from spirl.rl.policies.mlp_policies import MLPPolicy, TanhLogstd_MLPPolicy
from spirl.rl.components.critic import MLPCritic
from spirl.rl.components.replay_buffer import UniformReplayBuffer

from spirl.rl.envs.gts_corner2.gts_corner2_double import GTSEnv_Corner2_Double

from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.components.sampler_batched import SamplerBatched


from spirl.rl.components.normalization import Normalizer
from spirl.configs.default_data_configs.gts import data_spec
from spirl.utils.gts_utils import double_reward_function, corner2_done_function, single_reward_function


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'non-hierarchical RL experiments in gts env'

# Environment
env_config = AttrDict(
    reward_norm=1.,
    # do_init = False,
    action_standard = False,

    reward_function = double_reward_function,
    done_function = corner2_done_function,

    # num_cars = 3,
    # builtin_controlled = [0, 2],
    # store_states = False,
    # initial_velocity = [55*3.6, 65*3.6, 200],
    # initial_course_v = [1400, 1200, 1000],
    # bop = [[0.8, 1.2], [1, 1], [1,1]],
    
    # num_cars = 1,
    # builtin_controlled = [],
    # store_states = False,
    # initial_velocity = [55*3.6],
    # initial_course_v = [1400],
    # bop = [[1, 1]],
    
    
    num_cars = 2,
    builtin_controlled = [1],
    store_states = False,
    initial_velocity = [65*3.6, 55*3.6],
    initial_course_v = [1200, 1250],
    bop = [[1.0, 1.0], [1.0, 1.0]],
)   

configuration = {
    'seed': 2,
    'agent': SACAgent,
    
    'data_dir': '.',
    'num_epochs': 2000,
    'max_rollout_len': 1000,
    'n_steps_per_epoch': 1000 ,
    'n_warmup_steps': 5000 ,
    'use_update_after_sampling':True,

    'environment': GTSEnv_Corner2_Double,
    'sampler':SamplerBatched,

    # 'n_steps_per_epoch': 2000 ,
    # 'n_warmup_steps': 2000 ,
}

configuration = AttrDict(configuration)


# Policy
policy_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim,
    n_layers=2,      #  number of policy network layers
    nz_mid=256,
    max_action_range=1.,
)

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    input_dim=policy_params.input_dim,
    output_dim=1,
    n_layers=1,      #  number of critic network layers
    nz_mid=256,
    action_input=True,
)

# Replay Buffer
replay_params = AttrDict(
    capacity=4000000,
    dump_replay=True,
)

# Observation Normalization
obs_norm_params = AttrDict(
)

sampler_config = AttrDict(
    select_agent_id = [0]
)

# Agent
agent_config = AttrDict(
    policy=MLPPolicy,
    # policy=TanhLogstd_MLPPolicy,
    policy_params=policy_params,
    critic=MLPCritic,
    critic_params=critic_params,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    # obs_normalizer=Normalizer,
    # obs_normalizer_params=obs_norm_params,
    clip_q_target=False,
    batch_size=4096,
    log_videos=False,

    discount_factor = 0.98,
    fixed_alpha = 0.01,
    update_iterations = 64 * 20,

    # target_entropy = 0,
    visualize_values = True,
    
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec


