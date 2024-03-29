import os

from spirl.utils.general_utils import AttrDict
from spirl.rl.policies.mlp_policies import MLPPolicy, TanhLogstd_MLPPolicy
from spirl.rl.components.critic import MLPCritic
from spirl.rl.components.replay_buffer import UniformReplayBuffer

from spirl.rl.envs.gts_corner2.gts_corner2_single import GTSEnv_Corner2_Single

from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.components.sampler_batched import SamplerBatched


from spirl.rl.components.normalization import Normalizer
from spirl.configs.default_data_configs.gts import data_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'non-hierarchical RL experiments in gts env'

# Environment
env_config = AttrDict(
    reward_norm=1.,
    # do_init = False,
    # action_standard = True,

    # reward_function = eval_time_trial_reward_function,
    # done_function = eval_time_trial_done_function,

    # num_cars = 1,
    # store_states = True,

)

configuration = {
    'seed': 2,
    'agent': SACAgent,
    
    'data_dir': '.',
    'num_epochs': 2000,
    'max_rollout_len': 10000,
    'n_steps_per_epoch': 10000 ,
    'n_warmup_steps': 10000 ,
    'use_update_after_sampling':True,

    'environment': GTSEnv_Corner2_Single,
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
    fixed_alpha = 0.1,
    update_iterations = 64 * 20,

    # target_entropy = 0,
    # visualize_values = True,
    
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec


