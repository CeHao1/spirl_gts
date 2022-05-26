import os

from spirl.utils.general_utils import AttrDict
from spirl.rl.policies.mlp_policies import MLPPolicy
from spirl.rl.components.critic import MLPCritic
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.envs.gts import GTSEnv_Base
from spirl.rl.envs.gts_multi import GTSEnv_Multi

from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.components.sampler import Sampler
from spirl.rl.components.sampler_multi import SamplerMulti

from spirl.rl.components.normalization import Normalizer
from spirl.configs.default_data_configs.gts import data_spec

from spirl.utils.gts_utils import reward_function, sampling_done_function
from spirl.utils.gts_utils import eval_time_trial_done_function, eval_time_trial_reward_function


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'non-hierarchical RL experiments in gts env'

# Environment
env_config = AttrDict(
    reward_norm=1.,
    # do_init = False,

    # reward_function = eval_time_trial_reward_function,
    # done_function = eval_time_trial_done_function,
)

configuration = {
    'seed': -1,
    'agent': SACAgent,
    
    'data_dir': '.',
    'num_epochs': 300,
    'max_rollout_len': 20000,
    'n_steps_per_epoch': 21000,
    'n_warmup_steps': 80000,
    'use_update_after_sampling':True,

    # 'environment': GTSEnv_Base,
    # 'sampler' : Sampler,
    
    'environment': GTSEnv_Multi,
    'sampler':SamplerMulti
}

configuration = AttrDict(configuration)

sampler_config = AttrDict(
    number_of_agents = 20,
)

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
    n_layers=2,      #  number of policy network layers
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
    update_iterations = 64 * 20
    
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec


