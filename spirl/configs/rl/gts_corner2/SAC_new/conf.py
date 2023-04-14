import os
import copy

from spirl.utils.general_utils import AttrDict
from spirl.rl.policies.mlp_policies import MLPPolicy
from spirl.rl.components.critic import MLPCritic
from spirl.rl.components.replay_buffer import UniformReplayBuffer

from spirl.rl.envs.gts_corner2.gts_corner2_single import GTSEnv_Corner2_Single
from spirl.rl.envs.env_list import EnvList

# from spirl.rl.agents.ac_agent import SACAgent
from spirl.data.gts.src.gts_agent import GTSSACAgent
from spirl.rl.components.sampler_wrap import SamplerWrapped
from spirl.rl.components.sampler_batched import AgentDetached_SampleBatched


from spirl.rl.components.normalization import Normalizer
from spirl.configs.default_data_configs.gts import data_spec
from spirl.utils.gts_utils import  corner2_done_function, corner2_spare_reward_function


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'non-hierarchical RL experiments in gts env'




configuration = {
    'seed': 2,
    'agent': GTSSACAgent,
    
    'data_dir': '.',
    'num_epochs': 2000,
    'max_rollout_len': 200,
    # 'n_steps_per_epoch': 1000 ,
    # 'n_warmup_steps': 5000 ,
    # 'use_update_after_sampling':True,

    'environment': EnvList,
    'sampler':SamplerWrapped,

    # 'n_steps_per_epoch': 1000 ,
    # 'n_steps_per_update': 200 ,
    # 'n_warmup_steps': 1000 ,

    'log_output_interval': 400,
    'log_image_interval' : 400,
    
    # debug
    'n_steps_per_epoch': 800 ,
    'n_steps_per_update': 400 ,
    'n_warmup_steps': 400 ,
}

configuration = AttrDict(configuration)


# Environment
sub_env_config = AttrDict(
    reward_norm=1.,
    # do_init = False,
    reward_function = corner2_spare_reward_function,
    done_function = corner2_done_function,
    initial_velocity = 65*3.6, 
)

ip_address_list = ['192.168.1.104', '192.168.1.106']

sub_env_config_list = []
for ip in ip_address_list:
    sub_env_config.update({'ip_address': ip})
    sub_env_config_list.append(copy.deepcopy(sub_env_config))

env_config = AttrDict(
    env_class = GTSEnv_Corner2_Single,
    sub_env_configs = sub_env_config_list
)

# sampler
sampler_config = AttrDict(
    sub_sampler = AgentDetached_SampleBatched,
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
    policy_params=policy_params,
    critic=MLPCritic,
    critic_params=critic_params,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    # obs_normalizer=Normalizer,
    # obs_normalizer_params=obs_norm_params,
    clip_q_target=False,
    batch_size=256,
    log_videos=False,

    discount_factor = 0.98,
    fixed_alpha = 0.01,
    update_iterations = 64,

    # target_entropy = 0,
    # visualize_values = True,
    
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec


