from spirl.configs.rl.gts_corner2.prior_initialized.base_conf import *
from spirl.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent

import copy

from spirl.rl.envs.gts_corner2.gts_corner2_single import GTSEnv_Corner2_Single
from spirl.rl.envs.env_list import EnvList

# from spirl.rl.agents.ac_agent import SACAgent
from spirl.data.gts.src.gts_agents import GTSSACAgent, GTSActionPriorSACAgent
from spirl.rl.components.sampler_wrap import SamplerWrapped
from spirl.rl.components.sampler_batched import AgentDetached_SampleBatched

agent_config.update(AttrDict(
    td_schedule_params=AttrDict(p=1.),
))

agent_config.policy = LearnedPriorAugmentedPIPolicy
configuration.agent = ActionPriorSACAgent

# ip_address_list = [
#     "192.168.1.125",
#     "192.168.1.117",
#     "192.168.1.121",
#     "192.168.1.120",
#     "192.168.1.123",
#     "192.168.1.119",
#     '192.168.1.115',]

ip_address_list = ["192.168.1.107"]

num_of_sampler = len(ip_address_list)

configuration = {
    'seed': 2,
    'agent': GTSActionPriorSACAgent,
    
    'data_dir': '.',
    'num_epochs': 2000,
    'max_rollout_len': 200,

    'environment': EnvList,
    'sampler':SamplerWrapped,

    'max_rollout_len': 600,
    'n_steps_per_epoch': 600*num_of_sampler*2,
    'n_steps_per_update': 600*num_of_sampler,
    'n_warmup_steps': 600*num_of_sampler*5,
    
    'log_output_interval': 600*num_of_sampler,
    'log_image_interval': 600*num_of_sampler,

}

configuration = AttrDict(configuration)


# Environment
sub_env_config = AttrDict(
    reward_norm=1.,
    do_init = False,
    reward_function = corner2_spare_reward_function,
    done_function = corner2_done_function,
    # initial_velocity = 65*3.6, 
    initial_velocity = 10*3.6,
)



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
