
from spirl.configs.hrl.gts_corner2.sh.conf import *

from spirl.rl.envs.env_list import EnvList
from spirl.rl.components.sampler_wrap import HierarchicalSamplerWrapped
from spirl.rl.components.sampler_batched import AgentDetached_HierarchicalSamplerBatched

ip_address_list = \
                [   '192.168.1.103', 
                    '192.168.1.104',
                    '192.168.1.113', 
                    '192.168.1.117',
                    '192.168.1.121', 
                    '192.168.1.101']
                
num_of_sampler = len(ip_address_list)

configuration.update(AttrDict(
    environment=EnvList,
    sampler=HierarchicalSamplerWrapped,
    
    max_rollout_len = 600,
    n_steps_per_epoch= 600*num_of_sampler*2,
    n_steps_per_update= 600*num_of_sampler,
    n_warmup_steps = 600*num_of_sampler*5,
    # n_warmup_steps = 600*num_of_sampler*1,
    
    log_output_interval = 600*num_of_sampler,
    log_image_interval = 600*num_of_sampler,
))

# Environment
sub_env_config = AttrDict(
    reward_norm=1.,
    do_init = False,
    reward_function = corner2_spare_reward_function,
    done_function = corner2_done_function,
    initial_velocity = 65*3.6, 
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
    sub_sampler = AgentDetached_HierarchicalSamplerBatched,
)

agent_config.update_iterations = num_of_sampler * 128
initial_train_stage = skill_critic_stages.HL_TRAIN