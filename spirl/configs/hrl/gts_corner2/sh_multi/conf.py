
from spirl.configs.hrl.gts_corner2.sh.conf import *

from spirl.rl.envs.env_list import EnvList
from spirl.rl.components.sampler_wrap import HierarchicalSamplerWrapped
from spirl.rl.components.sampler_batched import AgentDetached_HierarchicalSamplerBatched

ip_address_list = \
                [ '192.168.1.105',
                 '192.168.1.100',
                 '192.168.1.106',
                 '192.168.1.110',
                 '192.168.1.107',
                 '192.168.1.101',
                 '192.168.1.108',
                 '192.168.1.109',
                 ]
                
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
    # reward_function = single_reward_function,
    done_function = corner2_done_function,

    initial_velocity = 10*3.6,
    # initial_velocity = 65*3.6, 
    # standardize_observations = True,
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
# agent_config.update_iterations = num_of_sampler * 64


agent_config.initial_train_stage = skill_critic_stages.HL_TRAIN
# agent_config.initial_train_stage = skill_critic_stages.HYBRID
# agent_config.initial_train_stage = skill_critic_stages.LL_TRAIN
# agent_config.initial_train_stage = skill_critic_stages.HL_LLVAR
ll_policy_params.manual_log_sigma = [-3.5, -2.5]
# ll_agent_config.fixed_alpha = 0.1
ll_agent_config.td_schedule_params=AttrDict(p=10.)

'''
exp()
-1: 0.36787944117144233
-1.5: 0.22313016014842982
-2: 0.1353352832366127
-2.5: 0.0820849986238988
-3: 0.049787068367863944
-3.5: 0.0301973834223185
-4: 0.01831563888873418

'''