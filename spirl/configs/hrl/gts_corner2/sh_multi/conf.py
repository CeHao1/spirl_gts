
from spirl.configs.hrl.gts_corner2.sh.conf import *

from spirl.rl.envs.env_list import EnvList
from spirl.rl.components.sampler_wrap import HierarchicalSamplerWrapped
from spirl.rl.components.sampler_batched import HierarchicalSamplerBatched


configuration.update(AttrDict(
    environment=EnvList,
    sampler=HierarchicalSamplerWrapped,
    
    max_rollout_len = 200,
    n_steps_per_epoch= 10000,
    n_steps_per_update= 800,
    n_warmup_steps = 400,
))

# Environment
sub_env_config = AttrDict(
    reward_norm=1.,
    do_init = False,
    reward_function = corner2_spare_reward_function,
    done_function = corner2_done_function,
    initial_velocity = 65*3.6, 
)

ip_address_list = ['192.168.1.112', '192.168.1.110']

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
    sub_sampler = HierarchicalSamplerBatched,
)