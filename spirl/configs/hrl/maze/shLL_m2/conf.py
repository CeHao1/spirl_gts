import os
import copy

from spirl.configs.hrl.maze.shLL.conf import *
from spirl.rl.envs.maze import  ACmMaze2

from spirl.data.maze.src.maze_agents import MazeSACAgent
MazeSACAgent.chosen_maze = ACmMaze2

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'skill critic on the maze env'

configuration.update({
    'environment': ACmMaze2,
    'num_epochs': 73,
    'max_rollout_len': 2000,
    'n_steps_per_epoch': 1e5,
    'n_warmup_steps': 5e3,
})

# hl_replay_params.capacity *= 0.7
# ll_replay_params.capacity *= 0.7


# agent_config.initial_train_stage = skill_critic_stages.HL_TRAIN
# agent_config.initial_train_stage = skill_critic_stages.HYBRID
agent_config.initial_train_stage = skill_critic_stages.LL_TRAIN

# from spirl.utils.general_utils import DelayedLinearSchedule
# ll_agent_config.td_schedule = DelayedLinearSchedule
# ll_agent_config.td_schedule_params = AttrDict(initial_p=1.,
#                                 final_p=80.,
#                                 schedule_timesteps=int(5e5),
#                                 delay = int(10e5))

'''
(12e5) -> 5 ->(5e5) -> 80
(10e5) -> 1 ->(5e5) -> 80
(12e5) -> 10 ->(5e5) -> 80
'''

# ll_agent_config.td_schedule_params = AttrDict(p=10.)
# ll_agent_config.td_schedule_params = AttrDict(p=20.)
# ll_agent_config.td_schedule_params = AttrDict(p=50.)
# ll_agent_config.td_schedule_params = AttrDict(p=80.)

# ll_policy_params.manual_log_sigma = [-1, -1]
# ll_policy_params.manual_log_sigma = [-2, -2]
ll_policy_params.manual_log_sigma = [-3, -3]
# ll_policy_params.manual_log_sigma = [-4, -4]
# ll_policy_params.manual_log_sigma = [-5, -5]