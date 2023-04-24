import os
import copy

from spirl.configs.hrl.maze.shLL.conf import *
from spirl.rl.envs.maze import  ACmMaze3

from spirl.data.maze.src.maze_agents import MazeSACAgent
MazeSACAgent.chosen_maze = ACmMaze3

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'skill critic on the maze env'

configuration.update({
    'environment': ACmMaze3,
    'max_rollout_len': 800,
    'n_steps_per_epoch': 8000,
    'n_warmup_steps': 800*10,
    
    'log_output_interval': 800,
    'log_image_interval': 8000,
})

hl_replay_params.capacity = 3e4
ll_replay_params.capacity = 1.5e5

# agent_config.initial_train_stage = skill_critic_stages.HL_TRAIN
agent_config.initial_train_stage = skill_critic_stages.HYBRID

ll_agent_config.td_schedule_params=AttrDict(p=10.)
agent_config.update_iterations = 1
ll_policy_params.manual_log_sigma = [-2, -2]
# ll_policy_params.manual_log_sigma = [-5, -5]