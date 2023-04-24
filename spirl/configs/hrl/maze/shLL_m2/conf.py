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
    'max_rollout_len': 2000,
    'n_steps_per_epoch': 1e5,
    'n_warmup_steps': 5e3,
})

# agent_config.initial_train_stage = skill_critic_stages.HL_TRAIN
agent_config.initial_train_stage = skill_critic_stages.HYBRID

ll_agent_config.td_schedule_params=AttrDict(p=80.)
agent_config.update_iterations = 1
ll_policy_params.manual_log_sigma = [-3, -3]