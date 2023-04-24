import os
import copy

from spirl.configs.hrl.maze.shLL.conf import *
from spirl.rl.envs.maze import  ACmMaze1

from spirl.data.maze.src.maze_agents import MazeSACAgent
MazeSACAgent.chosen_maze = ACmMaze1

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'skill critic on the maze env'

configuration.update({
    'environment': ACmMaze1,
    'max_rollout_len': 2000,
    'n_steps_per_epoch': 1e5,
    'n_warmup_steps': 5e3,
})
