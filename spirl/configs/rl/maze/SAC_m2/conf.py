
import imp
from spirl.configs.rl.maze.SAC.conf import *
from spirl.rl.envs.maze import ACmMaze2

from spirl.data.maze.src.maze_agents import MazeSACAgent
MazeSACAgent.chosen_maze = ACmMaze2

configuration.environment = ACmMaze2