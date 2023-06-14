
import imp
from spirl.configs.rl.maze.SAC.conf import *
from spirl.rl.envs.maze import ACmMaze1
from spirl.data.maze.src.maze_agents import MazeSACAgent
MazeSACAgent.chosen_maze = ACmMaze1

configuration.environment = ACmMaze1
