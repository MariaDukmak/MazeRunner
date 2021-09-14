"""Class file for mazerunner environment."""
from typing import List, Tuple
import gym
import numpy as np

from gym_mazerunner.maze_generator import generate_maze


class MazeRunnerEnv(gym.Env):
    maze: np.array
    # 4 possible actions: 0:Up, 1:Down, 2:Left, 3:Right
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Box(low=False, high=True, shape=(8,), dtype=bool)

    def __init__(self, maze_size: int = 16, center_size: int = 4):
        super(MazeRunnerEnv, self).__init__()
        self.reset(maze_size, center_size)

    def step(self, actions: List[np.int64]) -> Tuple[List[np.Array], float, bool, dict]:
        """

        :param actions:
        :return: observations, reward, done, info
        """
        pass

    def reset(self, maze_size: int = 16, center_size: int = 4):
        self.maze = generate_maze(maze_size, center_size)

    def render(self, mode="human"):
        pass
