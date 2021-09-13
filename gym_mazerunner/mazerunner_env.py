from typing import List, Tuple
import gym
import numpy as np

ExploredMap = np.Array()


class MazeRunnerEnv(gym.Env):
    def __init__(self):
        pass

    def step(self, action: List[]) -> Tuple[List[ExploredMap], float, bool, dict]:
        """

        :param action:
        :return: Observation, reward, done, stats
        """
        pass

    def reset(self):
        pass

    def render(self):
        pass
