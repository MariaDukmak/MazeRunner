"""Example agent that takes random actions."""

import numpy as np

from gym_mazerunner.mazerunner_env import MazeRunnerEnv

env = MazeRunnerEnv(day_length=100000, n_agents=2)
done = False

while not done:
    observations, reward, done, info = env.step([np.random.randint(4), 1])
    print(env.runners[0].location)
