import numpy as np

from gym_mazerunner.mazerunner_env import MazeRunnerEnv

env = MazeRunnerEnv(day_length=100000)
done = False

while not done:
    observations, reward, done, info = env.step([np.random.randint(4)])
    print(env.runners[0].location)
