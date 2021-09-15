"""Example agent that takes random actions."""

from gym_mazerunner.mazerunner_env import MazeRunnerEnv

import numpy as np
import cv2

WINDOW_NAME = 'Mazerunner-sim'

env = MazeRunnerEnv(day_length=100000, n_agents=2500)
done = False

while not done:
    actions = [np.random.randint(4) for _ in range(2500)]
    observations, reward, done, info = env.step(actions)
    render = env.render()

    cv2.imshow(WINDOW_NAME, cv2.cvtColor(np.array(render), cv2.COLOR_BGR2RGB))
    cv2.waitKey(25)

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
