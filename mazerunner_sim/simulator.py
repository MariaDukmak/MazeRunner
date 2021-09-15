"""Run simulation with given parameters."""
from typing import List

import cv2

from mazerunner_sim.agents.Agent import Agent
from mazerunner_sim.envs.mazerunner_env import MazeRunnerEnv

import numpy as np


def run_simulation(env: MazeRunnerEnv,
                   agents: List[Agent],
                   window_name: str = 'MazeRunner Simulation',
                   wait_key: int = 10) -> float:
    """
    Run the simulation with given parameters.

    :param env: Environment used for the simulation
    :param agents: List of agents used in the experiments
    :param window_name: Name used for the simulation
    :param wait_key: Time in milliseconds used as interval for displaying steps
    :return: Total reward gotten
    """
    done = False
    total_reward = 0

    observations = env.get_observation()

    while not done:
        # For every agent, decide an action according to the observation
        actions = [agent.decide_action(observations) for agent in agents]

        # Let the actions take place in the environment
        observations, reward, done, info = env.step(actions)

        total_reward += reward

        # Render current time in simulation for visual output
        render = env.render()

        # Display render of current time in the environment
        cv2.imshow(window_name, cv2.cvtColor(np.array(render), cv2.COLOR_BGR2RGB))

        # Delay between renders of the simulation
        cv2.waitKey(wait_key)

        # On window [X] button press: stop the simulation and destroy the window
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    return total_reward
