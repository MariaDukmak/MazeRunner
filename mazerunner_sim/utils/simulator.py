"""Run simulation with given parameters."""
from typing import List, Union

import cv2

from mazerunner_sim.policies import BasePolicy
from mazerunner_sim.envs import MazeRunnerEnv

import numpy as np


def run_simulation(env: MazeRunnerEnv,
                   policies: List[BasePolicy],
                   window_name: Union[str, None] = 'MazeRunner Simulation',
                   wait_key: int = 10,
                   follow_runner_id: int = None) -> List[dict]:
    """
    Run the simulation with given parameters.

    :param env: Environment used for the simulation
    :param policies: List of policies used in the experiments
    :param window_name: Name used for the simulation, don't visualize when window_name is None
    :param wait_key: Time in milliseconds used as interval for displaying steps
    :param follow_runner_id: Id used to follow runner
    :return: The collected stats/info from each step
    """
    done = False
    total_reward = 0
    visualize = window_name is not None
    collected_info = []

    observations = env.get_observations()

    if visualize:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(window_name, 800, 800)

    while not done:
        # For every agent, decide an action according to the observation
        actions = {runner_id: policies[runner_id].decide_action(observation) for runner_id, observation in observations.items()}

        # Let the actions take place in the environment
        observations, reward, done, info = env.step(actions)
        collected_info.append(info)

        total_reward += reward

        if visualize:
            # Render current time in simulation for visual output
            render = env.render(follow_runner_id=follow_runner_id)

            # Display render of current time in the environment
            cv2.imshow(window_name, cv2.cvtColor(np.array(render), cv2.COLOR_BGR2RGB))

            # Delay between renders of the simulation
            cv2.waitKey(wait_key)

            # On window [X] button press: stop the simulation and destroy the window
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    if visualize:
        cv2.destroyAllWindows()

    return collected_info
