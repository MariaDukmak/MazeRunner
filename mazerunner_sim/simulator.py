"""Run simulation with given parameters."""
from typing import List, Union

import cv2

from mazerunner_sim.policies import BasePolicy
from mazerunner_sim.envs import MazeRunnerEnv

import numpy as np


def run_simulation(env: MazeRunnerEnv,
                   agents: List[BasePolicy],
                   window_name: Union[str, None] = 'MazeRunner Simulation',
                   wait_key: int = 10,
                   follow_runner_id: int = None) -> float:
    """
    Run the simulation with given parameters.

    :param env: Environment used for the simulation
    :param agents: List of policies used in the experiments
    :param window_name: Name used for the simulation, don't visualize when window_name is None
    :param wait_key: Time in milliseconds used as interval for displaying steps
    :param follow_runner_id: Id used to follow runner
    :return: Total reward gotten
    """
    done = False
    total_reward = 0
    visualize = window_name is not None

    observations = env.get_observations()

    while not done:
        # For every agent, decide an action according to the observation
        actions = [agent.decide_action(observation) for agent, observation in zip(agents, observations)]

        # Let the actions take place in the environment
        observations, reward, done, info = env.step(actions)
        print(f"Time till end of day: {observations[0].time_till_end_of_day}")

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

    return total_reward
