"""Batch runner function."""
# from typing import Sequence, Any, List
#
# from mazerunner_sim.envs.mazerunner_env import MazeRunnerEnv
# from mazerunner_sim.utils.simulator import run_simulation
# from mazerunner_sim.policies import BasePolicy
#
# from multiprocessing import Pool, cpu_count
#
# from copy import deepcopy
# import pickle
#
#
# def run_batch(envs: Sequence[MazeRunnerEnv], policies: List[BasePolicy], batch_size: int = 1, filepath: str = 'batch_data.p') -> None:
#     """
#     Run a batch of simulation, kind of a Monte Carlo simulation, but also using multiprocessing.
#
#     :param envs: A sequence of environments used for the simulations,
#                  this can be infinitely many or just one that is repeated for each simulation.
#                  Using multiple different environments can make the simulation results more robust,
#                  because it doesn't use the same maze each time for example.
#     :param policies: A list of policies used for the simulations.
#     :param batch_size: Number of simulations to run, aka the batch size.
#     :param filepath: The file to write the pickled collected batch info to.
#     """
#     simulator_params = [(deepcopy(envs[i % len(envs)]), policies, None) for i in range(batch_size)]
#
#     with Pool(cpu_count()) as pool:
#         batch_collected_info = pool.starmap(run_simulation, simulator_params)
#
#     pickle_and_save(batch_collected_info, filepath)
#
#
# def pickle_and_save(data: Any, filepath: str) -> None:
#     """
#     Save the data into a pickle file.
#
#     :param data: Any type of Python data-container
#     :param filepath: The path to save the pickled file to
#     """
#     pickle.dump(data, open(filepath, 'wb'))

import abc
from multiprocessing import Pool


class BatchRunner(metaclass=abc.ABCMeta):
    def __init__(self, filename: str):
        self.filename = filename

    @staticmethod
    @abc.abstractmethod
    def update(env, data) -> data:
        pass

    @staticmethod
    @abc.abstractmethod
    def finish(env, data) -> tuple:
        pass

    def _run_single(self, env, pol):
        data = None
        done = False
        observations = env.get_observations()

        while not done:
            # For every agent, decide an action according to the observation
            actions = {runner_id: pol[runner_id].decide_action(observation) for runner_id, observation in observations.items()}

            # Let the actions take place in the environment
            observations, reward, done, info = env.step(actions)

            # Update data
            data = self.update(env, data)

        summary = self.finish(env, data)
        return summary

    def run_batch(self, envs, policies, batch_size: int):
        simulator_params = [(deepcopy(envs[i % len(envs)]), policies, None) for i in range(batch_size)]

        with Pool() as pool:
            results = pool.starmap(self._run_single, [])
        # save results
