"""Batch runner function."""
from typing import Sequence, Any, List
from mazerunner_sim.envs.mazerunner_env import MazeRunnerEnv
from mazerunner_sim.utils.simulator import run_simulation
from mazerunner_sim.policies import BasePolicy
from multiprocessing import Pool, cpu_count

from copy import deepcopy
import pickle


def run_batch(envs: Sequence[MazeRunnerEnv], policies: List[BasePolicy], batch_size: int = 1, filepath: str = 'batch_data.p') -> None:
    """
    Run a batch of simulation, kind of a Monte Carlo simulation, but also using multiprocessing.

    :param envs: A sequence of environments used for the simulations,
                 this can be infinitely many or just one that is repeated for each simulation.
                 Using multiple different environments can make the simulation results more robust,
                 because it doesn't use the same maze each time for example.
    :param policies: A list of policies used for the simulations.
    :param batch_size: Number of simulations to run, aka the batch size.
    :param filepath: The file to write the pickled collected batch info to.
    """
    simulator_params = [(deepcopy(envs[i % len(envs)]), policies, None) for i in range(batch_size)]

    with Pool(cpu_count()) as pool:
        batch_collected_info = pool.starmap(run_simulation, simulator_params)

    pickle_and_save(batch_collected_info, filepath)


def pickle_and_save(data: Any, filepath: str) -> None:
    """
    Save the data into a pickle file.

    :param data: Any type of Python data-container
    :param filepath: The path to save the pickled file to
    """
    pickle.dump(data, open(filepath, 'wb'))


#
# ### een schets :))
#
# def make_simulation_args():
#     """
#     Prepare all combinations of parameter values for `batch_run`
#
#     Returns:Tuple with the form: (total_iterations, all_kwargs, all_param_values)
#     """
#
#     total_iterations = ...
#     instellingen_list = ...
#     all_kwargs = []
#     fixed_parameters = None
#
#     count = len(instellingen_list)
#     if count:
#         for instelling in instellingen_list:
#             kwarg = instelling.copy()
#             kwarg.update(fixed_parameters)
#
#
