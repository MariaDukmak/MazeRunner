"""Batch runner function."""
from typing import Union
from mazerunner_sim.envs.mazerunner_env import MazeRunnerEnv
from mazerunner_sim.utils.simulator import run_simulation
from mazerunner_sim.policies import BasePolicy
from multiprocessing import Pool, cpu_count

from copy import deepcopy
import pickle
from typing import List


def run_batch(policies: List[BasePolicy],
              batch_size: int = 1,
              env: Union[MazeRunnerEnv, None] = None) -> None:

    simulator_params = [(deepcopy(env), policies, None) for _ in range(batch_size)]

    with Pool(cpu_count()) as pool:
        batch_collected_info = pool.starmap(run_simulation, simulator_params)

    export_batch(batch_collected_info)


def export_batch(data: List[List[dict]], filepath: str = 'batch_data.p') -> None:
    """Save the data into a pickle file."""
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
