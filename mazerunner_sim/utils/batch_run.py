"""Batch runner function."""
from typing import Union
from mazerunner_sim.envs.mazerunner_env import MazeRunnerEnv
from mazerunner_sim.policies.pure_random_policy import PureRandomPolicy
from mazerunner_sim.policies.path_finding_policy import PathFindingPolicy
from mazerunner_sim.utils.simulator import run_simulation
from mazerunner_sim.policies import BasePolicy
from multiprocessing import Pool, Process, cpu_count

import copy
import pickle
from typing import List

data_path = '../../experiments/'


def batch_runner(policies: List[BasePolicy],
                 day_length: int = 1000,
                 n_env: int = 1,
                 env: Union[MazeRunnerEnv, None] = None) -> List[dict]:

    available_processors = cpu_count()
    # Maak env aan, meerdere kan ook
    with Pool(n_env) as pool:
        for en in range(n_env):
            env = MazeRunnerEnv(n_runners=len(policies), day_length=day_length)

    # Run simulatie
    # data = run_simulation(env, policies, window_name=None, wait_key=100, follow_runner_id=0)

    # return data


def export_batch(data: List[dict], file_name: str = 'batch_data.p') -> None:
    """Save the data into a pickle file."""
    pickle.dump(data, open(f'{data_path + file_name}', 'wb'))


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
