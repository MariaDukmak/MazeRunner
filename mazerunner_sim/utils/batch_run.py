"""Batch runner function."""
from mazerunner_sim.envs.mazerunner_env import MazeRunnerEnv
from mazerunner_sim.policies.pure_random_policy import PureRandomPolicy
from mazerunner_sim.policies.path_finding_policy import PathFindingPolicy
from mazerunner_sim.utils.simulator import run_simulation
from mazerunner_sim.policies import BasePolicy

import pickle
from typing import List

data_path = '../../experiments/'

# TODO: fix deze functie dat er meerder env worden aangemaakt met meerdere polices


def batch_runner(n_runners: int = 5,
                 day_length: int = 1000,
                 n_env: int = 1,
                 policy: List[BasePolicy] = PureRandomPolicy) -> List[dict]:

    # Maak env aan, meerdere kan ook
    for en in range(n_env):
        env = MazeRunnerEnv(n_runners=n_runners, day_length=day_length)

    # Maak policy aan, meerdere kan ook
    # for p in policy:
    policies = [policy() for _ in range(env.n_runners)]

    # Run simulatie
    data = run_simulation(env, policies, window_name=None, wait_key=100, follow_runner_id=0)

    return data


def export_batch(data: List[dict], file_name: str = 'batch_data.p') -> None:
    """Save the data into a pickle file."""
    pickle.dump(data, open(f'{data_path + file_name}', 'wb'))
