"""Baseline experiment with normal speed and without memory decay"""
from typing import Union

from mazerunner_sim import BatchRunner, HiddenState
from mazerunner_sim.envs import MazeRunnerEnv, Runner
from mazerunner_sim.policies import PathFindingPolicy, PureRandomPolicy, LeafTrackerPolicy


class CustomBatchRunner(BatchRunner):
    """Custom batch runner."""

    def __init__(self, filename: str):
        """
        Initialize the batch.
        :param filename: Name of the data file.
        """
        super().__init__(filename)

    @staticmethod
    def update(env: MazeRunnerEnv, data: Union[HiddenState, None]) -> HiddenState:
        """
        Update function per batch.
        :param env: Mazeenvironment.
        :param data: data that have been generated from a simulation
        """
        return {}

    @staticmethod
    def finish(env: MazeRunnerEnv, data: HiddenState) -> dict:
        """
        Finish data of the batch after simulation.
        :param env: Mazeenvironment.
        :param data: data that have been generated from a simulation
        """
        runner_id_at_exit = env.runners.index(env.found_exit) if env.found_exit is not None else -1
        n_alive = sum([r.alive for r in env.runners])
        explored = [r.explored.tolist() for r in env.runners]
        return {'time': env.time, 'found_exit': runner_id_at_exit, 'n_alive': n_alive, 'explored': explored}


runners = [
    Runner(action_speed=0, memory_decay_percentage=0),
]
policies = [
    LeafTrackerPolicy(),
    PathFindingPolicy(),
    PureRandomPolicy(),
]
env_list = [MazeRunnerEnv(runners=runners, day_length=300, maze_size=10) for _ in range(100)]

if __name__ == '__main__':
    batch_runner_lt = CustomBatchRunner('baseline_leaftracker_small.feather')
    batch_runner_lt.run_batch(envs=env_list, policies=[policies[0]], batch_size=100)

    batch_runner_pf = CustomBatchRunner('baseline_pathfinder_small.feather')
    batch_runner_pf.run_batch(envs=env_list, policies=[policies[1]], batch_size=100)

    batch_runner_pr = CustomBatchRunner('baseline_purerandom_small.feather')
    batch_runner_pr.run_batch(envs=env_list, policies=[policies[2]], batch_size=100)
