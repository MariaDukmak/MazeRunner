"""Example of a batch runner experiment to compare the speed of agents."""
from typing import Union

import pyarrow.feather as feather

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
    Runner(action_speed=5, memory_decay_percentage=5),
    Runner(action_speed=6, memory_decay_percentage=5),
    Runner(action_speed=6, memory_decay_percentage=5),
    Runner(action_speed=6, memory_decay_percentage=5),
]
policies = [
    PathFindingPolicy(),
    PathFindingPolicy(),
    PureRandomPolicy(),
    LeafTrackerPolicy(),
]
env = MazeRunnerEnv(runners=runners, day_length=300, maze_size=10)

if __name__ == '__main__':
    batch_runner = CustomBatchRunner('snelheid_test.feather')
    batch_runner.run_batch(envs=[env], policies=policies, batch_size=10)

    print(feather.read_feather('snelheid_test.feather'))
