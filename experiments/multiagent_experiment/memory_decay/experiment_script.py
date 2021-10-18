"""Example of a batch runner experiment to compare the memory deacy of agents.
we were unable to use this experiment due to extremely long run time :(
"""
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
        Update function per step in batch.
        :param env: Mazeenvironment.
        :param data: data that have been generated from a simulation
        """
        n_alive = sum([r.alive for r in env.runners])
        explored = [r.explored.tolist() for r in env.runners]
        return {'time': env.time, 'n_alive': n_alive, 'explored': explored}

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
    Runner(action_speed=0, memory_decay_percentage=0),
    Runner(action_speed=0, memory_decay_percentage=0),
]

env = MazeRunnerEnv(runners=runners, day_length=120, maze_size=16)


if __name__ == '__main__':
    for decay in range(0, 31, 5):
        for r in runners:
            r.memory_decay_percentage = decay

        batch_runner = CustomBatchRunner(f'memory_decay_pathfinder_{str(decay)}.feather')
        batch_runner.run_batch(envs=[env], policies=[PathFindingPolicy() for _ in range(3)], batch_size=100)

        batch_runner = CustomBatchRunner(f'memory_decay_leaftracker_{str(decay)}.feather')
        batch_runner.run_batch(envs=[env], policies=[LeafTrackerPolicy() for _ in range(3)], batch_size=100)

        batch_runner = CustomBatchRunner(f'memory_decay_purerandom_{str(decay)}.feather')
        batch_runner.run_batch(envs=[env], policies=[PureRandomPolicy() for _ in range(3)], batch_size=100)
