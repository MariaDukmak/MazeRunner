from typing import Union

from mazerunner_sim import BatchRunner, HiddenState
from mazerunner_sim.envs import MazeRunnerEnv, Runner
from mazerunner_sim.policies import PathFindingPolicy, PureRandomPolicy, LeafTrackerPolicy
import pyarrow.feather as feather


class CustomBatchRunner(BatchRunner):
    def __init__(self, filename: str):
        super().__init__(filename)

    @staticmethod
    def update(env: MazeRunnerEnv, data: Union[HiddenState, None]) -> HiddenState:
        return {}

    @staticmethod
    def finish(env: MazeRunnerEnv, data: HiddenState) -> dict:
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
