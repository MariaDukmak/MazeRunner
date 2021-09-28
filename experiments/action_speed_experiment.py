"""Expirement for testing different action speeds."""

from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PathFindingPolicy
from mazerunner_sim.policies import PureRandomPolicy
from mazerunner_sim.envs import MazeRunnerEnv, Runner

runners = [
    Runner(action_speed=20),
    Runner(action_speed=1),
]

policies = [
    PathFindingPolicy(),
    PureRandomPolicy(),
]

env = MazeRunnerEnv(runners, day_length=1000, maze_size=10)
run_simulation(env, policies, wait_key=1, follow_runner_id=1)
