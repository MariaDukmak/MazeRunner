"""Script file for path-finding policies."""

from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PathFindingPolicy
from mazerunner_sim.envs import MazeRunnerEnv, Runner


runners = [
    Runner(action_speed=5),
    Runner(action_speed=10),
]
policies = [
    PathFindingPolicy(),
    PathFindingPolicy(),
]

env = MazeRunnerEnv(runners, day_length=300)

run_simulation(env, policies, wait_key=10, follow_runner_id=0)
