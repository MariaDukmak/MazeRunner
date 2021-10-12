"""Script file for path-finding policies."""

from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PathFindingPolicy
from mazerunner_sim.envs import MazeRunnerEnv, Runner


runners = [
    Runner(action_speed=1),
    Runner(action_speed=0),
]
policies = [
    PathFindingPolicy(),
    PathFindingPolicy(),
]

env = MazeRunnerEnv(runners, day_length=100)

run_simulation(env, policies, wait_key=100, follow_runner_id=0)
