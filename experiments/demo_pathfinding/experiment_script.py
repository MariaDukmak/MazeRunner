"""Script file for path-finding policies."""

from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PathFindingPolicy
from mazerunner_sim.envs import MazeRunnerEnv, Runner


runners = [
    Runner(action_speed=i//2)
    for i in range(10)
]
policies = [
    PathFindingPolicy(task_weight=1.0)
    for _ in range(10)
]

env = MazeRunnerEnv(runners, day_length=200, maze_size=28)

run_simulation(env, policies, wait_key=50, follow_runner_id=0)
