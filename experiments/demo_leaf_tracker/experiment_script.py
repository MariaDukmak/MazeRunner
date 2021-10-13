"""Script file for path-finding policies."""

from mazerunner_sim import run_simulation
from mazerunner_sim.policies import LeafTrackerPolicy
from mazerunner_sim.envs import MazeRunnerEnv, Runner


runners = [
    Runner(action_speed=0),
    # Runner(action_speed=0),
    # Runner(action_speed=1),
    # Runner(action_speed=2),
]
policies = [
    LeafTrackerPolicy(leaf_weight=1, task_weight=10),
    # LeafTrackerPolicy(leaf_weight=1, task_weight=10),
    # LeafTrackerPolicy(leaf_weight=1, task_weight=10),
    # LeafTrackerPolicy(leaf_weight=1, task_weight=10),
]

env = MazeRunnerEnv(runners, day_length=100, maze_size=26)

run_simulation(env, policies, wait_key=1, follow_runner_id=0)
