"""Script file for path-finding policies."""

from mazerunner_sim import run_simulation
from mazerunner_sim.policies import LeafTrackerPolicy
from mazerunner_sim.envs import MazeRunnerEnv, Runner


runners = [
    Runner(action_speed=0),
    Runner(action_speed=0),
    Runner(action_speed=1),
    Runner(action_speed=2),
]
policies = [
    LeafTrackerPolicy(),
    LeafTrackerPolicy(),
    LeafTrackerPolicy(),
    LeafTrackerPolicy(),
]

env = MazeRunnerEnv(runners, day_length=200, maze_size=26)

run_simulation(env, policies, wait_key=1, follow_runner_id=0)
