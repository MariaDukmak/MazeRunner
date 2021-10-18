"""This experiment tests 4 runners with different speeds, all the runners take random actions."""

from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PureRandomPolicy
from mazerunner_sim.envs import MazeRunnerEnv, Runner

runners = [
    Runner(action_speed=5),
    Runner(action_speed=3),
    Runner(action_speed=1),
    Runner(action_speed=10),
]
policies = [
    PureRandomPolicy(),
    PureRandomPolicy(),
    PureRandomPolicy(),
    PureRandomPolicy(),
]
env = MazeRunnerEnv(runners, day_length=100)

run_simulation(env, policies, wait_key=25, follow_runner_id=0)
