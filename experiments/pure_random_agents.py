"""
This experiment tests 4 runners with different speeds,
all the runners take random actions.
"""
from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PureRandomPolicy
from mazerunner_sim.envs import MazeRunnerEnv, Runner

runners = [
    Runner(5),
    Runner(3),
    Runner(1),
    Runner(10),
]
policies = [
    PureRandomPolicy(),
    PureRandomPolicy(),
    PureRandomPolicy(),
    PureRandomPolicy(),
]
env = MazeRunnerEnv(runners, day_length=1000)

run_simulation(env, policies, wait_key=100, follow_runner_id=0)
