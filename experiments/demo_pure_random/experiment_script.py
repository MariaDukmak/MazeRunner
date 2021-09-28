"""Script file for random policies."""
from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PureRandomPolicy
from mazerunner_sim.envs import MazeRunnerEnv

policies = [
    PureRandomPolicy(),
    PureRandomPolicy(),
    PureRandomPolicy(),
    PureRandomPolicy(),
    PureRandomPolicy(),
]
env = MazeRunnerEnv(n_runners=5, day_length=1000)

run_simulation(env, policies, wait_key=100, follow_runner_id=0)
