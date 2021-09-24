"""Script file for random policies."""
from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PureRandomPolicy
from mazerunner_sim.envs import MazeRunnerEnv


env = MazeRunnerEnv(n_runners=5, day_length=1000)
agents = [PureRandomPolicy() for _ in range(env.n_runners)]

run_simulation(env, agents, wait_key=100, follow_runner_id=0)
