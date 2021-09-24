"""Script file for random policies."""
from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PureRandomPolicy
from mazerunner_sim.envs import MazeRunnerEnv

def batchrunner(n_runners, day_lenth, percentage_policies: Tuple[float, float, float], n_simulations, ) -> None:
    pass

env = MazeRunnerEnv(n_runners=5, day_length=1000)
agents = [PureRandomPolicy() for _ in range(env.n_runners)]

run_simulation(env, agents, wait_key=100, follow_runner_id=0)
