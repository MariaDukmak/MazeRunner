"""Script file for path-finding policies."""
from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PathFindingPolicy
from mazerunner_sim.envs import MazeRunnerEnv


env = MazeRunnerEnv(n_runners=5, day_length=300, maze_size=10)
agents = [PathFindingPolicy() for _ in range(env.n_runners)]

run_simulation(env, agents, wait_key=10, follow_runner_id=0)
