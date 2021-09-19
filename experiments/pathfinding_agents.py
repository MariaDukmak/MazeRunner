"""Script file for path-finding policies."""
from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PathFindingAgent
from mazerunner_sim.envs import MazeRunnerEnv


env = MazeRunnerEnv(n_runners=5, day_length=150, maze_size=20)
agents = [PathFindingAgent() for _ in range(env.n_runners)]

run_simulation(env, agents, wait_key=100, follow_runner_id=0)
