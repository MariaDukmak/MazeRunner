"""Script file for random agents."""
from mazerunner_sim import run_simulation
from mazerunner_sim.agents import PureRandomAgent
from mazerunner_sim.envs import MazeRunnerEnv


env = MazeRunnerEnv(n_agents=10, day_length=10000)
agents = [PureRandomAgent() for _ in range(env.n_agents)]

run_simulation(env, agents, wait_key=100)
