from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PureRandomPolicy
from mazerunner_sim.policies import PathFindingPolicy
from mazerunner_sim.envs import MazeRunnerEnv

env = MazeRunnerEnv(n_runners=2, day_length=100)
agents = [PureRandomPolicy(), PathFindingPolicy()]


# agents = [[PureRandomPolicy(), 3], [PathFindingPolicy, 5]]
run_simulation(env, agents, wait_key=100, follow_runner_id=1)
