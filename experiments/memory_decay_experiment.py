"""blabla"""

from mazerunner_sim import run_simulation
from mazerunner_sim.policies import PathFindingPolicy
from mazerunner_sim.envs import MazeRunnerEnv, Runner


runners = [
    Runner(action_speed=0, memory_decay_percentage=50),
    # Runner(action_speed=0, memory_decay_percentage=50),
    # Runner(action_speed=0, memory_decay_percentage=0),
]
policies = [
    PathFindingPolicy(),
    # PathFindingPolicy(),
    # PathFindingPolicy(),
]

env = MazeRunnerEnv(runners, day_length=50)

run_simulation(env, policies, wait_key=40, follow_runner_id=0)
