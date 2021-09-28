"""
Run a batch experiment with two pathfinding runners,
a day-length of 300,
a maze-size of 10.
Run the simulation 10 times.
"""
from mazerunner_sim import run_batch
from mazerunner_sim.policies import PathFindingPolicy
from mazerunner_sim.envs import MazeRunnerEnv

policies = [
    PathFindingPolicy(),
    PathFindingPolicy(),
]
env = MazeRunnerEnv(n_runners=2, day_length=300, maze_size=10)

run_batch([env], policies, batch_size=10)
