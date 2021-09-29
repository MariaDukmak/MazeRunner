"""
Run a batch experiment with with one pathfinding & one random runners that have speed,
speed of 5, 6,
a day-length of 300,
a maze-size of 10.
Run the simulation 10 times.
"""
from mazerunner_sim import run_batch
from mazerunner_sim.policies import PathFindingPolicy, PureRandomPolicy
from mazerunner_sim.envs import MazeRunnerEnv
from mazerunner_sim.envs import Runner

runners = [
    Runner(action_speed=5),
    Runner(action_speed=6),
]
policies = [
    PathFindingPolicy(),
    PureRandomPolicy(),
]
env = MazeRunnerEnv(runners=runners, day_length=300, maze_size=10)

if __name__ == "__main__":
    run_batch([env], policies, batch_size=10)
