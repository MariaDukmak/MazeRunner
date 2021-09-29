"""
Run a batch experiment with with one pathfinding & one random runners that have speed,
speed of 5, 6,
memory delay of 5%,
a day-length of 300,
a maze-size of 10.
Run the simulation 10 times.
"""
from mazerunner_sim import run_batch
from mazerunner_sim.policies import PathFindingPolicy, PureRandomPolicy, LeafTrackerPolicy
from mazerunner_sim.envs import MazeRunnerEnv
from mazerunner_sim.envs import Runner

runners = [
    Runner(action_speed=5, memory_decay_percentage=5),
    Runner(action_speed=6, memory_decay_percentage=5),
    Runner(action_speed=7, memory_decay_percentage=5)
]
policies = [
    PathFindingPolicy(),
    PureRandomPolicy(),
    LeafTrackerPolicy()
]
env = MazeRunnerEnv(runners=runners, day_length=300, maze_size=10)

if __name__ == "__main__":
    run_batch([env], policies, batch_size=10)
