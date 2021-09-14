"""Class file for mazerunner environment."""
from typing import List, Tuple
import gym
import numpy as np
import numpy.typing as npt

from gym_mazerunner.maze_generator import generate_maze
from gym_mazerunner.runner import Runner


class MazeRunnerEnv(gym.Env):
    # 4 possible actions: 0:Up, 1:Down, 2:Left, 3:Right
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Box(low=False, high=True, shape=(3, 3), dtype=bool)
    metadata = {'render.modes': ['human']}

    done: bool
    time: int
    total_rewards_given: float
    runners: List[Runner]

    DEATH_PUNISHMENT = 99999

    def __init__(self, maze_size: int = 16, center_size: int = 4, n_agents: int = 1, day_length: int = 20):
        super(MazeRunnerEnv, self).__init__()
        self.maze, self.safe_zone = generate_maze(maze_size, center_size)
        self.day_length = day_length
        self.n_agents = n_agents
        self.reset()

    def step(self, actions: List[np.int64]) -> Tuple[List[npt.NDArray], float, bool, dict]:
        """

        :param actions:
        :return: observations, reward, done, info
        """
        # Increment time
        self.time += 1

        # Let the runners take a step
        for runner, action in zip(self.runners, actions):
            if runner.alive:
                step = np.array([[0, -1],
                                 [0, 1],
                                 [-1, 0],
                                 [1, 0]][action])
                # if the step is actually possible, take the step
                if self.maze[tuple(runner.location + step)]:
                    runner.location += step

        # Kill the runners that are still in the maze at the end of the day
        if self.time % self.day_length == 0:
            for runner in self.runners:
                if not self.safe_zone[tuple(runner.location)]:
                    runner.alive = False

        # Observations
        observations = [
            self.maze[r.location[1]-1:r.location[1]+2, r.location[0]-1:r.location[0]+2]
            if r.alive else
            np.full((3, 3), False)
            for r in self.runners
        ]

        # Reward & done
        reward = -1
        # if a runner found the exit
        if any(r.location[0] == 0 or r.location[0] == self.maze.shape[1]-1 or
               r.location[1] == 0 or r.location[1] == self.maze.shape[0]-1
               for r in self.runners):
            self.done = True
        # if all runners are dead
        elif not all(runner.alive for runner in self.runners):
            self.done = True
            reward -= self.DEATH_PUNISHMENT + self.total_rewards_given
        self.total_rewards_given += reward

        return observations, reward, self.done, {}

    def reset(self):
        self.done = False
        self.time = 0
        self.total_rewards_given = 0.

        center_coord = np.array([self.maze.shape[0] // 2] * 2)
        self.runners = [Runner(center_coord) for _ in range(self.n_agents)]

    def render(self, mode="human"):
        pass
