"""OpenAI gym environment for the MazeRunner."""

from typing import List, Tuple

from PIL import Image

import gym

from mazerunner_sim.envs.maze_generator import generate_maze

from mazerunner_sim.envs.maze_render import render_agent_in_step, render_background

from mazerunner_sim.envs.runner import Runner

from mazerunner_sim.observation_and_action import Observation, Action

import numpy as np

from functools import reduce


class MazeRunnerEnv(gym.Env):
    """OpenAI gym environment for the MazeRunner."""

    # 4 possible actions: 0:Up, 1:Down, 2:Left, 3:Right
    action_space = gym.spaces.Discrete(4)
    metadata = {'render.modes': ['human']}

    done: bool
    time: int
    total_rewards_given: float
    runners: List[Runner]

    DEATH_PUNISHMENT = 99999

    def __init__(self, maze_size: int = 16, center_size: int = 4, n_agents: int = 1, day_length: int = 20):
        """
        Initialize the MazeRunner environment.

        :param maze_size: Size of the maze
        :param center_size: Size of the glade (center)
        :param n_agents: Number of agents, results in the number of runners
        :param day_length: Length of a day, at the end of the day, all the runners not in a safe spot are going to a better place
        """
        super(MazeRunnerEnv, self).__init__()
        self.maze, self.safe_zone = generate_maze(maze_size, center_size)
        self.day_length = day_length
        self.n_agents = n_agents
        self.reset()

        self.rendered_background = render_background(self.maze)

    def step(self, actions: List[Action]) -> Tuple[List[Observation], float, bool, dict]:
        """
        Taken an step in the environment.

        :param actions: list of actions
        :return: observations, reward, done, info.
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
                # if the step is actually possible, take the step # TODO add extra code for checking if space is accupied
                if self.maze[tuple(runner.location + step)[::-1]]:
                    runner.location += step

        # Update maps
        for r in self.runners:
            if r.alive:
                r.update_map(self.maze[r.location[1] - 1:r.location[1] + 2, r.location[0] - 1:r.location[0] + 2])

        # At the end of the day
        if self.time % self.day_length == 0:
            # kill the ones still in the maze
            for runner in self.runners:
                if not self.safe_zone[tuple(runner.location)]:
                    runner.alive = False
            # share memory map between those still alive
            combined_explored_map = reduce(np.logical_or, [r.explored for r in self.runners if r.alive])
            combined_maze_map = reduce(np.logical_or, [r.known_maze for r in self.runners if r.alive])
            for runner in self.runners:
                if runner.alive:
                    runner.explored = combined_explored_map.copy()
                    runner.known_maze = combined_maze_map.copy()

        # Observations
        observations = self.get_observations()

        # Reward & done
        reward = -1
        # if a runner found the exit
        if any(r.location[0] == 0 or r.location[0] == self.maze.shape[1] - 1 or
               r.location[1] == 0 or r.location[1] == self.maze.shape[0] - 1
               for r in self.runners):
            self.done = True
        # if all runners are dead
        elif not all(runner.alive for runner in self.runners):
            self.done = True
            reward -= self.DEATH_PUNISHMENT + self.total_rewards_given
        self.total_rewards_given += reward

        return observations, reward, self.done, {}

    def reset(self):
        """
        Reset the environment.

        The maze stays the same, so does the number of runners and the day-length,
        but the agents are reset, the time and the `done` flag.
        """
        self.done = False
        self.time = 0
        self.total_rewards_given = 0.

        center_coord = np.array([self.maze.shape[0] // 2] * 2)
        self.runners = [Runner(center_coord.copy(), self.maze.shape) for _ in range(self.n_agents)]

    def get_observations(self) -> List[Observation]:
        """
        Get information about the environment location, returns walls.

        :return A list of runner-observations, take a look at it's documentation for more detail
        """
        return [
            Observation(
                explored=runner.explored.copy(),
                known_maze=runner.known_maze.copy(),
                runner_location=(runner.location[0], runner.location[1]),
                time_till_end_of_day=self.day_length - (self.time % self.day_length) - 1,
            )
            for runner in self.runners
        ]

    def render(self, mode="human", follow_runner_id: int = None) -> Image:
        """
        Render the state of the environment.

        :param follow_runner_id: The index of the agent to follow what has been explored
        :param mode: Mode of rendering, choose between: ['human']
        """
        return render_agent_in_step(self.maze, self.rendered_background, self.runners, follow_runner_id)
