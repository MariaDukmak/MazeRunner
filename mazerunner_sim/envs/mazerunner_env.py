"""OpenAI gym environment for the MazeRunner."""

from typing import List, Tuple, Dict

from PIL import Image

import gym

from mazerunner_sim.envs.maze_generator import generate_maze

from mazerunner_sim.envs.visualisation.maze_render import render_agent_in_step, render_background

from mazerunner_sim.envs.agents.runner import Runner

from mazerunner_sim.utils.observation_and_action import Observation, Action

import numpy as np

from functools import reduce


class MazeRunnerEnv(gym.Env):
    """OpenAI gym environment for the MazeRunner."""

    action_space = gym.spaces.Discrete(5)
    metadata = {'render.modes': ['human']}

    done: bool
    time: int
    total_rewards_given: float
    runners: List[Runner]

    DEATH_PUNISHMENT = 99999

    def __init__(self, runners: List[Runner], maze_size: int = 16, center_size: int = 4, day_length: int = 20):
        """
        Initialize the MazeRunner environment.

        :param runners: The runners in the maze with their properties
        :param maze_size: Size of the maze
        :param center_size: Size of the glade (center)
        :param day_length: Length of a day, at the end of the day, all the runners not in a safe spot are going to a better place
        """
        super(MazeRunnerEnv, self).__init__()
        self.maze, self.safe_zone, self.leaves = generate_maze(maze_size, center_size)
        self.day_length = day_length
        self.runners = runners
        self.reset()

        self.rendered_background = render_background(self.maze, self.leaves, self.safe_zone)

    def step(self, actions: Dict[int, Action]) -> Tuple[Dict[int, Observation], float, bool, dict]:
        """
        Taken an step in the environment.

        :param actions: list of actions
        :return: observations, reward, done, info.
        """
        # Increment time
        self.time += 1

        # Let the runners take a step
        for runner_id, action in actions.items():
            runner = self.runners[runner_id]
            if runner.alive:
                step = np.array([[0, -1],
                                 [0, 1],
                                 [-1, 0],
                                 [1, 0],
                                 [0, 0]][action])
                # if the step is actually possible, take the step # TODO add extra code for checking if space is accupied
                if self.maze[tuple(runner.location + step)[::-1]]:
                    runner.location += step

        # Update maps
        for r in self.runners:
            if r.alive:
                r.update_map(
                    self.maze[r.location[1] - 1:r.location[1] + 2, r.location[0] - 1:r.location[0] + 2],
                    self.leaves[r.location[1] - 1:r.location[1] + 2, r.location[0] - 1:r.location[0] + 2]
                )

        # Kill the runners that are still in the maze at the end of the day
        if self.time % self.day_length == 0:
            for runner in self.runners:
                if not self.safe_zone[tuple(runner.location)]:
                    runner.alive = False

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

        # Share memory map between those still alive at the end of the day
        if self.time % self.day_length == 0 and not self.done:
            combined_explored_map = reduce(np.logical_or, [r.explored for r in self.runners if r.alive])
            combined_maze_map = reduce(np.logical_or, [r.known_maze for r in self.runners if r.alive])
            combined_leaves_map = reduce(np.logical_or, [r.known_leaves for r in self.runners if r.alive])
            for runner in self.runners:
                if runner.alive:
                    runner.explored = combined_explored_map.copy()
                    runner.known_maze = combined_maze_map.copy()
                    runner.known_leaves = combined_leaves_map.copy()

        # Observations
        observations = self.get_observations()

        return observations, reward, self.done, self.get_info()

    def reset(self):
        """
        Reset the environment.

        The maze stays the same, so does the number of runners and the day-length,
        but the policies are reset, the time and the `done` flag.
        """
        self.done = False
        self.time = 0
        self.total_rewards_given = 0.

        center_coord = np.array([self.maze.shape[0] // 2] * 2)
        for runner in self.runners:
            runner.reset(center_coord.copy(), self.safe_zone, self.leaves)

    def get_observations(self) -> Dict[int, Observation]:
        """
        Get information about the environment location, returns walls.

        :return A list of runner-observations, take a look at it's documentation for more detail
        """
        return {
            runner_id: Observation(
                explored=runner.explored.copy(),
                known_maze=runner.known_maze.copy(),
                known_leaves=runner.known_leaves.copy(),
                safe_zone=self.safe_zone,
                runner_location=(runner.location[0], runner.location[1]),
                time_till_end_of_day=self.day_length - (self.time % self.day_length) - 1,
            )
            for runner_id, runner in enumerate(self.runners)
            if runner.check_status_speed()
        }

    def render(self, mode="human", follow_runner_id: int = None) -> Image:
        """
        Render the state of the environment.

        :param follow_runner_id: The index of the agent to follow what has been explored
        :param mode: Mode of rendering, choose between: ['human']
        """
        print(self.time)
        return render_agent_in_step(self.maze, self.rendered_background, self.runners, follow_runner_id)

    def get_info(self) -> dict:
        """Get the environment and the agents info. Needed for the batch run."""
        return {
            'time': self.time,
            'agents_n': len(self.runners),
            'explored': [r.explored for r in self.runners],
        }
