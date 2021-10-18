"""OpenAI gym environment for the MazeRunner."""

from typing import List, Tuple, Dict, Union, Sequence
import math
import random
from functools import reduce

import gym
import numpy as np
from PIL import Image

from mazerunner_sim.envs.maze_generator import generate_maze
from mazerunner_sim.envs.visualisation.maze_render import render_agent_in_step, render_background
from mazerunner_sim.envs.agents.runner import Runner
from mazerunner_sim.utils.observation_and_action import Observation, Action


class MazeRunnerEnv(gym.Env):
    """OpenAI gym environment for the MazeRunner."""

    metadata = {'render.modes': ['human']}

    done: bool
    found_exit: Union[Runner, None]
    time: int
    total_rewards_given: float
    runners: List[Runner]
    tasks: List[Tuple[int, int]]

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
        if self.time % self.day_length == 0:    # Night
            reward = self._night_step(actions)
        else:   # Day
            reward = self._day_step(actions)

        self.total_rewards_given += reward

        # Observations
        observations = self.get_observations()

        # Increment time
        self.time += 1

        # End simulation if agents surpassed a year, the sim wil end
        if self.time > self.day_length * 365:
            self.done = True

        return observations, reward, self.done, {}

    def _day_step(self, actions: Dict[int, Action]) -> float:
        """Let the runners run during a day step"""
        reward = -1

        # Let the runners take a step
        for runner_id, action in actions.items():
            runner = self.runners[runner_id]
            if runner.alive:
                step = np.array([[0, -1],
                                 [0, 1],
                                 [-1, 0],
                                 [1, 0],
                                 [0, 0]][action.step_direction])
                # if the step is actually possible, take the step
                if self.maze[tuple(runner.location + step)[::-1]]:
                    runner.location += step

                    # update map
                    runner.update_map(
                        self.maze[runner.location[1] - 1:runner.location[1] + 2, runner.location[0] - 1:runner.location[0] + 2],
                        self.leaves[runner.location[1] - 1:runner.location[1] + 2, runner.location[0] - 1:runner.location[0] + 2]
                    )

                    # if found the exit
                    if runner.location[0] == 0 or runner.location[0] == self.maze.shape[1] - 1 or \
                            runner.location[1] == 0 or runner.location[1] == self.maze.shape[0] - 1:
                        self.done = True
                        self.found_exit = runner

        return reward

    def _night_step(self, actions: Dict[int, Action]) -> float:
        reward = 0

        # Kill the runners that are still in the maze
        if self.time % self.day_length == 0:
            for runner in self.runners:
                if not self.safe_zone[tuple(runner.location)]:
                    runner.alive = False

        # if all runners are dead
        if not any(runner.alive for runner in self.runners):
            self.done = True
            reward -= self.DEATH_PUNISHMENT + self.total_rewards_given

        if not self.done:
            # Share maps between those alive
            combined_explored_map = reduce(np.logical_or, [r.explored for r in self.runners if r.alive])
            combined_maze_map = reduce(np.logical_or, [r.known_maze for r in self.runners if r.alive])
            combined_leaves_map = reduce(np.logical_or, [r.known_leaves for r in self.runners if r.alive])
            for runner in self.runners:
                if runner.alive:
                    forget_mask = runner.memory_decay_map_generator()

                    runner.explored = np.logical_and(combined_explored_map.copy(), forget_mask.copy())
                    runner.known_maze = np.logical_and(combined_maze_map.copy(), forget_mask.copy())
                    runner.known_leaves = np.logical_and(combined_leaves_map.copy(), forget_mask.copy())

            # Assign tasks according to an auction
            worths = {i: [w / sum(action.task_worths) for w in action.task_worths] for i, action in actions.items()}
            self._auction_tasks(worths, self.tasks)

        return reward

    def _auction_tasks(self, worths: Dict[int, Sequence[float]], tasks: Sequence[Tuple[int, int]]):
        assignments: Dict[int, Tuple[Runner, float]] = {}  # key: task_id, value: (runner, bid)
        small_value = 1 / (len(worths) + 1)
        while len(assignments) < len(worths):
            for runner_id, runner in enumerate(self.runners):
                if runner_id in worths and not any(r == runner for r, _ in assignments.values()):
                    highest, second_highest = -math.inf, -math.inf
                    highest_task = None
                    for task_id in range(len(tasks)):
                        if task_id in assignments:
                            relative_value = worths[runner_id][task_id] - assignments[task_id][1]
                        else:
                            relative_value = worths[runner_id][task_id]
                        if relative_value >= highest:
                            highest, second_highest = relative_value, highest
                            highest_task = task_id
                        elif relative_value >= second_highest:
                            second_highest = relative_value
                    assignments[highest_task] = (runner, assignments.get(highest_task, (0, 0))[1] + highest - second_highest + small_value)

        for task_id, (runner, bid) in assignments.items():
            runner.assigned_task = tasks[task_id]

    def reset(self):
        """
        Reset the environment.

        The maze stays the same, so does the number of runners and the day-length,
        but the policies are reset, the time and the `done` flag.
        """
        self.done = False
        self.found_exit = None
        self.time = 0
        self.total_rewards_given = 0.

        center_coord = np.array([self.maze.shape[0] // 2] * 2)
        for runner in self.runners:
            runner.reset(center_coord.copy(), self.safe_zone, self.leaves)

    def get_observations(self, first_observation: bool = False) -> Dict[int, Observation]:
        """
        Get information about the environment location, returns walls.

        :return A list of runner-observations, take a look at it's documentation for more detail
        """
        tasks: List[Tuple[int, int]] = []

        if (self.time + 1) % self.day_length == 0 or first_observation:    # time-step before night
            # Make tasks from unexplored area
            combined_explored_map: np.array = reduce(np.logical_or, [r.explored for r in self.runners if r.alive])

            if np.all(combined_explored_map):
                tasks = self.tasks
            else:
                r_alive = sum([1 for r in self.runners if r.alive])
                while len(tasks) < math.ceil(r_alive * 2):
                    rand_x = random.randint(0, combined_explored_map.shape[1] - 1)
                    rand_y = random.randint(0, combined_explored_map.shape[0] - 1)
                    if not combined_explored_map[rand_y, rand_x]:
                        tasks.append((rand_x, rand_y))
                self.tasks = tasks

        return {
            runner_id: Observation(
                explored=runner.explored.copy(),
                known_maze=runner.known_maze.copy(),
                known_leaves=runner.known_leaves.copy(),
                safe_zone=self.safe_zone,
                runner_location=(runner.location[0], runner.location[1]),
                time_till_end_of_day=self.time_till_end_of_day(),
                action_speed=runner.action_speed,
                assigned_task=runner.assigned_task,
                tasks=tasks
            )
            for runner_id, runner in enumerate(self.runners)
            if runner.alive
        }

    def time_till_end_of_day(self) -> int:
        """Get number of time-steps left till the end of the day"""
        return self.day_length - (self.time % self.day_length) - 1

    def render(self, mode="human", follow_runner_id: int = None) -> Image:
        """
        Render the state of the environment.

        :param follow_runner_id: The index of the agent to follow what has been explored
        :param mode: Mode of rendering, choose between: ['human']
        """
        if self.done:
            print("Done")

        return render_agent_in_step(self, follow_runner_id)
