"""
Observations and actions are the most basic information flow in an agent system.

An agent takes an action, the environment uses that action to update and returns a new observation, rinse and repeat.
Observations and actions are just data containers.
"""

from typing import NamedTuple, Tuple, Sequence

import numpy as np


class Observation(NamedTuple):
    """
    The Maze Runner environment returns observations, policies need this to make decisions.

    An observation consist of the following thing:
        explored: map of booleans the size of the entire maze, True means explored, False means not explored
        known_maze: map of booleans the size of the entire maze, True means open to walk, False means a wall or unexplored
        safe_zone: map of booleans the size of the entire maze, True means safe to be at the end of the day
        runner_location: Coordinate location of the runner (x, y)
        time_till_end_of_day: Time till the end of the day, decreases from `day_length` to 0
        action_speed: Number of simulation steps between each action, this effectively forms the runners speed
        assigned_task: The location the runner has been assigned to explore
        tasks: The tasks available to be assigned, the policy should return a value for each tasks what it thinks it's worth

    This can be expanded to have more observation parameters in the future as the simulation development continuous.
    """

    explored: np.array
    known_maze: np.array
    known_leaves: np.array
    safe_zone: np.array
    runner_location: Tuple[int, int]
    time_till_end_of_day: int
    action_speed: int
    assigned_task: Tuple[int, int]
    tasks: Sequence[Tuple[int, int]]


class Action(NamedTuple):
    """
    The Maze Runner environment takes actions to change the environment, policies create these actions.

    An action consist of the following thing:
        step_direction: The direction to step to, it's an integer between 0 and 4
        task_worths: A value that the policy thinks a task is worth, for each task in the observation
    """

    step_direction: int
    task_worths: Sequence[float]
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
