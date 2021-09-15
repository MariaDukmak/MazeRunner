"""
Observations and actions are the most basic information flow in an agent system.
An agent takes an action, the environment uses that action to update and returns a new observation, rinse and repeat.

Observations and actions are just data containers.
"""

from typing import NamedTuple, Tuple
import numpy as np


class Observation(NamedTuple):
    """
    The Maze Runner environment returns observations, agents need this to make decisions.

    An observation consist of the following thing:
        explored: map of booleans the size of the entire maze, True means explored, False means not explored
        known_maze: map of booleans the size of the entire maze, True means open to walk, False means a wall or unexplored
        runner_location: Coordinate location of the runner (x, y)
        time_till_end_of_day: Time till the end of the day, decreases from `day_length` to 0

    This can be expanded to have more observation parameters in the future as the simulation development continuous.
    """
    explored: np.array
    known_maze: np.array
    runner_location: Tuple[int, int]
    time_till_end_of_day: int


class Action(int):
    """An action in the MazeRunner environment is just an integer between 0 and 3."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
