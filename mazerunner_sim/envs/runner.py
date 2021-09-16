"""The Runner class."""
from typing import Tuple

import numpy as np


class Runner:
    """
    The runner class for in the maze environment.
    """

    def __init__(self, start_location: np.array, safe_zone: np.array):
        """
        Initialize a Runner.

        :param start_location: Numpy vector with shape [2], it's x and y coordinate.
        """
        self.location = start_location
        self.alive = True
        self.explored = safe_zone.copy()
        self.known_maze = safe_zone.copy()

    def update_map(self, sensor_input: np.array):
        """
        Update the locally known maps.

        :param sensor_input: 3x3 block of booleans surrounding the runner where False is a wall and True is an open space
        """
        x, y = self.location
        self.explored[y - 1:y + 2, x - 1:x + 2] = True
        self.known_maze[y - 1:y + 2, x - 1:x + 2] = sensor_input
