"""The Runner class."""

import numpy as np


class Runner:
    """The runner class for in the maze environment."""

    def __init__(self, start_location: np.array, safe_zone: np.array, leaves: np.array):
        """
        Initialize a Runner.

        :param start_location: Numpy vector with shape [2], it's x and y coordinate.
        """
        self.location = start_location
        self.alive = True
        self.explored = safe_zone.copy()
        self.known_maze = safe_zone.copy()
        self.known_leaves = np.logical_and(leaves, self.explored)

    def update_map(self, maze_input: np.array, leaves_input: np.array):
        """
        Update the locally known maps.

        :param leaves_input: 3x3 block of booleans surrounding the runner where False is no leaf and True is a leaf
        :param maze_input: 3x3 block of booleans surrounding the runner where False is a wall and True is an open space
        """
        x, y = self.location
        self.explored[y - 1:y + 2, x - 1:x + 2] = True
        self.known_maze[y - 1:y + 2, x - 1:x + 2] = maze_input
        self.known_leaves[y - 1:y + 2, x - 1:x + 2] = leaves_input
