"""The Runner class."""

import numpy as np


class Runner:
    """The runner class for in the maze environment."""

    location: np.array
    alive: bool
    explored: np.array
    known_maze: np.array
    known_leaves: np.array
    memory_decay_map: np.array

    def __init__(self, action_speed: int = 10, memory_decay_percentage: int = 0):
        """
        Initialize a Runner.

        :param action_speed: Runner speed in the simulation.
        """
        self.action_speed = action_speed  # Base speed
        self.action_wait_time = self.action_speed  # Current speed in a step
        self.memory_decay_percentage = memory_decay_percentage

    def update_map(self, maze_input: np.array, leaves_input: np.array) -> None:
        """
        Update the locally known maps.

        :param leaves_input: 3x3 block of booleans surrounding the runner where False is no leaf and True is a leaf
        :param maze_input: 3x3 block of booleans surrounding the runner where False is a wall and True is an open space
        """
        x, y = self.location
        self.explored[y - 1:y + 2, x - 1:x + 2] = True



        self.known_maze[y - 1:y + 2, x - 1:x + 2] = maze_input
        self.known_leaves[y - 1:y + 2, x - 1:x + 2] = leaves_input

    def check_status_speed(self) -> bool:
        """Update the status of the agent."""
        if self.action_wait_time == 0:
            self.action_wait_time = self.action_speed
            return True
        else:
            self.action_wait_time -= 1
            return False

    def memory_decay_map_generator(self) -> np.array:
        """
        Generates a memory decay map.

        return: numpy array of decayed map
        """
        amount_cells_decayed = self.known_maze.size * (self.memory_decay_percentage / 100)
        zeros = np.zeros(self.known_maze.size - amount_cells_decayed, dtype=bool)
        ones = np.ones(amount_cells_decayed, dtype=bool)

        filter_array = np.concatenate((ones, zeros), axis=0, out=None)

        np.random.shuffle(filter_array)
        return filter_array



    def reset(self, start_location: np.array, safe_zone: np.array, leaves: np.array) -> None:
        """Reset the status of the agent."""
        self.location = start_location
        self.alive = True
        self.explored = safe_zone.copy()
        self.known_maze = safe_zone.copy()
        self.known_leaves = np.logical_and(leaves, self.explored)
