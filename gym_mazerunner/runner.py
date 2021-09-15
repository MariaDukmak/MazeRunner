"""The Runner class."""

import numpy.typing as npt


class Runner:
    """
    The runner class for in the maze.

    Just holds it's location and whether or not it's still alive.
    """
    def __init__(self, start_location: npt.NDArray):
        self.location = start_location
        self.alive = True
