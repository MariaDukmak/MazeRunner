import numpy as np
import numpy.typing as npt


class Runner:
    def __init__(self, start_location: npt.NDArray):
        self.location = start_location
        self.alive = True
