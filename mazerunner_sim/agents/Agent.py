"""Class file for making agents."""
import abc

import numpy.typing as npt


class Agent(metaclass=abc.ABCMeta):
    """Initialization of agent class."""

    @abc.abstractmethod
    def decide_action(self, observation: npt.NDArray) -> int:
        """Decide action function."""
        raise NotImplementedError
