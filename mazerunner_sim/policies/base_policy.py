"""Class file for making policies."""
import abc

from mazerunner_sim.utils.observation_and_action import Observation, Action


class BasePolicy(metaclass=abc.ABCMeta):
    """
    Most generic agent class.

    Each agent should have the function observation_action.
    """

    @abc.abstractmethod
    def decide_action(self, observation: Observation) -> Action:
        """Take an action based on the given observation."""
        raise NotImplementedError
