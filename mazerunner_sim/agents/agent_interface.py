"""Class file for making agents."""
import abc

from mazerunner_sim.observation_and_action import RunnerObservation, Action


class Agent(metaclass=abc.ABCMeta):
    """
    Most generic agent class.

    Each agent should have the function observation_action.
    """

    @abc.abstractmethod
    def observation_action(self, observation: RunnerObservation) -> Action:
        """Take an action based on the given observation."""
        raise NotImplementedError
