"""Policy that uses Bayes Theorem on the leaves."""

from typing import List

from mazerunner_sim.policies import PathFindingPolicy
from mazerunner_sim.utils.observation_and_action import Observation
from mazerunner_sim.utils.pathfinder import Coord


class LeafTrackerPolicy(PathFindingPolicy):
    """Policy that extends the pathfinding-policy to also consider the leaves in it's q-value function."""

    def __init__(self):
        """Initialize the policy."""
        super().__init__()

    @staticmethod
    def q_value_path(target_path: List[Coord], observation: Observation) -> float:
        """
        Calculate the estimated quality of that action/taking that path.

        The pathfinder policy will take the path of the highest expected quality,
        so the performance of the policy is highly dependant of the content of this function.

        This policy uses Bayes Theorem on the observed leaves to predict where the exit is more likely to be.

        :param target_path: path to evaluate the q-value of
        :param observation: observation to use as extra info to estimate the q-value
        :return: a float representing the q-value/quality of following the given path, higher = better
        """
        raise NotImplementedError()
