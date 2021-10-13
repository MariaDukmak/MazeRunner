"""Policy that uses Bayes Theorem on the leaves."""

from typing import List, Tuple

import numpy as np

from mazerunner_sim.policies import PathFindingPolicy
from mazerunner_sim.utils.observation_and_action import Observation, Action
from mazerunner_sim.utils.pathfinder import Coord, manhattan_distance


class LeafTrackerPolicy(PathFindingPolicy):
    """Policy that extends the pathfinding-policy to also consider the leaves in it's q-value function."""

    def __init__(self, path_length_weight: float = 1., leaf_weight: float = 100.):
        """
        Initialize the policy.

        :param path_length_weight: How much the distance to the outside of the maze should weigh in the evaluation of a path
        :param leaf_weight: How much the expected exist according to the leaves should weigh in the evaluation of a path
        """
        super().__init__(path_length_weight=path_length_weight)
        self.leaf_weight = leaf_weight

    def decide_action(self, observation: Observation) -> Action:
        """Take an action, using the observed leaves."""
        known_leaves, known_not_leaves = [], []
        for y, x in zip(*np.where(observation.known_maze)):
            if observation.known_leaves[y, x]:
                known_leaves.append((x, y))
            else:
                known_not_leaves.append((x, y))

        # This part could be more efficient
        # Calculate the probability for each possible exit
        height, width = observation.known_maze.shape
        self.probs_exits = {
            proposed_exit: self.calc_probability_exit(proposed_exit, known_leaves, known_not_leaves, observation.known_maze.shape)
            for proposed_exit in [
                (0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1),
                (0, height // 2), (width // 2, 0), (width - 1, height // 2), (height // 2, height - 1)
            ]
        }

        return super().decide_action(observation)

    def q_value_path(self, target_path: List[Coord], observation: Observation) -> float:
        """
        Calculate the estimated quality of that action/taking that path.

        The pathfinder policy will take the path of the highest expected quality,
        so the performance of the policy is highly dependant of the content of this function.

        This policy uses Bayes Theorem on the observed leaves to predict where the exit is more likely to be.

        :param target_path: path to evaluate the q-value of
        :param observation: observation to use as extra info to estimate the q-value
        :return: a float representing the q-value/quality of following the given path, higher = better
        """
        target_x, target_y = target_path[-1]
        q_value = 0

        # Distance from current position
        q_value -= len(target_path) * self.path_length_weight

        # # Distance to outside
        # map_width, map_height = observation.known_maze.shape
        # q_value -= min(target_x, map_width - target_x, target_y, map_height - target_y)

        # Expected exit according to the leaves
        for anchor, prob in self.probs_exits.items():
            q_value += prob / (manhattan_distance((target_x, target_y), anchor) + 0.01) * self.leaf_weight

        return q_value

    @staticmethod
    def calc_probability_exit(proposed_exit: Coord, known_leaves: List[Coord],
                              known_not_leaves: List[Coord], maze_shape: Tuple[int, int]) -> float:
        """
        Calculate the probability of the proposed_exit being the actual exit given the known leaves.

        :param proposed_exit: the proposed exit location
        :param known_leaves: the coordinates known to have leaves
        :param known_not_leaves: the coordinates known to not have leaves
        :return: The probability of the proposed exit being the actual exit,
                 the probability isn't on the 0.0 - 1.0 scale, but relative to other exits.
                 Because this is less calculation and all that's needed to compare probable exits.
        :param maze_shape: Shape of the maze
        """
        largest_dist = sum(maze_shape) - 4
        return float(np.prod([1 - manhattan_distance(coord, proposed_exit) / largest_dist for coord in known_leaves] +
                             [manhattan_distance(coord, proposed_exit) / largest_dist for coord in known_not_leaves]
                             )) * 2**(len(known_not_leaves) + len(known_leaves))

    def q_task(self, observation: Observation) -> List[float]:
        """
        Calculate the estimated quality of each of the given tasks.

        :param observation:
        :return:
        """
        # return [
        #     manhattan_distance(, task) / (observation.action_speed + 1)
        #     for task in observation.tasks
        # ]
        raise NotImplementedError()
