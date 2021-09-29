"""Policy that uses pathfinding."""

from typing import List
from math import ceil

import numpy as np

from mazerunner_sim.policies import BasePolicy
from mazerunner_sim.utils.observation_and_action import Observation, Action
from mazerunner_sim.utils.pathfinder import Coord, paths_origin_targets


def next_coord_to_action(next_coord: Coord, old_coord: Coord) -> Action:
    """
    Convert a adjacent next coordinate to an action.

    :param next_coord: next adjacent coordinate
    :param old_coord: old coordinate as a reference
    :return: action that can be used in the environment
    """
    step = next_coord[0] - old_coord[0], next_coord[1] - old_coord[1]
    return {
        (0, -1): Action.UP,
        (0, 1): Action.DOWN,
        (-1, 0): Action.LEFT,
        (1, 0): Action.RIGHT,
        (0, 0): Action.STAY
    }[step]


def find_edge_of_knowledge_tiles(known_maze: np.array, explored: np.array) -> List[Coord]:
    """
    Find which tiles are on the edge of the known maze.

    :param known_maze: the known maze of the runner
    :param explored: a mask of what's explored by the runner
    :return: list of coordinates of all the tiles at the edge of the known maze (and not a wall)
    """
    explored = np.pad(explored, (1, 1), 'constant', constant_values=False)
    return [(x, y)
            for y, x in zip(*np.where(known_maze)) if not
            (explored[y + 2, x + 1] and explored[y, x + 1] and explored[y + 1, x + 2] and explored[y + 1, x])
            ]


def clip_retreat_path(safe_zone: np.array, path: List[Coord]) -> List[Coord]:
    """
    Clip the retreat path so it stops as soon as it's in a safe spot.

    :param safe_zone: map of which tiles are safe
    :param path: original path to safe zone
    :return: new path to safe zone that stops at the first safe spot
    """
    clipped = [(x, y) for x, y in path if not safe_zone[y, x]]
    return clipped + [path[len(clipped)]]


class PathFindingPolicy(BasePolicy):
    """Policy that uses path finding to retreat at the right time and plans new tiles to explore."""

    def __init__(self):
        """Initialize the policy."""
        self.planned_path = []

    def decide_action(self, observation: Observation) -> Action:
        """Take an action, using path finding."""

        # When there is no path planned, plan a new plan
        if len(self.planned_path) == 0:
            # Tiles at the edge of knowledge can be explored
            explorable_tiles = find_edge_of_knowledge_tiles(observation.known_maze, observation.explored)

            center = observation.known_maze.shape[1] // 2, observation.known_maze.shape[0] // 2
            # Compute the paths to the explorable tiles and the path back to the center
            *explore_paths, center_path = paths_origin_targets(observation.runner_location, explorable_tiles + [center],
                                                               observation.known_maze)
            # When the runner is at the explore tile, what is the path to retreat to the center
            retreat_paths = paths_origin_targets(center, explorable_tiles, observation.known_maze)
            *retreat_paths, center_path = [clip_retreat_path(observation.safe_zone, p) for p in retreat_paths + [center_path]]

            # Only keep the paths that can be done and retreated within the time left
            target_validation_mask = [len(tp) + len(tcp) < observation.time_till_end_of_day / (observation.action_speed+1)
                                      for tp, tcp in zip(explore_paths, retreat_paths)]
            if sum(target_validation_mask) > 0:
                # filter those paths
                explorable_tiles = [x for x, valid in zip(explorable_tiles, target_validation_mask) if valid]
                explore_paths = [x for x, valid in zip(explore_paths, target_validation_mask) if valid]

                q_values_paths = [self.q_value_path(target_path, observation) for target_path in explore_paths]
                best_paths = [path for path, score in zip(explore_paths, q_values_paths) if score == max(q_values_paths)]

                self.planned_path.extend(best_paths[np.random.randint(len(best_paths))])
            else:
                self.planned_path.extend(center_path)
                wait_place = center_path[-1] if len(center_path) > 0 else observation.runner_location
                self.planned_path.extend([wait_place] * ceil(observation.time_till_end_of_day / (observation.action_speed+1) - len(center_path)))

        # Follow the planned path
        return next_coord_to_action(self.planned_path.pop(0), observation.runner_location)

    @staticmethod
    def q_value_path(target_path: List[Coord], observation: Observation) -> float:
        """
        Calculate the estimated quality of that action/taking that path.

        The pathfinder policy will take the path of the highest expected quality,
        so the performance of the policy is highly dependant of the content of this function.

        This policy uses the distance to the outside of the maze at the end and the length of the path.

        :param target_path: path to evaluate the q-value of
        :param observation: observation to use as extra info to estimate the q-value
        :return: a float representing the q-value/quality of following the given path, higher = better
        """
        target_x, target_y = target_path[-1]
        map_width, map_height = observation.known_maze.shape

        distance_to_outside = min(target_x, map_width - target_x, target_y, map_height - target_y)
        return -(distance_to_outside + len(target_path))

    def reset(self):
        """Reset the planned path of the policy."""
        self.planned_path = []
