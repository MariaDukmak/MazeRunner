"""Policy that uses pathfinding."""

from typing import List

import numpy as np

from mazerunner_sim.policies import BasePolicy
from mazerunner_sim.observation_and_action import Observation, Action
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
        map_height, map_width = observation.known_maze.shape

        # if observation.status_speed != 0:
        #     return Action.STAY

        # When there is no path planned, plan a new plan
        if len(self.planned_path) == 0:
            explorable_tiles = find_edge_of_knowledge_tiles(observation.known_maze, observation.explored)

            center = observation.known_maze.shape[1] // 2, observation.known_maze.shape[0] // 2
            *explore_paths, center_path = paths_origin_targets(observation.runner_location, explorable_tiles + [center],
                                                               observation.known_maze)
            retreat_paths = paths_origin_targets(center, explorable_tiles, observation.known_maze)
            *retreat_paths, center_path = [clip_retreat_path(observation.safe_zone, p) for p in retreat_paths + [center_path]]

            target_validation_mask = [len(tp) + len(tcp) <= observation.time_till_end_of_day
                                      for tp, tcp in zip(explore_paths, retreat_paths)]
            if sum(target_validation_mask) > 0:
                explorable_tiles = [x for x, valid in zip(explorable_tiles, target_validation_mask) if valid]
                explore_paths = [x for x, valid in zip(explore_paths, target_validation_mask) if valid]
                tile_scores = [
                    min(x, map_width - x, y, map_height - y) + len(target_path)
                    for (x, y), target_path in zip(explorable_tiles, explore_paths)
                ]

                best_paths = [path for path, score in zip(explore_paths, tile_scores) if score == min(tile_scores)]
                self.planned_path.extend(best_paths[np.random.randint(len(best_paths))])
            else:
                self.planned_path.extend(center_path)
                wait_place = center_path[-1] if len(center_path) > 0 else observation.runner_location
                self.planned_path.extend([wait_place] * (observation.time_till_end_of_day + 1 - len(center_path)))

        # Follow the planned path
        return next_coord_to_action(self.planned_path.pop(0), observation.runner_location)
