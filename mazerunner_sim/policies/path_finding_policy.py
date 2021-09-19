"""Semi-random agent, that goes back in time using pathfinding."""

from typing import List
import numpy as np

from mazerunner_sim.policies import BasePolicy
from mazerunner_sim.observation_and_action import Observation, Action
from mazerunner_sim.utils.pathfinder import Coord, steps_to_targets


def next_coord_to_action(next_coord: Coord, old_coord: Coord) -> Action:
    step = next_coord[0] - old_coord[0], next_coord[1] - old_coord[1]
    return {
        (0, -1): Action.UP,
        (0, 1): Action.DOWN,
        (-1, 0): Action.LEFT,
        (1, 0): Action.RIGHT,
        (0, 0): Action.STAY
    }[step]


def explorable_tiles(known_maze: np.array, explored: np.array) -> List[Coord]:
    explored = np.pad(explored, (1, 1), 'constant', constant_values=False)
    return [(x, y)
            for y, x in zip(*np.where(known_maze)) if not
            (explored[y+2, x+1] and explored[y, x+1] and explored[y+1, x+2] and explored[y+1, x])
            ]


def clip_retreat_path(safe_zone: np.array, path: List[Coord]) -> List[Coord]:
    clipped = [(x, y) for x, y in path if not safe_zone[y, x]]
    return clipped + [path[len(clipped)]]


class PathFindingAgent(BasePolicy):
    """Semi-random agent, that goes back in time using pathfinding."""
    def __init__(self):
        self.planned_path = []

    def decide_action(self, observation: Observation) -> Action:
        map_height, map_width = observation.known_maze.shape
        if len(self.planned_path) == 0:
            # When there is no path planned, plan a new plan
            explore_tiles = explorable_tiles(observation.known_maze, observation.explored)

            center = observation.known_maze.shape[1] // 2, observation.known_maze.shape[0] // 2
            *target_paths, center_path = steps_to_targets(observation.runner_location, explore_tiles+[center], observation.known_maze)
            retreat_paths = steps_to_targets(center, explore_tiles, observation.known_maze)
            *retreat_paths, center_path = [clip_retreat_path(observation.safe_zone, p) for p in retreat_paths + [center_path]]

            target_validation_mask = [len(tp) + len(tcp) <= observation.time_till_end_of_day for tp, tcp in zip(target_paths, retreat_paths)]
            if sum(target_validation_mask) > 0:
                explore_tiles = [x for x, valid in zip(explore_tiles, target_validation_mask) if valid]
                target_paths = [x for x, valid in zip(target_paths, target_validation_mask) if valid]
                tile_scores = [
                    min(x, map_width-x, y, map_height-y) + len(target_path)
                    for (x, y), target_path in zip(explore_tiles, target_paths)
                ]

                best_paths = [path for path, score in zip(target_paths, tile_scores) if score == min(tile_scores)]
                self.planned_path.extend(best_paths[np.random.randint(len(best_paths))])
            else:
                self.planned_path.extend(center_path)
                wait_place = center_path[-1] if len(center_path) > 0 else observation.runner_location
                self.planned_path.extend([wait_place]*(observation.time_till_end_of_day+1 - len(center_path)))

        # Follow the planned path
        return next_coord_to_action(self.planned_path.pop(0), observation.runner_location)
