"""Policy that uses pathfinding."""

from typing import List
from math import ceil

import numpy as np

from mazerunner_sim.policies import BasePolicy
from mazerunner_sim.utils.observation_and_action import Observation, Action
from mazerunner_sim.utils.pathfinder import Coord, paths_origin_targets, compute_explore_paths, manhattan_distance


def next_coord_to_step(next_coord: Coord, old_coord: Coord) -> int:
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

    def __init__(self, outside_weight: float = 1., path_length_weight: float = 1., task_weight: float = 1.):
        """
        Initialize the policy.

        :param outside_weight: How much the distance to the outside of the maze should weigh in the evaluation of a path
        :param path_length_weight: How much the length of the disputed path should weigh in the evaluation of the path
        """
        self.planned_path = []
        self.outside_weight = outside_weight
        self.path_length_weight = path_length_weight
        self.task_weight = task_weight

    def decide_action(self, observation: Observation) -> Action:
        """Take an action, using path finding."""
        # When there is no path planned, plan a new plan
        if len(self.planned_path) == 0:

            explore_paths = compute_explore_paths(observation.runner_location, observation.known_maze, observation.explored)
            explorable_tiles = [path[-1] for path in explore_paths]

            center = observation.known_maze.shape[1] // 2, observation.known_maze.shape[0] // 2
            *retreat_paths, center_path = paths_origin_targets(center,
                                                               explorable_tiles + [observation.runner_location],
                                                               observation.known_maze)
            center_path = center_path[::-1]

            *retreat_paths, center_path = [clip_retreat_path(observation.safe_zone, p) for p in retreat_paths + [center_path]]

            # Only keep the paths that can be done and retreated within the time left
            target_validation_mask = [len(tp) + len(tcp) < observation.time_till_end_of_day / (observation.action_speed + 1)
                                      for tp, tcp in zip(explore_paths, retreat_paths)]
            if sum(target_validation_mask) > 0:
                # filter those paths
                explore_paths = [x for x, valid in zip(explore_paths, target_validation_mask) if valid]

                q_values_paths = [self.q_value_path(target_path, observation) for target_path in explore_paths]
                best_paths = [path for path, score in zip(explore_paths, q_values_paths) if score == max(q_values_paths)]

                self.planned_path.extend(best_paths[np.random.randint(len(best_paths))])
            else:
                self.planned_path.extend(center_path)
                wait_place = center_path[-1] if len(center_path) > 0 else observation.runner_location
                self.planned_path.extend([wait_place] * ceil(observation.time_till_end_of_day /
                                                             (observation.action_speed + 1) - len(center_path)))

        # Follow the planned path
        if len(observation.tasks) == 0:
            step_direction = next_coord_to_step(self.planned_path.pop(0), observation.runner_location)
        else:
            step_direction = Action.STAY
        action = Action(
            step_direction=step_direction,
            task_worths=self.q_task(observation)
        )
        return action

    def q_value_path(self, target_path: List[Coord], observation: Observation) -> float:
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
        distance_to_task = manhattan_distance(observation.assigned_task, (target_x, target_y))
        return -(distance_to_outside * self.outside_weight +
                 len(target_path) * self.path_length_weight +
                 distance_to_task * self.task_weight)

    def q_task(self, observation: Observation) -> List[float]:
        """
        Calculate the estimated quality of each of the given tasks.

        :return:
        """
        return [
            -(manhattan_distance(observation.runner_location, task) - observation.time_till_end_of_day / (observation.action_speed+1))**2
            for task in observation.tasks
        ]

    def reset(self):
        """Reset the planned path of the policy."""
        self.planned_path = []
