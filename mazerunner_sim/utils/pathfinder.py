"""Module containing some useful functions for pathfinding."""

from typing import Tuple, List, Callable, Dict
from queue import PriorityQueue

import numpy as np

Coord = Tuple[int, int]
DistanceFunc = Callable[[Coord], float]


def manhattan_distance(p1: Coord, p2: Coord) -> int:
    """
    Calculate the Manhattan distance between two coordinates.

    :param p1: First point
    :param p2: Second point
    :return: Distance between the two points
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def target_to_distance_func(target: Coord) -> DistanceFunc:
    """
    Create a manhattan distance function to the given coordinate.

    :param target: the reference point
    :return: distance function that returns the manhattan distance to target
    """
    return lambda p: manhattan_distance(p, target)


def traceback_visited(visited_tree: Dict[Coord, Coord], end: Coord, origin: Coord) -> List[Coord]:
    """
    Use the visited_log to trace the path back to the beginning.

    :param visited_tree: dictionary that represents the relation between next-tile as key and previous-tile as value
    :param end: end of the path
    :param origin: the original coordinate shouldn't get included in the path
    :return: a list of coordinates from the origin to the end
    """
    path = [end]
    while visited_tree[end] != origin and end != origin:
        path.append(visited_tree[end])
        end = visited_tree[end]
    path.reverse()
    return path

def check_boundary(maze: np.array, next_tile: np.array):
    """
    Check if next tile lays within maze boundarys

    :param maze: a 2D boolean array where True means walkable tile and False means a wall
    :param next_tile: ...
    """
    boundery = range(0, maze.shape[0] - 1)

    if not next_tile[0] in boundery or not next_tile[1] in boundery:
        return False
    else:
        return True


def paths_origin_targets(origin: Coord, targets: List[Coord], maze: np.array) -> List[List[Coord]]:
    """
    Find the paths from the origin to the targets.

    Finding the path to multiple targets from the same origin is faster than doing them all separately,
    by avoiding redoing a bunch of computations.
    The path to one target can still be done (and just as fast) by giving a list with one item.

    Warning:
    This implementation assumes there these paths are possible,
    otherwise the function will hang.

    :param origin: the origin coordinate from which each path will start
    :param targets: the ends where each path should end
    :param maze: a 2D boolean array where True means walkable tile and False means a wall
    :return: A list of paths corresponding with each end
    """
    for x, y in targets + [origin]:
        assert maze[y, x]

    targets_togo = list(targets)
    if origin in targets_togo:
        targets_togo.remove(origin)
    visited_tree = {origin: None}  # key: next tile, value: previous tile
    q = PriorityQueue()
    q.put((0, origin))



    # Dijkstra to all the targets, until one target is left
    while len(targets_togo) > 1:
        priority, tile = q.get()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            next_tile = tile[0] + dx, tile[1] + dy

            if not check_boundary(maze, next_tile):
                break

            if maze[next_tile[1], next_tile[0]] and next_tile not in visited_tree:
                q.put((priority + 1, next_tile))
                visited_tree[next_tile] = tile
                if next_tile in targets_togo:
                    targets_togo.remove(next_tile)

    if len(targets_togo) == 1:
        done = False
        distance_func = target_to_distance_func(targets_togo[0])
    else:
        done = True
        distance_func = None

    # A* for the last target for better efficiency
    while not done:
        _, tile = q.get()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            next_tile = tile[0] + dx, tile[1] + dy
            if maze[next_tile[1], next_tile[0]] and next_tile not in visited_tree:
                q.put((distance_func(next_tile), next_tile))
                visited_tree[next_tile] = tile
                if next_tile in targets_togo:
                    done = True
                    break

    # Trace back the path for each of the targets
    return [traceback_visited(visited_tree, target, origin) for target in targets]
