"""Module containing some useful functions for pathfinding."""

from typing import Callable, Dict, List, Tuple
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


def coord_within_boundary(maze: np.array, coord: Coord) -> bool:
    """
    Check if the given coordinate lays within maze boundaries.

    :param maze: a 2D boolean array where True means walkable tile and False means a wall
    :param coord: a 2D coordinate
    :return: false if out of bounds, true if inbounds
    """
    x, y = coord
    height, width = maze.shape
    return 0 <= x < width and 0 <= y < height


def surrounding_tiles(coord: Coord) -> List[Coord]:
    """
    Get surrounding tiles from giving tile.

    :param coord: Given coord to check surrounding
    :return: np array of surrounding_tiles
    """
    x, y = coord
    surrounding = [(x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)]
    return surrounding


def compute_explore_paths(start_coord: Coord, runner_known_map: np.array, runner_explored_map: np.array):
    edge_tiles = []
    visited_tiles = {start_coord: None}  # key: next_tile, value: previous_tile

    q = PriorityQueue()
    q.put((0, start_coord))
    while not q.empty():
        # gets the tile with the lowest priority value
        priority, tile = q.get()

        for next_tile in surrounding_tiles(tile):
            # Check coords with maze if you can walk there
            if runner_known_map[next_tile[1], next_tile[0]] and next_tile not in visited_tiles:
                if next_tile == (16, 12):
                    x = 0
                    print('debug')

                x, y = tile[0]+1, tile[1]+1
                explored = np.pad(runner_explored_map, (1, 1), 'constant', constant_values=False)
                #     UP                          LEFT                   RIGHT                       DOWN
                if not(explored[y - 1, x] and explored[y, x - 1] and explored[y, x + 1] and explored[y + 1, x]):
                    edge_tiles.append(next_tile)
                q.put((priority + 1, next_tile))

                visited_tiles[next_tile] = tile

    paths = [traceback_visited(visited_tiles, edge_tile, start_coord) for edge_tile in edge_tiles]
    return paths


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
            if coord_within_boundary(maze, next_tile):
                if maze[next_tile[1], next_tile[0]] and (next_tile not in visited_tree):
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
            if coord_within_boundary(maze, next_tile):
                if maze[next_tile[1], next_tile[0]] and (next_tile not in visited_tree):
                    q.put((distance_func(next_tile), next_tile))
                    visited_tree[next_tile] = tile
                    if next_tile in targets_togo:
                        done = True
                        break


    # Trace back the path for each of the targets
    return [traceback_visited(visited_tree, target, origin) for target in targets]
