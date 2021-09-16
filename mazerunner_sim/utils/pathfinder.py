from typing import Tuple, List, Callable, Dict
from queue import PriorityQueue

import numpy as np

Coord = Tuple[int, int]
DistanceFunc = Callable[[Coord], float]


def manhattan_distance(p1: Coord, p2: Coord) -> int:
    """Calculate the Manhattan distance between two locations"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def target_to_distance_func(target: Coord) -> DistanceFunc:
    return lambda p: manhattan_distance(p, target)


def traceback_visited(visited_log: Dict[Coord, Coord], end: Coord, origin: Coord) -> List[Coord]:
    path = [end]
    while visited_log[end] != origin and end != origin:
        path.append(visited_log[end])
        end = visited_log[end]
    path.reverse()
    return path


def steps_to_targets(origin: Coord, targets: List[Coord], maze: np.array) -> List[List[Coord]]:
    for x, y in targets + [origin]:
        assert maze[y, x]

    targets_togo = list(targets)
    if origin in targets_togo:
        targets_togo.remove(origin)
    visited_log = {origin: None}  # key: next tile, value: previous tile
    q = PriorityQueue()
    q.put((0, origin))
    while len(targets_togo) > 1:
        priority, tile = q.get()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            next_tile = tile[0] + dx, tile[1] + dy
            if maze[next_tile[1], next_tile[0]] and next_tile not in visited_log:
                q.put((priority + 1, next_tile))
                visited_log[next_tile] = tile
                if next_tile in targets_togo:
                    targets_togo.remove(next_tile)
    if len(targets_togo) == 1:
        done = False
        distance_func = target_to_distance_func(targets_togo[0])
    else:
        done = True
        distance_func = None
    while not done:
        _, tile = q.get()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            next_tile = tile[0] + dx, tile[1] + dy
            if maze[next_tile[1], next_tile[0]] and next_tile not in visited_log:
                q.put((distance_func(next_tile), next_tile))
                visited_log[next_tile] = tile
                if next_tile in targets_togo:
                    done = True
                    break

    return [traceback_visited(visited_log, target, origin) for target in targets]
