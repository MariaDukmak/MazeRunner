"""Generator for a maze for the envirement."""
# Maze generation using recursive backtracker algorithm
# https://en.wikipedia.org/wiki/Maze_generation_algorithm#Recursive_backtracker
from queue import LifoQueue

from random import choice
from random import randint

from typing import Tuple

from PIL import Image

import numpy as np

from mazerunner_sim.utils.pathfinder import manhattan_distance


def generate_maze(size: int = 16, center_size: int = 4) -> Tuple[np.array, np.array, np.array]:
    """
    Generate maze.

    :param size: size of the maze
    :param center_size: size of center section
    :return: numpy array of booleans with shape = [size * 2 + 1, size * 2 + 1]
    """
    # Modified from https://github.com/ravenkls/Maze-Generator-and-Solver/blob/master/maze_generator.py
    pixels = np.zeros((2 * size + 1, 2 * size + 1), dtype=bool)
    pixels[size - center_size + 1:size + center_size, size - center_size + 1:size + center_size] = True
    safe_zone = pixels.copy()

    # Creating exit
    random_height = randint(1, size * 2 - 1)
    exit_x, exit_y, offset_x, offset_y = choice([
        [0, random_height, 1, 0],  # left side
        [size * 2, random_height, -1, 0],  # right side
        [random_height, size * 2, 0, -1],  # bottom side
        [random_height, 0, 0, 1]  # top side
    ])
    pixels[exit_y, exit_x] = True
    pixels[exit_y + offset_y, exit_x + offset_x] = True

    stack = LifoQueue()
    cells = np.zeros((size, size), dtype=bool)
    cells[size // 2 - center_size // 2:size // 2 + center_size // 2, size // 2 - center_size // 2:size // 2 + center_size // 2] = True
    stack.put((size // 2 + center_size // 2, size // 2))

    while not stack.empty():
        x, y = stack.get()

        adjacents = []
        if x > 0 and not cells[x - 1, y]:
            adjacents.append((x - 1, y))
        if x < size - 1 and not cells[x + 1, y]:
            adjacents.append((x + 1, y))
        if y > 0 and not cells[x, y - 1]:
            adjacents.append((x, y - 1))
        if y < size - 1 and not cells[x, y + 1]:
            adjacents.append((x, y + 1))

        if adjacents:
            stack.put((x, y))

            neighbour = choice(adjacents)
            neighbour_on_img = (neighbour[0] * 2 + 1, neighbour[1] * 2 + 1)
            current_on_img = (x * 2 + 1, y * 2 + 1)
            wall_to_remove = (neighbour[0] + x + 1, neighbour[1] + y + 1)

            pixels[neighbour_on_img] = True
            pixels[current_on_img] = True
            pixels[wall_to_remove] = True

            cells[neighbour] = True
            stack.put(neighbour)

    # Creating the entries
    for x_b, y_b in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
        pixels[size + center_size * x_b, size + center_size * y_b] = True
        pixels[size + (center_size + 1) * x_b, size + (center_size + 1) * y_b] = True
    # Creating random openings in the walls
    corrupt_mask = np.zeros(pixels.shape)
    corrupt_mask[1:-1, 1:-1] = np.random.rand(*(pixels.shape - np.array([2, 2])))
    corrupt_mask = corrupt_mask > 0.95
    pixels = np.logical_or(pixels, corrupt_mask)

    # Leaves
    leaves = np.zeros((2 * size + 1, 2 * size + 1), dtype=float)
    for y in range(leaves.shape[0]):
        for x in range(leaves.shape[1]):
            if pixels[y, x]:
                leaves[y, x] = manhattan_distance((x, y), (exit_x, exit_y))
    leaves = np.random.rand(*leaves.shape) > leaves / (sum(leaves.shape) - 4)

    return pixels, safe_zone, leaves


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("size", nargs="?", type=int, default=16)
    parser.add_argument('--output', '-o', nargs='?', type=str, default='generated_maze.png')
    args = parser.parse_args()

    maze = generate_maze(args.size)
    image = Image.fromarray(maze * 255).convert('RGB')
    image.save(args.output)
