"""This file contains code to render a maze."""

import os

from PIL import Image


def render_background(maze, save_background: bool = False, file_name: str = "maze.png"):
    """
    Render the maze and save it if specified.

    :param maze: Maze from generate maze function in np 2D array
    """
    # Declare with images used for walls and path
    walls = Image.open("textures/stonebrick.png")
    path = Image.open("textures/dirt.png")

    # Get width and height of the walls
    tile_size_width, tile_size_height = walls.size

    # Calculate canvas size of the maze
    width = maze.shape[0] * tile_size_width
    height = maze.shape[1] * tile_size_height

    # Create canvas
    background = Image.new(mode="RGB", size=(width, height))

    # Loop through values from the maze and determine where the walls and path are
    for height_row, width_values in enumerate(maze):
        for index, value in enumerate(width_values):
            if value:
                background.paste(path, (index * tile_size_width, height_row * tile_size_height))
            else:
                background.paste(walls, (index * tile_size_width, height_row * tile_size_height))

    # Save maze
    if save_background:
        if not os.path.exists("maze_output"):
            os.makedirs("maze_output")
        background.save(os.path.join("maze_output", file_name))


if __name__ == '__main__':
    from maze_generator import generate_maze

    render_background(generate_maze(), True)
