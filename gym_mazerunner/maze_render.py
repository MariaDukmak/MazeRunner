"""This file contains code to render a maze."""

import os
from pathlib import Path

from PIL import Image

from gym_mazerunner.runner import Runner

from typing import List

import numpy as np

textures_path = Path(__file__) / '..' / 'textures'


def render_background(maze: np.array, save_background: bool = False, file_name: str = "maze.png") -> Image:
    """
    Render the maze and save it if specified.

    :param maze: Maze from generate maze function in np 2D array
    :param save_background
    :param file_name
    """
    # Declare with images used for walls and path
    walls = Image.open(textures_path / "stonebrick.png")
    path = Image.open(textures_path / "dirt.png")

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

    return background


def render_agent_in_step(maze_information: np.array, background_image: Image, runners: List[Runner],
                         step: int = 0, want_to_save: bool = False) -> Image:
    """
    Render all agents from step.

    :param maze_information: Information about the maze
    :param background_image: Rendered background from maze information
    :param runners: List of all runners
    :param step: Current step of the environment
    :param want_to_save: Save current step to folder
    :return:
    """
    # Copy from background
    copy_background = background_image.copy()
    # Declare with images used for the agent
    agent_icon = Image.open(textures_path / "agent.png")

    # Calculate canvas size of the maze
    tile_width = background_image.width // maze_information.shape[0]
    tile_height = background_image.height // maze_information.shape[1]

    for index, runner in enumerate(runners):
        agent_icon_copy = agent_icon.copy()

        # TODO change each agent with index variable

        copy_background.paste(agent_icon_copy, (runner.location[0] * tile_width, runner.location[1] * tile_height), agent_icon)

    if want_to_save:
        if not os.path.exists("maze_output"):
            os.makedirs("maze_output")
        copy_background.save(os.path.join("maze_output", f"testsim-{step}.png"))

    return copy_background


def main():
    """Test this class."""
    from maze_generator import generate_maze

    maze_background = generate_maze()

    rendered_background = render_background(maze_background, True)

    agent_location_list = np.array([[8, 8], [9, 9], [16, 16], [0, 0]])

    step = 42

    render_agent_in_step(maze_background, rendered_background, agent_location_list, step, True)


if __name__ == '__main__':
    main()
