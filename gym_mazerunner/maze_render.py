"""This file contains code to render a maze."""

from pathlib import Path

from typing import List

from PIL import Image

from gym_mazerunner.runner import Runner

import numpy as np

textures_path = Path(__file__) / '..' / 'textures'


def render_background(maze: np.array) -> Image:
    """
    Render the maze and save it if specified.

    :param maze: Maze from generate maze function in np 2D array
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
        for index, is_open_tile in enumerate(width_values):
            if is_open_tile:
                background.paste(path, (index * tile_size_width, height_row * tile_size_height))
            else:
                background.paste(walls, (index * tile_size_width, height_row * tile_size_height))

    return background


def render_agent_in_step(maze_information: np.array, background_image: Image, runners: List[Runner]) -> Image:
    """
    Render all agents from step.

    :param maze_information: Information about the maze
    :param background_image: Rendered background from maze information
    :param runners: List of all runners
    :param step: Current step of the environment
    :param want_to_save: Save current step to folder
    :return: Returns image of current step
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

        copy_background.paste(
            agent_icon_copy,
            (runner.location[0] * tile_width, runner.location[1] * tile_height),
            agent_icon
        )

    return copy_background
