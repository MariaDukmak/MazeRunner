"""This file contains code to render a maze."""
import copy

from pathlib import Path

from typing import List

from PIL import Image

from mazerunner_sim.envs.runner import Runner

import numpy as np

textures_path = Path(__file__) / '..' / 'textures'


def render_background(maze: np.array, leaves: np.array) -> Image:
    """
    Render the maze and save it if specified.

    :param maze: Maze from generate maze function in np 2D array
    """
    # Declare with images used for walls and path
    walls = Image.open(textures_path / "stonebrick.png")
    path = Image.open(textures_path / "dirt.png")
    leaf = Image.open(textures_path / "leaf.png")

    # Declare walls with directions
    # Corners
    wall_zw = Image.open(textures_path / "tile_corner.png")  # ↴
    wall_no = copy.deepcopy(wall_zw).rotate(180)             # ↳
    wall_nw = copy.deepcopy(wall_zw).rotate(270)             # ↵
    wall_zo = copy.deepcopy(wall_zw).rotate(90)              # ↱

    # Ends
    wall_w = Image.open(textures_path / "tile_end.png")    # ←
    wall_n = copy.deepcopy(wall_w).rotate(270)             # ↑
    wall_o = copy.deepcopy(wall_w).rotate(180)             # →
    wall_z = copy.deepcopy(wall_w).rotate(90)              # ↓

    # Straight
    wall_ow = Image.open(textures_path / "tile_straight.png")   # ⇄
    wall_nz = copy.deepcopy(wall_ow).rotate(90)                 # ⇅

    # T-cross
    wall_ozw = Image.open(textures_path / "tile_tcross.png")   # ▽
    wall_now = copy.deepcopy(wall_ozw).rotate(180)              # △
    wall_noz = copy.deepcopy(wall_ozw).rotate(90)              # ▷
    wall_nzw = copy.deepcopy(wall_ozw).rotate(270)              # ◁

    # Cross
    wall_nozw = Image.open(textures_path / "tile_cross.png")  # +

    # Full block
    wall_solid = Image.open(textures_path / "tile_solid.png")  # +

    # Get width and height of the walls
    tile_size_width, tile_size_height = walls.size

    # Calculate canvas size of the maze
    width = maze.shape[0] * tile_size_width
    height = maze.shape[1] * tile_size_height

    # Get maze grind size
    maze_width, maze_height = maze.shape

    # Create canvas
    background = Image.new(mode="RGB", size=(width, height))

    # Loop through values from the maze and determine where the walls and path are
    for height_row, width_values in enumerate(maze):
        for index, is_open_tile in enumerate(width_values):
            if is_open_tile:
                if leaves[height_row, index]:
                    texture = leaf
                else:
                    texture = path
            else:
                # Borders
                # Conners
                # Top left
                if height_row == 0 and index == 0 and not maze[height_row, min(index + 1, maze_width - 1)] and not maze[min(height_row + 1, maze_height - 1), index]:
                    texture = wall_zo
                # Top right
                elif height_row == 0 and index == maze_width - 1 and not maze[height_row, max(index - 1, 0)] and not maze[min(height_row + 1, maze_height-1), index]:
                    texture = wall_zw
                # Bottom left
                elif height_row == maze_height - 1 and index == 0 and not maze[height_row, max(index - 1, 0)] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_no
                # Bottom right
                elif height_row == maze_height -1 and index == maze_width -1 and not maze[height_row, max(index - 1, 0)] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_nw

                # Top side
                elif height_row == 0 and not maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width - 1)] and not maze[min(height_row + 1, maze_height - 1), index]:
                    texture = wall_ozw
                elif height_row == 0 and not maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width - 1)]:
                    texture = wall_ow

                # Left side
                elif index == 0 and not maze[height_row, min(index + 1, maze_width - 1)] and not maze[min(height_row + 1, maze_height - 1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_noz
                elif index == 0 and maze[height_row, min(index + 1, maze_width - 1)] and not maze[min(height_row + 1, maze_height - 1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_nz

                # Right side
                elif index == maze_width - 1 and not maze[height_row, max(index - 1, 0)] and not maze[min(height_row + 1, maze_height - 1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_nzw
                elif index == maze_width - 1 and maze[height_row, max(index - 1, 0)]and not maze[min(height_row + 1, maze_height - 1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_nz

                # Bottom row
                elif height_row == maze_height - 1 and not maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width - 1)] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_now
                elif height_row == maze_height - 1 and not maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width - 1)] and maze[max(height_row - 1, 0), index]:
                    texture = wall_ow

                # Corners
                elif not maze[height_row, max(index - 1, 0)] and maze[height_row, min(index + 1, maze_width-1)] and not maze[min(height_row + 1, maze_height-1), index] and maze[max(height_row - 1, 0), index]:
                    texture = wall_zw
                elif maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width-1)] and maze[min(height_row + 1, maze_height-1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_no
                elif not maze[height_row, max(index - 1, 0)] and maze[height_row, min(index + 1, maze_width - 1)] and maze[min(height_row + 1, maze_height - 1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_nw
                elif maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width - 1)] and not maze[min(height_row + 1, maze_height - 1), index] and maze[max(height_row - 1, 0), index]:
                    texture = wall_zo

                # Cross
                elif not maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width - 1)] and not maze[min(height_row + 1, maze_height - 1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_nozw

                # Endings
                elif not maze[height_row, max(index - 1, 0)] and maze[height_row, min(index + 1, maze_width - 1)] and maze[min(height_row + 1, maze_height - 1), index] and maze[max(height_row - 1, 0), index]:
                    texture = wall_w
                elif maze[height_row, max(index - 1, 0)] and maze[height_row, min(index + 1, maze_width - 1)] and maze[min(height_row + 1, maze_height - 1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_n
                elif maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width - 1)] and maze[min(height_row + 1, maze_height - 1), index] and maze[max(height_row - 1, 0), index]:
                    texture = wall_o
                elif maze[height_row, max(index - 1, 0)] and maze[height_row, min(index + 1, maze_width - 1)] and not maze[min(height_row + 1, maze_height - 1), index] and maze[max(height_row - 1, 0), index]:
                    texture = wall_z

                # Straights
                elif not maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width - 1)] and maze[min(height_row + 1, maze_height - 1), index] and maze[max(height_row - 1, 0), index]:
                    texture = wall_ow
                elif maze[height_row, max(index - 1, 0)] and maze[height_row, min(index + 1, maze_width - 1)] and not maze[min(height_row + 1, maze_height - 1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_nz

                # T crosses
                elif not maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width - 1)] and not maze[min(height_row + 1, maze_height - 1), index] and maze[max(height_row - 1, 0), index]:
                    texture = wall_ozw
                elif not maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width - 1)] and maze[min(height_row + 1, maze_height - 1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_now
                elif maze[height_row, max(index - 1, 0)] and not maze[height_row, min(index + 1, maze_width - 1)] and not maze[min(height_row + 1, maze_height - 1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_noz
                elif not maze[height_row, max(index - 1, 0)] and maze[height_row, min(index + 1, maze_width - 1)] and not maze[min(height_row + 1, maze_height - 1), index] and not maze[max(height_row - 1, 0), index]:
                    texture = wall_nzw

                # Solid
                elif maze[height_row, max(index - 1, 0)] and maze[height_row, min(index + 1, maze_width - 1)] and maze[min(height_row + 1, maze_height - 1), index] and maze[max(height_row - 1, 0), index]:
                    texture = wall_solid

                else:
                    texture = walls
            background.paste(texture, (index * tile_size_width, height_row * tile_size_height))

    return background


def render_agent_in_step(maze_information: np.array, background_image: Image, runners: List[Runner], render_agent_id: int = None) -> Image:
    """
    Render all agents from step.

    :param maze_information: Information about the maze
    :param background_image: Rendered background from maze information
    :param runners: List of all runners
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

    if render_agent_id is not None:
        copy_background = render_explored_maze(copy_background, runners[render_agent_id])

    return copy_background


def render_explored_maze(background_image: Image, runner: Runner) -> Image:
    """
    Render explored maze attribute from runner.

    :param maze_information: Information about the maze
    :param background_image: Rendered background from maze information
    :param runner: Runner used for rendering
    :return: Returns image of current step
    """
    pixels = np.array(background_image)

    # Calculate canvas size of the maze
    tile_width = background_image.width // runner.explored.shape[0]
    tile_height = background_image.height // runner.explored.shape[1]

    # Loop through values from the maze and determine where the walls and path are
    for y in range(runner.explored.shape[0]):
        for x in range(runner.explored.shape[1]):
            if not runner.explored[y, x]:
                pixels[int(y*tile_height):int((y+1)*tile_height), int(x*tile_width):int((x+1)*tile_width)] = 0

    return Image.fromarray(pixels, 'RGB')

    # # Copy from background
    # copy_background = background_image.copy()
    # # Declare with images used for the agent
    # cloud_img = Image.open(textures_path / "cloud.png")
    #
    # # Calculate canvas size of the maze
    # tile_width = background_image.width // runner.explored.shape[0]
    # tile_height = background_image.height // runner.explored.shape[1]
    #
    # # Create cloud canvas layer
    # cloud_layer = Image.new(mode="RGB", size=(background_image.width, background_image.height))
    #
    # # Loop through values from the maze and determine where the walls and path are
    # for height_row, width_values in enumerate(runner.explored):
    #     for index, is_known_tile in enumerate(width_values):
    #         if is_known_tile:
    #             cloud_layer.paste(cloud_img, (index * tile_width, height_row * tile_height))
    #
    # # Blend cloud layer with background
    # blended_background = Image.blend(copy_background, cloud_layer, alpha=0.25)
    # return blended_background
