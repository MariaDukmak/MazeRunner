"""This file contains code to render a maze."""
import copy

from pathlib import Path

from typing import List, Union

from PIL import Image, ImageDraw

from mazerunner_sim.envs.agents.runner import Runner

import numpy as np

textures_path = Path(__file__) / '..' / 'textures'


def render_background(maze: np.array, leaves: np.array, safe_zone: np.array) -> Image:
    """
    Render the maze and save it if specified.

    :param maze: Maze from generate maze function in np 2D array
    :param leaves:
    :param safe_zone:
    """
    # Declare with images used for walls and path
    walls = Image.open(textures_path / "stonebrick.png")
    path = Image.open(textures_path / "dirt.png")
    leaf = Image.open(textures_path / "leaf.png")
    safe_zone_texture = Image.new(mode="RGBA", size=walls.size, color=(250, 100, 0, 100))

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
                    background.paste(path, (index * tile_size_width, height_row * tile_size_height), path)
                    texture = leaf
                else:
                    texture = path
            else:
                # Declaring surrounding tiles
                left_tile = maze[height_row, max(index - 1, 0)]
                right_tile = maze[height_row, min(index + 1, maze_width - 1)]
                top_tile = maze[max(height_row - 1, 0), index]
                bottom_tile = maze[min(height_row + 1, maze_height - 1), index]
                # Borders Corners
                # Top left
                if height_row == 0 and index == 0 and not right_tile and not bottom_tile:
                    texture = wall_zo
                # Top right
                elif height_row == 0 and index == maze_width - 1 and not left_tile and not bottom_tile:
                    texture = wall_zw
                # Bottom left
                elif height_row == maze_height - 1 and index == 0 and not left_tile and not top_tile:
                    texture = wall_no
                # Bottom right
                elif height_row == maze_height - 1 and index == maze_width - 1 and not left_tile and not top_tile:
                    texture = wall_nw

                # Top side
                elif height_row == 0 and not left_tile and not right_tile and not bottom_tile:
                    texture = wall_ozw
                elif height_row == 0 and not left_tile and not right_tile:
                    texture = wall_ow

                # Left side
                elif index == 0 and not right_tile and not bottom_tile and not top_tile:
                    texture = wall_noz
                elif index == 0 and right_tile and not bottom_tile and not top_tile:
                    texture = wall_nz

                # Right side
                elif index == maze_width - 1 and not left_tile and not bottom_tile and not top_tile:
                    texture = wall_nzw
                elif index == maze_width - 1 and left_tile and not bottom_tile and not top_tile:
                    texture = wall_nz

                # Bottom row
                elif height_row == maze_height - 1 and not left_tile and not right_tile and not top_tile:
                    texture = wall_now
                elif height_row == maze_height - 1 and not left_tile and not right_tile and top_tile:
                    texture = wall_ow

                # Corners
                elif not left_tile and right_tile and not bottom_tile and top_tile:
                    texture = wall_zw
                elif left_tile and not right_tile and bottom_tile and not top_tile:
                    texture = wall_no
                elif not left_tile and right_tile and bottom_tile and not top_tile:
                    texture = wall_nw
                elif left_tile and not right_tile and not bottom_tile and top_tile:
                    texture = wall_zo

                # Cross
                elif not left_tile and not right_tile and not bottom_tile and not top_tile:
                    texture = wall_nozw

                # Endings
                elif not left_tile and right_tile and bottom_tile and top_tile:
                    texture = wall_w
                elif left_tile and right_tile and bottom_tile and not top_tile:
                    texture = wall_n
                elif left_tile and not right_tile and bottom_tile and top_tile:
                    texture = wall_o
                elif left_tile and right_tile and not bottom_tile and top_tile:
                    texture = wall_z

                # Straights
                elif not left_tile and not right_tile and bottom_tile and top_tile:
                    texture = wall_ow
                elif left_tile and right_tile and not bottom_tile and not top_tile:
                    texture = wall_nz

                # T crosses
                elif not left_tile and not right_tile and not bottom_tile and top_tile:
                    texture = wall_ozw
                elif not left_tile and not right_tile and bottom_tile and not top_tile:
                    texture = wall_now
                elif left_tile and not right_tile and not bottom_tile and not top_tile:
                    texture = wall_noz
                elif not left_tile and right_tile and not bottom_tile and not top_tile:
                    texture = wall_nzw

                # Solid
                elif left_tile and right_tile and bottom_tile and top_tile:
                    texture = wall_solid

                else:
                    texture = walls
            background.paste(texture, (index * tile_size_width, height_row * tile_size_height), texture)
            if safe_zone[height_row, index]:
                background.paste(safe_zone_texture, (index * tile_size_width, height_row * tile_size_height),
                                 safe_zone_texture)

    return background


def render_agent_in_step(env, render_agent_id: Union[int, None] = None) -> Image:
    """
    Render all agents from step.

    :param env: The environment
    :param render_agent_id: Id of the agent to follow the explored map of
    :return: Returns image of current step
    """
    # Copy from background
    copy_background = env.rendered_background.copy()
    # Declare with images used for the agent
    agent_icon = Image.open(textures_path / "agent.png")

    # Calculate canvas size of the maze
    tile_width = env.rendered_background.width // env.maze.shape[0]
    tile_height = env.rendered_background.height // env.maze.shape[1]

    for index, runner in enumerate(env.runners):
        agent_icon_copy = agent_icon.copy()

        # Split into 3 channels
        r, g, b, alph = agent_icon_copy.split()

        # Increase Reds
        r = r.point(lambda i: i * index)

        # Decrease Greens
        g = g.point(lambda i: i * index)

        # Recombine back to RGB image
        agent_icon_copy = Image.merge('RGBA', (r, g, b, alph))

        copy_background.paste(
            agent_icon_copy,
            (runner.location[0] * tile_width, runner.location[1] * tile_height),
            agent_icon_copy
        )

    if render_agent_id is not None:
        copy_background = render_explored_maze(copy_background, env.runners[render_agent_id])

    ImageDraw.Draw(copy_background).text((5, 0), f"Time: {env.time}, time left till end of day: {env.time_till_end_of_day()}")

    return copy_background


def render_explored_maze(background_image: Image, runner: Runner) -> Image:
    """
    Render explored maze attribute from runner.

    :param background_image: Rendered background from maze information
    :param runner: Runner used for rendering
    :return: Returns image of current step
    """
    # Copy from background
    copy_background = background_image.copy().convert("RGBA")
    # Declare with images used for the agent
    cloud_img = Image.open(textures_path / "cloud.png")

    # Calculate canvas size of the maze
    tile_width = background_image.width // runner.explored.shape[0]
    tile_height = background_image.height // runner.explored.shape[1]

    # Create cloud canvas layer
    cloud_layer = Image.new(mode="RGBA", size=(background_image.width, background_image.height))

    # Loop through values from the maze and determine where the walls and path are
    for height_row, width_values in enumerate(runner.explored):
        for index, is_known_tile in enumerate(width_values):
            if not is_known_tile:
                cloud_layer.paste(cloud_img, (index * tile_width, height_row * tile_height))

    # Blend cloud layer with background
    blended_background = Image.alpha_composite(copy_background, cloud_layer)
    return blended_background
