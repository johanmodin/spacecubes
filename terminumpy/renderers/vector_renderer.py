from collections import defaultdict
import curses
import curses.panel
import itertools
import math
import os
import sys
import time
import numpy as np


from devices.device import OutputDevice
from renderer import Renderer


class VectorRenderer(Renderer):
    def __init__(self, arg, *args, **kwargs):
        super(VectorRenderer, self).__init__(arg, *args, **kwargs)
        """Initializes the vector renderer.
        """

    def render(self, world_array, camera):
        """Render a scene to the output device.

        Args:
            world_array (np.array): A numpy array which
                specifies the 3D world in which the camera's perspective
                is rendered.
            camera (Camera): A Camera object that allows the render-function
                to get the translation and rotation of the camera
                with regards to world_array
        """

        """
        # Remove old image contents
        camera.panel.window().erase()
        
        """
        world_surfaces = self.render_surfaces(world_array, camera)


    def project_lines(self, surfaces, camera, image_size):
        """Projects points from camera coordinate system (XYZ) to
        image plane (UV).

            Args:
                points (np.array): An array of (3, N) points specified
                    in the camera's coordinate system
                camera (Camera): The points will be projected onto
                    this camera's image plane.
                image_size ((int, int)): A tuple of two ints that
                    describe the image frame's (height, width).
        """
        scale_mat = np.array([[image_size[0], 0, 0], [0, image_size[1], 0], [0, 0, 1]])
        h_points_i = scale_mat @ camera.intrinsic_matrix @ points

        h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
        h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

        # Find points behind the camera
        visible_indices = np.where(h_points_i[2, :] >= 0)

        # Remove the last column
        points_im = h_points_i[:2, :]
        return points_im, visible_indices


if __name__ == "__main__":
    from camera import Camera

    world_size = 5

    # Some kind of horse shoe shape
    thickness = 3
    length = 1
    origin = world_size // 2
    cube_world = np.zeros((world_size + 1, world_size + 1, world_size + 1))
    cube_world[
        origin : origin + thickness + length,
        origin : origin + thickness,
        origin : origin + thickness,
    ] = 6
    cube_world[
        origin : origin + thickness,
        origin : origin + thickness + length,
        origin : origin + thickness,
    ] = 6
    cube_world[
        origin + length : origin + length + thickness,
        origin : origin + thickness + length,
        origin : origin + thickness,
    ] = 6

    # cube_world[:] = 0
    # cube_world[0,0,0] = 1
    # Some interesting place to look at
    object_center = np.average(np.argwhere(cube_world > 0), axis=0)

    # Create a renderer that is to render the Camera cam in the np array cube_world
    colors = {
        2: curses.COLOR_BLUE,
        3: curses.COLOR_CYAN,
        4: curses.COLOR_GREEN,
        5: curses.COLOR_MAGENTA,
        6: curses.COLOR_RED,
        7: curses.COLOR_YELLOW,
        "border": curses.COLOR_CYAN,
    }
    r = Renderer(fps=30, colors=colors)

    # Create a camera
    cam_offset = 5
    cam = Camera(r, x=-cam_offset, y=-cam_offset, z=-cam_offset)
    move_size = 0.1
    n_moves = 100
    moves = (
        [(move_size, 0, 0) for _ in range(n_moves)]
        + [(0, move_size, 0) for _ in range(n_moves)]
        + [(0, 0, move_size) for _ in range(n_moves)]
        + [(0, 0, move_size) for _ in range(n_moves)]
    )

    t1 = time.time()
    for i in range(len(moves)):
        # Move the camera in a circle-ish pattern to visualize the 3d
        # information more clearly
        cam.move(*moves[i])

        # Redirect the camera to look at the center of the object
        cam.look_at(*object_center)

        # Render the cube_world array as seen by cam
        r.render(cube_world, cam)

        # cam.move(x=0, y=3 * math.sin(i % 100 / 100 * math.pi * 2),
        #         z=3 * math.cos(i % 100 / 100 * math.pi * 2))
        # cam.rotate(pitch=0.1)

    t2 = time.time()
    exit_curses()
    print(
        f"Rendered {i} steps at resolution {os.get_terminal_size().lines, os.get_terminal_size().columns} in {t2 - t1} seconds"
    )


# TODO:
# Fler backends
# Terminal, cairo? blend2D? qt? drawsvg?
