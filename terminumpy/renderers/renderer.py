from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
import math
import os
import sys
import time
import numpy as np


class Renderer(ABC):
    def __init__(
        self,
        show_border=True,
        border_thickness_frac=0.05,
        border_value=np.finfo(np.float64).eps,
        *args,
        **kwargs
    ):
        """Initializes the renderer.
        Args:
            output_device (OutputDevice): An OutputDevice which
                receives the rendered output. If None, a default
                OutputDevice for the current terminal will be created.
            fps (int): The maximum number of frames per second to render
            colors ({int: int}): A mapping from world point value to a curses
                color index. E.g., if the world at XYZ position (2, 5, 1) has value
                5, this can be rendered as green by specifying colors={5: curses.COLOR_GREEN}.
                Can also take the 'border' string as value, which is then used to color
                the border's of the world array's cells.
                See https://docs.python.org/3/library/curses.html#constants for
                more information on curses colors. Defaults to white for all values.
            show_border (bool): Whether to draw a border on the edge of each cell
                of the numpy world array
            border_thickness_frac (float): A value in the [0, 1] range that defines
                the fraction of a world cell's size that is replaced with border
            border value (float): Currently used to represent border's internally.
                Should just be set to not conflict with other values/colors.
            max_cell_surface_side_px (int): The maximum rendered side length of a cell
                measured in pixels. Thus, the maximum resolution of a cell surface is
                 max_cell_surface_side_px * max_cell_surface_side_px. Can be set to
                 e.g., math.inf to allow infite resolution, but this might be unstable as
                 points outside the image plane are currently included in
                 the surface fill operation which could lead to trying to allocate
                 inf * inf image points, which does not fit in memory.
        """
        super(Renderer, self).__init__(*args, **kwargs)

        self.show_border = show_border
        self.border_frac = border_thickness_frac
        self.border_value = border_value

    def world2image(self, world_points, camera, image_size):
        """Takes a set of world points and a Camera object and
        projects these points into the camera's image frame.
        """
        # Our world_points are grouped into 4-point surfaces
        # which we will need to temporarily undo
        original_surface_shape = world_points.shape[:-1]
        world_points = world_points.reshape((-1, 3))
        camera_points = camera.world_to_camera_coordinates(world_points.T)

        # Project the points into the image frame
        h, w = image_size
        image_points, visible_indices = self.project_points(
            camera_points, camera, (h, w)
        )
        image_points = image_points.T.reshape(original_surface_shape + (2,))
        return image_points, visible_indices


    def quad_area(self, points):
        """Area of 4-point quadrilateral

        Args:
            points (np.array): A Nx4x2 array containing N quadrilaterals
                which each consist of 4 points that have (x, y) coordinates
                Point order matters, so we define it with indices as follows:
                0   2
                1   3
        """
        ac = points[:, 0, :] - points[:, 3, :]
        bd = points[:, 1, :] - points[:, 2, :]
        return np.abs(np.cross(ac, bd)) / 2

    def cube2surface(self, points, dim):
        """Takes a set of Nx3 points and a dimension and produces
        a square for every point that represents the cell's surface
        in that dimension. If this is done with a positive and a negative offset
        for every dimension, we produce all surfaces of the cube located at the point.
        """
        other_dims_offsets = np.array(list(itertools.product([0.5, -0.5], repeat=2)))
        surface_corner_offsets = np.insert(other_dims_offsets, dim, 0.5, axis=1)

        surface_corner_offsets = np.repeat(
            surface_corner_offsets[np.newaxis, :, :], points.shape[0], axis=0
        )
        surfaces = surface_corner_offsets + points[:, np.newaxis, :]
        return surfaces

    def fill_surfaces(
        self, world_quads, image_quads, quad_values, fixed_dim, min_area=1
    ):
        """Takes N sets of 4 world points that each make up a square in a numpy array cell
        and N sets of 4 image frame points that correspond to the same points
        in the image space. The surface area of the image frame quadrilateral
        determines how many points to put in the world cell square, which can then
        be projected onto the image plane again, but with a resolution that is more apt
        for the surface.

        Args:
            world_quads (np.array): A Nx4x3 array which contains N sets of
                4 xyz points in the world coordinate system. Each such set
                represents a numpy array cell surface.
            image_quads (np.array): A Nx4x2 array which contains N sets of
                4 uv points in the image frame. Each such set represents
                the image frame projection of the numpy array cell surface.
        """
        new_world_points = []
        new_quad_values = []

        # Vectorize some operations for performance
        fixed_dim_values = world_quads[:, 0, fixed_dim]
        world_quad_min = np.min(world_quads, axis=1)
        world_quad_max = np.max(world_quads, axis=1)

        # Calculate the area of the image quadrilaterals to determine
        # which point sets that are actually worth reprojecting with a
        # higher resolution.
        image_areas = self.quad_area(image_quads)
        for i in range(len(world_quads)):
            world_quad = world_quads[i]
            image_area = image_areas[i]
            quad_value = quad_values[i]

            if image_area <= min_area:
                new_world_points.append(world_quad)
                new_quad_values.extend([quad_value for _ in range(len(world_quad))])
                continue

            # Get the value of the surface's constant dimension
            fixed_dim_value = fixed_dim_values[i]
            nonfixed_dims = [d for d in range(3) if d != fixed_dim]

            # Create linspaces for the nonconstant dimensions to fill the
            # world cell surface
            # The 1.5 constant is just an estimated factor that increases the
            # number of points such that missing points in the rendered surface is rare
            side = int(
                min(
                    math.ceil(math.sqrt(image_area)) * 1.5,
                    self.max_cell_surface_side_px,
                )
            )

            # Faster version of np.linspace
            step = (
                world_quad_max[i, nonfixed_dims[0]]
                - world_quad_min[i, nonfixed_dims[0]]
            ) / side
            dim_a = itertools.accumulate(
                [world_quad_min[i, nonfixed_dims[0]]] + [step for _ in range(side)]
            )

            step = (
                world_quad_max[i, nonfixed_dims[1]]
                - world_quad_min[i, nonfixed_dims[1]]
            ) / side
            dim_b = itertools.accumulate(
                [world_quad_min[i, nonfixed_dims[1]]] + [step for _ in range(side)]
            )

            # Make sure that our coordinates are actually
            # ordered x y z
            coordinates = [dim_a, dim_b]
            coordinates.insert(fixed_dim, [fixed_dim_value])

            # Create the meshgrid containing the new surface points
            new_points = np.array(list(itertools.product(*coordinates)))
            new_world_points.append(new_points)

            values = np.full(len(new_points), quad_value)

            # Draw cell borders to ease 3d understanding
            n_points_for_border = 9
            if self.show_border and len(new_points) >= n_points_for_border:
                dim_a_border_size = (
                    world_quad_max[i, nonfixed_dims[0]]
                    - world_quad_min[i, nonfixed_dims[0]]
                ) * self.border_frac
                dim_b_border_size = (
                    world_quad_max[i, nonfixed_dims[1]]
                    - world_quad_min[i, nonfixed_dims[1]]
                ) * self.border_frac

                # Set all values on the cell's border to the border value
                # TODO: Break out these conditions
                values[
                    (
                        new_points[:, nonfixed_dims[0]]
                        <= world_quad_min[i, nonfixed_dims[0]] + dim_a_border_size
                    )
                    | (
                        new_points[:, nonfixed_dims[0]]
                        >= world_quad_max[i, nonfixed_dims[0]] - dim_a_border_size
                    )
                    | (
                        new_points[:, nonfixed_dims[1]]
                        <= world_quad_min[i, nonfixed_dims[1]] + dim_b_border_size
                    )
                    | (
                        new_points[:, nonfixed_dims[1]]
                        >= world_quad_max[i, nonfixed_dims[1]] - dim_b_border_size
                    )
                ] = self.border_value

            # Add values per new point stemming from the original cell's value
            # or from the border value
            new_quad_values.extend(values)

        # Return all new points as a new array
        return np.concatenate(new_world_points), np.array(new_quad_values)

    @abstractmethod
    def render(self, world_array, camera, image_size):
        ''' Renders a numpy world array from the perspective
            of camera with the method '''
        pass

    def render_surfaces(self, world_array, camera):
        """Render a scene to the output device.

        Args:
            world_array (np.array): A numpy array which
                specifies the 3D world in which the camera's perspective
                is rendered.
            camera (Camera): A Camera object that allows the render-function
                to get the translation and rotation of the camera
                with regards to world_array

        Returns a dict where each key is a dimension and a direction and the corresponding
        values are dicts with two keys: 'coordinates' and 'values'. 
        Under the coordinates key are t he surface quads that have the 
        key's dimension as its constant dimension and the key's direction as 
        its "outer" direction. E.g., the key (0, -1, 0) holds all surface 
        quads that make up the surfaces in the negative y direction if the 
        coordinates are ordered as (x, y, z). The surfaces are numpy arrays 
        of sets of four points that make up the quad defining a cell's surface
        in one dimension and direction. The 'values' key holds the corresponding cell
        values as defined in the world array. 
        """
        # Below is our algorithm for finding surfaces in the numpy array world
        # Essentially, we pad the world with a zeros in each dimension
        # to the be able to subtract the entire world from itself offset by 1
        # for each dimension. As the world array is divided into background
        # cells valued 0 and nonbackground cells valued 1, surface cells
        # can be found by looking for nonzero results. The nonzero results
        # can be further divided into -1 and 1, which represents the two
        # opposing surfaces of the dimension.

        # Make a new world array that is 0 and 1 valued
        # as we need it for our surface finding algorithm
        # The dtype is set to int8 as this speeds things up significantly
        world_array_int = (world_array > 0).astype(np.int8)

        # Keep track of the values to allow rendering options depending on value
        # world_points = world_array > 0
        # world_values = world_array[world_points]

        # Pad with 0s in all dimensions to include boundaries of world
        world_array_int = np.pad(world_array_int, 1)

        # Calculate surface regions through the offseted subtraction
        surface_x = world_array_int[:-1, :, :] - world_array_int[1:, :, :]
        surface_y = world_array_int[:, :-1, :] - world_array_int[:, 1:, :]
        surface_z = world_array_int[:, :, :-1] - world_array_int[:, :, 1:]

        # Find the surfaces in the positive and negative direction of each dimension
        # and compensate for padding
        x_outer_points_pos_w = np.argwhere(surface_x == 1) + [-1, -1, -1]
        x_outer_points_neg_w = np.argwhere(surface_x == -1) + [-1, -1, -1]
        y_outer_points_pos_w = np.argwhere(surface_y == 1) + [-1, -1, -1]
        y_outer_points_neg_w = np.argwhere(surface_y == -1) + [-1, -1, -1]
        z_outer_points_pos_w = np.argwhere(surface_z == 1) + [-1, -1, -1]
        z_outer_points_neg_w = np.argwhere(surface_z == -1) + [-1, -1, -1]

        # Keep track of surface cell values
        x_pos_values = world_array[tuple((x_outer_points_pos_w).T)]
        x_neg_values = world_array[tuple((x_outer_points_neg_w + [1, 0, 0]).T)]
        y_pos_values = world_array[tuple((y_outer_points_pos_w).T)]
        y_neg_values = world_array[tuple((y_outer_points_neg_w + [0, 1, 0]).T)]
        z_pos_values = world_array[tuple((z_outer_points_pos_w).T)]
        z_neg_values = world_array[tuple((z_outer_points_neg_w + [0, 0, 1]).T)]

        # Turn the point & direction data into sets of 4 points
        # that define a surface of the cell
        x_surfaces_pos_w = self.cube2surface(x_outer_points_pos_w, 0)
        x_surfaces_neg_w = self.cube2surface(x_outer_points_neg_w, 0)
        y_surfaces_pos_w = self.cube2surface(y_outer_points_pos_w, 1)
        y_surfaces_neg_w = self.cube2surface(y_outer_points_neg_w, 1)
        z_surfaces_pos_w = self.cube2surface(z_outer_points_pos_w, 2)
        z_surfaces_neg_w = self.cube2surface(z_outer_points_neg_w, 2)

        surface_data = {(1, 0, 0): {'world_coordinates': x_surfaces_pos_w, 'values': x_pos_values},
                        (-1, 0, 0): {'world_coordinates': x_surfaces_neg_w, 'values': x_neg_values},
                        (0, 1, 0): {'world_coordinates': y_surfaces_pos_w, 'values': y_pos_values},
                        (0, -1, 0): {'world_coordinates': y_surfaces_neg_w, 'values': y_neg_values},
                        (0, 0, 1): {'world_coordinates': z_surfaces_pos_w, 'values': z_pos_values},
                        (0, 0, -1): {'world_coordinates': z_surfaces_neg_w, 'values': z_neg_values},
                        }
        return surface_data



    def project_points(self, points, camera, image_size):
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


def exit_curses():
    curses.nocbreak()
    curses.echo()
    curses.endwin()


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
