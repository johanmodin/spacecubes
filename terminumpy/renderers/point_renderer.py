from collections import defaultdict
import curses
import curses.panel
import itertools
import math
import os
import sys
import time
import numpy as np


from .renderer import Renderer


class PointRenderer(Renderer):
    def __init__(
        self,
        max_cell_surface_side_px=128,
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
        super(PointRenderer, self).__init__(*args, **kwargs)

        self.max_cell_surface_side_px = max_cell_surface_side_px

    def filter_image_points(self, camera, image_points, visible_indices, image_size):
        """Removes bad image points in order to make sure that the
        point set is printable/renderable.

        This is done by checking for inf/nans, removing points behind the
        camera and removing points that are outside of the image plane.
        """
        h, w = image_size
        indices_remaining = np.arange(len(image_points))

        # Filter out points behind the camera
        indices_remaining = indices_remaining[visible_indices]
        image_points = image_points[visible_indices]

        # Filter out of image frame points with some margin to allow for rounding and curses requirements
        inframe_points = np.where(
            (image_points[:, 0] >= 0)
            & (image_points[:, 1] >= 0)
            & (image_points[:, 0] < h - 2)
            & (image_points[:, 1] < w - 1)
        )
        indices_remaining = indices_remaining[inframe_points]
        image_points = image_points[inframe_points]

        # Filter NaN, inf, etc
        finite_points = np.where(
            np.isfinite(image_points[:, 0]) & np.isfinite(image_points[:, 1])
        )
        indices_remaining = indices_remaining[finite_points]

        return indices_remaining

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
        # This method accounts for like 85% of the exec. time
        # so if we could vectorize it that'd be great..
        # TODO: Optimize
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

    def render(self, world_array, camera, image_size):
        """Render a scene to the output device.

        Args:
            world_array (np.array): A numpy array which
                specifies the 3D world in which the camera's perspective
                is rendered.
            camera (Camera): A Camera object that allows the render-function
                to get the translation and rotation of the camera
                with regards to world_array
        """
        surfaces = self.render_surfaces(world_array, camera)

        # Project the surface points to the image frame
        indices_of_points_ahead = {}
        for surface_direction in surfaces:
            image_coordinates, point_indices_ahead = self.world2image(
                surfaces[surface_direction]['world_coordinates'], camera, image_size
            )
            surfaces[surface_direction]['image_coordinates'] = image_coordinates
            indices_of_points_ahead[surface_direction] = point_indices_ahead
            
        # Measure the area of the projected world frame squares that result in quadrilaterals
        # in the image frame. Use this information to potentially add more points into the
        # world frame square such that its resolution makes it seem like a surface
        # in the image frame
        for surface_direction in surfaces:
            dimension = np.argmax(np.abs(surface_direction))
            new_world_surface_coordinates, new_surface_values = self.fill_surfaces(
                surfaces[surface_direction]['world_coordinates'],
                surfaces[surface_direction]['image_coordinates'],
                surfaces[surface_direction]['values'],
                dimension,
            )
            surfaces[surface_direction]['world_coordinates'] = new_world_surface_coordinates
            surfaces[surface_direction]['values'] = new_surface_values

            # Project the new, fleshed out world into the image frame
            new_image_surface_coordinates, point_indices_ahead = self.world2image(
                surfaces[surface_direction]['world_coordinates'], camera, image_size
            )
            surfaces[surface_direction]['image_coordinates'] = new_image_surface_coordinates
            indices_of_points_ahead[surface_direction] = point_indices_ahead

        # Filter out bad points
        for surface_direction in surfaces:
            selected_indices = self.filter_image_points(
                camera, surfaces[surface_direction]['image_coordinates'],
                indices_of_points_ahead[surface_direction],
                image_size
            )

            # Remove the corresponding values as well so that the values indices match
            # their points indices
            filtered_world_surfaces = surfaces[surface_direction]['world_coordinates'][selected_indices]
            filtered_image_surfaces = surfaces[surface_direction]['image_coordinates'][selected_indices]
            filtered_values = surfaces[surface_direction]['values'][selected_indices]

            surfaces[surface_direction]['world_coordinates'] = filtered_world_surfaces
            surfaces[surface_direction]['image_coordinates'] = filtered_image_surfaces
            surfaces[surface_direction]['values'] = filtered_values

        return surfaces


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
