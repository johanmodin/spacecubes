from collections import defaultdict
import curses
import curses.panel
import itertools
import math
import os
import sys
import time


from output_device import OutputDevice


class Renderer:
    def __init__(
        self,
        output_device=None,
        fps=30,
        colors={1: curses.COLOR_BLUE, 2: curses.COLOR_RED},
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
                See https://docs.python.org/3/library/curses.html#constants for
                more information on curses colors. Defaults to white for all values.
        """
        if output_device:
            self.output_device = output_device
        else:
            self.output_device = OutputDevice()
        self.fps = fps
        self.next_frame = time.time()

        # Curses setup
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()

        # Initialize all colors and save the color string -> pair index mapping
        color_to_pair_idx = {}
        for color_idx in range(curses.COLORS):
            pair_idx = color_idx + 1
            curses.init_pair(pair_idx, color_idx, -1)
            color_to_pair_idx[color_idx] = pair_idx

        # Create a map from value to the correct color pair object
        self.colors = defaultdict(lambda: curses.color_pair(curses.COLOR_WHITE))
        if colors is not None:
            self.colors.update(
                {
                    value: curses.color_pair(color_to_pair_idx[color])
                    for value, color in colors.items()
                }
            )

    def create_camera_panel(self):
        """Used to create a new curses panel for a new camera"""
        window = curses.newwin(*os.get_terminal_size()[::-1])
        panel = curses.panel.new_panel(window)
        return panel

    def world2image(self, world_points, camera):
        """Takes a set of world points and a Camera object and
        projects these points into the camera's image frame.
        """
        # Our world_points are grouped into 4-point surfaces
        # which we will need to temporarily undo
        original_surface_shape = world_points.shape[:-1]
        world_points = world_points.reshape((-1, 3))
        camera_points = camera.world_to_camera_coordinates(world_points.T)

        # Project the points into the image frame
        h, w = camera.panel.window().getmaxyx()
        image_points, visible_indices = self.project_points(
            camera_points, camera, (h, w)
        )
        image_points = image_points.T.reshape(original_surface_shape + (2,))
        return image_points, visible_indices

    def filter_image_points(self, camera, image_points, visible_indices):
        """Removes bad image points in order to make sure that the
        point set is printable/renderable.

        This is done by checking for inf/nans, removing points behind the
        camera and removing points that are outside of the image plane.
        """
        h, w = camera.panel.window().getmaxyx()
        indices_remaining = np.arange(len(image_points))

        # Filter out points behind the camera
        image_points = image_points[visible_indices]
        indices_remaining = indices_remaining[visible_indices]

        # Filter out of image frame points with some margin to allow for rounding and curses requirements
        inframe_points = np.where(
            (image_points[:, 0] >= 0)
            & (image_points[:, 1] >= 0)
            & (image_points[:, 0] < h - 2)
            & (image_points[:, 1] < w - 1)
        )
        image_points = image_points[inframe_points]
        indices_remaining = indices_remaining[inframe_points]

        # Filter NaN, inf, etc
        finite_points = np.where(
            np.isfinite(image_points[:, 0]) & np.isfinite(image_points[:, 1])
        )
        image_points = image_points[finite_points]
        indices_remaining = indices_remaining[finite_points]

        return image_points, indices_remaining

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

    def fill_surfaces(self, world_quads, image_quads, quad_values, min_area=1):
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

        # Calculate the area of the image quadrilaterals to determine
        # which point sets that are actually worth reprojecting with a
        # higher resolution.
        image_areas = self.quad_area(image_quads)
        for world_quad, image_quad, image_area, value in zip(
            world_quads, image_quads, image_areas, quad_values
        ):
            if image_area <= min_area:
                new_world_points.append(world_quad)
                new_quad_values.extend([value for _ in range(len(world_quad))])
                continue

            # Find which dimension of the point set that
            # is constant
            fixed_dim = np.argmax(
                np.all(world_quad[:-1, :] == world_quad[1:, :], axis=0)
            )
            fixed_dim_value = world_quad[0, fixed_dim]
            nonfixed_dims = [0, 1, 2]
            nonfixed_dims.remove(fixed_dim)

            # Create linspaces for the nonconstant dimensions to fill the
            # world cell surface
            side = math.ceil(math.sqrt(image_area))
            dim_a = np.linspace(
                np.min(world_quad[:, nonfixed_dims[0]]),
                np.max(world_quad[:, nonfixed_dims[0]]),
                side,
            )
            dim_b = np.linspace(
                np.min(world_quad[:, nonfixed_dims[1]]),
                np.max(world_quad[:, nonfixed_dims[1]]),
                side,
            )

            # Make sure that our coordinates are actually
            # ordered x y z
            coordinates = [dim_a, dim_b]
            coordinates.insert(fixed_dim, fixed_dim_value)

            # Create the meshgrid containing the new surface points
            x, y, z = np.meshgrid(*coordinates)

            # Reorder the coordinates into (N, 3)
            new_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
            new_world_points.append(new_points)

            # Add values per new point stemming from the original cell's value
            new_quad_values.extend([value for _ in range(len(new_points))])

        # Return all new points as a new array
        return np.concatenate(new_world_points), np.array(new_quad_values)

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
        camera.panel.top()

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

        # Project the world points to the image frame
        x_surface_pos_ip, x_surface_pos_indices = self.world2image(
            x_surfaces_pos_w, camera
        )
        x_surface_neg_ip, x_surface_neg_indices = self.world2image(
            x_surfaces_neg_w, camera
        )
        y_surface_pos_ip, y_surface_pos_indices = self.world2image(
            y_surfaces_pos_w, camera
        )
        y_surface_neg_ip, y_surface_neg_indices = self.world2image(
            y_surfaces_neg_w, camera
        )
        z_surface_pos_ip, z_surface_pos_indices = self.world2image(
            z_surfaces_pos_w, camera
        )
        z_surface_neg_ip, z_surface_neg_indices = self.world2image(
            z_surfaces_neg_w, camera
        )

        # Measure the area of the projected world frame squares that result in quadrilaterals
        # in the image frame. Use this information to potentially add more points into the
        # world frame square such that its resolution makes it seem like a surface
        # in the image frame
        x_surfaces_pos_w, x_pos_values = self.fill_surfaces(
            x_surfaces_pos_w, x_surface_pos_ip, x_pos_values
        )
        x_surfaces_neg_w, x_neg_values = self.fill_surfaces(
            x_surfaces_neg_w, x_surface_neg_ip, x_neg_values
        )
        y_surfaces_pos_w, y_pos_values = self.fill_surfaces(
            y_surfaces_pos_w, y_surface_pos_ip, y_pos_values
        )
        y_surfaces_neg_w, y_neg_values = self.fill_surfaces(
            y_surfaces_neg_w, y_surface_neg_ip, y_neg_values
        )
        z_surfaces_pos_w, z_pos_values = self.fill_surfaces(
            z_surfaces_pos_w, z_surface_pos_ip, z_pos_values
        )
        z_surfaces_neg_w, z_neg_values = self.fill_surfaces(
            z_surfaces_neg_w, z_surface_neg_ip, z_neg_values
        )

        # Project the new, fleshed out world into the image frame
        x_surface_pos_ip, x_surface_pos_indices = self.world2image(
            x_surfaces_pos_w, camera
        )
        x_surface_neg_ip, x_surface_neg_indices = self.world2image(
            x_surfaces_neg_w, camera
        )
        y_surface_pos_ip, y_surface_pos_indices = self.world2image(
            y_surfaces_pos_w, camera
        )
        y_surface_neg_ip, y_surface_neg_indices = self.world2image(
            y_surfaces_neg_w, camera
        )
        z_surface_pos_ip, z_surface_pos_indices = self.world2image(
            z_surfaces_pos_w, camera
        )
        z_surface_neg_ip, z_surface_neg_indices = self.world2image(
            z_surfaces_neg_w, camera
        )

        # Filter out bad points
        x_surface_pos_ip, x_surface_pos_indices = self.filter_image_points(
            camera, x_surface_pos_ip, x_surface_pos_indices
        )
        x_surface_neg_ip, x_surface_neg_indices = self.filter_image_points(
            camera, x_surface_neg_ip, x_surface_neg_indices
        )
        y_surface_pos_ip, y_surface_pos_indices = self.filter_image_points(
            camera, y_surface_pos_ip, y_surface_pos_indices
        )
        y_surface_neg_ip, y_surface_neg_indices = self.filter_image_points(
            camera, y_surface_neg_ip, y_surface_neg_indices
        )
        z_surface_pos_ip, z_surface_pos_indices = self.filter_image_points(
            camera, z_surface_pos_ip, z_surface_pos_indices
        )
        z_surface_neg_ip, z_surface_neg_indices = self.filter_image_points(
            camera, z_surface_neg_ip, z_surface_neg_indices
        )

        # Remove the corresponding values as well so that the values indices match
        # their points indices
        x_pos_values = x_pos_values[x_surface_pos_indices]
        x_neg_values = x_neg_values[x_surface_neg_indices]
        y_pos_values = y_pos_values[y_surface_pos_indices]
        y_neg_values = y_neg_values[y_surface_neg_indices]
        z_pos_values = z_pos_values[z_surface_pos_indices]
        z_neg_values = z_neg_values[z_surface_neg_indices]

        # Remove the corresponding world points as well so that
        # the world point indices match the image points indices
        x_surfaces_pos_w = x_surfaces_pos_w[x_surface_pos_indices]
        x_surfaces_neg_w = x_surfaces_neg_w[x_surface_neg_indices]
        y_surfaces_pos_w = y_surfaces_pos_w[y_surface_pos_indices]
        y_surfaces_neg_w = y_surfaces_neg_w[y_surface_neg_indices]
        z_surfaces_pos_w = z_surfaces_pos_w[z_surface_pos_indices]
        z_surfaces_neg_w = z_surfaces_neg_w[z_surface_neg_indices]

        # Sleep as needed to not exceed the frame rate of self.fps
        if self.fps is not None and self.fps > 0:
            t_wait = self.next_frame - time.time()
            self.next_frame = time.time() + 1 / self.fps
            if t_wait > 0:
                time.sleep(t_wait)

        camera.panel.window().erase()

        world_points = np.concatenate(
            [
                x_surfaces_pos_w,
                x_surfaces_neg_w,
                y_surfaces_pos_w,
                y_surfaces_neg_w,
                z_surfaces_pos_w,
                z_surfaces_neg_w,
            ]
        )
        image_points = np.concatenate(
            [
                x_surface_pos_ip,
                x_surface_neg_ip,
                y_surface_pos_ip,
                y_surface_neg_ip,
                z_surface_pos_ip,
                z_surface_neg_ip,
            ]
        )
        values = np.concatenate(
            [
                x_pos_values,
                x_neg_values,
                y_pos_values,
                y_neg_values,
                z_pos_values,
                z_neg_values,
            ]
        )

        # Sort world points by distance so that we can paint the
        # further points first in order to not overwrite a nearer surface
        camera_position = np.array([[camera.x, camera.y, camera.z]])
        point_distances = np.linalg.norm(camera_position - world_points, axis=1)
        distance_sorting = np.argsort(-point_distances)
        image_points = image_points[distance_sorting]
        values = values[distance_sorting]

        # Print new screen content
        self.paint_points(image_points, values, camera)

        # Draw a box around the screen because it's neat and refresh the panel's contents
        camera.panel.window().box()
        camera.panel.window().refresh()
        curses.panel.update_panels()

    def paint_points(self, image_points, values, camera):
        for i, p in enumerate(image_points):
            # Find the value of the world point corresponding to the
            # image frame point p in order to decide color/attributes
            pair_index = self.colors[values[i]]

            # Add some kind of character at the pixel position
            camera.panel.window().addch(int(p[0]), int(p[1]), "\U000025A9", pair_index)

    def project_points(self, points, camera, image_size):
        """Projects points from camera coordinate system (XYZ) to
        image plane (UV).

            Args:
                camera (Camera): The points will be projected onto
                    this camera's image plane.
                points (np.array): An array of (3, N) points specified
                    in the camera's coordinate system
        """
        scale_mat = np.array([[image_size[0], 0, 0], [0, image_size[1], 0], [0, 0, 1]])
        h_points_i = scale_mat @ camera.intrinsic_matrix @ points

        h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
        h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

        # Remove points behind the camera
        cond = h_points_i[2, :] >= 0
        visible_indices = np.where(cond)[0]

        # Remove the last column
        points_im = h_points_i[:2, :]
        return points_im, visible_indices


if __name__ == "__main__":
    import numpy as np
    from camera import Camera

    world_size = 30

    # Some kind of horse shoe shape
    thickness = 2
    length = 10
    origin = world_size // 2
    cube_world = np.zeros((world_size + 1, world_size + 1, world_size + 1))
    cube_world[
        origin : origin + thickness + length,
        origin : origin + thickness,
        origin : origin + thickness,
    ] = 2
    cube_world[
        origin : origin + thickness,
        origin : origin + thickness + length,
        origin : origin + thickness,
    ] = 3
    cube_world[
        origin + length : origin + length + thickness,
        origin : origin + thickness + length,
        origin : origin + thickness,
    ] = 4

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
    }
    r = Renderer(fps=30, colors=colors)

    # Create a camera
    cam_offset = 5
    cam = Camera(r, x=-cam_offset, y=-cam_offset, z=-cam_offset)
    move_size = 0.1
    n_moves = 1000
    moves = (
        [(move_size, 0, 0) for _ in range(n_moves)]
        + [(0, move_size, 0) for _ in range(n_moves)]
        + [(0, 0, move_size) for _ in range(n_moves)]
    )

    t1 = time.time()
    for i in range(len(moves)):
        cam.look_at(*object_center)

        # Render the cube_world array as seen by cam
        r.render(cube_world, cam)

        # Move the camera in a circle-ish pattern to visualize the 3d
        # information more clearly
        cam.move(*moves[i])
        # cam.move(x=0, y=3 * math.sin(i % 100 / 100 * math.pi * 2),
        #         z=3 * math.cos(i % 100 / 100 * math.pi * 2))
        # cam.rotate(pitch=0.1)

        # Redirect the camera to look at the center of the object
        cam.look_at(*object_center)
    t2 = time.time()
    curses.endwin()
    print(
        f"Rendered {i} steps at resolution {os.get_terminal_size().lines, os.get_terminal_size().columns} in {t2 - t1} seconds"
    )
