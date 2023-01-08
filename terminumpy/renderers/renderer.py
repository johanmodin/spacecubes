from abc import ABC, abstractmethod
import itertools
import numpy as np


class Renderer(ABC):
    def __init__(
        self,
        show_border=True,
        border_thickness=0.01,
        border_value=np.finfo(np.float64).eps,
        *args,
        **kwargs
    ):
        """Initializes the renderer.
        Args:
            show_border (bool): Whether to draw a border on the edge of each cell
                of the numpy world array
            border_thickness (float): A value in the [0, 1] range that defines
                the fraction of a world cell's size that is replaced with border
            border value (float): Currently used to represent border's internally.
                Should just be set to not conflict with other values/colors.
        """
        super(Renderer, self).__init__(*args, **kwargs)

        self.show_border = show_border
        self.border_thickness = border_thickness
        self.border_value = border_value

    def world_points_to_image_points(self, world_points, camera, image_size):
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

    @abstractmethod
    def render(self, world_array, camera, image_size):
        """Renders a numpy world array from the perspective
        of camera with the method"""
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
        # Below is the algorithm for finding surfaces in the numpy array world
        # Essentially, we pad the world with a zeros in each dimension
        # to the be able to subtract the entire world from itself offset by 1
        # in each dimension. As the world array is divided into background
        # cells valued 0 and nonbackground cells valued 1, surface cells
        # can be found by looking for nonzero results. The nonzero results
        # are further divided into -1 and 1, which represents the two
        # opposing surfaces of the dimension.

        # Make a new world array that is 0 and 1 valued
        # as we need it for our surface finding algorithm
        # The dtype is set to int8 as this speeds things up significantly
        world_array_int = (world_array > 0).astype(np.int8)

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

        surface_data = {
            (1, 0, 0): {"world_coordinates": x_surfaces_pos_w, "values": x_pos_values},
            (-1, 0, 0): {"world_coordinates": x_surfaces_neg_w, "values": x_neg_values},
            (0, 1, 0): {"world_coordinates": y_surfaces_pos_w, "values": y_pos_values},
            (0, -1, 0): {"world_coordinates": y_surfaces_neg_w, "values": y_neg_values},
            (0, 0, 1): {"world_coordinates": z_surfaces_pos_w, "values": z_pos_values},
            (0, 0, -1): {"world_coordinates": z_surfaces_neg_w, "values": z_neg_values},
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
