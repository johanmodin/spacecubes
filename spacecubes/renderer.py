import itertools
import numpy as np
import math


class Renderer:
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
        self.cell_grid_image_size = None

    def world_points_to_image_points(self, world_points, camera, image_size):
        """Takes a set of world points and a Camera object and
        projects these points into the camera's image frame.
        """
        # Our world_points are grouped into 4-point surfaces
        # which we will need to temporarily undo
        original_surface_shape = world_points.shape[:-1]
        world_points = world_points.reshape((-1, 3))
        camera_points = camera._world_to_camera_coordinates(world_points.T)

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

    def points_to_surfaces(self, world_array):
        """Create world surfaces from world points

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

        # Make the data somewhat easier to interpret by putting it in a dict
        # where each axis positive and negative direction have their own keys
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
        visible_indices = h_points_i[2, :] >= 0

        # Remove the last column
        points_im = h_points_i[:2, :]
        return points_im, visible_indices

    def render(self, world_array, camera, image_size):
        """Create a image_size-sized np array that represents
            the projection of the world_array as seen from camera.

        Args:
            world_array (np.array): A numpy array which
                specifies the 3D world in which the camera's perspective
                is rendered.
            camera (Camera): A Camera object that allows the render-function
                to get the translation and rotation of the camera
                with regards to world_array
            image_size ((int, int)): A tuple of two ints representing the
                desired (height, width) of the output numpy array/image.

        Returns an np array shaped like image_size that represents the scene
            as seen by the camera.
        """
        # TODO: Allow a list of objects (np arrays) to be rendered, each with world position and rotation

        # Adjust camera to image resolution aspect ratio
        camera._regenerate_intrinsic_matrix(aspect_ratio=image_size[0] / image_size[1])

        # Get world coordinate system surfaces from the render-function of the parent Renderer
        surfaces = self.points_to_surfaces(world_array)

        # Project the world points onto the image
        for surface_direction in surfaces:
            (
                surfaces[surface_direction]["image_coordinates"],
                visible_indices,
            ) = self.world_points_to_image_points(
                surfaces[surface_direction]["world_coordinates"], camera, image_size
            )

            # Only retain surfaces that have any point infront of the camera
            visible_indices_per_surface = np.reshape(
                visible_indices,
                (len(surfaces[surface_direction]["image_coordinates"]), 4),
            )
            retained_surface_indices = np.all(visible_indices_per_surface, axis=1)
            surfaces[surface_direction]["image_coordinates"] = surfaces[
                surface_direction
            ]["image_coordinates"][retained_surface_indices]
            surfaces[surface_direction]["world_coordinates"] = surfaces[
                surface_direction
            ]["world_coordinates"][retained_surface_indices]
            surfaces[surface_direction]["values"] = surfaces[surface_direction][
                "values"
            ][retained_surface_indices]

        # Reorder y-axis elements to align them with the orders of other axes
        surfaces[(0, 1, 0)]["image_coordinates"] = np.take(
            surfaces[(0, 1, 0)]["image_coordinates"], np.array([3, 1, 2, 0]), axis=1
        )
        surfaces[(0, -1, 0)]["image_coordinates"] = np.take(
            surfaces[(0, -1, 0)]["image_coordinates"], np.array([3, 1, 2, 0]), axis=1
        )

        # Batch preprocess position differences to allow
        # determining surface corner order in a vectorized way
        camera_position = np.array([[camera.x, camera.y, camera.z]])
        pos_diffs = {}
        positive_pos_along_surface_axis = {}
        for surf_dir in surfaces:
            world_points = surfaces[surf_dir]["world_coordinates"]
            pos_diffs[surf_dir] = np.mean(camera_position - world_points, axis=1)
            dimension = np.argmax(np.abs(surf_dir))
            positive_pos_along_surface_axis[surf_dir] = (
                pos_diffs[surf_dir][:, dimension] >= 0
            )

        # Remove the surface direction key and concatenate everything
        # for easier handling
        image_points = np.concatenate(
            [surfaces[d]["image_coordinates"] for d in surfaces]
        )
        values = np.concatenate([surfaces[d]["values"] for d in surfaces])
        image_surface_order = np.concatenate(
            [
                positive_pos_along_surface_axis[d]
                for d in positive_pos_along_surface_axis
            ]
        )
        image_points_distance = np.linalg.norm(
            np.concatenate([pos_diffs[d] for d in pos_diffs]), axis=1
        )

        # Sort the surfaces and accompanying data by the surface's
        # average distance to the camera. This seems to be a
        # good enough approximation for drawing order to avoid artifacts.
        distance_sorting = np.argsort(-image_points_distance)
        image_points = image_points[distance_sorting]
        image_surface_order = image_surface_order[distance_sorting]
        values = values[distance_sorting]

        # Reorder the surface points so that we can form polygons of all quads
        # by using the vectors (b - a), (c - b), (d - c) and (a - d)
        # Depending on the sign of the position difference calculated above, different
        # orders are used
        if any(image_surface_order):
            image_points[image_surface_order] = image_points[image_surface_order][
                :, [2, 0, 1, 3], :
            ]

        if any(~image_surface_order):
            image_points[~image_surface_order] = image_points[~image_surface_order][
                :, [2, 3, 1, 0], :
            ]

        def point_pair_to_line(p1, p2):
            # Formula to get the line coefficients from two (potentially sets of) points
            kdiv = p1[:, [1]] - p2[:, [1]]
            kdiv[kdiv == 0] = 1
            k = (p1[:, [0]] - p2[:, [0]]) / kdiv
            m = (p1[:, [1]] * p2[:, [0]] - p2[:, [1]] * p1[:, [0]]) / kdiv
            return k, m

        frame = np.zeros(image_size)
        for surface_idx in range(len(image_points)):
            # Give the vertices of the quad variable names
            # to make visualizing the edges a little clearer
            a = image_points[[surface_idx], 0]
            b = image_points[[surface_idx], 1]
            c = image_points[[surface_idx], 2]
            d = image_points[[surface_idx], 3]

            min_y = np.clip(
                math.floor(np.min(image_points[[surface_idx], :, 0])),
                0,
                image_size[0] - 2,
            )
            max_y = np.clip(
                math.ceil(np.max(image_points[[surface_idx], :, 0])),
                0,
                image_size[0] - 1,
            )
            min_x = np.clip(
                math.floor(np.min(image_points[[surface_idx], :, 1])),
                0,
                image_size[1] - 2,
            )
            max_x = np.clip(
                math.ceil(np.max(image_points[[surface_idx], :, 1])),
                0,
                image_size[1] - 1,
            )

            # Readjust the coordinate system to fit inside our smaller rectangle of interest
            a[:, 0] -= min_y
            b[:, 0] -= min_y
            c[:, 0] -= min_y
            d[:, 0] -= min_y
            a[:, 1] -= min_x
            b[:, 1] -= min_x
            c[:, 1] -= min_x
            d[:, 1] -= min_x

            # Create an y,x index grid to plug into our "quad fill" formula
            coords = np.mgrid[: max_y - min_y, 0 : max_x - min_x]

            # Paint the quad's inside
            within_line_1 = (
                (coords[1] - a[:, 1]) * (b[:, 0] - a[:, 0])
                - (coords[0] - a[:, 0]) * (b[:, 1] - a[:, 1])
            ) >= 0
            within_line_2 = (
                (coords[1] - b[:, 1]) * (c[:, 0] - b[:, 0])
                - (coords[0] - b[:, 0]) * (c[:, 1] - b[:, 1])
            ) >= 0
            within_line_3 = (
                (coords[1] - c[:, 1]) * (d[:, 0] - c[:, 0])
                - (coords[0] - c[:, 0]) * (d[:, 1] - c[:, 1])
            ) >= 0
            within_line_4 = (
                (coords[1] - d[:, 1]) * (a[:, 0] - d[:, 0])
                - (coords[0] - d[:, 0]) * (a[:, 1] - d[:, 1])
            ) >= 0
            frame[min_y:max_y, min_x:max_x][
                within_line_1 & within_line_2 & within_line_3 & within_line_4
            ] = values[surface_idx]

            # Paint border
            # Get coefficients of each quad's edge lines
            ba_k, ba_m = point_pair_to_line(a, b)
            cb_k, cb_m = point_pair_to_line(b, c)
            dc_k, dc_m = point_pair_to_line(c, d)
            ad_k, ad_m = point_pair_to_line(d, a)

            # Some heuristic on how wide the line should be related to the area of the cell
            area = (max_y - min_y) * (max_x - min_x)
            lw = self.border_thickness * math.sqrt(area)

            # Paint all points that are within the quad and within lw of an edge
            frame[min_y:max_y, min_x:max_x][
                within_line_1
                & within_line_2
                & within_line_3
                & within_line_4
                & (
                    (np.abs(ba_k * coords[1] + ba_m - coords[0]) < lw)
                    | (np.abs(cb_k * coords[1] + cb_m - coords[0]) < lw)
                    | (np.abs(dc_k * coords[1] + dc_m - coords[0]) < lw)
                    | (np.abs(ad_k * coords[1] + ad_m - coords[0]) < lw)
                )
            ] = self.border_value

        return frame
