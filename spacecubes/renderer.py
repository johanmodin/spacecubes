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
            border_value (float): Currently used to represent border's internally.
                Should just be set to not conflict with other values/colors.
        """
        super(Renderer, self).__init__(*args, **kwargs)

        self.show_border = show_border
        self.border_thickness = border_thickness
        self.border_value = border_value
        self.cell_grid_image_size = None

    def _world_points_to_image_points(self, world_points, camera, image_size):
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
        image_points, visible_indices = self._project_points(
            camera_points, camera, (h, w)
        )
        image_points = image_points.T.reshape(original_surface_shape + (2,))
        return image_points, visible_indices

    def _cube2surface(self, points, dim):
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

    def _points_to_surfaces(self, world_array):
        """Create world surfaces from world points

        Returns a world point dict and their corresponding values as dict.
        Both of these dictionaries or organized such that each surface normal
        direction has its own key, e.g., (-1, 0, 0) and (1, 0, 0) are the two
        keys for the surfaces in the x-direction.
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
        x_surfaces_pos_w = self._cube2surface(x_outer_points_pos_w, 0)
        x_surfaces_neg_w = self._cube2surface(x_outer_points_neg_w, 0)
        y_surfaces_pos_w = self._cube2surface(y_outer_points_pos_w, 1)
        y_surfaces_neg_w = self._cube2surface(y_outer_points_neg_w, 1)
        z_surfaces_pos_w = self._cube2surface(z_outer_points_pos_w, 2)
        z_surfaces_neg_w = self._cube2surface(z_outer_points_neg_w, 2)

        # Make the data somewhat easier to interpret by putting it in dicts
        # where each axis positive and negative direction have their own keys
        world_points = {
            (1, 0, 0): x_surfaces_pos_w,
            (-1, 0, 0): x_surfaces_neg_w,
            (0, 1, 0): y_surfaces_pos_w,
            (0, -1, 0): y_surfaces_neg_w,
            (0, 0, 1): z_surfaces_pos_w,
            (0, 0, -1): z_surfaces_neg_w,
        }

        values = {
            (1, 0, 0): x_pos_values,
            (-1, 0, 0): x_neg_values,
            (0, 1, 0): y_pos_values,
            (0, -1, 0): y_neg_values,
            (0, 0, 1): z_pos_values,
            (0, 0, -1): z_neg_values,
        }
        return world_points, values

    def _project_points(self, points, camera, image_size):
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

    def _point_pair_to_line(self, p1, p2):
        """Formula to get the line coefficients from two (potentially sets of) points
        on the form (N, 2)
        """
        kdiv = p1[:, [1]] - p2[:, [1]]
        kdiv[kdiv == 0] = 1
        k = (p1[:, [0]] - p2[:, [0]]) / kdiv
        m = (p1[:, [1]] * p2[:, [0]] - p2[:, [1]] * p1[:, [0]]) / kdiv
        return k, m

    def _filter_nonvisible_surfaces(self, world_points, values, camera):
        """Remove surfaces that have normals that are pointing away from us as these will be
        covered by other surfaces
        """
        for surface_dir in world_points:
            cam_to_surface_dir = (
                np.mean(world_points[surface_dir], axis=1) - camera.position
            )
            angles_to_surfaces = np.arccos(
                np.dot(cam_to_surface_dir, surface_dir)
                / (np.linalg.norm(cam_to_surface_dir, axis=1))
            )
            visible_surfaces = angles_to_surfaces > np.pi / 2
            world_points[surface_dir] = world_points[surface_dir][visible_surfaces]
            values[surface_dir] = values[surface_dir][visible_surfaces]
        return world_points, values

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
        # TODO: Break this function up into more aptly sized functions

        # Adjust camera to image resolution aspect ratio
        # TODO: This should be done cleaner. Renderer shouldnt have to remake camera's variables.
        camera._regenerate_intrinsic_matrix(aspect_ratio=image_size[0] / image_size[1])

        # Get world coordinate system surfaces from the render-function of the parent Renderer
        world_points, values = self._points_to_surfaces(world_array)

        # Ignore surfaces that are pointing away from us
        world_points, values = self._filter_nonvisible_surfaces(
            world_points, values, camera
        )

        # Project the world points onto the image
        image_points = {}
        for surface_direction in world_points:
            (
                image_points[surface_direction],
                visible_indices,
            ) = self._world_points_to_image_points(
                world_points[surface_direction], camera, image_size
            )

            # Only retain surfaces that have any point infront of the camera
            visible_indices_per_surface = np.reshape(
                visible_indices,
                (len(image_points[surface_direction]), 4),
            )
            retained_surface_indices = np.all(visible_indices_per_surface, axis=1)
            world_points[surface_direction] = world_points[surface_direction][
                retained_surface_indices
            ]
            image_points[surface_direction] = image_points[surface_direction][
                retained_surface_indices
            ]
            values[surface_direction] = values[surface_direction][
                retained_surface_indices
            ]

        # Reorder y-axis elements to align them with the orders of other axes
        image_points[(0, 1, 0)] = np.take(
            image_points[(0, 1, 0)], np.array([3, 1, 2, 0]), axis=1
        )
        image_points[(0, -1, 0)] = np.take(
            image_points[(0, -1, 0)], np.array([3, 1, 2, 0]), axis=1
        )

        # Batch preprocess position differences to allow
        # determining surface corner order in a vectorized way
        camera_position = np.array([[camera.x, camera.y, camera.z]])
        surface_distance = {}
        image_surface_order = {}
        for surface_direction in world_points:
            # Get the distance in each dimension from the camera to
            # each surface
            surface_distance[surface_direction] = np.mean(
                camera_position - world_points[surface_direction], axis=1
            )

            # Get the difference in position between
            # the camera and each surface in its normal's axis
            # to determine the order of the edges
            dimension = np.argmax(np.abs(surface_direction))
            pos_dir = surface_distance[surface_direction][:, dimension] >= 0
            image_surface_order[surface_direction] = pos_dir

        # Remove the surface direction key and concatenate everything
        # for easier handling
        world_points = np.concatenate([world_points[d] for d in world_points])
        image_points = np.concatenate([image_points[d] for d in image_points])
        values = np.concatenate([values[d] for d in values])
        image_surface_order = np.concatenate(
            [image_surface_order[d] for d in image_surface_order]
        )
        surface_distance = np.concatenate(
            [surface_distance[d] for d in surface_distance]
        )
        surface_distance = np.linalg.norm(surface_distance, axis=1)

        # Sort the surfaces and accompanying data by the surface's
        # average distance to the camera. This seems to be a
        # good enough approximation for drawing order to avoid artifacts.
        distance_sorting = np.argsort(surface_distance)
        image_points = image_points[distance_sorting]
        image_surface_order = image_surface_order[distance_sorting]
        values = values[distance_sorting]

        # Reorder the surface points so that we can form polygons of all quads
        # by using the vectors (b - a), (c - b), (d - c) and (a - d)
        # Depending on the sign of the position difference calculated above, different
        # orders are used
        # TODO: This and the calculation of image_surface_order is likely unnecessarily
        # complicated, simplify
        if any(image_surface_order):
            image_points[image_surface_order] = image_points[image_surface_order][
                :, [2, 0, 1, 3], :
            ]

        if any(~image_surface_order):
            image_points[~image_surface_order] = image_points[~image_surface_order][
                :, [2, 3, 1, 0], :
            ]

        # Find the mins and maxes of each surface in order to find the
        # "rectangle of interest" that we will work on
        min_y_per_surface = np.clip(
            np.floor(np.min(image_points[:, :, 0], axis=1)), 0, image_size[0] - 2
        ).astype(int)
        max_y_per_surface = np.clip(
            np.ceil(np.max(image_points[:, :, 0], axis=1)),
            0,
            image_size[0] - 1,
        ).astype(int)
        min_x_per_surface = np.clip(
            np.floor(np.min(image_points[:, :, 1], axis=1)),
            0,
            image_size[1] - 2,
        ).astype(int)
        max_x_per_surface = np.clip(
            np.ceil(np.max(image_points[:, :, 1], axis=1)),
            0,
            image_size[1] - 1,
        ).astype(int)

        # Readjust the coordinate system to fit inside our smaller rectangle of interest
        image_points[:, :, 0] -= min_y_per_surface[:, np.newaxis]
        image_points[:, :, 1] -= min_x_per_surface[:, np.newaxis]

        # Give the vertices of the quad variable names
        # to make visualizing the edges a little clearer
        a_per_surface = image_points[:, 0]
        b_per_surface = image_points[:, 1]
        c_per_surface = image_points[:, 2]
        d_per_surface = image_points[:, 3]

        # Calculate coefficients of each quad's edge lines
        ba_k_per_surface, ba_m_per_surface = self._point_pair_to_line(
            a_per_surface, b_per_surface
        )
        cb_k_per_surface, cb_m_per_surface = self._point_pair_to_line(
            b_per_surface, c_per_surface
        )
        dc_k_per_surface, dc_m_per_surface = self._point_pair_to_line(
            c_per_surface, d_per_surface
        )
        ad_k_per_surface, ad_m_per_surface = self._point_pair_to_line(
            d_per_surface, a_per_surface
        )

        # Some heuristic on how wide the line should be related to the area of the cell
        area_per_surface = (max_y_per_surface - min_y_per_surface) * (
            max_x_per_surface - min_x_per_surface
        )
        lw_per_surface = self.border_thickness * np.sqrt(area_per_surface)

        # Create some background to paint on
        frame = np.zeros(image_size)

        # Boolean "depth buffer" for if pixel is free (drawable) or not
        # This is possible as the drawing is done front to back
        depth_buffer = np.ones(image_size, dtype="bool")

        frame_coord_grid = np.mgrid[0 : image_size[0], 0 : image_size[1]]
        # The main draw loop
        for surface_idx in range(len(image_points)):
            min_y = min_y_per_surface[surface_idx]
            max_y = max_y_per_surface[surface_idx]
            min_x = min_x_per_surface[surface_idx]
            max_x = max_x_per_surface[surface_idx]

            # If there are no free pixels
            # in the RoI, skip further calculations as they could
            # not result in any pixels drawn
            # This can increase performance manifold when
            # there is a lot of occlusion
            depth_buffer_roi = depth_buffer[min_y:max_y, min_x:max_x]
            if not np.any(depth_buffer_roi):
                continue

            lw = lw_per_surface[surface_idx]

            a = a_per_surface[surface_idx]
            b = b_per_surface[surface_idx]
            c = c_per_surface[surface_idx]
            d = d_per_surface[surface_idx]

            # Create a view of an y, x index grid to plug into our "quad fill" formula
            coords = frame_coord_grid[:, 0 : max_y - min_y, 0 : max_x - min_x]

            # Paint the quad's inside
            # Find all points that are inside the quad's exterior edges
            within_edges = (
                depth_buffer_roi
                & (
                    (coords[1] - a[1]) * (b[0] - a[0])
                    - (coords[0] - a[0]) * (b[1] - a[1])
                    >= 0
                )
                & (
                    (coords[1] - b[1]) * (c[0] - b[0])
                    - (coords[0] - b[0]) * (c[1] - b[1])
                    >= 0
                )
                & (
                    (coords[1] - c[1]) * (d[0] - c[0])
                    - (coords[0] - c[0]) * (d[1] - c[1])
                    >= 0
                )
                & (
                    (coords[1] - d[1]) * (a[0] - d[0])
                    - (coords[0] - d[0]) * (a[1] - d[1])
                    >= 0
                )
            )

            # Put the value of the surface within the surface's edges
            # on the image frame
            frame[min_y:max_y, min_x:max_x][within_edges] = values[surface_idx]
            depth_buffer[min_y:max_y, min_x:max_x][within_edges] = False

            if not self.show_border:
                continue

            # Paint the edges

            # Get the y=k*x+m line coefficients for each
            # clockwise edge: b->a, c->b, d->c, a->d
            ba_k = ba_k_per_surface[surface_idx]
            ba_m = ba_m_per_surface[surface_idx]

            cb_k = cb_k_per_surface[surface_idx]
            cb_m = cb_m_per_surface[surface_idx]

            dc_k = dc_k_per_surface[surface_idx]
            dc_m = dc_m_per_surface[surface_idx]

            ad_k = ad_k_per_surface[surface_idx]
            ad_m = ad_m_per_surface[surface_idx]

            # Paint the border by setting the border value on all points that
            # are within the quad and within lw of an edge
            frame[min_y:max_y, min_x:max_x][
                within_edges
                & (
                    (np.abs(ba_k * coords[1] + ba_m - coords[0]) <= lw)
                    | (np.abs(cb_k * coords[1] + cb_m - coords[0]) <= lw)
                    | (np.abs(dc_k * coords[1] + dc_m - coords[0]) <= lw)
                    | (np.abs(ad_k * coords[1] + ad_m - coords[0]) <= lw)
                )
            ] = self.border_value

        return frame
