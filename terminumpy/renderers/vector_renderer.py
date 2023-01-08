import math
import numpy as np

from .renderer import Renderer


class VectorRenderer(Renderer):
    def __init__(self, *args, **kwargs):
        super(VectorRenderer, self).__init__(*args, **kwargs)
        """Initializes the vector renderer.
        """
        self.cell_grid_image_size = None

    def render(self, world_array, camera, image_size):
        """ Create a image_size-sized np array that represents
            the projection of the world_array as seen from camera.

        Args:
            world_array (np.array): A numpy array which
                specifies the 3D world in which the camera's perspective
                is rendered.
            camera (Camera): A Camera object that allows the render-function
                to get the translation and rotation of the camera
                with regards to world_array
            image_size ((int, int)): A tuple of two ints representing the
                desired (height, width) of the output numpy array.
        """

        # Get world coordinate system surfaces from the render-function of the parent Renderer
        surfaces = self.render_surfaces(world_array, camera)

        # Project the world points to the image
        for surface_direction in surfaces:
            surfaces[surface_direction]['image_coordinates'], visible_indices = self.world_points_to_image_points(
                surfaces[surface_direction]['world_coordinates'], camera, image_size)

        # Reorder y-axis elements to align them with the orders of other axes
        surfaces[(0, 1, 0)]['image_coordinates'] = np.take(surfaces[(
            0, 1, 0)]['image_coordinates'], np.array([3, 1, 2, 0]), axis=1)
        surfaces[(0, -1, 0)]['image_coordinates'] = np.take(surfaces[(
            0, -1, 0)]['image_coordinates'], np.array([3, 1, 2, 0]), axis=1)

        camera_position = np.array([[camera.x, camera.y, camera.z]])
        pos_diffs = {}
        positive_pos_along_surface_axis = {}
        for surf_dir in surfaces:
            world_points = surfaces[surf_dir]['world_coordinates']
            pos_diffs[surf_dir] = np.mean(camera_position - world_points, axis=1)
            dimension = np.argmax(np.abs(surf_dir))
            positive_pos_along_surface_axis[surf_dir] = pos_diffs[surf_dir][:, dimension] >= 0

        # Remove the surface direction key and concatenate everything into bigger
        # piles of data to make handling it a bit easier
        image_points = np.concatenate([surfaces[d]['image_coordinates'] for d in surfaces])
        values = np.concatenate([surfaces[d]['values'] for d in surfaces ])
        image_surface_order = np.concatenate([positive_pos_along_surface_axis[d] for d in positive_pos_along_surface_axis])
        image_points_distance = np.linalg.norm(np.concatenate(
            [pos_diffs[d] for d in pos_diffs]), axis=1)


        # Sort the surfaces and accompanying data by the surface's
        # average distance to the camera. This seems to be a 
        # good enough approximation for drawing order to avoid artifacts.
        distance_sorting = np.argsort(-image_points_distance)
        image_points = image_points[distance_sorting]
        image_surface_order = image_surface_order[distance_sorting]
        values = values[distance_sorting]

        # Reorder the surface points so that we can form polygons of all quads
        # by using the vectors (b - a), (c - b), (d - c) and (a - d)
        if any(image_surface_order):
            image_points[image_surface_order] = image_points[image_surface_order][:, [2, 0, 1, 3], :]

        if any(~image_surface_order):
            image_points[~image_surface_order] = image_points[~image_surface_order][:, [2, 3, 1, 0], :]


        def point_pair_to_line(p1, p2):
            # Formula to get the line coefficients from two points
            kdiv = (p1[:, [1]] - p2[:, [1]])
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

            # Extract a rectangle of interest region that is enveloping the quad
            # and work on that instead in order to lower the amount of
            # data shuffled around in each op

            # TODO: Handle clipping better. Right now we're just trying to avoid
            # crashing, but image points outside of the image frame should be
            # handled more gracefully, i.e., take care of the portion
            # of the quads that are within the frame as expected in a 3d world
            min_y = np.clip(math.floor(
                np.min(image_points[[surface_idx], :, 0])), 0, image_size[0])
            max_y = np.clip(
                math.ceil(np.max(image_points[[surface_idx], :, 0])), 0, image_size[0])
            min_x = np.clip(math.floor(
                np.min(image_points[[surface_idx], :, 1])), 0, image_size[1])
            max_x = np.clip(math.ceil(np.max(image_points[[surface_idx], :, 1])), 0, image_size[1])

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
            coords = np.mgrid[: max_y - min_y, 0: max_x - min_x]

            # Paint the quad's inside
            within_line_1 = (((coords[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) -
                             (coords[0] - a[:, 0]) * (b[:, 1] - a[:, 1])) >= 0)
            within_line_2 = (((coords[1] - b[:, 1]) * (c[:, 0] - b[:, 0]) -
                             (coords[0] - b[:, 0]) * (c[:, 1] - b[:, 1])) >= 0)
            within_line_3 = (((coords[1] - c[:, 1]) * (d[:, 0] - c[:, 0]) -
                             (coords[0] - c[:, 0]) * (d[:, 1] - c[:, 1])) >= 0)
            within_line_4 = (((coords[1] - d[:, 1]) * (a[:, 0] - d[:, 0]) -
                             (coords[0] - d[:, 0]) * (a[:, 1] - d[:, 1])) >= 0)
            frame[min_y: max_y, min_x: max_x][within_line_1 & within_line_2 & 
                                              within_line_3 & within_line_4] = values[surface_idx]

            # Paint border
            # Get coefficients of each quad's edge lines
            ba_k, ba_m = point_pair_to_line(a, b)
            cb_k, cb_m = point_pair_to_line(b, c)
            dc_k, dc_m = point_pair_to_line(c, d)
            ad_k, ad_m = point_pair_to_line(d, a)
            
            # Soem heuristic on how wide the line should be related to the area of the cell
            area = (max_y - min_y) * (max_x - min_x)
            lw = self.border_thickness * math.sqrt(area)

            # Paint all points that are within the quad and within lw of an edge 
            frame[min_y: max_y, min_x: max_x][within_line_1 & within_line_2 &
                                              within_line_3 & within_line_4 &
                                              ((np.abs(ba_k * coords[1] + ba_m - coords[0]) < lw) |
                                              (np.abs(cb_k * coords[1] + cb_m - coords[0]) < lw) |
                                              (np.abs(dc_k * coords[1] + dc_m - coords[0]) < lw) |
                                              (np.abs(ad_k * coords[1] + ad_m - coords[0]) < lw))] = self.border_value

        return frame


