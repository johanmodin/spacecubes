from collections import defaultdict
import curses
import curses.panel
import itertools
import math
import os
import scipy
import sys
from scipy.ndimage import convolve
import time


from output_device import OutputDevice


class Renderer:
    def __init__(self, output_device=None, fps=30, colors={1: curses.COLOR_BLUE, 2: curses.COLOR_RED}):
        ''' Initializes the renderer.
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
        '''
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
        self.colors = defaultdict(
            lambda: curses.color_pair(curses.COLOR_WHITE))
        if colors is not None:
            self.colors.update({value: curses.color_pair(color_to_pair_idx[color]) 
                                for value, color in colors.items()})


    def create_camera_panel(self):
        ''' Used to create a new curses panel for a new camera'''
        window = curses.newwin(*os.get_terminal_size()[::-1])
        panel = curses.panel.new_panel(window)
        return panel

    def world2image(self, world_points, camera):
        # Our world_points are grouped into 4-point surfaces
        # which we will need to temporarily undo
        n_points = world_points.shape[0]
        world_points = world_points.reshape((-1, 3))
        camera_points = camera.world_to_camera_coordinates(world_points.T)

        # Project the points into the image frame
        h, w = camera.panel.window().getmaxyx()
        image_size = (h, w)
        image_points, visible_indices = self.project_points(camera_points, camera, image_size)
        #image_points = image_points.reshape((2,) + original_shape[:2])
        #visible_indices = visible_indices.reshape(original_shape[:2])
        image_points = image_points.T
        # Filter out of image frame points with some margin to allow for rounding and curses requirements
        inframe_points = np.where((image_points[:, 0] >= 0) & (image_points[:, 1] >= 0) & (
            image_points[:, 0] < h - 2) & (image_points[:, 1] < w - 1))

        # Group such that we remove points only if all four corners are out of frame
        #image_points = image_points[inframe_points]
        #visible_indices = visible_indices[inframe_points]

        # Filter NaN, inf, etc     
        finite_points = np.where(np.isfinite(image_points[:, 0]) & np.isfinite(image_points[:, 1]))
        #image_points = image_points[finite_points]
        #visible_indices = visible_indices[finite_points]


        #print(visible_indices.shape)
        #print(inframe_points.shape)
        #print(finite_points.shape)
        #visible_indices = visible_indices.reshape(4, n_points)
        #inframe_points = inframe_points.reshape(4, n_points)
        #finite_points = finite_points.reshape(4, n_points)

        # All points in a cell must be finite for us to keep it
        keep_finite = np.zeros((4 * n_points), dtype='bool')
        keep_finite[finite_points] = True
        keep_finite = keep_finite.reshape((n_points, 4))
        keep_finite = np.all(keep_finite, axis=1)

        # If any point in a cell is visible or inframe, we keep it
        keep_inframe = np.zeros((4 * n_points), dtype='bool')
        keep_inframe[inframe_points] = True
        keep_inframe = keep_inframe.reshape((n_points, 4))
        keep_inframe = np.any(keep_inframe, axis=1)

        # If any point in a cell is visible or inframe, we keep it
        keep_visible = np.zeros((4 * n_points), dtype='bool')
        keep_visible[visible_indices] = True
        keep_visible = keep_visible.reshape((n_points, 4))
        keep_visible = np.any(keep_visible, axis=1)

        # Keep the image points matching our requirements
        image_points = image_points.reshape((n_points, 4, 2))
        image_points = image_points[keep_finite & keep_visible & keep_inframe, :]
        
        # Round pixel coordinates to nearest integer position
        image_points = np.around(image_points).astype(int)
        return image_points, visible_indices
        
    #@profile
    def render(self, world_array, camera):
        ''' Render a scene to the output device.
        
            Args:
                world_array (np.array): A numpy array which 
                    specifies the 3D world in which the camera's perspective
                    is rendered.
                camera (Camera): A Camera object that allows the render-function
                    to get the translation and rotation of the camera
                    with regards to world_array
        '''
        camera.panel.top()
        

        #world_points = np.where(world_array)

        world_array_int = (world_array > 0).astype(np.int8)
        world_array_int = np.pad(world_array_int, 1)

        surface_x = world_array_int[:-1, :, :] - world_array_int[1:, :, :]
        surface_y = world_array_int[:, :-1, :] - world_array_int[:, 1:, :]
        surface_z = world_array_int[:, :, :-1] - world_array_int[:, :, 1:]

        x_outer_points_pos_w = np.argwhere(surface_x == 1)
        x_outer_points_neg_w = np.argwhere(surface_x == -1)
        y_outer_points_pos_w = np.argwhere(surface_y == 1)
        y_outer_points_neg_w = np.argwhere(surface_y == -1)
        z_outer_points_pos_w = np.argwhere(surface_z == 1)
        z_outer_points_neg_w = np.argwhere(surface_z == -1)
        
        def cube2surface(points, sign, dim):
            # TODO: Figure out which order (x1, y1, x2, y2) these come in
            other_dims_offsets = np.array(list(itertools.product([0.5, -0.5], repeat=2)))
            surface_corner_offsets = np.insert(
                other_dims_offsets, dim, sign * 0.5, axis=1)

            surface_corner_offsets = np.repeat(
                surface_corner_offsets[np.newaxis, :, :], points.shape[0], axis=0)
            surfaces = surface_corner_offsets + points[:, np.newaxis, :]
            return surfaces

        x_surfaces_pos_w = cube2surface(x_outer_points_pos_w, 1, 0)
        x_surfaces_neg_w = cube2surface(x_outer_points_neg_w, -1, 0)
        y_surfaces_pos_w = cube2surface(y_outer_points_pos_w, 1, 1)
        y_surfaces_neg_w = cube2surface(y_outer_points_neg_w, -1, 1)
        z_surfaces_pos_w = cube2surface(z_outer_points_pos_w, 1, 2)
        z_surfaces_neg_w = cube2surface(z_outer_points_neg_w, -1, 2)

        # Order the points by distance such that we render the 
        # farthest points first. Useful to not overwrite
        # e.g., edges which should be visible from the camera perspective
        #camera_position = np.array([[camera.x, camera.y, camera.z]])
        #point_distances=scipy.spatial.distance.cdist(camera_position, world_points)[0]
        #distance_sorting = np.argsort(-point_distances)
        #world_points = world_points[distance_sorting]

        # Convert the points into the camera coordinate system
        x_surface_pos_ip, x_surface_pos_indices = self.world2image(x_surfaces_pos_w, camera)
        x_surface_neg_ip, x_surface_neg_indices = self.world2image(x_surfaces_neg_w, camera)
        y_surface_pos_ip, y_surface_pos_indices = self.world2image(y_surfaces_pos_w, camera)
        y_surface_neg_ip, y_surface_neg_indices = self.world2image(y_surfaces_neg_w, camera)
        z_surface_pos_ip, z_surface_pos_indices = self.world2image(z_surfaces_pos_w, camera)
        z_surface_neg_ip, z_surface_neg_indices = self.world2image(z_surfaces_neg_w, camera)

        # If the image points are far enough apart
        # we can fill the surface by finding
        # a perspective transform from a meshgrid
        # to the trapezoid that makes up the np array cell surface
        # in the image plane. We shouldn't always do this
        # as it is costly to do for many sets of points
        # TODO: do this


        # Multiple points mapping to the same pixel can be discarded
        # Actually they probably cant because we dont know
        # if we kept the closest point
        #image_points = np.unique(image_points, axis=0)

        # Sleep as needed to not exceed the frame rate of self.fps
        if self.fps is not None and self.fps > 0:
            t_wait = self.next_frame - time.time()
            self.next_frame = time.time() + 1 / self.fps
            if t_wait > 0:
                time.sleep(t_wait)

        # Print new screen content
        camera.panel.window().erase()

        self.paint_planes(x_surface_pos_ip, camera)
        self.paint_planes(x_surface_neg_ip, camera)
        self.paint_planes(y_surface_pos_ip, camera)
        self.paint_planes(y_surface_neg_ip, camera)
        self.paint_planes(z_surface_pos_ip, camera)
        self.paint_planes(z_surface_neg_ip, camera)


        # Draw a box around the screen because it's neat and refresh the panel's contents
        camera.panel.window().box()
        camera.panel.window().refresh()
        curses.panel.update_panels()

    def paint_points(self, image_points, camera):
        for i, p in enumerate(image_points):
            # TODO: Fix value to color map thing
            # Find the value of the world point corresponding to the 
            # image frame point p in order to decide color/attributes
            #wp = world_points[indices_remaining[i]]
            #wx, wy, wz = wp[0], wp[1], wp[2]
            #point_value = int(world_array[wx, wy, wz])
            #pair_index = self.colors[point_value]

            # Add some kind of character at the pixel position
            #camera.panel.window().addch(*p, '\U000025A9', pair_index)
            camera.panel.window().addch(int(p[0]), int(p[1]), '\U000025A9', 2)


    def paint_planes(self, surfaces, camera):
        # TODO: add pruning by size here
        # TODO: add meshgrid size depending on cell size on screen
        sx, sy = 10, 10
        default_mesh = np.mgrid[:sy, :sx].T.reshape(1, -1, 2)
        p1 = [0, 0]
        p2 = [0, sx]
        p3 = [sy, 0]
        p4 = [sy, sx]

        if len(surfaces) == 0:
            return

        def unique_2d(a):
            n = a.max()+1
            a_off = a + (np.arange(a.shape[0])[:, None])*n
            M = a.shape[0]*n
            out = (np.bincount(a_off.ravel(), minlength=M).reshape(-1,n)!=0).sum(1)
            return out
        #uniques = unique_2d(x_surface_pos_ip[:, :, 0] + 100000) + unique_2d(x_surface_pos_ip[:, :, 1] + 100000)
        #big_surfaces = x_surface_pos_ip[uniques > 6]
        '''
        big_surfaces = surfaces

        cell_surfaces = None
        if len(big_surfaces) > 0:
            default_bbox_arr = np.repeat(
                np.array([[p1, p2, p3, p4]]), len(big_surfaces), axis=0)
            params, good_matrices = self.make_projective_multi(
                default_bbox_arr, big_surfaces)
            big_surfaces = big_surfaces[good_matrices]
            if len(params) > 0:
                if len(params.shape) == 1:
                    params = params[np.newaxis, :]
                new_surfaces = self.projective_transform_multi(np.repeat(default_mesh, len(big_surfaces), axis=0), params)
                cell_surfaces = np.round(new_surfaces).reshape(-1, 2)
        '''


        cell_surfaces = None
        default_bbox_arr = np.repeat(
            np.array([[p1, p2, p3, p4]]), len(surfaces), axis=0)
        params, good_matrices = self.make_projective_multi(
            default_bbox_arr, surfaces)
        big_surfaces = surfaces[good_matrices]
        surfaces = surfaces[~good_matrices]
        if len(params) > 0:
            if len(params.shape) == 1:
                params = params[np.newaxis, :]
            new_surfaces = self.projective_transform_multi(np.repeat(default_mesh, len(big_surfaces), axis=0), params)
            cell_surfaces = np.round(new_surfaces).reshape(-1, 2)
        
        h, w = camera.panel.window().getmaxyx()
        # TODO: Probably check for isfinite here because of cell surfaces 
        # not being checked
        surfaces = surfaces.mean(axis=1).astype(int)

        
        if cell_surfaces is not None:
            surfaces = np.concatenate(
                [surfaces, cell_surfaces], axis=0)

        inframe_points = np.where((surfaces[:, 0] >= 0) & (surfaces[:, 1] >= 0) & (
            surfaces[:, 0] < h - 2) & (surfaces[:, 1] < w - 1))
        surfaces = surfaces[inframe_points]

        # Filter NaN, inf, etc
        finite_points = np.where(np.isfinite(
            surfaces[:, 0]) & np.isfinite(surfaces[:, 1]))
        surfaces = surfaces[finite_points]

        self.paint_points(surfaces, camera)


    def project_points(self, points, camera, image_size):    
        ''' Projects points from camera coordinate system (XYZ) to
            image plane (UV).

                Args:
                    camera (Camera): The points will be projected onto
                        this camera's image plane. 
                    points (np.array): An array of (3, N) points specified
                        in the camera's coordinate system
        '''
        scale_mat = np.array([[image_size[0], 0, 0], [0, image_size[1], 0], [0, 0, 1]])
        h_points_i = scale_mat @ camera.intrinsic_matrix @ points

        h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
        h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]
        
        # Remove points behind the camera
        cond = h_points_i[2, :] >= 0
        visible_indices = np.where(cond)[0]
        #points_im = h_points_i[:, cond]

        # Remove the last column
        points_im = h_points_i[:2, :]
        return points_im, visible_indices


    def make_projective_multi(self, src, dst):
        ''' Find M projection transform matrices for 
            M corresponding point sets src and dst.
            
            Args:
                src (np.array): A MxNx2 array with M sets of 
                    N points that each have two coordinates.
                    src[i, :, :] should correspond to dst[i, :, :].
                dst (np.array): A MxNx2 array with M sets of 
                    N points that each have two coordinates.
                    dst[i, :, :] should correspond to src[i, :, :].
        '''

        xs = src[:, :, 0]
        ys = src[:, :, 1]
        rows = src.shape[1]
        A = np.zeros((len(src), rows*2, 8))
        A[:, :rows, 0] = 1
        A[:, :rows, 1] = xs
        A[:, :rows, 2] = ys
        A[:, rows:, 3] = 1
        A[:, rows:, 4] = xs
        A[:, rows:, 5] = ys
        A[:, :rows, 6] = - dst[:, :, 0] * xs
        A[:, :rows, 7] = - dst[:, :, 0] * ys
        A[:, rows:, 6] = - dst[:, :, 1] * xs
        A[:, rows:, 7] = - dst[:, :, 1] * ys
        b = np.zeros((len(src), rows*2,))
        b[:, :rows] = dst[:, :, 0]
        b[:, rows:] = dst[:, :, 1]

        # TODO: Find a good way to make sure these matrices aint shit
        good_matrices = np.linalg.cond(A) < (1/sys.float_info.epsilon)
        A = A[good_matrices]
        b = b[good_matrices]

        params = np.squeeze(np.matmul(np.linalg.inv(np.matmul(A.swapaxes(1, 2), A)),
                                    np.matmul(A.swapaxes(1, 2), np.atleast_3d(b))))
        return params, good_matrices

    def projective_transform_multi(self, coords, params, inverse=False):
        ''' Vectorized projective transforms for performing
            projective transforms multiple sets of points.

            Args:
                coords (np.array): A MxNx2 array, where M is the number
                    of sets of points, each set with its own corresponding
                    set of params, and N being the number of points to transform.
                params (np.array): A Mx8 array of projective transform
                    parameters, where params[i, :] will be applied to the
                    points in coords[i, :, :]
        '''
        a0, a1, a2, b0, b1, b2, c0, c1 = zip(*params[:, :, np.newaxis])
        x = coords[:, :,0]
        y = coords[:, :,1]
        out = np.zeros(coords.shape)
        if inverse:
            out[:, :,0] = (a2*b0-a0*b2+(b2-b0*c1)*x+(a0*c1-a2)*y) \
                / (a1*b2-a2*b1+(b1*c1-b2*c0)*x+(a2*c0-a1*c1)*y)
            out[:, :,1] = (a0*b1-a1*b0+(b0*c0-b1)*x+(a1-a0*c0)*y) \
                / (a1*b2-a2*b1+(b1*c1-b2*c0)*x+(a2*c0-a1*c1)*y)
        else:
            out[:, :,0] = (a0+a1*x+a2*y) / (1+c0*x+c1*y)
            out[:, :,1] = (b0+b1*x+b2*y) / (1+c0*x+c1*y)
        return out




if __name__ == '__main__':
    import numpy as np
    from camera import Camera
    world_size = 30

    # Some kind of horse shoe shape
    thickness = 3
    length = 10
    origin = world_size // 2
    cube_world = np.zeros((world_size+1, world_size+1, world_size+1))
    cube_world[origin:origin + thickness + length, origin:origin + thickness, origin:origin + thickness] = 1
    cube_world[origin:origin + thickness, origin:origin + thickness + length, origin:origin + thickness] = 1
    cube_world[origin + length:origin + length + thickness, origin:origin + thickness + length, origin:origin + thickness] = 1

    # Perform 3d convolution to find AABB edges
    # This is done "offline" for now until we have a faster solution
    # and can do it for every render or for every time the world updates
    filter = np.ones((3, 3, 3))
    filter[1,1,1] = 0

    # Seems to work ok but 3d conv is pretty slow over large volumes
    # Find cell edges by looking at how many neighbours each cell has
    #c1 = convolve((cube_world > 0).astype(int), filter, mode='constant')
    #cube_world[np.where((cube_world > 0) & (c1 <= 16))] = 2

    # Probably does not work with complex geometries
    # Find inward cell edges by looking at how many neighbours each cell has
    #c2 = convolve((cube_world == 0).astype(int), filter, mode='constant')
    #cube_world[np.where((cube_world > 0) & (c1 > 16) & (c2 == 3))] = 2
    
    #cube_world_bool = cube_world.astype(bool)
    #cube_world_bool = np.pad(cube_world_bool, 1)
    #face_x = np.where(np.bitwise_xor(cube_world_bool[:-1, :, :], cube_world_bool[1:, :, :]))
    #face_y = np.where(np.bitwise_xor(cube_world_bool[:, :-1, :], cube_world_bool[:, 1:, :]))
    #face_z = np.where(np.bitwise_xor(cube_world_bool[:, :, :-1], cube_world_bool[:, :, 1:]))

    volume = cube_world
    volume_int = (volume > 0).astype(np.int8)
    volume_int = np.pad(volume_int, 1)

    surface_x = volume_int[:-1, :, :] - volume_int[1:, :, :]
    surface_y = volume_int[:, :-1, :] - volume_int[:, 1:, :]
    surface_z = volume_int[:, :, :-1] - volume_int[:, :, 1:]

    points_x_pos = np.where(surface_x == 1)
    points_x_neg = np.where(surface_x == -1)
    points_y_pos = np.where(surface_y == 1)
    points_y_neg = np.where(surface_y == -1)
    points_z_pos = np.where(surface_z == 1)
    points_z_neg = np.where(surface_z == -1)

    cube_world[:] = 0
    cube_world[points_x_pos] = 2
    cube_world[points_x_neg] = 3
    cube_world[points_y_pos] = 4 # why no work
    cube_world[points_y_neg] = 5
    cube_world[points_z_pos] = 6
    cube_world[points_z_neg] = 7

    # Some interesting place to look at
    object_center = np.average(np.argwhere(cube_world > 0), axis=0)

    # Create a renderer that is to render the Camera cam in the np array cube_world
    colors = {2: curses.COLOR_BLUE, 3: curses.COLOR_CYAN, 4: curses.COLOR_GREEN, 
              5: curses.COLOR_MAGENTA, 6: curses.COLOR_RED, 7: curses.COLOR_YELLOW}
    r = Renderer(fps=30, colors=colors)

    # Create a camera
    cam = Camera(r, x=30, y=30, z=30)
    move_size = 0.5
    n_moves = 100
    moves = [(move_size, 0, 0) for _ in range(n_moves)] + [(0, move_size, 0)
                                                           for _ in range(n_moves)] + [(0, 0, move_size) for _ in range(n_moves)]

    t1 = time.time()
    for i in range(len(moves)):
        # Render the cube_world array as seen by cam
        r.render(cube_world, cam)

        # Move the camera in a circle-ish pattern to visualize the 3d
        # information more clearly
        cam.move(*moves[i])
        #cam.move(x=0, y=3 * math.sin(i % 100 / 100 * math.pi * 2),
        #         z=3 * math.cos(i % 100 / 100 * math.pi * 2))
        #cam.rotate(pitch=0.1)

        # Redirect the camera to look at the center of the object
        cam.look_at(*object_center)
    t2 = time.time()
    curses.endwin()
    print(
        f'Rendered {i} steps at resolution {os.get_terminal_size().lines, os.get_terminal_size().columns} in {t2 - t1} seconds')

    # ncplane_gradient



