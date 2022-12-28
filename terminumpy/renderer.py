import os
import time
import sys
import curses
import curses.panel
from output_device import OutputDevice
import scipy
from scipy.ndimage import convolve


class Renderer:
    def __init__(self, output_device=None, fps=30):
        ''' Initializes the renderer.
            Args:
                output_device (OutputDevice): An OutputDevice which 
                    receives the rendered output. If None, a default
                    OutputDevice for the current terminal will be created.
                fps (int): The maximum number of frames per second to render
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
        for i in range(0, curses.COLORS):
            curses.init_pair(i + 1, i, -1)

    def create_camera_panel(self):
        ''' Used to create a new curses panel for a new camera'''
        window = curses.newwin(*os.get_terminal_size()[::-1])
        panel = curses.panel.new_panel(window)
        return panel

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
        h, w = camera.panel.window().getmaxyx()

        # Perform 3d convolution to find AABB edges
        #filter = np.ones((3, 3, 3))
        #filter[1,1,1] = 0

        # Seems to work ok but 3d conv is pretty slow over large volumes
        # Find cell edges by looking at how many neighbours each cell has
        #c1 = convolve((world_array > 0).astype(int), filter, mode='constant')
        #world_array[np.where((world_array > 0) & (c1 <= 16))] = 3

        # Probably does not work with complex geometries
        # Find inward cell edges by looking at how many neighbours each cell has
        #c2 = convolve((world_array == 0).astype(int), filter, mode='constant')
        #world_array[np.where((world_array > 0) & (c1 > 16) & (c2 == 3))] = 3

        # Find what points to render
        world_points = np.argwhere(world_array)

        # Order the points by distance such that we render the 
        # farthest points first. Useful to not overwrite
        # e.g., edges which should be visible from the camera perspective
        camera_position = np.array([[camera.x, camera.y, camera.z]])
        point_distances=scipy.spatial.distance.cdist(camera_position, world_points)[0]
        distance_sorting = np.argsort(-point_distances)
        world_points = world_points[distance_sorting]

        # Convert the points into the camera coordinate system
        camera_points = camera.world_to_camera_coordinates(world_points.T)

        # Project the points into the image frame
        image_size = (h, w)
        image_points, indices_remaining = self.project_points(camera_points, camera, image_size)
        image_points = image_points.T

        # Filter out of image frame points with some margin to allow for rounding and curses requirements
        inframe_points = np.where((image_points[:, 0] >= 0) & (image_points[:, 1] >= 0) & (
            image_points[:, 0] < h - 2) & (image_points[:, 1] < w - 1))
        image_points = image_points[inframe_points]
        indices_remaining = indices_remaining[inframe_points]

        # Filter NaN, inf, etc     
        finite_points = np.where(np.isfinite(image_points[:, 0]) & np.isfinite(image_points[:, 1]))
        image_points = image_points[finite_points]
        indices_remaining = indices_remaining[finite_points]

        # Round pixel coordinates to nearest integer position
        image_points = np.around(image_points).astype(int)

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
        for i, p in enumerate(image_points):
            # Find the value of the world point corresponding to the 
            # image frame point p in order to decide color/attributes
            wp = world_points[indices_remaining[i]]
            wx, wy, wz = wp[0], wp[1], wp[2]
            point_value = int(world_array[wx, wy, wz])
            color = curses.color_pair(point_value)

            # Add some kind of character at the pixel position
            camera.panel.window().addch(*p, '\U000025A9', color)

        # Draw a box around the screen because it's neat and refresh the panel's contents
        camera.panel.window().box()
        camera.panel.window().refresh()
        curses.panel.update_panels()
        
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
        indices_remaining = np.where(cond)[0]
        points_im = h_points_i[:, cond]

        # Remove the last column
        points_im = points_im[:2, :]
        return points_im, indices_remaining


if __name__ == '__main__':
    t1 = time.time()
    import numpy as np
    from camera import Camera
    world_size = 150

    # Some kind of horse shoe shape
    thickness = 10
    length = 50
    mid = world_size // 2
    cube_world = np.zeros((world_size+1, world_size+1, world_size+1))
    cube_world[mid:mid + thickness + length, mid:mid + thickness, mid:mid + thickness] = 5
    cube_world[mid:mid + thickness, mid:mid + thickness + length, mid:mid + thickness] = 5
    cube_world[mid + length:mid + length + thickness, mid:mid + thickness + length, mid:mid + thickness] = 5
    cube_center = np.array([s // 2 for s in cube_world.shape])

    # Perform 3d convolution to find AABB edges
    # This is done "offline" for now until we have a faster solution
    # and can do it for every render or for every time the world updates
    filter = np.ones((3, 3, 3))
    filter[1,1,1] = 0

    # Seems to work ok but 3d conv is pretty slow over large volumes
    # Find cell edges by looking at how many neighbours each cell has
    c1 = convolve((cube_world > 0).astype(int), filter, mode='constant')
    cube_world[np.where((cube_world > 0) & (c1 <= 16))] = 3

    # Probably does not work with complex geometries
    # Find inward cell edges by looking at how many neighbours each cell has
    c2 = convolve((cube_world == 0).astype(int), filter, mode='constant')
    cube_world[np.where((cube_world > 0) & (c1 > 16) & (c2 == 3))] = 3

    r = Renderer(fps=10)
    import math
    # We're looking down the z-axis
    cam = Camera(r, x=0, y=0, z=0)
    move_size = 3
    moves = [move_size for i in range(100)]
    i = 0
    for _ in range(100):
        r.render(cube_world, cam)
        cam.move(x=0, y=moves[i % 100] * math.sin(i % 100 / 100 * math.pi * 2),
                 z=moves[i % 100] * math.cos(i % 100 / 100 * math.pi * 2))
        #cam.rotate(pitch=0.1)

        cam.look_at(*cube_center)
        i+=1
    curses.endwin()
    t2 = time.time()
    print(os.get_terminal_size(), t2 - t1)

    # ncplane_gradient



