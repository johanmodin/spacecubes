import os
import time
import sys

from output_device import OutputDevice


LINE_UP = '\033[1A'


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

        w, h = self.output_device.get_resolution()

        # Find what points to render
        world_points = np.argwhere(world_array).T

        # Convert the points into the camera coordinate system
        camera_points = camera.world_to_camera_coordinates(world_points)

        # Project the points into the image frame
        image_size = (h, w)
        image_points = self.project_points(camera_points, camera, image_size).T

        # Filter NaN, inf, etc        
        image_points = np.where(np.isfinite(image_points), image_points, 0)
        point_set = set([tuple([round(c) for c in p]) for p in image_points if np.isfinite])
        
        next_frame = time.time()
        # Put this in class -- async?
        if self.fps is not None and self.fps > 0:
            t_wait = self.next_frame - time.time()
            self.next_frame = time.time() + 1 / self.fps
            if t_wait > 0:
                time.sleep(t_wait)

        # Print new screen content
        m = ''
        for line in range(h):
            for column in range(w):
                if (line, column) in point_set:
                    m += '#'
                else:
                    m += ' '
            m += '\n'

        # Reset the output pointer to the beginning of the screen
        self.set_pointer_n_lines_up(h)

        # Print new content
        print(m, end='', file=self.output_device.file)


    def set_pointer_n_lines_up(self, n):
        ''' Sets the pointer n lines up. Used to overwrite old content. '''
        print(LINE_UP * n, end='')


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
        points_im = h_points_i[:, h_points_i[2, :] >= 0]

        # Remove the last column
        points_im = points_im[:2, :]
        return points_im


if __name__ == '__main__':
    import numpy as np
    from camera import Camera
    cube_world = np.zeros((50, 50, 50))
    cube_world[0:20, 0:20, 0:20] = 1
    cube_center = np.array([7.5, 7.5, 7.5])

    r = Renderer(fps=25)
    import math
    # We're looking down the z-axis
    cam = Camera(x = -15, y = -15, z = -15)
    move_size = 4
    moves = [move_size for i in range(100)]
    i = 0
    while True:
        r.render(cube_world, cam)
        cam.move(x=moves[i % 100] * math.cos(i % 100 / 100 * math.pi * 2), y=moves[i % 100] * math.sin(i % 100 / 100 * math.pi * 2), z=np.random.random())
        #cam.rotate(pitch=0.1)

        cam.look_at(*cube_center)
        i+=1



