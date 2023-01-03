
import os
import sys
import time
from abc import ABC, abstractmethod

class IODevice(ABC):
    def __init__(self, renderer=None, fps=None, *args, **kwargs):
        """ Create an OutputDevice.

        Args:
            resolution (int, int): A (width, height) tuple specifying
                the resolution of the rendered output
        """
        super(IODevice, self).__init__(*args, **kwargs)
        if renderer is None:
            self.renderer = self.suggested_renderer()
        else:
            self.renderer = renderer
        self.fps = fps
        self.time_to_next_frame = time.time()


    @abstractmethod
    def get_resolution(self):
        """ Returns the resolution of the device"""
        pass

    @abstractmethod
    def _output_frame_data(frame_data, camera):
        ''' Outputs frame data as defined by the specific IODevice'''
        pass

    def show(self, world_array, camera):
        ''' Displays the world_array from the perspective of camera.

            Args:
                world_array (np.array): A three dimensional numpy array
                    with non-zero elements that define the world to display
                camera (Camera): A Camera object which defines the perspective
                    to use when displaying the world
        '''
        frame_data = self.renderer.render(world_array, camera, self.get_resolution())
        self._output_frame_data(frame_data, camera)


    def limit_fps(self):
        ''' Sleep as needed to not exceed the frame rate of self.fps '''
        if self.fps is not None and self.fps > 0:
            t_wait = self.time_to_next_frame - time.time()
            self.time_to_next_frame = time.time() + 1 / self.fps
            if t_wait > 0:
                time.sleep(t_wait)
