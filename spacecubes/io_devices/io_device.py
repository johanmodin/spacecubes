import time
from abc import ABC, abstractmethod

from ..renderer import Renderer


class IODevice(ABC):
    def __init__(self, renderer=None, fps=None, *args, **kwargs):
        """Create an IODevice.

        Args:
            renderer (Renderer): The renderer used to produce
                frames for this IODevice to show
            fps (float): A value specifying the maximum number
                of frames per second to show.
        """
        super(IODevice, self).__init__(*args, **kwargs)
        if renderer is None:
            self.renderer = Renderer()
        else:
            self.renderer = renderer
        self.fps = fps
        self.time_to_next_frame = time.time()

    @abstractmethod
    def get_resolution(self):
        """Returns the resolution of the device"""
        pass

    @abstractmethod
    def _output_frame_data(frame_data, camera):
        """Outputs frame data as defined by the specific IODevice and/or returns
        data related to the frame.

        """
        pass

    def get_input(self, timeout=0):
        """Returns input received by the device, e.g., key strokes"""
        raise NotImplementedError(
            f"This function does not yet have an implementation in {self.__class__.__name__}"
        )

    def close(self):
        """Performs a graceful shutdown of the IODevice"""
        raise NotImplementedError(
            f"This function does not yet have an implementation in {self.__class__.__name__}"
        )

    def render(self, world_array, camera):
        """Displays the world_array from the perspective of camera.

        Args:
            world_array (np.array): A three dimensional numpy array
                with non-zero elements that define the world to display
            camera (Camera): A Camera object which defines the perspective
                to use when displaying the world

        Optionally returns an output from the _output_frame_data function.
        """
        frame_data = self.renderer.render(world_array, camera, self.get_resolution())
        return self._output_frame_data(frame_data)

    def limit_fps(self):
        """Sleep as needed to not exceed the frame rate of self.fps"""
        if self.fps is not None and self.fps > 0:
            t_wait = self.time_to_next_frame - time.time()
            self.time_to_next_frame = time.time() + 1 / self.fps
            if t_wait > 0:
                time.sleep(t_wait)
