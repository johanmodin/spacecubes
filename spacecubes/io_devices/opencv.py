from collections import UserDict

import numpy as np
import os
import cv2
from zlib import crc32

from .io_device import IODevice


class OpenCV(IODevice):
    """An I/O device that outputs frame data through the
       OpenCV imshow functionality
    """

    def __init__(self, resolution=(600, 800), colors={}, *args, **kwargs):
        super(OpenCV, self).__init__(*args, **kwargs)
        self.resolution = resolution

        if "border" in colors:
            colors[self.renderer.border_value] = colors["border"]
            del colors["border"]
        else:
            colors[self.renderer.border_value] = (245, 240, 103)

        self.colors = {0: (0, 0, 0)}
        self.colors.update(colors.items())

        self.window_name = 'spacecubes'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def get_resolution(self):
        return self.resolution

    def close(self):
        cv2.destroyAllWindows()

    def _output_frame_data(self, frame_data, camera):
        # Paint points in terminal
        self.update_window(frame_data)

    def update_window(self, frame_data):
        """ Updates the OpenCV window with new frame data.

        Args:
            frame_data (np.array): The (h, w) numpy array to be displayed
        """

        def replace_values_with_bgr(frame, value_translation):
            k = np.array(list(value_translation.keys()))
            v = np.array(list(value_translation.values()))
            sidx = k.argsort()
            ks = k[sidx]
            vs = v[sidx]
            idx = np.searchsorted(ks, frame)
            return vs[idx]
        
        # Replace the values image frames' values with BGR colors
        # by the user-supplied value -> color mapping 
        out = replace_values_with_bgr(frame_data, self.colors).astype(np.uint8)

        # Optional FPS limiter here as all calculations are done
        self.limit_fps()

        # Update the window with new frame data
        cv2.imshow(self.window_name, out)
        cv2.waitKey(1)
