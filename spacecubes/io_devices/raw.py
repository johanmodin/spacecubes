from collections import UserDict

import numpy as np
import os
import cv2
from zlib import crc32

from .io_device import IODevice


class Raw(IODevice):
    """An I/O device that returns the raw numpy frame
    data, as opposed to showing it in some visualized fashion
    """

    def __init__(self, resolution=(600, 800), colors={}, *args, **kwargs):
        super(Raw, self).__init__(*args, **kwargs)
        self.resolution = resolution

        if "border" in colors:
            colors[self.renderer.border_value] = colors["border"]
            del colors["border"]
        else:
            colors[self.renderer.border_value] = (245, 240, 103)

        self.colors = {0: (0, 0, 0)}
        self.colors.update(colors.items())

    def get_resolution(self):
        return self.resolution

    def close(self):
        pass

    def _output_frame_data(self, frame_data, replace_values_with_colors=True):
        def replace_values_with_bgr(frame, value_translation):
            k = np.array(list(value_translation.keys()))
            v = np.array(list(value_translation.values()))
            sidx = k.argsort()
            ks = k[sidx]
            vs = v[sidx]
            idx = np.searchsorted(ks, frame)
            return vs[idx]

        if replace_values_with_colors:
            # Replace the values image frames' values with BGR colors
            # by the user-supplied value -> color mapping
            frame_data = replace_values_with_bgr(frame_data, self.colors).astype(
                np.uint8
            )

        # Optional FPS limiter here as all calculations are done
        self.limit_fps()

        return frame_data
