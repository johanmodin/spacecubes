import numpy as np

from .io_device import IODevice


class Raw(IODevice):
    """An I/O device that returns the raw numpy frame
    data, as opposed to showing it in some visualized fashion

    Args:
    colors (dict): A dict mapping numpy array values to rendered
        (B, G, R) colors, i.e., to map all voxels with value 5 to
        green, the colors dict would be: {5: (0, 255, 0)}.
        Additionally, the 'border' keyword can be used to set
        a certain color to the voxel borders:
        {'border': (50, 150, 250)}
    """

    def __init__(self, colors=None, resolution=(600, 800), *args, **kwargs):
        super(Raw, self).__init__(*args, **kwargs)
        self.resolution = resolution

        if colors is not None:
            if "border" in colors:
                colors[self.renderer.border_value] = colors["border"]
                del colors["border"]
            else:
                colors[self.renderer.border_value] = (245, 240, 103)

            self.colors = {0: (0, 0, 0)}
            self.colors.update(colors.items())
        else:
            self.colors = colors

    def get_resolution(self):
        return self.resolution

    def close(self):
        pass

    def _output_frame_data(self, frame_data):
        def replace_values_with_bgr(frame, value_translation):
            k = np.array(list(value_translation.keys()))
            v = np.array(list(value_translation.values()))
            sidx = k.argsort()
            ks = k[sidx]
            vs = v[sidx]
            idx = np.searchsorted(ks, frame)
            return vs[idx]

        if self.colors is not None:
            # Replace the values image frames' values with BGR colors
            # by the user-supplied value -> color mapping
            frame_data = replace_values_with_bgr(frame_data, self.colors).astype(
                np.uint8
            )

        # Optional FPS limiter here as all calculations are done
        self.limit_fps()

        return frame_data
