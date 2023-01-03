
from collections import defaultdict
import curses
import curses.panel

import numpy as np
import os

from .io_device import IODevice
from ..renderers.point_renderer import PointRenderer


class Terminal(IODevice):
    ''' An I/O device that outputs frame data by writing characters
        to the terminal using the ncurses module
    '''

    def __init__(self, colors=None, *args, **kwargs):
        self.suggested_renderer = PointRenderer
        super(Terminal, self).__init__(*args, **kwargs)

        # Curses setup
        # TODO: Find a solution so we can sort of wrap the evil curses terminal
        # state modifying code to avoid having bad terminals after a crash
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
            curses.init_pair(pair_idx, 0, color_idx)
            color_to_pair_idx[color_idx] = pair_idx

        # Create a map from value to the correct color pair object
        self.colors = defaultdict(
            lambda: curses.color_pair(color_to_pair_idx[curses.COLOR_RED]))
        self.colors.update({'border': curses.color_pair(color_to_pair_idx[curses.COLOR_CYAN])})
        if colors is not None:
            self.colors.update(
                {
                    value: curses.color_pair(color_to_pair_idx[color])
                    for value, color in colors.items()
                }
            )
            
        self.colors[self.renderer.border_value] = self.colors["border"]

    def get_resolution(self):
        # Return the _writable_ area of the terminal
        w, h = os.get_terminal_size()
        return (h - 1, w)

    def _output_frame_data(self, frame_data, camera):
        # Concatenate data as the direction is not needed here
        world_points = np.concatenate([frame_data[d]['world_coordinates'] for d in frame_data])
        image_points = np.concatenate([frame_data[d]['image_coordinates'] for d in frame_data])
        values = np.concatenate([frame_data[d]['values'] for d in frame_data])

        # Sort world points by distance so that we can paint the
        # further points first in order to not overwrite a nearer surface
        camera_position = np.array([[camera.x, camera.y, camera.z]])
        point_distances = np.linalg.norm(
            camera_position - world_points, axis=1)
        distance_sorting = np.argsort(point_distances)
        image_points = image_points[distance_sorting]
        values = values[distance_sorting]

        # Round to nearest integer position
        image_points = (np.round(image_points)).astype(int)

        # Paint points in terminal
        self.paint_points(image_points, values)

    def paint_points(self, image_points, values):
        """Paints characters at positions defined by image_points and with
        colors as defined by corresponding element in values. However,
        the character used here is just a space and the actual color comes
        from changing the foreground to the background color and vice versa,
        thus filling the entire character cell with color.

        Args:
            image_points (np.array): A Nx2 array that consists of N points
                with (u, v) coordinates. The (u, v) coordinates define the pixel
                position on the screen on which to draw the point
            values (np.array): A (N,)-shaped array that consists of one
                value for every point in image_points. The point and value
                correspondence is done by index, i.e., image_points[i] will
                be drawn with the color that corresponds to the value of values[i].
            camera (Camera): The Camera object that holds the window in which
                to draw the pixels/characters.

        """

        # Remove old image contents
        self.stdscr.erase()

        # Paint each image point coordinate that has not already been painted
        # This is ok since the image points are distance sorted
        painted_coordinates = set()
        for i, p in enumerate(image_points):
            p = (p[0], p[1])
            if p in painted_coordinates:
                continue
            painted_coordinates.add(p)
            # Find the value of the world point corresponding to the
            # image frame point p in order to decide color/attributes
            pair_index = self.colors[values[i]]

            # Add some kind of character at the pixel position
            self.stdscr.addch(*p, " ", pair_index)

        # Draw a box around the screen because it's neat and refresh the panel's contents
        self.stdscr.box()
        self.stdscr.refresh()
