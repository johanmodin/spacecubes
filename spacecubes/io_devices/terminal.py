from collections import UserDict
import curses

import numpy as np
import os

from .io_device import IODevice


class Terminal(IODevice):
    """An I/O device that outputs frame data by writing characters
    to the terminal using the ncurses module

    Args:
        colors (dict): A color mapping from numpy array values to
            ncurses color values (0: black, 1: red, 2: green, 3: yellow,
            4: blue, 5: magenta, 6: cyan, and 7: white). Nonspecified
            colors will get a random color
    """

    def __init__(self, colors={}, *args, **kwargs):
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
        if "border" in colors:
            colors[self.renderer.border_value] = curses.color_pair(
                color_to_pair_idx[colors["border"]]
            )
            del colors["border"]
        else:
            colors[self.renderer.border_value] = curses.COLOR_CYAN

        # Assign missing values (ints) to any color
        class ColorDict(UserDict):
            def __missing__(self, key):
                color_pair = curses.color_pair(
                    color_to_pair_idx[(int(key)) % (curses.COLORS)]
                )
                return color_pair

        self.colors = ColorDict()
        self.colors.update(
            {
                value: curses.color_pair(color_to_pair_idx[color])
                for value, color in colors.items()
            }
        )

    def get_resolution(self):
        # Return the writable area of the terminal, which
        # curiously does not include the last row
        w, h = os.get_terminal_size()
        return (h - 1, w)

    def _output_frame_data(self, frame_data):
        # Paint points in terminal
        self.paint_points(frame_data)

    def close(self):
        curses.endwin()

    def paint_points(self, frame_data):
        """Paints characters into the terminal that with some
            enthusiasm can be seen as representing the 3d view.
            This is done by printing strings of the space character
            that have their background color set to whatever color
            is assigned to the value of the pixel. This is done to
            fill the entire character cell and provide a better image.

            This can get pretty slow at high resolutions.

        Args:
            frame_data (np.array): The (h, w) numpy array to be displayed

        """

        # Remove old image contents
        self.stdscr.erase()

        # Find where the frame changes color along each row
        # then print strings that occupy these spaces
        # This is an optimization over printing every
        # character individually
        padded_frame_data = np.pad(frame_data, ((0, 0), (1, 0)), constant_values=-1)
        color_changes = np.argwhere(
            padded_frame_data[:, 1:] - padded_frame_data[:, :-1]
        )
        color_changes = zip(color_changes, color_changes[1:])
        for from_pos, to_pos in color_changes:
            value = frame_data[from_pos[0], from_pos[1]]
            length = (to_pos[0] - from_pos[0]) * frame_data.shape[1] + (
                to_pos[1] - from_pos[1]
            )
            self.stdscr.addstr(
                from_pos[0], from_pos[1], " " * length, self.colors[value]
            )

        # Optional FPS limiter here as all calculations are done
        self.limit_fps()

        # Refresh the window's contents
        self.stdscr.refresh()
