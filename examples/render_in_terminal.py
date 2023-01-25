''' This example shows how the use the Terminal io_device, which
    outputs the rendered graphics in the terminal through ncurses.

    The resolution of the render is the same as the resolution
    of the terminal. The resolution can often be increased/decrease with
    "CTRL -" and "CTRL +"
'''

import numpy as np

from spacecubes import Camera
from spacecubes.io_devices import Terminal

# Make three cubes with different values
world = np.zeros(((5, 5, 5)))
world[0, 0, 0] = 1
world[0, 0, 2] = 2
world[0, 0, 4] = 3

# Initialize the camera some distance from the cubes
camera = Camera(x=2, y=2, z=2)

# For the terminal, the colors dict is optional
# and is a mapping between the np array value
# and integer color code in the range [0, 7].
# The colors are normally:
# 0: black, 1: red, 2: green, 3: yellow,
# 4: blue, 5: magenta, 6: cyan, and 7: white

# This should color the cubes blue, magenta and red respectively
colors = {1: 4, 2: 5, 3: 1}

# Initialize the Terminal io_device
t = Terminal(colors, fps=30)

# Find the center of the object
object_center = np.average(np.argwhere(world > 0), axis=0)

# The render loop
# The resolution can be increased by increasing the terminal resolution
# i.e., decreasing the font size
for i in range(300):
    # Render the cubes
    t.render(world, camera)

    # Move the camera in the camera's perspective
    camera.move(up=0.01, right=0.01)

    # Point the camera at the center of the cubes
    camera.look_at(*object_center)

# Closing the terminal object when you're done with it
# is best, as ncurses can when interrupted make
# the terminal look a little funky
t.close()
