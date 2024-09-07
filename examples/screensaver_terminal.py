""" Recreating the classic Windows screensaver but with (space)cubes.

    This examples draws the graphics to the terminal using ncurses.
    Resolution is controlled by resizing and zooming in/out of the terminal (e.g., CTRL +, CTRL -).
    
    Note: you will currently have to call `reset` in the terminal after exiting the program with CTRL C
    due to curses changing the terminal state.
"""
import numpy as np
import random

from spacecubes import Camera
from spacecubes.io_devices import Terminal

# Create a world that is slightly deeper (z-axis) than it is wide
world = np.zeros((50, 50, 100))

# Create a world-shaped array with integers that will represent colors
color = np.random.randint(0, 1000, size=world.shape)

# Create a random array that is world-shaped that will be used as a random mask
mask = np.random.random(size=world.shape)

# Define the probability of a voxel being occupied
p_voxel = 0.001

# Create a random set of voxels using the mask, each with a random color
world[mask < p_voxel] = color[mask < p_voxel]

# Center the camera along the x and y axis of the world "tunnel"
camera = Camera(x=25, y=25, z=0)

# Create random colors that are mapped to the curses.COLOR
# (0: black, 1: red, 2: green, 3: yellow, 4: blue, 5: magenta, 6: cyan, and 7: white)
colors = {i: (random.randint(1, 8)) for i in range(1, 1000)}

# We output the graphics to the terminal directly using curses
device = Terminal(colors, fps=30)

# Set some interesting direction to look in
camera.look_at(x=25, y=25, z=100)

# Render loop
while True:
    if camera.position[-1] > 1:
        # Once the camera has moved 1 unit forward, it is
        # moved back one unit in world coordinates
        # and the world is rolled around using np.roll
        # which puts the first elements (behind us)
        # at the far side of the world, producing the
        # effect of the camera always traveling forward
        camera.move_xyz(z=-1)
        world = np.roll(world, -1, axis=-1)

    # Move forward and render
    camera.move(forward=0.1)
    device.render(world, camera)
