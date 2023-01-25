""" An example showing the very basic usage of spacecubes
    through a simple example of rendering a single voxel (i.e., a cube)
"""

import numpy as np

from spacecubes import Camera
from spacecubes.io_devices import OpenCV

# Create some world object
world = np.ones((1, 1, 1))

# Initialize the camera with some position
camera = Camera(x=-1, y=-1, z=-1)

# Define what color our cube should have
# The cube's value is 1 (as it's created with np.ones),
# so that's the key which we set the color tuple for
# In this case, we're going for a smooth bastard amber
# The cube's border color can also be set using the keyword 'border'
colors = {1: (135, 204, 255), "border": (234, 135, 0)}

# Initialize an output device to show the resulting frames
# In this case, OpenCV is used.
device = OpenCV(colors, resolution=(512, 512))

# Calculate where the non-empty voxel is in order to know where
# to point the camera (it's at (0, 0, 0))
object_center = np.average(np.argwhere(world > 0), axis=0)

# The render loop
while True:
    # Move the camera to make the visualization interesting
    camera.move(up=0.01, right=0.01)

    # Point the camera at the nonzero voxel
    # This way, the camera is moving in a circle pattern around the voxel
    camera.look_at(*object_center)

    # Show the world array from the perspective of the camera in the OpenCV window
    device.render(world, camera)
