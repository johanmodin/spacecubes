''' An example showing how a camera can be moved 
    interactively using OpenCV.

    The controls are set to:
    W: Forward
    S: Back
    A: Left
    D: Right
    Q: Up
    Z: Down

    Left arrow: Turn left
    Right arrow: Turn right
    Up arrow: Look up
    Down arrow: Look down
    ,: Twist counter-clockwise
    .: Twist clockwise
'''

import numpy as np

from spacecubes import Camera
from spacecubes.io_devices import OpenCV

# Create some world object
world = np.zeros((10, 10, 10))
world[5, 5, 5] = 1
world[6, 5, 5] = 2
world[7, 5, 5] = 3

# Initialize the camera with some position
camera = Camera(x=-1, y=-1, z=-1)

# Initialize an output device to show the resulting frames
# In this case, OpenCV is used. The colors are mapped
# from numpy array value -> BGR (in the case of OpenCV)
colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
device = OpenCV(colors=colors, resolution=(512, 512))

# Define how much the user input should rotate or move the camera
rotation_amount = 0.1
move_size = 0.1
rotations = {
    3: {"pitch": rotation_amount},
    2: {"pitch": -rotation_amount},
    1: {"yaw": rotation_amount},
    0: {"yaw": -rotation_amount},
    ord("."): {"roll": rotation_amount},
    ord(","): {"roll": -rotation_amount},
}
movements = {
    ord("w"): {"forward": move_size},
    ord("s"): {"forward": -move_size},
    ord("a"): {"right": -move_size},
    ord("d"): {"right": move_size},
    ord("q"): {"up": move_size},
    ord("z"): {"up": -move_size},
}

# Point the camera at where the non-zero voxels are
object_center = np.average(np.argwhere(world > 0), axis=0)
camera.look_at(*object_center)

# The render loop
while True:
    # Show the world array from the perspective of the camera in the OpenCV window
    device.render(world, camera)

    # Get the user input through the OpenCV window
    key = device.get_input()

    # Translate whatever keys were pressed into actions
    if key in rotations:
        camera.rotate(**rotations[key])
    elif key in movements:
        camera.move(**movements[key])
