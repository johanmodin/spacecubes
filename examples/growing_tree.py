""" A world which has a dynamic little tree that grows (i.e., the world object is changed) """

import numpy as np
import math

from spacecubes import Camera
from spacecubes.io_devices import OpenCV


def create_tree(world, x, y, timestep):
    """Creates a tree in a world array at position x, y
    that grows for each timestep.
    """
    max_height = world.shape[-1] - 1
    initial_height = 5
    grow_rate = 0.1
    # A tree has some initial length and then some grow rate
    tree_length = min(max_height, initial_height + round(timestep * grow_rate))

    # Set the tree trunk to value 2
    world[x, y, 1:tree_length] = 2

    # Leaves are very orderly - 3 units long and occur every 5 units in height
    leaves_every_n = 5
    leaf_length = 3

    # Do trees bring their old leaves up as they grow or do they create new ones at the top?
    # Who knows? Perhaps it comes down to where the tree grows?
    # In this case, the leaves are brought up with the tree
    leaf_heights = [
        leaves_every_n * h for h in range(int(tree_length / leaves_every_n))
    ]
    for h in leaf_heights:
        leaf_base = tree_length - h

        # Leaves grow from the trunk in the x and y axis
        for direction in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            for d in range(leaf_length):
                # Set voxels to be leaves
                world[x + direction[0] * d, y + direction[1] * d, leaf_base - d] = 3


# Initialize the camera with some position a bit away from the tree
camera = Camera(x=-30, y=0, z=10)

# Give the world tree-y colors
colors = {1: (12, 48, 21), 2: (3, 30, 54), 3: (17, 105, 7)}
device = OpenCV(colors, resolution=(512, 512))

# Setup some variables needed during the render loop
timestep = 0
world_size = 10
world_height = 30
tree_x = int(world_size / 2)
tree_y = int(world_size / 2)

# The render loop
for _ in range(1000):
    # Recreate the world every iteration to clean up old leaves etc
    world = np.zeros((world_size, world_size, world_height))
    world[:, :, 0] = 1

    # Grow the tree
    create_tree(world, tree_x, tree_y, timestep)

    # Look at the tree
    if np.linalg.norm([tree_x, tree_y] - camera.position[:-1]) < 20:
        camera.move(up=0.1, right=0.1, forward=0.03)
    else:
        camera.move(up=0.01, forward=0.1)

    # Find a nice position to look at, namely the very top of the tree
    tree_height = np.max(np.argwhere(world[tree_x, tree_y, :]))
    object_center = [5, 5, tree_height]

    # Point the camera in the direction of the object center, but interpolated
    # to make the motion smoother. The amount is 0.2, which will
    # point the camera 0.2 in between the *current* camera orientation and
    # looking at the target position object_center. As the
    # camera orientation inches closer to looking at object_center every
    # render and the object_center only changes once every few renders, this
    # amount is enough to both make the motion look smooth and yet not
    # lag behind the object_center.
    camera.look_at_interpolated(*object_center, amount=0.2)

    # Show the world array from the perspective of the camera
    # in the OpenCV window
    device.render(world, camera)
    timestep += 1
