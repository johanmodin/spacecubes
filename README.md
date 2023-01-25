
# spacecubes
*Now listen you Royal Highness, take only what you need to survive!*


## Overview
`spacecubes` is a simple voxel renderer for three-dimensional NumPy arrays. It is made to be easy to use and allowing fast visualization. It is not made to produce good looking images or be feature rich.

## Demo
![Alt Text](https://media.giphy.com/media/1XADnkAnPnnw2YyCAg/giphy.gif)

Below is how the Windows 95 screensaver-esque demo was created using spacecubes ([examples/windows_screensaver](examples/windows_screensaver.py)).
```python
import numpy as np
from spacecubes import Camera
from spacecubes.io_devices import OpenCV

world = np.zeros((50, 50, 100))
color_mask = np.random.randint(0, 1000, size=world.shape)
random_mask = np.random.random(size=world.shape)
world[random_mask < 0.001] = color_mask[random_mask < 0.001]
colors = {i: np.random.random(3) * 255 for i in range(1, 1000)}
device = OpenCV(colors, resolution=(1080, 1920))
camera = Camera(x=25, y=25, z=0)
camera.look_at(x=25, y=25, z=100)
while True:
    if camera.position[-1] > 1:
        camera.move_xyz(z=-1)
        world = np.roll(world, -1, axis=-1)
    camera.move(forward=0.1)
    device.render(world, camera)
```

## Examples
Rendering a single voxel (cube) in OpenCV and flying the camera around it can be done by running:
```python
import numpy as np
from spacecubes import Camera
from spacecubes.io_devices import OpenCV

world = np.ones((1, 1, 1))
camera = Camera(x=-1, y=-1, z=-1)
colors = {1: (0, 255, 0)}
device = OpenCV(colors, resolution=(1080, 1920))
while True:
    camera.move(up=0.01, right=0.01)
    camera.look_at(x=0, y=0, z=0)
    device.render(world, camera)
```

Other examples with more a fleshed out description can be found in the [examples](examples) directory.

## Features
Any NumPy array with 3 dimensions can be rendered. All non-zero values in the array are considered voxels, while elements with value 0 will be treated as empty space.

### IO Devices
An IO Device in spacecubes is what (optionally) [handles user input](examples/interactive_camera.py) and definitely handles image frame output. The output can be done e.g., through visualization or raw dump. The IO Device needs to know what colors to map each value in the numpy array with, which is what the `colors` argument does. The available io_devices are specified below along with how they are used:
```python
from spacecubes.io_devices import OpenCV, Raw, Terminal
from spacecubes import Camera
import numpy as np
world = np.ones((1,1,1))
camera = Camera()

# Output the frame using OpenCV imshow
opencv_device = OpenCV(colors={i: (0, 255, 0) for i in range(1, 100)}, resolution=(1080, 1920))
opencv_device.render(world, camera)

# Returns the frame as an numpy array
raw_device = Raw(colors={i: (0, 255, 0) for i in range(1, 100)}, resolution=(1080, 1920))
frame = raw_device.render(world, camera)

# Outputs the frame directly in the terminal using ncurses
terminal_device = Terminal(colors={i: 5 for i in range(1, 100)})
terminal_device.render(world, camera)
```

To render the output on the IO device, `device.render(world, camera)` is used, where world is a 3D NumPy array and Camera is..

### Camera
Camera is the object that handles the virtual camera which specifies the perspective through which the image is rendered. It supports some functions
related to moving, rotating and looking at world locations:
```python
from spacecubes import Camera

# Initialize a camera along with some world position
camera = Camera(x=1, y=2, z=3)

# Move the camera 1 unit back from the camera's perspective
camera.move(up=0, forward=-1, right=0)

# Move the camera -1 unit along the world y-axis
camera.move_xyz(x=0, y=-1, z=0)

# Move the camera to a specified world position (0, 5, 0)
camera.move_to_xyz(x=0, y=5, z=0)

# The camera can be rotated manually through yaw, pitch and roll given in radians
camera.rotate(yaw=-3.14/2, pitch=0, roll=0)

# Make the camera look at a specified world location (3, 5, 2)
camera.look_at(x=3, y=5, z=2)

# If camera.look_at is too snappy, the same can be done but interpolated.
# This is done by supplying an amount, which is a fraction between
# 0 and 1 that specifies where in the interpolation between the current camera
# pose and the target camera pose that the camera should look
for interp_amount in range(100):
    camera.look_at_interpolated(x=3, y=5, z=2, amount=interp_amount / 100)
    device.render(world, camera)
```




## Installation
spacecubes is available on PyPI: `pip install spacecubes`

### Dependencies:
- numpy
- pyquaternion
- opencv-python (optional)

