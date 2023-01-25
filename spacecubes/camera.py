import math
from pyquaternion import Quaternion
import numpy as np


class Camera:
    """The Camera class holds the extrinsic matrix, which specifies
    how the camera is rotated and translated with regards to the
    world, along with the intrinsic matrix, which specifies the
    focal length and internal translation of the camera. More
    information on how these matrices work can be found at
    e.g., https://en.wikipedia.org/wiki/Camera_resectioning

    The class also holds helper functions for common operations
    such as moving in the X, Y and Z directions as well as
    rotating the camera.

    Args:
        x (float): Initial world x position of camera
        y (float): Initial world y position of camera
        z (float): Initial world z position of camera
    """

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

        # Create a quaternion from the initial roll, pitch and yaw
        self.Q = Quaternion(1, 0, 0, 0)

        # Generate the camera matrices
        self._regenerate_extrinsic_matrix()
        self._regenerate_intrinsic_matrix()

    def _quaternion_from_yaw_pitch_roll(self, yaw=0, pitch=0, roll=0):
        """Create a quaternion from yaw, pitch and roll given in radians

        Args:
            yaw (float): Yaw in radians
            pitch (float): Pitch in radians
            roll (float): Roll in radians

        Adapted from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        """
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return Quaternion(w=w, x=-y, y=z, z=-x)

    def _regenerate_extrinsic_matrix(self):
        """Recalculates and sets the camera's extrinsic matrix"""
        T = self._generate_translation_matrix(self.x, self.y, self.z)
        R = np.identity(4)
        R[:3, :3] = self.Q.rotation_matrix
        self.extrinsic_matrix = self._generate_extrinsic_matrix(T, R)

    def _regenerate_intrinsic_matrix(
        self, focal_length=1, skew_factor=0, offset_x=0.5, offset_y=0.5, aspect_ratio=1
    ):
        """Set the camera's intrinsic matrix"""
        self.intrinsic_matrix = self._generate_intrinsic_matrix(
            focal_length,
            skew_factor=skew_factor,
            offset_x=offset_x,
            offset_y=offset_y,
            aspect_ratio=aspect_ratio,
        )

    def _generate_extrinsic_matrix(self, T, R):
        """Calculates the essential (E) matrix"""
        return np.linalg.inv(T @ R)

    def _generate_intrinsic_matrix(
        self, focal_length, skew_factor, offset_x, offset_y, aspect_ratio
    ):
        """Create an intrinsic matrix"""
        return np.array(
            [
                [focal_length, skew_factor, offset_x],
                [0, aspect_ratio * focal_length, offset_y],
                [0, 0, 1],
            ]
        )

    def _generate_translation_matrix(self, offset_x, offset_y, offset_z):
        """Generates the translation (T) matrix from the camera position"""
        T = np.identity(4)
        T[:3, 3] = [offset_x, offset_y, offset_z]
        return T

    def _world_to_camera_coordinates(self, world_points):
        """Changes the coordinate system of points from world coordinate
        system to camera coordinate system.

            Args:
                world_points (np.array): A numpy array of shape (3, N)
                    holding the points in the world coordinate system
                    that are to be converted to the camera coordinate system.
        """
        # Convert to homogenous coordinates
        world_points = np.vstack((world_points, np.ones(world_points.shape[1])))

        # Project world points to camera coordinate system
        camera_points = self.extrinsic_matrix @ world_points
        return camera_points[:-1, :]

    @property
    def position(self):
        return np.array([self.x, self.y, self.z])

    @property
    def rotation(self):
        return self.Q

    def look_at_interpolated(self, x, y, z, amount):
        """Interpolated version of look_at. By specifying the amount,
        the camera can either be pointed directly
        at the (x, y, z) location by using amount=1, or in the current
        direction by setting amount=0, or somewhere in between by
        specifying amount to be in the [0, 1] range.

        Can for example be used in a for loop to create smooth camera rotation.

        Note that as this method always uses the current camera
        quaternion, the interpolation will not be linear if called repeatedly
        with equally-sized amount-steps, but rather slow down as the
        camera approaches looking at the target position.

        Args:
            x (float): The x-coordinate of the world position to look at
            y (float): The y-coordinate of the world position to look at
            z (float): The z-coordinate of the world position to look at
            amount (float): The interpolation amount between the current
                camera pose and the target camera pose
        """
        look_at_quaternion = self._create_look_at_quaternion(x, y, z)
        self.Q = Quaternion.slerp(self.Q, look_at_quaternion, amount)
        self._regenerate_extrinsic_matrix()

    def look_at(self, x, y, z):
        """Sets the camera to look at a world position given as x, y, z.

        Args:
            x (float): The x-coordinate of the world position to look at
            y (float): The y-coordinate of the world position to look at
            z (float): The z-coordinate of the world position to look at
        """
        # Create a quatenion
        self.Q = self._create_look_at_quaternion(x, y, z)
        self._regenerate_extrinsic_matrix()

    def _create_look_at_quaternion(self, x, y, z):
        """Creates a quaternion that looks at the target world position"""
        # Finds a quaternion that has its axis pointed at the target position
        src = np.array([self.x, self.y, self.z], dtype=np.float64)
        dst = np.array([x, y, z], dtype=np.float64)

        camera_new_fwd = dst - src
        camera_new_fwd = camera_new_fwd / math.sqrt(np.sum(camera_new_fwd**2))
        world_fwd = np.array([0, 0, 1])
        d = np.dot(world_fwd, camera_new_fwd)
        if abs(d) > 1:
            # Maybe warn
            rot = 0
        else:
            rot = math.acos(d)
        rot_axis = np.cross(world_fwd, camera_new_fwd)
        if not np.any(rot_axis):
            # Rot axis is zero
            rot_axis = [0, 0, 0]
        else:
            rot_axis = rot_axis / math.sqrt(np.sum(rot_axis**2))
        s = math.sin(rot * 0.5)
        return Quaternion(
            w=math.cos(rot * 0.5),
            x=rot_axis[0] * s,
            y=rot_axis[1] * s,
            z=rot_axis[2] * s,
        )

    def move_xyz(self, x=0, y=0, z=0):
        """Translates the camera in world coordinates

        Args:
            x (float): The movement along the world x axis
            y (float): The movement along the world y axis
            z (float): The movement along the world z axis
        """
        self.x += x
        self.y += y
        self.z += z
        self._regenerate_extrinsic_matrix()

    def move_to_xyz(self, x, y, z):
        """Set the camera to a position given in world coordinates

        Args:
            x (float): The x-coordinate of the new position
            y (float): The y-coordinate of the new position
            z (float): The z-coordinate of the new position
        """
        self.x = x
        self.y = y
        self.z = z
        self._regenerate_extrinsic_matrix()

    def move(self, forward=0, right=0, up=0):
        """Translates the camera in camera coordinates. Values can be
            negative in order to allow moving back, left and down.

        Uses p' = QpQ^* to find the world position move corresponding
        to a move in the camera's frame

        Args:
            forward (float): The movement along the camera forward axis
            right (float): The movement along the camera side axis
            up (float): The movement along the camera up axis

        """
        # Calculate the i-component in the world frame
        world_forward = [0, 0, 1]
        dxf, dyf, dzf = (
            self.Q.unit * np.array([0, *world_forward]) * self.Q.unit.conjugate.unit
        ).axis * forward

        # Calculate the j-component in the world frame
        world_right = [0, 1, 0]
        dxr, dyr, dzr = (
            self.Q.unit * np.array([0, *world_right]) * self.Q.unit.conjugate.unit
        ).axis * right

        # Calculate the k-component in the world frame
        world_up = [-1, 0, 0]
        dxu, dyu, dzu = (
            self.Q.unit * np.array([0, *world_up]) * self.Q.unit.conjugate.unit
        ).axis * up

        # Add all of the components together to get the final coordinate change
        # in each world axis
        dx = dxf + dxr + dxu
        dy = dyf + dyr + dyu
        dz = dzf + dzr + dzu

        self.x += dx
        self.y += dy
        self.z += dz
        self._regenerate_extrinsic_matrix()

    def rotate(self, yaw=0, pitch=0, roll=0):
        """Rotates the camera the given amount of roll, pitch and yaw

        Args:
            yaw (float): Angle to yaw in radians
            pitch (float): Angle to pitch in radians
            roll (float): Angle to roll in radians
        """
        self.Q *= self._quaternion_from_yaw_pitch_roll(yaw=yaw, pitch=pitch, roll=roll)
        self._regenerate_extrinsic_matrix()
