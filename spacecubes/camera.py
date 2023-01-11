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
        roll (float): Initial camera roll
        pitch (float): Initial camera pitch
        yaw (float): Initial camera yaw
    """

    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        self.x = x
        self.y = y
        self.z = z

        # Create a quaternion from the initial roll, pitch and yaw
        # self.Q = self.quat_from_ypr(yaw=yaw, pitch=pitch, roll=roll)
        self.Q = Quaternion(1, 0, 0, 0)

        # Generate the camera matrices
        self._regenerate_extrinsic_matrix()
        self._regenerate_intrinsic_matrix()

    def _quaternion_from_yaw_pitch_roll(self, yaw=0, pitch=0, roll=0):
        """Create a quaternion from yaw, pitch and roll given in radians"""
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        return Quaternion(w=qw, x=qx, y=qy, z=qz)

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
                world_array (np.array): A numpy array of shape (3, N)
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

    def look_at(self, x, y, z):
        """Rotates the camera to look at a world position specified as (x, y, z)

        Find a quaternion that has the axis specified by the target and camera position
        """
        src = np.array([self.x, self.y, self.z], dtype=np.float64)
        dst = np.array([x, y, z], dtype=np.float64)

        fwd_v = (dst - src) / math.sqrt(np.sum((src - dst) ** 2))
        fwd = np.array([0, 0, 1])
        d = np.dot(fwd, fwd_v)
        if abs(d) > 1:
            # Maybe warn
            rot = 0
        else:
            rot = math.acos(d)
        rot_axis = np.cross(fwd, fwd_v)
        rot_axis = rot_axis / math.sqrt(np.sum(rot_axis**2))
        s = math.sin(rot * 0.5)
        self.Q = Quaternion(
            w=math.cos(rot * 0.5),
            x=rot_axis[0] * s,
            y=rot_axis[1] * s,
            z=rot_axis[2] * s,
        )
        self._regenerate_extrinsic_matrix()

    def move_xyz(self, x=0, y=0, z=0):
        """Translates the camera in world coordinates"""
        self.x += x
        self.y += y
        self.z += z
        self._regenerate_extrinsic_matrix()

    def move(self, forward=0, right=0, up=0):
        """Translates the camera in camera coordinates"""
        # Calculate the i-component in the world frame
        world_forward = [0, 0, 1]
        dxf, dyf, dzf = (
            self.Q.unit * np.array([0, *world_forward]) * self.Q.unit.conjugate.unit
        ).axis * forward

        # Calculate the j-component in the world frame
        world_side = [0, 1, 0]
        dxs, dys, dzs = (
            self.Q.unit * np.array([0, *world_side]) * self.Q.unit.conjugate.unit
        ).axis * right

        # Calculate the k-component in the world frame
        world_up = [-1, 0, 0]
        dxu, dyu, dzu = (
            self.Q.unit * np.array([0, *world_up]) * self.Q.unit.conjugate.unit
        ).axis * up

        dx = dxf + dxs + dxu
        dy = dyf + dys + dyu
        dz = dzf + dzs + dzu

        self.x += dx
        self.y += dy
        self.z += dz
        self._regenerate_extrinsic_matrix()

    def rotate(self, roll=0, pitch=0, yaw=0):
        """Rotates the camera the given amount of roll, pitch and yaw"""
        self.Q *= self._quaternion_from_yaw_pitch_roll(roll=roll, pitch=pitch, yaw=yaw)
        self._regenerate_extrinsic_matrix()
