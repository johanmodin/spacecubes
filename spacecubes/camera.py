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
        #self.Q = self.quat_from_ypr(yaw=yaw, pitch=pitch, roll=roll)
        self.Q = Quaternion(1, 0, 0, 0)

        # Generate the camera matrices
        self.regenerate_extrinsic_matrix()
        self.regenerate_intrinsic_matrix()

    def quat_from_ypr(self, yaw=0, pitch=0, roll=0):
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

    def regenerate_extrinsic_matrix(self):
        """Recalculates and sets the camera's extrinsic matrix"""
        T = self.generate_translation_matrix(self.x, self.y, self.z)
        R = np.identity(4)
        R[:3, :3] = self.Q.rotation_matrix
        self.extrinsic_matrix = self.generate_extrinsic_matrix(T, R)

    def regenerate_intrinsic_matrix(self):
        """Set the camera's intrinsic matrix"""
        self.intrinsic_matrix = self.generate_intrinsic_matrix()

    def generate_extrinsic_matrix(self, T, R):
        """Calculates the essential (E) matrix"""
        return np.linalg.inv(T @ R)

    def generate_intrinsic_matrix(
        self, focal_length=1, skew_factor=0, offset_x=0.5, offset_y=0.5, aspect_ratio=1
    ):
        """Create an intrinsic matrix"""
        return np.array(
            [
                [focal_length, skew_factor, offset_x],
                [0, aspect_ratio * focal_length, offset_y],
                [0, 0, 1],
            ]
        )

    def generate_translation_matrix(self, offset_x, offset_y, offset_z):
        """Generates the translation (T) matrix from the camera position"""
        T = np.identity(4)
        T[:3, 3] = [offset_x, offset_y, offset_z]
        return T

    def world_to_camera_coordinates(self, world_points):
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

    def look_at(self, x, y, z):
        """Rotates the camera to look at a world position specified as (x, y, z)"""
        # TODO: Handle this cleaner
        def normalize(v, tolerance=0.00001):
            mag2 = sum(n * n for n in v)
            if mag2 != 0 and abs(mag2 - 1.0) > tolerance:
                mag = math.sqrt(mag2)
                v = tuple(n / mag for n in v)
            return v

        src = np.array([self.x, self.y, self.z], dtype=np.float64)
        dst = np.array([x, y, z], dtype=np.float64)

        fwd_v = normalize(dst - src)
        fwd = np.array([0, 0, 1])
        d = np.dot(fwd, fwd_v)
        if abs(d) > 1:
            # Maybe warn
            rot = 0
        else:
            rot = math.acos(d)
        rot_axis = np.cross(fwd, fwd_v)
        rot_axis = normalize(rot_axis)

        half_rot = rot * 0.5
        s = math.sin(half_rot)
        self.Q = Quaternion(
            w=math.cos(half_rot),
            x=rot_axis[0] * s,
            y=rot_axis[1] * s,
            z=rot_axis[2] * s,
        )
        self.regenerate_extrinsic_matrix()

    def move(self, x=0, y=0, z=0):
        """Translates the camera in world coordinates"""
        self.x += x
        self.y += y
        self.z += z
        self.regenerate_extrinsic_matrix()

    def rotate(self, roll=0, pitch=0, yaw=0):
        """Rotates the camera the given amount of roll, pitch and yaw"""
        self.Q *= self.quat_from_ypr(roll=roll, pitch=pitch, yaw=yaw)
        self.regenerate_extrinsic_matrix()
