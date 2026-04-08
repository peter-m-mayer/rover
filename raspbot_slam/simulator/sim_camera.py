"""
Simulated camera: renders synthetic views from the PyBullet world.

Drop-in replacement for raspbot_slam.camera.Camera -- implements the same
interface (capture, capture_color, undistort_points, K, etc.) but renders
from the simulated rover's viewpoint.
"""

import math
import numpy as np
import pybullet as p

from .. import config
from .sim_world import SimWorld


class SimCamera:
    """Simulated camera using PyBullet's built-in renderer.

    Produces 640x480 images from the rover's camera viewpoint, accounting
    for pan/tilt servo angles.
    """

    def __init__(self, world: SimWorld, width: int = None, height: int = None,
                 fov: float = 60.0, near: float = 0.05, far: float = 20.0):
        """Initialize simulated camera.

        Args:
            world: SimWorld instance.
            width: Image width (default: config.CAMERA_WIDTH).
            height: Image height (default: config.CAMERA_HEIGHT).
            fov: Vertical field of view in degrees.
            near: Near clip plane (meters).
            far: Far clip plane (meters).
        """
        self._world = world
        self._width = width or config.CAMERA_WIDTH
        self._height = height or config.CAMERA_HEIGHT
        self._fov = fov
        self._near = near
        self._far = far

        # Compute intrinsic matrix from FOV
        aspect = self._width / self._height
        fy = self._height / (2.0 * math.tan(math.radians(fov / 2.0)))
        fx = fy  # square pixels
        cx = self._width / 2.0
        cy = self._height / 2.0

        self._K = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]], dtype=np.float64)
        self._dist_coeffs = np.zeros(5, dtype=np.float64)

        # Camera offset from rover center (top of chassis, centered)
        self._camera_offset_z = 0.10  # 10cm above rover base
        self._camera_offset_x = 0.08  # 8cm forward (front of rover)

        # PyBullet projection matrix
        self._projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=near, farVal=far)

    def capture(self) -> np.ndarray:
        """Capture a grayscale frame from the simulated camera.

        Returns:
            HxW uint8 grayscale numpy array.
        """
        color = self.capture_color()
        # Convert BGR to grayscale
        gray = np.mean(color[:, :, :3], axis=2).astype(np.uint8)
        return gray

    def capture_color(self) -> np.ndarray:
        """Capture a color (BGR) frame from the simulated camera.

        Returns:
            HxWx3 uint8 BGR numpy array.
        """
        view_matrix = self._compute_view_matrix()

        _, _, rgba, _, _ = p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=view_matrix,
            projectionMatrix=self._projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        # PyBullet returns RGBA as flat array
        rgba = np.array(rgba, dtype=np.uint8).reshape(self._height, self._width, 4)
        # Convert RGBA to BGR (OpenCV convention)
        bgr = rgba[:, :, [2, 1, 0]]
        return bgr

    def capture_depth(self) -> np.ndarray:
        """Capture a depth image from the simulated camera.

        Returns:
            HxW float32 array with depth in meters.
        """
        view_matrix = self._compute_view_matrix()

        _, _, _, depth_buffer, _ = p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=view_matrix,
            projectionMatrix=self._projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        depth_buffer = np.array(depth_buffer, dtype=np.float32).reshape(
            self._height, self._width)

        # Convert from normalized depth buffer to meters
        # depth = far * near / (far - (far - near) * buffer)
        depth = self._far * self._near / (
            self._far - (self._far - self._near) * depth_buffer)
        return depth

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """No distortion in simulation -- just normalize by K."""
        pts_float = pts.astype(np.float64)
        result = np.zeros_like(pts_float)
        result[:, 0] = (pts_float[:, 0] - self._K[0, 2]) / self._K[0, 0]
        result[:, 1] = (pts_float[:, 1] - self._K[1, 2]) / self._K[1, 1]
        return result

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """No distortion in simulation."""
        return frame

    @property
    def K(self) -> np.ndarray:
        return self._K.copy()

    @property
    def dist_coeffs(self) -> np.ndarray:
        return self._dist_coeffs.copy()

    @property
    def focal_length_px(self) -> float:
        return (self._K[0, 0] + self._K[1, 1]) / 2.0

    @property
    def principal_point(self) -> tuple:
        return (self._K[0, 2], self._K[1, 2])

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def open(self):
        pass  # no-op for sim

    def close(self):
        pass

    def is_open(self) -> bool:
        return True

    def set_calibration(self, K, dist_coeffs):
        self._K = np.array(K, dtype=np.float64)
        self._dist_coeffs = np.array(dist_coeffs, dtype=np.float64)

    def save_calibration(self, filepath):
        import json, os
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                "camera_matrix": self._K.tolist(),
                "dist_coeffs": self._dist_coeffs.tolist(),
                "image_size": [self._width, self._height],
            }, f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    # =========================================================================
    # View Matrix Computation
    # =========================================================================

    def _compute_view_matrix(self) -> list:
        """Compute the camera view matrix based on rover pose + servo angles."""
        # Get rover pose
        pos, orn = p.getBasePositionAndOrientation(self._world.rover_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # Pan/tilt servo angles
        pan_rad = math.radians(self._world.pan_angle - config.SERVO_PAN_CENTER)
        tilt_rad = math.radians(self._world.tilt_angle - config.SERVO_TILT_REST)

        # Camera position: offset from rover center
        cam_yaw = yaw + pan_rad
        cam_x = pos[0] + self._camera_offset_x * math.cos(yaw)
        cam_y = pos[1] + self._camera_offset_x * math.sin(yaw)
        cam_z = pos[2] + self._camera_offset_z

        # Camera look direction
        look_dist = 1.0
        target_x = cam_x + look_dist * math.cos(cam_yaw) * math.cos(tilt_rad)
        target_y = cam_y + look_dist * math.sin(cam_yaw) * math.cos(tilt_rad)
        target_z = cam_z + look_dist * math.sin(tilt_rad)

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[cam_x, cam_y, cam_z],
            cameraTargetPosition=[target_x, target_y, target_z],
            cameraUpVector=[0, 0, 1],
        )
        return view_matrix
