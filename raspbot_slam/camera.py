"""
Camera interface with calibration support.

Wraps cv2.VideoCapture and provides undistortion using stored intrinsics.
On non-Pi platforms (dev/test), falls back to a video file or synthetic frames.
"""

import json
import os
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from . import config


class Camera:
    """Calibrated monocular camera for the RASPBOT-V2."""

    def __init__(self, device=None, width=None, height=None, calibration_file=None):
        """Initialize camera capture and load calibration if available.

        Args:
            device: Camera device index or video file path. Defaults to config.CAMERA_DEVICE.
            width: Frame width. Defaults to config.CAMERA_WIDTH.
            height: Frame height. Defaults to config.CAMERA_HEIGHT.
            calibration_file: Path to camera_calibration.json. Defaults to config path.
        """
        self._device = device if device is not None else config.CAMERA_DEVICE
        self._width = width or config.CAMERA_WIDTH
        self._height = height or config.CAMERA_HEIGHT
        self._cap = None

        # Calibration data (loaded from file or set manually)
        self._K = None              # 3x3 intrinsic matrix
        self._dist_coeffs = None    # distortion coefficients
        self._new_K = None          # optimal new camera matrix for undistortion

        cal_file = calibration_file or config.CAMERA_CALIBRATION_FILE
        if os.path.exists(cal_file):
            self._load_calibration(cal_file)
        else:
            # Approximate intrinsics for a typical USB camera at 640x480
            fx = fy = 500.0
            cx, cy = self._width / 2.0, self._height / 2.0
            self._K = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0,  0,  1]], dtype=np.float64)
            self._dist_coeffs = np.zeros(5, dtype=np.float64)

    def open(self):
        """Open the camera device."""
        if cv2 is None:
            raise RuntimeError("OpenCV not available")
        self._cap = cv2.VideoCapture(self._device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        # Keep the driver queue at 1 frame: a control loop slower than the
        # camera FPS otherwise reads stale buffered frames (hundreds of ms
        # of hidden latency). Best-effort; not all backends honor it.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {self._device}")

    def close(self):
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def capture(self) -> np.ndarray:
        """Capture a single grayscale frame.

        Returns:
            Grayscale image as HxW uint8 numpy array.

        Raises:
            RuntimeError: If capture fails.
        """
        if not self.is_open():
            self.open()
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Camera capture failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def capture_color(self) -> np.ndarray:
        """Capture a single color (BGR) frame."""
        if not self.is_open():
            self.open()
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Camera capture failed")
        return frame

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """Undistort 2D keypoint coordinates using calibration.

        Args:
            pts: Nx2 array of (x, y) pixel coordinates.

        Returns:
            Nx2 array of undistorted coordinates in normalized camera frame.
        """
        if cv2 is None:
            return pts
        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float64)
        undistorted = cv2.undistortPoints(pts_reshaped, self._K, self._dist_coeffs)
        return undistorted.reshape(-1, 2)

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Undistort an entire frame. More expensive than undistort_points."""
        if cv2 is None or self._dist_coeffs is None:
            return frame
        if self._new_K is None:
            h, w = frame.shape[:2]
            self._new_K, _ = cv2.getOptimalNewCameraMatrix(
                self._K, self._dist_coeffs, (w, h), 1, (w, h))
        return cv2.undistort(frame, self._K, self._dist_coeffs, None, self._new_K)

    @property
    def K(self) -> np.ndarray:
        """3x3 camera intrinsic matrix."""
        return self._K.copy()

    @property
    def dist_coeffs(self) -> np.ndarray:
        """Distortion coefficients [k1, k2, p1, p2, k3]."""
        return self._dist_coeffs.copy()

    @property
    def focal_length_px(self) -> float:
        """Average focal length in pixels: (fx + fy) / 2."""
        return (self._K[0, 0] + self._K[1, 1]) / 2.0

    @property
    def principal_point(self) -> tuple:
        """Principal point (cx, cy) in pixels."""
        return (self._K[0, 2], self._K[1, 2])

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _load_calibration(self, filepath: str):
        """Load camera calibration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self._K = np.array(data["camera_matrix"], dtype=np.float64)
        self._dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)

    def save_calibration(self, filepath: str):
        """Save current calibration to JSON file."""
        data = {
            "camera_matrix": self._K.tolist(),
            "dist_coeffs": self._dist_coeffs.tolist(),
            "image_size": [self._width, self._height],
        }
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def set_calibration(self, K: np.ndarray, dist_coeffs: np.ndarray):
        """Manually set calibration parameters."""
        self._K = np.array(K, dtype=np.float64)
        self._dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
        self._new_K = None  # reset cached optimal matrix

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
