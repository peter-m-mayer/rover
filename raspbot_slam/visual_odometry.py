"""
Monocular visual odometry using ORB features and Essential matrix decomposition.

Estimates frame-to-frame relative pose (rotation + translation direction).
Translation magnitude (scale) comes from synthetic stereo -- this module
provides direction only, scaled by the last known scale factor.

Manages a keyframe buffer for matching stability and loop closure support.
"""

import time
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from . import config
from .camera import Camera
from .feature_extractor import FeatureExtractor


@dataclass
class Keyframe:
    """A stored frame with features and estimated pose."""
    id: int
    timestamp: float
    frame: np.ndarray                  # grayscale HxW
    keypoints_xy: np.ndarray           # Nx2 pixel coordinates
    descriptors: np.ndarray            # Nx32 ORB descriptors
    pose: np.ndarray = field(          # 4x4 homogeneous transform (world frame)
        default_factory=lambda: np.eye(4, dtype=np.float64))


class VisualOdometry:
    """Monocular VO: frame-to-frame pose estimation from ORB features.

    Usage:
        vo = VisualOdometry(camera, feature_extractor)
        while running:
            frame = camera.capture()
            delta = vo.process_frame(frame)
            if delta is not None:
                dx, dy, dtheta = delta
                # feed to EKF
    """

    def __init__(self, camera: Camera, feature_extractor: FeatureExtractor):
        self._camera = camera
        self._fe = feature_extractor
        self._K = camera.K

        # Previous frame state
        self._prev_kp = None
        self._prev_desc = None
        self._prev_frame = None

        # Keyframe management
        self._keyframes: List[Keyframe] = []
        self._keyframe_counter = 0
        self._last_keyframe_id = -1

        # Cumulative pose (4x4 homogeneous, world frame)
        self._pose = np.eye(4, dtype=np.float64)

        # Scale factor (updated by synthetic stereo module).
        # Default is very small -- the unit vector from recoverPose has
        # magnitude 1.0, which is meaningless. Synthetic stereo will
        # calibrate this to real meters. Until then, keep it tiny so
        # the EKF doesn't diverge from uncalibrated VO.
        self._scale = 0.001

        # Tracking state
        self._frames_since_keyframe = 0
        self._cumulative_translation = 0.0
        self._cumulative_rotation = 0.0
        self._tracking = False
        self._last_n_inliers = 0

    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Process a new grayscale frame and estimate relative motion.

        Args:
            frame: HxW uint8 grayscale image.

        Returns:
            (dx, dy, dtheta) relative motion in robot frame (meters, meters, radians),
            or None if tracking is lost.
            dx = forward, dy = leftward, dtheta = CCW rotation.
        """
        # Detect features
        kp, desc = self._fe.detect_and_compute(frame)

        if desc is None or len(kp) < config.MIN_INLIER_MATCHES:
            self._tracking = False
            self._prev_kp = kp
            self._prev_desc = desc
            self._prev_frame = frame
            return None

        # First frame: just store, no motion estimate
        if self._prev_desc is None:
            self._prev_kp = kp
            self._prev_desc = desc
            self._prev_frame = frame
            self._tracking = True
            # Create initial keyframe
            self._create_keyframe(frame, kp, desc)
            return (0.0, 0.0, 0.0)

        # Match against previous frame with geometric verification
        inlier_matches, mask, E = self._fe.match_with_geometric_check(
            self._prev_kp, self._prev_desc,
            kp, desc,
            self._K,
        )

        if E is None or len(inlier_matches) < config.MIN_INLIER_MATCHES:
            self._tracking = False
            self._prev_kp = kp
            self._prev_desc = desc
            self._prev_frame = frame
            return None

        # Recover rotation and translation direction from Essential matrix
        pts1, pts2 = self._fe.matches_to_points(self._prev_kp, kp, inlier_matches)
        n_inliers, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, self._K)

        self._last_n_inliers = n_inliers
        self._tracking = True

        # --- Minimum parallax gate ---
        # Check if enough features moved significantly. During pure forward
        # motion, features near the focus of expansion (image center) have
        # near-zero displacement while edge features move a lot. Use the
        # 75th percentile to avoid the stationary center features dominating.
        pixel_displacements = np.sqrt(np.sum((pts2 - pts1)**2, axis=1))
        p75_displacement = float(np.percentile(pixel_displacements, 75))

        if p75_displacement < 1.5:
            # Scene looks the same -- no significant motion
            self._prev_kp = kp
            self._prev_desc = desc
            self._prev_frame = frame
            return (0.0, 0.0, 0.0)

        # t is a unit vector (3x1) -- direction only.
        # Scale by the current scale factor. The default scale (0.001) is
        # intentionally small -- synthetic stereo will calibrate it to
        # real-world meters. Until then, we accumulate small relative units.
        t_scaled = t.ravel() * self._scale

        # Extract yaw rotation
        dtheta = self._rotation_to_yaw(R)

        # Convert to 2D robot-frame motion.
        # OpenCV recoverPose returns t as "translation of camera 2 relative
        # to camera 1", i.e., where did the camera move TO. When the robot
        # moves forward, the scene moves backward in camera frame, so
        # recoverPose returns negative Z. We negate to get robot motion.
        # Camera convention: X=right, Y=down, Z=forward.
        # Robot convention: dx=forward, dy=left, dtheta=CCW.
        dx = float(-t_scaled[2])     # negate: camera -Z → robot forward
        dy = float(t_scaled[0])      # camera X → robot right → negate for left

        # Update cumulative pose
        delta_T = np.eye(4, dtype=np.float64)
        delta_T[:3, :3] = R
        delta_T[:3, 3] = t_scaled
        self._pose = self._pose @ delta_T

        # Track cumulative displacement since last keyframe
        self._cumulative_translation += np.sqrt(dx**2 + dy**2)
        self._cumulative_rotation += abs(dtheta)
        self._frames_since_keyframe += 1

        # Update previous frame
        self._prev_kp = kp
        self._prev_desc = desc
        self._prev_frame = frame

        return (dx, dy, dtheta)

    def is_keyframe_needed(self) -> bool:
        """Check if a new keyframe should be created based on motion thresholds."""
        if self._cumulative_translation > config.KEYFRAME_TRANSLATION_M:
            return True
        if np.degrees(self._cumulative_rotation) > config.KEYFRAME_ROTATION_DEG:
            return True
        if self._frames_since_keyframe >= config.KEYFRAME_MAX_INTERVAL:
            return True
        # Check feature overlap with last keyframe
        if self._keyframes and self._prev_desc is not None:
            last_kf = self._keyframes[-1]
            matches = self._fe.match(last_kf.descriptors, self._prev_desc)
            overlap = len(matches) / max(len(self._prev_desc), 1)
            if overlap < config.KEYFRAME_OVERLAP_THRESHOLD:
                return True
        return False

    def create_keyframe(self) -> Optional[Keyframe]:
        """Promote current frame to keyframe if we have valid data.

        Returns:
            The new Keyframe, or None if no valid frame data.
        """
        if self._prev_frame is None or self._prev_desc is None:
            return None
        return self._create_keyframe(
            self._prev_frame, self._prev_kp, self._prev_desc)

    def _create_keyframe(self, frame, kp, desc) -> Keyframe:
        """Internal keyframe creation."""
        kf = Keyframe(
            id=self._keyframe_counter,
            timestamp=time.time(),
            frame=frame.copy(),
            keypoints_xy=FeatureExtractor.keypoints_to_array(kp),
            descriptors=desc.copy(),
            pose=self._pose.copy(),
        )
        self._keyframes.append(kf)
        self._keyframe_counter += 1
        self._last_keyframe_id = kf.id

        # Trim buffer to max size
        if len(self._keyframes) > config.KEYFRAME_BUFFER_SIZE:
            self._keyframes.pop(0)

        # Reset counters
        self._cumulative_translation = 0.0
        self._cumulative_rotation = 0.0
        self._frames_since_keyframe = 0

        return kf

    @property
    def is_tracking(self) -> bool:
        """Whether VO currently has enough feature matches."""
        return self._tracking

    @property
    def last_n_inliers(self) -> int:
        """Number of inlier matches in the last frame."""
        return self._last_n_inliers

    @property
    def pose(self) -> np.ndarray:
        """Current 4x4 cumulative pose in world frame."""
        return self._pose.copy()

    @property
    def position_2d(self) -> Tuple[float, float]:
        """Current (x, y) position projected to ground plane."""
        return (float(self._pose[0, 3]), float(self._pose[2, 3]))

    @property
    def heading_rad(self) -> float:
        """Current heading angle in radians (yaw, CCW from forward)."""
        return self._rotation_to_yaw(self._pose[:3, :3])

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float):
        """Set scale factor (called by SyntheticStereo after calibration)."""
        self._scale = max(0.01, value)

    def get_keyframe_buffer(self) -> List[Keyframe]:
        """Return current keyframe buffer (most recent KEYFRAME_BUFFER_SIZE)."""
        return list(self._keyframes)

    def get_all_keyframe_count(self) -> int:
        """Total keyframes created (including those evicted from buffer)."""
        return self._keyframe_counter

    def reset(self):
        """Reset VO state (for reinitialization after relocalization)."""
        self._prev_kp = None
        self._prev_desc = None
        self._prev_frame = None
        self._keyframes.clear()
        self._pose = np.eye(4, dtype=np.float64)
        self._tracking = False
        self._cumulative_translation = 0.0
        self._cumulative_rotation = 0.0
        self._frames_since_keyframe = 0

    def set_pose(self, x: float, y: float, theta: float):
        """Set pose directly (after relocalization)."""
        self._pose = np.eye(4, dtype=np.float64)
        self._pose[0, 3] = x
        self._pose[2, 3] = y
        c, s = np.cos(theta), np.sin(theta)
        self._pose[0, 0] = c
        self._pose[0, 2] = s
        self._pose[2, 0] = -s
        self._pose[2, 2] = c

    @staticmethod
    def _rotation_to_yaw(R: np.ndarray) -> float:
        """Extract yaw angle (rotation about Y axis) from a 3x3 rotation matrix.
        Returns angle in radians. Assumes ground-plane motion."""
        return float(np.arctan2(R[0, 2], R[0, 0]))
