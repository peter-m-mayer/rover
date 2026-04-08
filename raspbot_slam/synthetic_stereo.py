"""
Depth estimation via synthetic stereo using mecanum lateral strafe.

The rover stops, captures a frame, strafes right by a calibrated distance D,
captures a second frame, then triangulates matched features. This resolves
the scale ambiguity inherent in monocular visual odometry.

The mecanum wheels strafe purely laterally (no rotation), creating clean
horizontal epipolar geometry -- disparity is purely in the x-coordinate.
"""

import json
import os
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from . import config
from .camera import Camera
from .feature_extractor import FeatureExtractor
from .actuators import Actuators
from .sensors import Sensors


@dataclass
class DepthObservation:
    """A single triangulated 3D point from synthetic stereo."""
    pixel_xy: np.ndarray       # 2D position in left image (x, y)
    position_3d: np.ndarray    # 3D position in camera frame (X, Y, Z)
    depth: float               # Z distance in meters
    disparity: float           # pixel disparity (left_x - right_x)
    descriptor: np.ndarray     # ORB descriptor (32 bytes)
    confidence: float          # 0-1, inversely proportional to depth


class SyntheticStereo:
    """Depth estimation by strafing the mecanum rover laterally.

    Usage:
        stereo = SyntheticStereo(camera, feature_extractor, actuators, sensors)
        observations = stereo.capture_and_triangulate()
        # observations is a list of DepthObservation with 3D positions
    """

    def __init__(self, camera: Camera, feature_extractor: FeatureExtractor,
                 actuators: Actuators, sensors: Sensors,
                 baseline_m: float = None):
        """Initialize synthetic stereo.

        Args:
            camera: Calibrated camera.
            feature_extractor: ORB feature extractor.
            actuators: Motor/servo controller.
            sensors: Sensor interface (for ultrasonic cross-validation).
            baseline_m: Known strafe baseline in meters. If None, uses config.
        """
        self._camera = camera
        self._fe = feature_extractor
        self._actuators = actuators
        self._sensors = sensors
        self._baseline_m = baseline_m or (config.STRAFE_DISTANCE_MM / 1000.0)
        self._focal_length = camera.focal_length_px

        # Scale correction factor (updated by cross-validation with ultrasonic)
        self._scale_correction = 1.0

        # Load strafe calibration if available
        self._strafe_speed = config.STRAFE_SPEED
        self._strafe_duration_s = config.STRAFE_DURATION_MS / 1000.0
        self._load_strafe_calibration()

    def capture_and_triangulate(self) -> List[DepthObservation]:
        """Perform full synthetic stereo capture and triangulation.

        Protocol:
        1. Stop motors, settle
        2. Capture left frame + ORB
        3. Strafe right by calibrated distance
        4. Settle, capture right frame + ORB
        5. Match and triangulate
        6. Strafe left to return to original position

        Returns:
            List of DepthObservation with 3D positions in camera frame.
        """
        # 1. Stop and settle
        self._actuators.stop()
        time.sleep(config.SETTLE_TIME_MS / 1000.0)

        # 2. Capture left frame
        left_frame = self._camera.capture()
        left_kp, left_desc = self._fe.detect_and_compute(left_frame)

        if left_desc is None or len(left_kp) < config.MIN_INLIER_MATCHES:
            return []

        # 3. Strafe right
        self._actuators.strafe_right_timed(self._strafe_speed, self._strafe_duration_s)
        time.sleep(config.SETTLE_TIME_MS / 1000.0)

        # 4. Capture right frame
        right_frame = self._camera.capture()
        right_kp, right_desc = self._fe.detect_and_compute(right_frame)

        # 5. Triangulate
        observations = []
        if right_desc is not None and len(right_kp) >= config.MIN_INLIER_MATCHES:
            observations = self._triangulate(
                left_kp, left_desc, right_kp, right_desc)

        # 6. Return to original position
        self._actuators.strafe_left_timed(self._strafe_speed, self._strafe_duration_s)
        time.sleep(config.SETTLE_TIME_MS / 1000.0)

        return observations

    def capture_stereo_pair(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Capture a stereo pair without triangulation.

        Returns:
            (left_frame, right_frame, baseline_meters)
        """
        self._actuators.stop()
        time.sleep(config.SETTLE_TIME_MS / 1000.0)

        left_frame = self._camera.capture()

        self._actuators.strafe_right_timed(self._strafe_speed, self._strafe_duration_s)
        time.sleep(config.SETTLE_TIME_MS / 1000.0)

        right_frame = self._camera.capture()

        self._actuators.strafe_left_timed(self._strafe_speed, self._strafe_duration_s)
        time.sleep(config.SETTLE_TIME_MS / 1000.0)

        return left_frame, right_frame, self._baseline_m

    def _triangulate(self, left_kp, left_desc, right_kp, right_desc
                     ) -> List[DepthObservation]:
        """Match features between stereo pair and compute depths.

        For pure lateral strafe, epipolar lines are horizontal:
        disparity = left_x - right_x  (should be positive for objects in front)
        depth = focal_length * baseline / disparity
        """
        # Match with ratio test (no Essential matrix needed -- geometry is known)
        matches = self._fe.match(left_desc, right_desc)
        if not matches:
            return []

        f = self._focal_length
        D = self._baseline_m * self._scale_correction
        observations = []

        for m in matches:
            left_pt = np.array(left_kp[m.queryIdx].pt)
            right_pt = np.array(right_kp[m.trainIdx].pt)

            # Disparity (horizontal shift)
            disparity = left_pt[0] - right_pt[0]

            # Sanity checks:
            # - disparity must be positive (object in front)
            # - disparity must exceed minimum (avoids huge depth errors)
            # - vertical displacement should be small (epipolar constraint)
            vertical_shift = abs(left_pt[1] - right_pt[1])
            if disparity < config.MIN_DISPARITY_PX:
                continue
            if vertical_shift > 10.0:
                # Large vertical shift means rotation occurred during strafe
                continue

            # Depth from stereo formula
            depth = (f * D) / disparity

            # 3D position in camera frame
            # Camera convention: X=right, Y=down, Z=forward
            cx, cy = self._camera.principal_point
            X = (left_pt[0] - cx) * depth / f
            Y = (left_pt[1] - cy) * depth / f
            Z = depth

            # Confidence: higher for closer objects (larger disparity)
            confidence = min(1.0, disparity / 20.0)

            observations.append(DepthObservation(
                pixel_xy=left_pt,
                position_3d=np.array([X, Y, Z]),
                depth=depth,
                disparity=disparity,
                descriptor=left_desc[m.queryIdx].copy(),
                confidence=confidence,
            ))

        return observations

    def cross_validate_with_ultrasonic(self, observations: List[DepthObservation],
                                       ultrasonic_mm: int) -> Optional[float]:
        """Compare triangulated depths to ultrasonic reading for scale correction.

        Looks for observations near the image center (roughly aligned with
        the ultrasonic beam direction) and compares their depth to the
        ultrasonic measurement.

        Args:
            observations: Depth observations from triangulation.
            ultrasonic_mm: Ultrasonic reading in millimeters.

        Returns:
            Updated scale correction factor, or None if no valid comparison.
        """
        if ultrasonic_mm <= 0 or not observations:
            return None

        us_depth_m = ultrasonic_mm / 1000.0
        cx, cy = self._camera.principal_point
        center_tolerance = 50  # pixels from image center

        # Find observations near image center (aligned with ultrasonic beam)
        center_obs = [
            obs for obs in observations
            if abs(obs.pixel_xy[0] - cx) < center_tolerance
            and abs(obs.pixel_xy[1] - cy) < center_tolerance
            and obs.confidence > 0.3
        ]

        if not center_obs:
            return None

        # Median triangulated depth of center observations
        median_depth = np.median([obs.depth for obs in center_obs])

        if median_depth <= 0 or us_depth_m <= 0:
            return None

        # Scale correction: if triangulated depth disagrees with ultrasonic,
        # adjust the baseline calibration
        correction = us_depth_m / median_depth
        # Smooth update (don't jump too aggressively)
        self._scale_correction = 0.8 * self._scale_correction + 0.2 * correction

        return self._scale_correction

    def estimate_vo_scale(self, observations: List[DepthObservation]) -> Optional[float]:
        """Estimate the VO scale factor from stereo depth observations.

        The median depth of well-observed features, combined with the
        known baseline, gives an absolute scale reference.

        Returns:
            Estimated scale (meters per VO unit), or None if insufficient data.
        """
        reliable = [obs for obs in observations if obs.confidence > 0.3]
        if len(reliable) < 5:
            return None
        # The median depth itself is the scale anchor. The VO scale is set
        # so that VO displacement units match real-world meters.
        # This is applied externally by the caller.
        return float(np.median([obs.depth for obs in reliable]))

    @property
    def baseline_m(self) -> float:
        """Current effective baseline in meters (raw * scale_correction)."""
        return self._baseline_m * self._scale_correction

    @property
    def scale_correction(self) -> float:
        return self._scale_correction

    def _load_strafe_calibration(self):
        """Load strafe calibration LUT if available."""
        cal_file = config.STRAFE_CALIBRATION_FILE
        if not os.path.exists(cal_file):
            return
        try:
            with open(cal_file, 'r') as f:
                data = json.load(f)
            for entry in data.get("calibrations", []):
                if entry["speed"] == self._strafe_speed:
                    self._baseline_m = entry["distance_mm"] / 1000.0
                    self._strafe_duration_s = entry["duration_ms"] / 1000.0
                    break
        except (json.JSONDecodeError, KeyError):
            pass
