"""
Goal-directed navigation on a known map.

Handles relocalization (finding the robot's pose on a stored map),
path planning, waypoint following, and dynamic map adaptation when
the environment has changed.
"""

import math
import time
from typing import Tuple, Optional, List
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from . import config
from .camera import Camera
from .feature_extractor import FeatureExtractor
from .state_estimator import EKFSLAM
from .map_manager import MapManager
from .motion_controller import MotionController
from .actuators import Actuators
from .sensors import Sensors
from .explorer import Explorer


class Navigator:
    """Navigation on a stored map with relocalization and change adaptation.

    Usage:
        nav = Navigator(camera, feature_extractor, actuators, sensors,
                        map_manager, state_estimator, motion_controller)
        if nav.relocalize():
            nav.navigate_to((3.0, 2.5))
    """

    def __init__(self, camera: Camera, feature_extractor: FeatureExtractor,
                 actuators: Actuators, sensors: Sensors,
                 map_manager: MapManager, state_estimator: EKFSLAM,
                 motion_controller: MotionController):
        self._camera = camera
        self._fe = feature_extractor
        self._actuators = actuators
        self._sensors = sensors
        self._map = map_manager
        self._ekf = state_estimator
        self._mc = motion_controller
        self._explorer = Explorer(map_manager)

        # Tracking state
        self._localized = False
        self._tracking_lost_frames = 0
        self._max_lost_frames = 5

        # Landmark change tracking: {landmark_id: missed_count}
        self._missed_landmarks = {}

    def relocalize(self, max_attempts: int = 4) -> bool:
        """Determine robot pose by matching camera views against stored map.

        Performs a panoramic servo sweep, extracts features from each view,
        matches against the landmark database, and uses PnP to estimate pose.

        Args:
            max_attempts: Number of 90-degree rotations to try.

        Returns:
            True if localization succeeded.
        """
        if cv2 is None:
            return False

        K = self._camera.K
        dist = self._camera.dist_coeffs

        for attempt in range(max_attempts):
            total_inliers = 0
            best_pose = None
            best_inliers = 0

            # Pan sweep across multiple angles
            for pan_angle in config.PAN_SWEEP_ANGLES:
                self._actuators.set_servo_pan(pan_angle)
                time.sleep(config.SERVO_SETTLE_MS / 1000.0)

                frame = self._camera.capture()
                kp, desc = self._fe.detect_and_compute(frame)

                if desc is None or len(kp) < 10:
                    continue

                # Match against all landmarks in the map
                object_points = []  # 3D landmark positions
                image_points = []   # 2D keypoint positions

                for i, d in enumerate(desc):
                    match = self._map.match_observation(d)
                    if match is not None:
                        object_points.append(match.position_3d)
                        image_points.append(kp[i].pt)

                if len(object_points) < 6:
                    continue

                obj_pts = np.array(object_points, dtype=np.float64)
                img_pts = np.array(image_points, dtype=np.float64).reshape(-1, 1, 2)

                # PnP with RANSAC
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, K, dist,
                    iterationsCount=1000,
                    reprojectionError=5.0,
                    confidence=0.99,
                )

                if success and inliers is not None and len(inliers) > best_inliers:
                    best_inliers = len(inliers)
                    # Convert rvec/tvec to robot pose
                    R, _ = cv2.Rodrigues(rvec)
                    # Camera position in world frame
                    cam_pos = -R.T @ tvec.ravel()
                    # Extract yaw from rotation
                    yaw = math.atan2(R[0, 2], R[0, 0])
                    # Adjust for pan servo offset
                    yaw_offset = math.radians(pan_angle - config.SERVO_PAN_CENTER)
                    best_pose = (float(cam_pos[0]), float(cam_pos[2]),
                                 yaw - yaw_offset)

            if best_inliers >= 20 and best_pose is not None:
                self._ekf.set_pose(*best_pose)
                self._localized = True
                self._tracking_lost_frames = 0
                self._actuators.center_camera()
                self._actuators.set_led_color("localizing")
                return True

            # Rotate 90 degrees and try again
            self._actuators.rotate_right(60)
            time.sleep(1.0)
            self._actuators.stop()
            time.sleep(0.3)

        self._actuators.center_camera()
        self._actuators.set_led_color("lost")
        return False

    def navigate_to(self, goal_xy: Tuple[float, float],
                    timeout_s: float = 120.0) -> bool:
        """Plan and execute a path to a goal position.

        Args:
            goal_xy: (x, y) target in world coordinates.
            timeout_s: Maximum time for navigation.

        Returns:
            True if goal was reached.
        """
        if not self._localized:
            return False

        start_time = time.time()
        pose = self._ekf.get_pose()

        # Plan initial path
        path = self._explorer.plan_path(pose[:2], goal_xy)
        if not path:
            return False

        waypoint_idx = 0

        while waypoint_idx < len(path):
            if time.time() - start_time > timeout_s:
                self._mc.stop()
                return False

            pose = self._ekf.get_pose()
            target = path[waypoint_idx]

            # Check for obstacles
            if self._sensors.is_obstacle_ahead():
                self._mc.stop()
                # Update map with new obstacle
                us_range = self._sensors.read_ultrasonic_m()
                if us_range > 0:
                    self._map.occupancy.update_ultrasonic(
                        pose[0], pose[1], pose[2], us_range)
                # Replan
                path = self._explorer.plan_path(pose[:2], goal_xy)
                if not path:
                    return False
                waypoint_idx = 0
                continue

            # Drive toward current waypoint
            reached = self._mc.drive_to_waypoint(pose, target)
            if reached:
                waypoint_idx += 1

            # Record trajectory
            self._map.record_pose(*pose)

        self._mc.stop()
        return True

    def update_tracking(self, frame: np.ndarray,
                        vo_delta: Optional[Tuple[float, float, float]]):
        """Update localization state with a new frame and VO estimate.

        Called each frame during navigation. Handles landmark matching
        against the map for EKF updates, and detects environmental changes.

        Args:
            frame: Current grayscale frame.
            vo_delta: Visual odometry estimate (dx, dy, dtheta), or None if lost.
        """
        if vo_delta is not None:
            self._ekf.predict(vo_delta)
            self._tracking_lost_frames = 0
        else:
            self._tracking_lost_frames += 1
            if self._tracking_lost_frames >= self._max_lost_frames:
                self._localized = False
                self._actuators.set_led_color("lost")
            return

        # Match current features against map landmarks for EKF update
        kp, desc = self._fe.detect_and_compute(frame)
        if desc is None:
            return

        pose = self._ekf.get_pose()
        visible = self._map.get_visible_landmarks(pose, fov_deg=90, max_range=5.0)

        # Track which landmarks we expected to see vs actually saw
        expected_ids = set(lm.id for lm in visible)
        seen_ids = set()

        for i, d in enumerate(desc):
            match = self._map.match_observation(
                d, max_descriptor_distance=50)
            if match is not None and match.id in expected_ids:
                # Compute bearing and range from robot to matched landmark
                lm_pos = match.position_3d
                dx = lm_pos[0] - pose[0]
                dy = lm_pos[1] - pose[1]
                bearing = math.atan2(dy, dx) - pose[2]
                range_m = math.sqrt(dx**2 + dy**2)

                self._ekf.update_landmark(match.id, bearing, range_m, d)
                seen_ids.add(match.id)

        # Track missed landmarks (environmental change detection)
        for lm_id in expected_ids - seen_ids:
            self._missed_landmarks[lm_id] = self._missed_landmarks.get(lm_id, 0) + 1

            if self._missed_landmarks[lm_id] >= 10:
                # Landmark likely moved/removed -- inflate its covariance
                rec = self._ekf.get_landmark_record(lm_id)
                if rec is not None and rec.status == "active":
                    self._ekf.freeze_landmark(lm_id)

        # Reset miss count for seen landmarks
        for lm_id in seen_ids:
            self._missed_landmarks.pop(lm_id, None)

    @property
    def is_localized(self) -> bool:
        """True if the robot has a confident pose estimate."""
        if not self._localized:
            return False
        return self._ekf.pose_uncertainty < 0.5  # trace threshold

    @property
    def pose(self) -> Tuple[float, float, float]:
        return self._ekf.get_pose()
