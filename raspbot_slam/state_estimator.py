"""
EKF-SLAM: Extended Kalman Filter for simultaneous localization and mapping.

Fuses visual odometry deltas, synthetic stereo landmark observations, and
ultrasonic range readings into a coherent robot pose estimate with a sparse
set of 3D landmark positions.

State vector: [x, y, theta, scale, lm1_x, lm1_y, lm1_z, ..., lmN_x, lmN_y, lmN_z]
where N <= MAX_ACTIVE_LANDMARKS (30).

The EKF runs in 2D+scale for the robot (ground plane assumption) but maintains
3D positions for landmarks (walls, furniture have vertical extent).
"""

import math
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
import numpy as np

from . import config


@dataclass
class LandmarkRecord:
    """Metadata for a tracked landmark."""
    id: int
    descriptor: np.ndarray           # ORB descriptor (32 bytes)
    observation_count: int = 0
    last_seen_keyframe: int = -1
    status: str = "candidate"        # 'candidate', 'active', 'frozen'
    frozen_position: Optional[np.ndarray] = None  # stored when frozen
    frozen_covariance: Optional[np.ndarray] = None
    state_index: int = -1            # index in state vector (-1 if not active)


class EKFSLAM:
    """Extended Kalman Filter for robot pose + landmark positions.

    Robot state is 2D: (x, y, theta) on the ground plane, plus a scale factor.
    Landmarks are 3D: (lx, ly, lz) in world frame.
    Active landmark count is bounded at MAX_ACTIVE_LANDMARKS for real-time performance.

    Usage:
        ekf = EKFSLAM(initial_pose=(0, 0, 0))
        # In main loop:
        ekf.predict(vo_delta)
        for each observed landmark:
            ekf.update_landmark(lm_id, bearing, range_m, descriptor)
        ekf.update_ultrasonic(range_mm, occupancy_grid)
    """

    # Indices in state vector
    IX = 0
    IY = 1
    ITHETA = 2
    ISCALE = 3
    ROBOT_DIM = 4  # x, y, theta, scale
    LM_DIM = 3     # x, y, z per landmark

    def __init__(self, initial_pose: Tuple[float, float, float] = (0, 0, 0)):
        """Initialize EKF at given pose.

        Args:
            initial_pose: (x, y, theta) in meters and radians.
        """
        self._max_landmarks = config.MAX_ACTIVE_LANDMARKS
        max_dim = self.ROBOT_DIM + self.LM_DIM * self._max_landmarks

        # State vector and covariance
        self._x = np.zeros(max_dim, dtype=np.float64)
        self._P = np.zeros((max_dim, max_dim), dtype=np.float64)

        # Initialize robot pose
        self._x[self.IX] = initial_pose[0]
        self._x[self.IY] = initial_pose[1]
        self._x[self.ITHETA] = initial_pose[2]
        self._x[self.ISCALE] = 1.0

        # Initial robot pose uncertainty
        self._P[self.IX, self.IX] = 0.01
        self._P[self.IY, self.IY] = 0.01
        self._P[self.ITHETA, self.ITHETA] = 0.01
        self._P[self.ISCALE, self.ISCALE] = 0.1

        # Landmark management
        self._active_count = 0  # number of landmarks in active state
        self._landmarks: Dict[int, LandmarkRecord] = {}
        self._next_landmark_id = 0

        # Dimension of currently used state
        self._dim = self.ROBOT_DIM

    # =========================================================================
    # Prediction
    # =========================================================================

    def predict(self, vo_delta: Tuple[float, float, float]):
        """Prediction step using visual odometry motion estimate.

        Args:
            vo_delta: (dx, dy, dtheta) in robot frame.
                dx = forward, dy = leftward, dtheta = CCW rotation.
        """
        dx, dy, dtheta = vo_delta
        scale = self._x[self.ISCALE]
        theta = self._x[self.ITHETA]

        # Scale the VO translation
        dx_scaled = dx * scale
        dy_scaled = dy * scale

        # Transform to world frame
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx_world = dx_scaled * cos_t - dy_scaled * sin_t
        dy_world = dx_scaled * sin_t + dy_scaled * cos_t

        # State prediction
        self._x[self.IX] += dx_world
        self._x[self.IY] += dy_world
        self._x[self.ITHETA] += dtheta
        # Normalize angle to [-pi, pi]
        self._x[self.ITHETA] = self._normalize_angle(self._x[self.ITHETA])

        # Jacobian of motion model w.r.t. robot state
        n = self._dim
        F = np.eye(n, dtype=np.float64)
        # d(x_world)/d(theta)
        F[self.IX, self.ITHETA] = -dx_scaled * sin_t - dy_scaled * cos_t
        # d(y_world)/d(theta)
        F[self.IY, self.ITHETA] = dx_scaled * cos_t - dy_scaled * sin_t
        # d(x_world)/d(scale)
        F[self.IX, self.ISCALE] = dx * cos_t - dy * sin_t
        # d(y_world)/d(scale)
        F[self.IY, self.ISCALE] = dx * sin_t + dy * cos_t

        # Process noise
        displacement = math.sqrt(dx_scaled**2 + dy_scaled**2)
        Q = np.zeros((n, n), dtype=np.float64)
        Q[self.IX, self.IX] = (config.PROCESS_NOISE_POSITION * max(displacement, 0.001))**2
        Q[self.IY, self.IY] = (config.PROCESS_NOISE_POSITION * max(displacement, 0.001))**2
        Q[self.ITHETA, self.ITHETA] = (config.PROCESS_NOISE_HEADING * max(abs(dtheta), 0.001))**2
        Q[self.ISCALE, self.ISCALE] = config.PROCESS_NOISE_SCALE**2

        # Covariance prediction
        P = self._P[:n, :n]
        self._P[:n, :n] = F @ P @ F.T + Q

    # =========================================================================
    # Landmark Update
    # =========================================================================

    def update_landmark(self, landmark_id: int,
                        bearing: float, range_m: float,
                        descriptor: np.ndarray) -> bool:
        """Update EKF with a landmark observation.

        Args:
            landmark_id: Internal landmark ID (from add_landmark or match).
            bearing: Bearing angle to landmark from robot heading (radians, CCW+).
            range_m: Distance to landmark in meters.
            descriptor: ORB descriptor for this observation.

        Returns:
            True if update was applied, False if gated out.
        """
        if landmark_id not in self._landmarks:
            return False

        rec = self._landmarks[landmark_id]
        rec.observation_count += 1

        if rec.status != "active" or rec.state_index < 0:
            return False

        idx = rec.state_index
        n = self._dim

        # Predicted landmark position (2D projection for bearing/range)
        lm_x = self._x[idx]
        lm_y = self._x[idx + 1]
        rx = self._x[self.IX]
        ry = self._x[self.IY]
        rtheta = self._x[self.ITHETA]

        # Predicted observation
        dx = lm_x - rx
        dy = lm_y - ry
        pred_range = math.sqrt(dx**2 + dy**2)
        if pred_range < 1e-6:
            return False
        pred_bearing = self._normalize_angle(math.atan2(dy, dx) - rtheta)

        # Innovation
        z = np.array([bearing - pred_bearing, range_m - pred_range])
        z[0] = self._normalize_angle(z[0])

        # Chi-squared gating
        # Compute innovation covariance first
        H = np.zeros((2, n), dtype=np.float64)
        # d(bearing)/d(rx)
        H[0, self.IX] = dy / (pred_range**2)
        H[0, self.IY] = -dx / (pred_range**2)
        H[0, self.ITHETA] = -1.0
        H[0, idx] = -dy / (pred_range**2)
        H[0, idx + 1] = dx / (pred_range**2)
        # d(range)/d(rx)
        H[1, self.IX] = -dx / pred_range
        H[1, self.IY] = -dy / pred_range
        H[1, idx] = dx / pred_range
        H[1, idx + 1] = dy / pred_range

        R = np.diag([config.OBSERVATION_NOISE_BEARING**2,
                      config.OBSERVATION_NOISE_RANGE**2])

        P = self._P[:n, :n]
        S = H @ P @ H.T + R

        # Chi-squared test
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        mahal = float(z @ S_inv @ z)
        if mahal > config.CHI_SQUARED_GATE:
            return False

        # Kalman gain
        K = P @ H.T @ S_inv

        # State update
        self._x[:n] += K @ z
        self._x[self.ITHETA] = self._normalize_angle(self._x[self.ITHETA])

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(n) - K @ H
        self._P[:n, :n] = I_KH @ P @ I_KH.T + K @ R @ K.T

        return True

    def update_ultrasonic(self, range_mm: float, occupancy_grid: np.ndarray = None):
        """Update EKF with ultrasonic range measurement.

        Simple 1D constraint along the robot's heading direction.

        Args:
            range_mm: Ultrasonic reading in millimeters.
            occupancy_grid: Optional occupancy grid for ray-cast prediction.
                If None, update is skipped (no predicted range to compare against).
        """
        if range_mm <= 0:
            return

        range_m = range_mm / 1000.0
        R_us = np.array([[((config.ULTRASONIC_NOISE_MM / 1000.0)**2)]])

        # Without a map to ray-cast against, we can still use the ultrasonic
        # as a soft constraint on position. For now, just store the reading
        # for use by the map manager.
        self._last_ultrasonic_m = range_m

    # =========================================================================
    # Landmark Management
    # =========================================================================

    def add_landmark(self, position_3d: np.ndarray,
                     descriptor: np.ndarray) -> int:
        """Add a new landmark. Starts as candidate; promoted after enough observations.

        Args:
            position_3d: (x, y, z) in world frame.
            descriptor: ORB descriptor (32 bytes).

        Returns:
            Landmark ID.
        """
        lm_id = self._next_landmark_id
        self._next_landmark_id += 1

        self._landmarks[lm_id] = LandmarkRecord(
            id=lm_id,
            descriptor=descriptor.copy(),
            observation_count=1,
            status="candidate",
        )
        return lm_id

    def promote_landmark(self, landmark_id: int, position_3d: np.ndarray) -> bool:
        """Promote a candidate landmark to active EKF state.

        Args:
            landmark_id: ID of the candidate.
            position_3d: Best estimate of 3D position.

        Returns:
            True if promoted, False if at capacity or not a candidate.
        """
        if landmark_id not in self._landmarks:
            return False
        rec = self._landmarks[landmark_id]
        if rec.status != "candidate":
            return False
        if self._active_count >= self._max_landmarks:
            # Try to freeze the least-recently-observed active landmark
            if not self._freeze_oldest():
                return False

        # Add to state vector
        idx = self.ROBOT_DIM + self._active_count * self.LM_DIM
        self._x[idx] = position_3d[0]
        self._x[idx + 1] = position_3d[1]
        self._x[idx + 2] = position_3d[2]

        # Initialize landmark covariance (large uncertainty)
        init_var = 1.0  # 1 meter uncertainty
        self._P[idx, idx] = init_var
        self._P[idx + 1, idx + 1] = init_var
        self._P[idx + 2, idx + 2] = init_var

        rec.state_index = idx
        rec.status = "active"
        self._active_count += 1
        self._dim = self.ROBOT_DIM + self._active_count * self.LM_DIM

        return True

    def freeze_landmark(self, landmark_id: int):
        """Remove a landmark from the active EKF state, preserving its estimate."""
        if landmark_id not in self._landmarks:
            return
        rec = self._landmarks[landmark_id]
        if rec.status != "active" or rec.state_index < 0:
            return

        idx = rec.state_index

        # Save position and covariance
        rec.frozen_position = self._x[idx:idx + 3].copy()
        rec.frozen_covariance = self._P[idx:idx + 3, idx:idx + 3].copy()
        rec.status = "frozen"

        # Remove from state vector by shifting everything after it
        self._remove_from_state(idx)
        rec.state_index = -1

    def reactivate_landmark(self, landmark_id: int) -> bool:
        """Restore a frozen landmark to the active state with inflated covariance."""
        if landmark_id not in self._landmarks:
            return False
        rec = self._landmarks[landmark_id]
        if rec.status != "frozen" or rec.frozen_position is None:
            return False
        if self._active_count >= self._max_landmarks:
            if not self._freeze_oldest():
                return False

        idx = self.ROBOT_DIM + self._active_count * self.LM_DIM
        self._x[idx:idx + 3] = rec.frozen_position
        inflated_cov = rec.frozen_covariance * config.LANDMARK_REACTIVATION_INFLATION
        self._P[idx:idx + 3, idx:idx + 3] = inflated_cov

        rec.state_index = idx
        rec.status = "active"
        rec.frozen_position = None
        rec.frozen_covariance = None
        self._active_count += 1
        self._dim = self.ROBOT_DIM + self._active_count * self.LM_DIM
        return True

    def _freeze_oldest(self) -> bool:
        """Freeze the active landmark with the oldest last_seen_keyframe."""
        oldest_id = None
        oldest_kf = float('inf')
        for lm_id, rec in self._landmarks.items():
            if rec.status == "active" and rec.last_seen_keyframe < oldest_kf:
                oldest_kf = rec.last_seen_keyframe
                oldest_id = lm_id
        if oldest_id is not None:
            self.freeze_landmark(oldest_id)
            return True
        return False

    def _remove_from_state(self, idx: int):
        """Remove a 3-element block at idx from state vector and covariance."""
        n = self._dim
        end = idx + self.LM_DIM

        # Shift state
        self._x[idx:n - self.LM_DIM] = self._x[end:n]
        self._x[n - self.LM_DIM:n] = 0

        # Shift covariance (rows and columns)
        # Remove rows
        self._P[idx:n - self.LM_DIM, :] = self._P[end:n, :]
        self._P[n - self.LM_DIM:n, :] = 0
        # Remove columns
        self._P[:, idx:n - self.LM_DIM] = self._P[:, end:n]
        self._P[:, n - self.LM_DIM:n] = 0

        # Update state indices of landmarks after the removed one
        for rec in self._landmarks.values():
            if rec.status == "active" and rec.state_index > idx:
                rec.state_index -= self.LM_DIM

        self._active_count -= 1
        self._dim -= self.LM_DIM

    # =========================================================================
    # Queries
    # =========================================================================

    def get_pose(self) -> Tuple[float, float, float]:
        """Return current (x, y, theta) estimate."""
        return (float(self._x[self.IX]),
                float(self._x[self.IY]),
                float(self._x[self.ITHETA]))

    def get_pose_covariance(self) -> np.ndarray:
        """Return 3x3 pose covariance (x, y, theta)."""
        return self._P[:3, :3].copy()

    def get_scale_factor(self) -> float:
        """Return current estimated scale factor."""
        return float(self._x[self.ISCALE])

    def get_landmark_position(self, landmark_id: int) -> Optional[np.ndarray]:
        """Return 3D position of a landmark, active or frozen."""
        if landmark_id not in self._landmarks:
            return None
        rec = self._landmarks[landmark_id]
        if rec.status == "active" and rec.state_index >= 0:
            idx = rec.state_index
            return self._x[idx:idx + 3].copy()
        elif rec.status == "frozen" and rec.frozen_position is not None:
            return rec.frozen_position.copy()
        return None

    def get_active_landmarks(self) -> Dict[int, np.ndarray]:
        """Return dict of {landmark_id: position_3d} for all active landmarks."""
        result = {}
        for lm_id, rec in self._landmarks.items():
            if rec.status == "active" and rec.state_index >= 0:
                idx = rec.state_index
                result[lm_id] = self._x[idx:idx + 3].copy()
        return result

    def get_landmark_record(self, landmark_id: int) -> Optional[LandmarkRecord]:
        return self._landmarks.get(landmark_id)

    def get_all_landmark_ids(self) -> List[int]:
        return list(self._landmarks.keys())

    @property
    def active_landmark_count(self) -> int:
        return self._active_count

    @property
    def total_landmark_count(self) -> int:
        return len(self._landmarks)

    @property
    def state_dimension(self) -> int:
        return self._dim

    @property
    def pose_uncertainty(self) -> float:
        """Trace of the 3x3 pose covariance -- scalar measure of uncertainty."""
        return float(np.trace(self._P[:3, :3]))

    # =========================================================================
    # Utilities
    # =========================================================================

    def set_pose(self, x: float, y: float, theta: float):
        """Set pose directly (after relocalization). Resets pose covariance."""
        self._x[self.IX] = x
        self._x[self.IY] = y
        self._x[self.ITHETA] = theta
        self._P[self.IX, self.IX] = 0.05
        self._P[self.IY, self.IY] = 0.05
        self._P[self.ITHETA, self.ITHETA] = 0.02

    def set_scale(self, scale: float):
        """Set scale factor directly (from synthetic stereo calibration)."""
        self._x[self.ISCALE] = max(0.01, scale)
        self._P[self.ISCALE, self.ISCALE] = 0.01

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def match_descriptor(self, descriptor: np.ndarray,
                         max_distance: int = 60) -> Optional[int]:
        """Find the closest matching landmark by ORB descriptor.

        Args:
            descriptor: 1x32 ORB descriptor.
            max_distance: Maximum Hamming distance for a match.

        Returns:
            Landmark ID of best match, or None if no match.
        """
        best_id = None
        best_dist = max_distance + 1
        desc = descriptor.reshape(1, -1)

        for lm_id, rec in self._landmarks.items():
            if rec.status in ("active", "frozen"):
                d = cv2_hamming_distance(desc[0], rec.descriptor)
                if d < best_dist:
                    best_dist = d
                    best_id = lm_id

        return best_id if best_dist <= max_distance else None


def cv2_hamming_distance(d1: np.ndarray, d2: np.ndarray) -> int:
    """Compute Hamming distance between two binary descriptors."""
    return int(np.sum(np.unpackbits(np.bitwise_xor(d1, d2))))
