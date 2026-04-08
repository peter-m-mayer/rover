"""Tests for state_estimator.py -- EKF-SLAM predict/update/landmark management."""

import math
import numpy as np
import pytest

from raspbot_slam.state_estimator import EKFSLAM, LandmarkRecord


class TestEKFInit:

    def test_initial_pose(self):
        ekf = EKFSLAM(initial_pose=(1.0, 2.0, 0.5))
        x, y, theta = ekf.get_pose()
        assert abs(x - 1.0) < 1e-10
        assert abs(y - 2.0) < 1e-10
        assert abs(theta - 0.5) < 1e-10

    def test_initial_scale(self):
        ekf = EKFSLAM()
        assert ekf.get_scale_factor() == 1.0

    def test_initial_covariance_small(self):
        ekf = EKFSLAM()
        P = ekf.get_pose_covariance()
        assert P.shape == (3, 3)
        assert np.trace(P) < 0.1  # small initial uncertainty

    def test_initial_no_landmarks(self):
        ekf = EKFSLAM()
        assert ekf.active_landmark_count == 0
        assert ekf.total_landmark_count == 0
        assert ekf.state_dimension == 4  # x, y, theta, scale


class TestEKFPredict:

    def test_forward_motion(self):
        ekf = EKFSLAM(initial_pose=(0, 0, 0))
        ekf.predict((1.0, 0.0, 0.0))  # 1m forward, heading=0
        x, y, theta = ekf.get_pose()
        assert abs(x - 1.0) < 0.01
        assert abs(y) < 0.01
        assert abs(theta) < 0.01

    def test_lateral_motion(self):
        ekf = EKFSLAM(initial_pose=(0, 0, 0))
        ekf.predict((0.0, 1.0, 0.0))  # 1m left, heading=0
        x, y, theta = ekf.get_pose()
        assert abs(x) < 0.01
        assert abs(y - 1.0) < 0.01

    def test_rotation(self):
        ekf = EKFSLAM(initial_pose=(0, 0, 0))
        ekf.predict((0.0, 0.0, math.pi / 2))  # 90 degrees CCW
        x, y, theta = ekf.get_pose()
        assert abs(x) < 0.01
        assert abs(y) < 0.01
        assert abs(theta - math.pi / 2) < 0.01

    def test_forward_then_rotate_then_forward(self):
        ekf = EKFSLAM(initial_pose=(0, 0, 0))
        ekf.predict((1.0, 0.0, 0.0))       # 1m forward (now at x=1)
        ekf.predict((0.0, 0.0, math.pi/2)) # rotate 90 CCW
        ekf.predict((1.0, 0.0, 0.0))       # 1m forward (now at x=1, y=1)
        x, y, theta = ekf.get_pose()
        assert abs(x - 1.0) < 0.1
        assert abs(y - 1.0) < 0.1

    def test_covariance_grows(self):
        ekf = EKFSLAM(initial_pose=(0, 0, 0))
        trace_before = ekf.pose_uncertainty
        ekf.predict((0.5, 0.0, 0.1))
        trace_after = ekf.pose_uncertainty
        assert trace_after > trace_before

    def test_scale_factor_applied(self):
        ekf = EKFSLAM(initial_pose=(0, 0, 0))
        ekf.set_scale(2.0)
        ekf.predict((1.0, 0.0, 0.0))  # 1m forward × scale 2 = 2m
        x, y, _ = ekf.get_pose()
        assert abs(x - 2.0) < 0.01

    def test_zero_motion_no_change(self):
        ekf = EKFSLAM(initial_pose=(5.0, 3.0, 1.0))
        ekf.predict((0.0, 0.0, 0.0))
        x, y, theta = ekf.get_pose()
        assert abs(x - 5.0) < 0.01
        assert abs(y - 3.0) < 0.01
        assert abs(theta - 1.0) < 0.01


class TestEKFLandmarks:

    def test_add_landmark(self):
        ekf = EKFSLAM()
        desc = np.random.randint(0, 256, 32, dtype=np.uint8)
        lm_id = ekf.add_landmark(np.array([1.0, 2.0, 0.5]), desc)
        assert lm_id == 0
        assert ekf.total_landmark_count == 1
        assert ekf.active_landmark_count == 0  # still candidate

    def test_promote_landmark(self):
        ekf = EKFSLAM()
        desc = np.random.randint(0, 256, 32, dtype=np.uint8)
        lm_id = ekf.add_landmark(np.array([1.0, 2.0, 0.5]), desc)

        success = ekf.promote_landmark(lm_id, np.array([1.0, 2.0, 0.5]))
        assert success
        assert ekf.active_landmark_count == 1
        assert ekf.state_dimension == 7  # 4 robot + 3 landmark

        pos = ekf.get_landmark_position(lm_id)
        assert pos is not None
        assert abs(pos[0] - 1.0) < 1e-10

    def test_freeze_and_reactivate(self):
        ekf = EKFSLAM()
        desc = np.random.randint(0, 256, 32, dtype=np.uint8)
        lm_id = ekf.add_landmark(np.array([3.0, 4.0, 1.0]), desc)
        ekf.promote_landmark(lm_id, np.array([3.0, 4.0, 1.0]))
        assert ekf.active_landmark_count == 1

        ekf.freeze_landmark(lm_id)
        assert ekf.active_landmark_count == 0
        # Position should still be retrievable (frozen)
        pos = ekf.get_landmark_position(lm_id)
        assert pos is not None
        assert abs(pos[0] - 3.0) < 0.1

        # Reactivate
        success = ekf.reactivate_landmark(lm_id)
        assert success
        assert ekf.active_landmark_count == 1

    def test_landmark_capacity_freeze_oldest(self):
        """When at capacity, promoting a new landmark should freeze the oldest."""
        from raspbot_slam import config
        ekf = EKFSLAM()

        # Fill to capacity
        for i in range(config.MAX_ACTIVE_LANDMARKS):
            desc = np.random.randint(0, 256, 32, dtype=np.uint8)
            lm_id = ekf.add_landmark(np.array([float(i), 0, 0]), desc)
            ekf.promote_landmark(lm_id, np.array([float(i), 0, 0]))
            rec = ekf.get_landmark_record(lm_id)
            rec.last_seen_keyframe = i  # ascending so first is "oldest"

        assert ekf.active_landmark_count == config.MAX_ACTIVE_LANDMARKS

        # Add one more -- should freeze the oldest (keyframe=0)
        desc = np.random.randint(0, 256, 32, dtype=np.uint8)
        new_id = ekf.add_landmark(np.array([99, 0, 0]), desc)
        success = ekf.promote_landmark(new_id, np.array([99, 0, 0]))
        assert success
        assert ekf.active_landmark_count == config.MAX_ACTIVE_LANDMARKS

    def test_get_active_landmarks(self):
        ekf = EKFSLAM()
        for i in range(3):
            desc = np.random.randint(0, 256, 32, dtype=np.uint8)
            lm_id = ekf.add_landmark(np.array([float(i), 0, 0]), desc)
            ekf.promote_landmark(lm_id, np.array([float(i), 0, 0]))

        active = ekf.get_active_landmarks()
        assert len(active) == 3


class TestEKFUpdate:

    def test_update_reduces_covariance(self):
        """Landmark update should reduce pose uncertainty."""
        ekf = EKFSLAM(initial_pose=(0, 0, 0))
        desc = np.random.randint(0, 256, 32, dtype=np.uint8)
        lm_id = ekf.add_landmark(np.array([2.0, 0.0, 1.0]), desc)
        ekf.promote_landmark(lm_id, np.array([2.0, 0.0, 1.0]))

        # Add some uncertainty via prediction
        ekf.predict((0.5, 0.0, 0.0))
        trace_before = ekf.pose_uncertainty

        # Update with a landmark observation (bearing=0, range=1.5m)
        ekf.update_landmark(lm_id, bearing=0.0, range_m=1.5, descriptor=desc)
        trace_after = ekf.pose_uncertainty

        assert trace_after < trace_before

    def test_gating_rejects_outlier(self):
        """An observation wildly inconsistent with prediction should be gated."""
        ekf = EKFSLAM(initial_pose=(0, 0, 0))
        desc = np.random.randint(0, 256, 32, dtype=np.uint8)
        lm_id = ekf.add_landmark(np.array([2.0, 0.0, 1.0]), desc)
        ekf.promote_landmark(lm_id, np.array([2.0, 0.0, 1.0]))

        # Observe with wildly wrong bearing (landmark at x=2 should be bearing~0)
        result = ekf.update_landmark(lm_id, bearing=math.pi, range_m=2.0,
                                     descriptor=desc)
        assert not result  # should be gated


class TestEKFSetPose:

    def test_set_pose_directly(self):
        ekf = EKFSLAM()
        ekf.set_pose(5.0, 3.0, 1.0)
        x, y, theta = ekf.get_pose()
        assert abs(x - 5.0) < 1e-10
        assert abs(y - 3.0) < 1e-10
        assert abs(theta - 1.0) < 1e-10

    def test_angle_normalization(self):
        ekf = EKFSLAM(initial_pose=(0, 0, 0))
        ekf.predict((0, 0, 4 * math.pi + 0.1))
        _, _, theta = ekf.get_pose()
        assert -math.pi <= theta <= math.pi


class TestDescriptorMatching:

    def test_exact_match(self):
        ekf = EKFSLAM()
        desc = np.array([42] * 32, dtype=np.uint8)
        lm_id = ekf.add_landmark(np.array([1, 0, 0]), desc)
        ekf.promote_landmark(lm_id, np.array([1, 0, 0]))

        result = ekf.match_descriptor(desc)
        assert result == lm_id

    def test_no_match_distant_descriptor(self):
        ekf = EKFSLAM()
        desc1 = np.zeros(32, dtype=np.uint8)
        desc2 = np.full(32, 255, dtype=np.uint8)  # maximum distance
        lm_id = ekf.add_landmark(np.array([1, 0, 0]), desc1)
        ekf.promote_landmark(lm_id, np.array([1, 0, 0]))

        result = ekf.match_descriptor(desc2, max_distance=60)
        assert result is None
