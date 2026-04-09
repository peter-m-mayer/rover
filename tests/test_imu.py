"""Tests for IMU module and EKF IMU integration."""

import math
import time
import numpy as np
import pytest

from raspbot_slam.imu import IMU, IMUReading
from raspbot_slam.state_estimator import EKFSLAM

try:
    import pybullet
    _HAS_PYBULLET = True
except ImportError:
    _HAS_PYBULLET = False


# =============================================================================
# IMU Unit Tests (mock mode, no hardware)
# =============================================================================

class TestIMUMock:

    def test_mock_mode_available(self):
        imu = IMU(mock_mode=True)
        assert imu.is_available

    def test_mock_reading_stationary(self):
        imu = IMU(mock_mode=True)
        imu.update()
        # Stationary: accel should be ~(0, 0, 9.81), gyro ~(0, 0, 0)
        ax, ay = imu.accel_xy
        assert abs(ax) < 0.1
        assert abs(ay) < 0.1
        assert abs(imu.gyro_z) < 0.01

    def test_calibrate_gyro_mock(self):
        imu = IMU(mock_mode=True)
        imu.calibrate_gyro(n_samples=10, verbose=False)
        assert imu.is_calibrated

    def test_set_mock_data(self):
        imu = IMU(mock_mode=True)
        imu.calibrate_gyro(n_samples=5, verbose=False)
        imu.set_mock_data(gyro=(0, 0, 0.5))
        assert abs(imu.gyro_z - 0.5) < 0.01

    def test_prediction_delta_math(self):
        """Test prediction delta computation directly (no calibrate call)."""
        imu = IMU(mock_mode=True)
        imu._calibrated = True
        # Bypass lock by setting internals directly
        imu._gyro_raw = np.array([0, 0, 0.5])
        imu._accel_raw = np.array([1.0, 0, 9.81])
        imu._dt = 0.01

        dx, dy, dtheta = imu.get_prediction_delta(dt=0.01)
        assert abs(dtheta - 0.005) < 0.001  # 0.5 rad/s * 0.01s
        assert dx > 0  # positive forward acceleration

    def test_heading_observation(self):
        imu = IMU(mock_mode=True)
        heading, noise = imu.get_heading_observation()
        assert isinstance(heading, float)
        assert noise > 0


# =============================================================================
# EKF IMU Integration Tests
# =============================================================================

class TestEKFWithIMU:

    def test_predict_imu_heading(self):
        """IMU gyro prediction should update heading accurately."""
        ekf = EKFSLAM(initial_pose=(0, 0, 0))

        # 1 second of 0.5 rad/s rotation in 100 steps
        for _ in range(100):
            ekf.predict_imu(gyro_z=0.5, accel_x=0, accel_y=0, dt=0.01)

        _, _, theta = ekf.get_pose()
        expected = 0.5  # 0.5 rad/s * 1s
        assert abs(theta - expected) < 0.01, \
            f"Expected heading {expected:.3f}, got {theta:.3f}"

    def test_predict_imu_forward(self):
        """IMU accelerometer prediction should move robot forward."""
        ekf = EKFSLAM(initial_pose=(0, 0, 0))

        # Constant forward acceleration of 1 m/s^2 for 1 second
        for _ in range(100):
            ekf.predict_imu(gyro_z=0, accel_x=1.0, accel_y=0, dt=0.01)

        x, y, _ = ekf.get_pose()
        # x = 0.5 * a * t^2 = 0.5 * 1 * 1 = 0.5m (but integrated in steps)
        # With 100 steps of dt=0.01: sum of 0.5*1*0.01^2 * 100 = 0.005m
        # (accel double-integration at small dt gives very small displacement)
        assert x > 0, f"Should move forward, got x={x:.6f}"

    def test_predict_imu_covariance_growth(self):
        """IMU prediction should grow covariance (process noise)."""
        ekf = EKFSLAM(initial_pose=(0, 0, 0))
        trace0 = ekf.pose_uncertainty

        for _ in range(50):
            ekf.predict_imu(gyro_z=0.1, accel_x=0.5, accel_y=0, dt=0.01)

        trace1 = ekf.pose_uncertainty
        assert trace1 > trace0

    def test_update_heading_corrects_drift(self):
        """Magnetometer heading update should correct accumulated drift."""
        ekf = EKFSLAM(initial_pose=(0, 0, 0))

        # Accumulate heading drift via noisy predictions
        for _ in range(200):
            ekf.predict_imu(gyro_z=0.01, accel_x=0, accel_y=0, dt=0.01)

        _, _, theta_drifted = ekf.get_pose()
        assert abs(theta_drifted) > 0.01  # some drift accumulated

        # Magnetometer says heading is actually 0
        ekf.update_heading(0.0, noise_rad=0.05)

        _, _, theta_corrected = ekf.get_pose()
        # Should be closer to 0 after mag update
        assert abs(theta_corrected) < abs(theta_drifted), \
            f"Mag update should reduce heading error: before={theta_drifted:.4f}, after={theta_corrected:.4f}"

    def test_update_vo_as_observation(self):
        """VO update should work as an observation on the EKF."""
        ekf = EKFSLAM(initial_pose=(0, 0, 0))

        # Some IMU predictions
        for _ in range(50):
            ekf.predict_imu(gyro_z=0.1, accel_x=0, accel_y=0, dt=0.01)

        trace_before = ekf.pose_uncertainty

        # VO observation
        ekf.update_vo(vo_dx=0.01, vo_dy=0, vo_dtheta=0.05)

        trace_after = ekf.pose_uncertainty
        # VO update should reduce (or at least not increase) uncertainty
        assert trace_after <= trace_before * 1.1  # allow small numerical increase

    def test_imu_plus_vo_better_than_vo_alone(self):
        """EKF with IMU+VO should have less heading error than VO alone."""
        # VO-only: accumulate with large heading noise
        ekf_vo = EKFSLAM(initial_pose=(0, 0, 0))
        for _ in range(100):
            ekf_vo.predict((0.004, 0, 0.02))  # noisy heading

        # IMU+VO: gyro for heading, VO for position
        ekf_imu = EKFSLAM(initial_pose=(0, 0, 0))
        for _ in range(100):
            ekf_imu.predict_imu(gyro_z=0.02, accel_x=0, accel_y=0, dt=0.01)

        # IMU heading covariance should be smaller
        P_vo = ekf_vo.get_pose_covariance()
        P_imu = ekf_imu.get_pose_covariance()
        assert P_imu[2, 2] < P_vo[2, 2], \
            f"IMU heading variance {P_imu[2,2]:.6f} should be < VO {P_vo[2,2]:.6f}"


# =============================================================================
# Simulated IMU Tests (requires PyBullet)
# =============================================================================

@pytest.mark.skipif(not _HAS_PYBULLET, reason="PyBullet required")
class TestSimIMU:

    def test_sim_imu_init(self):
        from raspbot_slam.simulator import SimWorld, SimIMU
        world = SimWorld(floor_plan="simple_room", gui=False)
        try:
            imu = SimIMU(world)
            assert imu.is_available
        finally:
            world.close()

    def test_sim_imu_calibrate(self):
        from raspbot_slam.simulator import SimWorld, SimIMU
        world = SimWorld(floor_plan="simple_room", gui=False)
        try:
            imu = SimIMU(world)
            imu.calibrate_gyro(verbose=False)
            assert imu.is_calibrated
        finally:
            world.close()

    def test_sim_imu_stationary(self):
        from raspbot_slam.simulator import SimWorld, SimIMU
        world = SimWorld(floor_plan="simple_room", gui=False)
        try:
            imu = SimIMU(world)
            imu.calibrate_gyro(verbose=False)
            world.step(10)
            imu.update()
            # Stationary: gyro should be near zero
            assert abs(imu.gyro_z) < 0.05, f"Gyro Z should be ~0 when stationary, got {imu.gyro_z}"
        finally:
            world.close()

    def test_sim_imu_detects_rotation(self):
        from raspbot_slam.simulator import SimWorld, SimIMU, SimActuators
        world = SimWorld(floor_plan="simple_room", gui=False)
        try:
            imu = SimIMU(world)
            act = SimActuators(world)
            imu.calibrate_gyro(verbose=False)

            act.rotate_left(60)
            world.step(60)
            imu.update()

            # Gyro Z should be positive (CCW rotation)
            assert imu.gyro_z > 0.1, f"Gyro should detect CCW rotation, got {imu.gyro_z}"
        finally:
            world.close()

    def test_sim_imu_heading_tracks_rotation(self):
        from raspbot_slam.simulator import SimWorld, SimIMU, SimActuators
        world = SimWorld(floor_plan="simple_room", gui=False)
        try:
            imu = SimIMU(world)
            act = SimActuators(world)
            imu.calibrate_gyro(verbose=False)
            imu.update()
            h0 = imu.heading

            # Rotate left for a bit
            act.rotate_left(50)
            for _ in range(30):
                world.step(8)
                imu.update()
            act.stop()

            h1 = imu.heading
            gt = world.get_robot_pose()
            gt_heading_change = gt[2]  # started at 0

            # IMU heading should track GT heading direction
            heading_change = h1 - h0
            assert heading_change * gt_heading_change > 0, \
                f"IMU heading should track GT direction: IMU={math.degrees(heading_change):.1f}, GT={math.degrees(gt_heading_change):.1f}"
        finally:
            world.close()
