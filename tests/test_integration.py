"""
Integration tests: run short SLAM sessions in the simulator and verify
that the pipeline produces reasonable results end-to-end.

These are slower than unit tests (~5-10 seconds each) but catch
interaction bugs between modules.
"""

import math
import numpy as np
import pytest

try:
    import pybullet
except ImportError:
    pytest.skip("PyBullet required for integration tests", allow_module_level=True)

from raspbot_slam.simulator import SimWorld, SimCamera, SimSensors, SimActuators
from raspbot_slam.feature_extractor import FeatureExtractor
from raspbot_slam.visual_odometry import VisualOdometry
from raspbot_slam.synthetic_stereo import SyntheticStereo
from raspbot_slam.state_estimator import EKFSLAM
from raspbot_slam.map_manager import MapManager
from raspbot_slam.explorer import Explorer
from raspbot_slam.motion_controller import MotionController
from raspbot_slam import config


class TestVOOnKnownTrajectory:
    """Test visual odometry on known simulated trajectories."""

    def _setup_vo(self):
        world = SimWorld(floor_plan="simple_room", gui=False)
        cam = SimCamera(world)
        act = SimActuators(world)
        fe = FeatureExtractor()
        vo = VisualOdometry(cam, fe)
        # Init VO with first frame
        vo.process_frame(cam.capture())
        return world, cam, act, vo

    def test_stationary_reports_zero(self):
        """VO should report zero motion when the robot is stationary."""
        world, cam, act, vo = self._setup_vo()
        try:
            deltas = []
            for _ in range(20):
                world.step(8)
                d = vo.process_frame(cam.capture())
                if d:
                    deltas.append(d)

            # All deltas should be (0, 0, 0) thanks to parallax gate
            for dx, dy, dtheta in deltas:
                assert abs(dx) < 0.001
                assert abs(dy) < 0.001
                assert abs(dtheta) < 0.01
        finally:
            world.close()

    def test_forward_motion_positive_dx(self):
        """Forward motion should produce positive dx accumulation."""
        world, cam, act, vo = self._setup_vo()
        vo.scale = 0.005  # reasonable scale for sim
        try:
            act.move_forward(80)
            total_dx = 0
            for _ in range(100):
                world.step(8)
                d = vo.process_frame(cam.capture())
                if d:
                    total_dx += d[0]
            act.stop()

            # Should accumulate positive forward motion
            assert total_dx > 0, f"Forward motion should be positive, got {total_dx}"
        finally:
            world.close()

    def test_rotation_detected(self):
        """In-place rotation should produce dtheta accumulation."""
        world, cam, act, vo = self._setup_vo()
        vo.scale = 0.005
        try:
            act.rotate_left(50)
            total_dtheta = 0
            for _ in range(80):
                world.step(8)
                d = vo.process_frame(cam.capture())
                if d:
                    total_dtheta += d[2]
            act.stop()

            # Should detect some rotation (not necessarily accurate magnitude)
            assert abs(total_dtheta) > 0.1, \
                f"Should detect rotation, got {math.degrees(total_dtheta):.1f} deg"
        finally:
            world.close()

    def test_feature_tracking_across_motion(self):
        """VO should maintain tracking (not lose features) during smooth motion."""
        world, cam, act, vo = self._setup_vo()
        try:
            act.move_forward(60)
            lost_count = 0
            total_frames = 50
            for _ in range(total_frames):
                world.step(8)
                d = vo.process_frame(cam.capture())
                if d is None:
                    lost_count += 1
            act.stop()

            # Should not lose tracking more than 20% of frames
            assert lost_count < total_frames * 0.2, \
                f"Lost tracking {lost_count}/{total_frames} frames"
        finally:
            world.close()


class TestSyntheticStereoInSim:
    """Test depth estimation from simulated stereo pairs."""

    def test_stereo_produces_observations(self):
        """Synthetic stereo should triangulate features in the simulated room."""
        world = SimWorld(floor_plan="simple_room", gui=False)
        cam = SimCamera(world)
        sens = SimSensors(world)
        act = SimActuators(world)
        fe = FeatureExtractor()
        stereo = SyntheticStereo(cam, fe, act, sens, baseline_m=0.05)

        try:
            obs = stereo.capture_and_triangulate()
            assert len(obs) > 5, f"Expected >5 stereo observations, got {len(obs)}"

            # Depths should be positive and reasonable for a 4m room
            depths = [o.depth for o in obs]
            assert min(depths) > 0.1, f"Min depth {min(depths):.2f} too small"
            assert max(depths) < 10.0, f"Max depth {max(depths):.2f} too large"
        finally:
            world.close()

    def test_stereo_depth_consistent_with_ultrasonic(self):
        """Median stereo depth should be roughly consistent with ultrasonic."""
        world = SimWorld(floor_plan="simple_room", gui=False)
        cam = SimCamera(world)
        sens = SimSensors(world)
        act = SimActuators(world)
        fe = FeatureExtractor()
        stereo = SyntheticStereo(cam, fe, act, sens, baseline_m=0.05)

        try:
            us_mm = sens.read_ultrasonic_mm()
            obs = stereo.capture_and_triangulate()

            if obs and us_mm > 0:
                median_depth = np.median([o.depth for o in obs])
                us_m = us_mm / 1000.0
                # Stereo median should be within 2x of ultrasonic
                # (they measure different things -- ultrasonic is forward-only,
                # stereo sees the whole scene)
                ratio = median_depth / us_m
                assert 0.2 < ratio < 5.0, \
                    f"Stereo/US ratio {ratio:.2f} out of range (stereo={median_depth:.2f}, US={us_m:.2f})"
        finally:
            world.close()


class TestEKFWithSimVO:
    """Test EKF integration with simulated VO inputs."""

    def test_ekf_bounded_error_forward(self):
        """EKF error should stay bounded during forward motion with VO input."""
        world = SimWorld(floor_plan="simple_room", gui=False)
        cam = SimCamera(world)
        act = SimActuators(world)
        fe = FeatureExtractor()
        vo = VisualOdometry(cam, fe)
        vo.scale = 0.005
        ekf = EKFSLAM(initial_pose=(0, 0, 0))

        try:
            vo.process_frame(cam.capture())
            act.move_forward(80)

            max_error = 0
            for _ in range(100):
                world.step(8)
                d = vo.process_frame(cam.capture())
                if d:
                    ekf.predict(d)

                gt = world.get_robot_pose()
                est = ekf.get_pose()
                err = math.sqrt((gt[0]-est[0])**2 + (gt[1]-est[1])**2)
                max_error = max(max_error, err)

            act.stop()
            # Error should not blow up (< 5m for a 4m room)
            assert max_error < 5.0, f"EKF error {max_error:.2f}m exceeded 5m"
        finally:
            world.close()


class TestMotionControllerInSim:
    """Test waypoint following in simulation."""

    def test_drive_to_waypoint_makes_progress(self):
        """Robot should move closer to the target over time."""
        world = SimWorld(floor_plan="simple_room", gui=False)
        act = SimActuators(world)
        mc = MotionController(act)

        try:
            target = (1.0, 0.0)
            initial_dist = 1.0

            for _ in range(100):
                world.step(8)
                gt = world.get_robot_pose()
                mc.drive_to_waypoint(gt, target)

            gt = world.get_robot_pose()
            final_dist = math.sqrt((target[0]-gt[0])**2 + (target[1]-gt[1])**2)
            assert final_dist < initial_dist * 0.7, \
                f"Should make progress: initial={initial_dist:.2f}, final={final_dist:.2f}"
        finally:
            world.close()

    def test_drive_to_angled_waypoint(self):
        """Robot should navigate to a waypoint requiring a turn."""
        world = SimWorld(floor_plan="simple_room", gui=False)
        act = SimActuators(world)
        mc = MotionController(act)

        try:
            target = (0.5, 0.5)  # 45 degree turn needed
            for _ in range(150):
                world.step(8)
                gt = world.get_robot_pose()
                mc.drive_to_waypoint(gt, target)

            gt = world.get_robot_pose()
            dist = math.sqrt((target[0]-gt[0])**2 + (target[1]-gt[1])**2)
            assert dist < 0.5, f"Should approach diagonal waypoint, dist={dist:.2f}m"
        finally:
            world.close()


class TestExplorationInSim:
    """Test frontier exploration in simulation."""

    def test_rotation_scan_discovers_space(self):
        """A rotation scan should populate the occupancy grid."""
        world = SimWorld(floor_plan="simple_room", gui=False)
        sens = SimSensors(world)
        act = SimActuators(world)
        mm = MapManager()

        try:
            # Do a rotation scan
            for _ in range(12):
                act.rotate_right(50)
                world.step(40)
                act.stop()
                world.step(10)
                gt = world.get_robot_pose()
                us = sens.read_ultrasonic_mm()
                if us > 0:
                    mm.occupancy.update_ultrasonic(gt[0], gt[1], gt[2], us/1000.0)
                    mm.occupancy.mark_traversed(gt[0], gt[1])

            # Grid should have discovered free space and walls
            prob = mm.occupancy.get_probability_grid()
            free_cells = np.sum(prob < 0.4)
            occupied_cells = np.sum(prob > 0.6)

            assert free_cells > 50, f"Should discover free space, got {free_cells} cells"
            assert occupied_cells > 5, f"Should discover walls, got {occupied_cells} cells"
        finally:
            world.close()

    def test_explorer_finds_frontiers_after_scan(self):
        """After scanning, the explorer should find viable frontiers."""
        world = SimWorld(floor_plan="simple_room", gui=False)
        sens = SimSensors(world)
        act = SimActuators(world)
        mm = MapManager()

        try:
            for _ in range(12):
                act.rotate_right(50)
                world.step(40)
                act.stop()
                world.step(10)
                gt = world.get_robot_pose()
                us = sens.read_ultrasonic_mm()
                if us > 0:
                    mm.occupancy.update_ultrasonic(gt[0], gt[1], gt[2], us/1000.0)
                    mm.occupancy.mark_traversed(gt[0], gt[1])

            explorer = Explorer(mm)
            gt = world.get_robot_pose()
            target = explorer.select_target(gt)

            assert target is not None, "Should find an exploration target"
            # Target should be within reasonable distance
            dist = math.sqrt(target[0]**2 + target[1]**2)
            assert dist < 5.0, f"Target too far: {dist:.2f}m"
        finally:
            world.close()
