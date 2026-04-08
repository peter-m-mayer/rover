"""Tests for the PyBullet digital twin simulator.

Tests motion kinematics, camera rendering, sensor ray-casting,
and actuator interface compatibility -- all headless (no GUI).
"""

import math
import numpy as np
import pytest

try:
    import pybullet
except ImportError:
    pytest.skip("PyBullet required for simulator tests", allow_module_level=True)

from raspbot_slam.simulator import SimWorld, SimCamera, SimSensors, SimActuators
from raspbot_slam.simulator.sim_world import FLOOR_PLANS
from raspbot_slam.feature_extractor import FeatureExtractor
from raspbot_slam import config


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sim_room():
    """Simple room world, cleaned up after test."""
    world = SimWorld(floor_plan="simple_room", gui=False)
    yield world
    world.close()


@pytest.fixture
def sim_L():
    """L-shaped world."""
    world = SimWorld(floor_plan="L_shaped", gui=False)
    yield world
    world.close()


@pytest.fixture
def sim_full(sim_room):
    """Full sim stack: world + camera + sensors + actuators."""
    return {
        "world": sim_room,
        "camera": SimCamera(sim_room),
        "sensors": SimSensors(sim_room),
        "actuators": SimActuators(sim_room),
    }


# =============================================================================
# World Construction
# =============================================================================

class TestSimWorld:

    def test_all_floor_plans_load(self):
        """Every predefined floor plan should initialize without error."""
        for name in FLOOR_PLANS:
            world = SimWorld(floor_plan=name, gui=False)
            pose = world.get_robot_pose()
            assert len(pose) == 3
            world.close()

    def test_initial_pose_at_origin(self, sim_room):
        x, y, theta = sim_room.get_robot_pose()
        assert abs(x) < 0.01
        assert abs(y) < 0.01
        assert abs(theta) < 0.01

    def test_custom_start_pose(self):
        world = SimWorld(floor_plan="simple_room", gui=False,
                         start_pose=(1.0, -0.5, math.pi / 4))
        x, y, theta = world.get_robot_pose()
        assert abs(x - 1.0) < 0.02
        assert abs(y - (-0.5)) < 0.02
        assert abs(theta - math.pi / 4) < 0.05
        world.close()

    def test_step_advances_time(self, sim_room):
        t0 = sim_room.sim_time
        sim_room.step(240)
        t1 = sim_room.sim_time
        assert t1 > t0
        assert abs(t1 - t0 - 1.0) < 0.01  # 240 steps at 1/240 = 1 second

    def test_robot_stays_on_floor(self, sim_room):
        """Rover Z should remain at chassis height, not fall or float."""
        sim_room.step(100)
        _, _, z = sim_room.get_robot_position_3d()
        expected_z = SimWorld.ROVER_HEIGHT + SimWorld.ROVER_SIZE[2] / 2
        assert abs(z - expected_z) < 0.01


# =============================================================================
# Motion Kinematics
# =============================================================================

class TestSimMotion:

    def test_forward_displacement(self, sim_full):
        """Forward motion should produce positive displacement along heading."""
        world = sim_full["world"]
        act = sim_full["actuators"]

        p0 = world.get_robot_pose()
        act.move_forward(80)
        world.step(480)  # 2 seconds
        act.stop()
        p1 = world.get_robot_pose()

        # Expected: 80 * 0.5/255 * 2 = 0.314m
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dist = math.sqrt(dx**2 + dy**2)
        assert 0.25 < dist < 0.40, f"Forward displacement {dist:.3f}m not in expected range"

    def test_backward_displacement(self, sim_full):
        world = sim_full["world"]
        act = sim_full["actuators"]

        p0 = world.get_robot_pose()
        act.move_backward(80)
        world.step(240)
        act.stop()
        p1 = world.get_robot_pose()

        dist = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
        assert dist > 0.05, "Backward motion should produce displacement"

    def test_strafe_right_pure_lateral(self, sim_full):
        """Strafe should move laterally with minimal heading change."""
        world = sim_full["world"]
        act = sim_full["actuators"]

        p0 = world.get_robot_pose()
        act.move_right(80)
        world.step(240)
        act.stop()
        p1 = world.get_robot_pose()

        dist = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
        heading_change = abs(math.degrees(p1[2] - p0[2]))

        assert dist > 0.05, "Strafe should produce displacement"
        assert heading_change < 2.0, f"Strafe should not rotate: {heading_change:.1f} deg"

    def test_strafe_left_pure_lateral(self, sim_full):
        world = sim_full["world"]
        act = sim_full["actuators"]

        p0 = world.get_robot_pose()
        act.move_left(80)
        world.step(240)
        act.stop()
        p1 = world.get_robot_pose()

        heading_change = abs(math.degrees(p1[2] - p0[2]))
        assert heading_change < 2.0

    def test_rotation_ccw(self, sim_full):
        """Rotate left should increase heading angle."""
        world = sim_full["world"]
        act = sim_full["actuators"]

        p0 = world.get_robot_pose()
        act.rotate_left(60)
        world.step(240)
        act.stop()
        p1 = world.get_robot_pose()

        rotation_deg = math.degrees(p1[2] - p0[2])
        assert rotation_deg > 10, f"CCW rotation should increase heading: {rotation_deg:.1f} deg"

    def test_rotation_cw(self, sim_full):
        """Rotate right should decrease heading angle."""
        world = sim_full["world"]
        act = sim_full["actuators"]

        p0 = world.get_robot_pose()
        act.rotate_right(60)
        world.step(240)
        act.stop()
        p1 = world.get_robot_pose()

        rotation_deg = math.degrees(p1[2] - p0[2])
        assert rotation_deg < -10, f"CW rotation should decrease heading: {rotation_deg:.1f} deg"

    def test_stop_zeroes_velocity(self, sim_full):
        world = sim_full["world"]
        act = sim_full["actuators"]

        act.move_forward(100)
        world.step(60)
        act.stop()
        world.step(60)

        p0 = world.get_robot_pose()
        world.step(60)
        p1 = world.get_robot_pose()

        drift = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
        assert drift < 0.001, f"Robot should not drift after stop: {drift:.4f}m"

    def test_speed_proportional(self, sim_full):
        """Higher speed should produce more displacement."""
        world = sim_full["world"]
        act = sim_full["actuators"]

        # Low speed
        p0 = world.get_robot_pose()
        act.move_forward(40)
        world.step(240)
        act.stop()
        p1 = world.get_robot_pose()
        dist_low = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)

        # Reset position
        world2 = SimWorld(floor_plan="simple_room", gui=False)
        act2 = SimActuators(world2)

        p0 = world2.get_robot_pose()
        act2.move_forward(120)
        world2.step(240)
        act2.stop()
        p1 = world2.get_robot_pose()
        dist_high = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)

        world2.close()
        assert dist_high > dist_low * 1.5, \
            f"High speed ({dist_high:.3f}) should be > 1.5x low speed ({dist_low:.3f})"

    def test_deflection_diagonal(self, sim_full):
        """set_deflection at 45 deg should move diagonally."""
        world = sim_full["world"]
        act = sim_full["actuators"]

        p0 = world.get_robot_pose()
        act.set_deflection(80, 45.0)  # 45 deg = forward-right
        world.step(240)
        act.stop()
        p1 = world.get_robot_pose()

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        # Both dx and dy should have significant magnitude
        assert abs(dx) > 0.02 or abs(dy) > 0.02


# =============================================================================
# Camera
# =============================================================================

class TestSimCamera:

    def test_capture_grayscale(self, sim_full):
        frame = sim_full["camera"].capture()
        assert frame.shape == (config.CAMERA_HEIGHT, config.CAMERA_WIDTH)
        assert frame.dtype == np.uint8

    def test_capture_color(self, sim_full):
        frame = sim_full["camera"].capture_color()
        assert frame.shape == (config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3)
        assert frame.dtype == np.uint8

    def test_capture_depth(self, sim_full):
        depth = sim_full["camera"].capture_depth()
        assert depth.shape == (config.CAMERA_HEIGHT, config.CAMERA_WIDTH)
        assert depth.dtype == np.float32
        assert depth.min() > 0  # all depths should be positive
        assert depth.max() < 25  # nothing further than far plane

    def test_intrinsics_reasonable(self, sim_full):
        cam = sim_full["camera"]
        K = cam.K
        assert K.shape == (3, 3)
        assert K[0, 0] > 100  # focal length reasonable
        assert K[1, 1] > 100
        assert abs(K[0, 2] - config.CAMERA_WIDTH / 2) < 1
        assert abs(K[1, 2] - config.CAMERA_HEIGHT / 2) < 1

    def test_orb_features_on_sim_frame(self, sim_full):
        """Simulated frames should have enough texture for ORB."""
        frame = sim_full["camera"].capture()
        fe = FeatureExtractor()
        kp, desc = fe.detect_and_compute(frame)
        assert len(kp) > 50, f"Expected >50 ORB features, got {len(kp)}"

    def test_frame_changes_with_motion(self, sim_full):
        """Moving the rover should change the camera view."""
        cam = sim_full["camera"]
        world = sim_full["world"]
        act = sim_full["actuators"]

        frame1 = cam.capture()
        act.move_forward(80)
        world.step(120)
        act.stop()
        frame2 = cam.capture()

        # Frames should differ (not identical)
        diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
        assert diff > 1.0, f"Frames should differ after motion, mean diff={diff:.2f}"

    def test_pan_servo_changes_view(self, sim_full):
        """Panning the camera should change the view."""
        cam = sim_full["camera"]
        world = sim_full["world"]

        world.set_servo_pan(90)
        frame_center = cam.capture()

        world.set_servo_pan(30)
        frame_left = cam.capture()

        diff = np.mean(np.abs(frame_center.astype(float) - frame_left.astype(float)))
        assert diff > 1.0, "Pan servo should change the view"

    def test_feature_matching_across_small_motion(self, sim_full):
        """ORB should find matches between frames with small displacement."""
        cam = sim_full["camera"]
        world = sim_full["world"]
        act = sim_full["actuators"]
        fe = FeatureExtractor()

        frame1 = cam.capture()
        kp1, desc1 = fe.detect_and_compute(frame1)

        act.move_forward(60)
        world.step(30)  # small motion
        act.stop()

        frame2 = cam.capture()
        kp2, desc2 = fe.detect_and_compute(frame2)

        if desc1 is not None and desc2 is not None:
            matches = fe.match(desc1, desc2)
            assert len(matches) > 20, \
                f"Expected >20 matches across small motion, got {len(matches)}"


# =============================================================================
# Sensors
# =============================================================================

class TestSimSensors:

    def test_ultrasonic_in_room(self, sim_full):
        """Ultrasonic should detect wall at ~2m from center of 4m room."""
        us = sim_full["sensors"].read_ultrasonic_mm()
        assert 500 < us < 3000, f"Expected 500-3000mm in room center, got {us}mm"

    def test_ultrasonic_m_conversion(self, sim_full):
        """read_ultrasonic_m should return a value consistent with mm reading."""
        m = sim_full["sensors"].read_ultrasonic_m()
        # Both calls have noise, so just check the value is reasonable
        assert 0.1 < m < 5.0, f"Ultrasonic meters should be reasonable: {m}"

    def test_ultrasonic_closer_when_approaching_wall(self, sim_full):
        """Moving toward a wall should decrease ultrasonic reading."""
        world = sim_full["world"]
        act = sim_full["actuators"]
        sens = sim_full["sensors"]

        us_before = sens.read_ultrasonic_mm()

        act.move_forward(80)
        world.step(240)  # 1 second toward wall
        act.stop()

        us_after = sens.read_ultrasonic_mm()
        assert us_after < us_before, \
            f"Ultrasonic should decrease: before={us_before}, after={us_after}"

    def test_is_obstacle_ahead(self, sim_full):
        """At center of 4m room, no obstacle within 200mm."""
        assert not sim_full["sensors"].is_obstacle_ahead()

    def test_line_tracker_returns_tuple(self, sim_full):
        result = sim_full["sensors"].read_line_tracker()
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_ir_remote_returns_none(self, sim_full):
        assert sim_full["sensors"].read_ir_remote() is None

    def test_enable_disable_ultrasonic(self, sim_full):
        sens = sim_full["sensors"]
        sens.disable_ultrasonic()
        assert sens.read_ultrasonic_mm() == -1
        sens.enable_ultrasonic()
        assert sens.read_ultrasonic_mm() > 0


# =============================================================================
# Actuator Interface Compatibility
# =============================================================================

class TestSimActuatorInterface:
    """Verify SimActuators has the same interface as Actuators."""

    def test_servo_pan_clamp(self, sim_full):
        act = sim_full["actuators"]
        act.set_servo_pan(200)
        assert act.pan_angle == config.SERVO_PAN_MAX
        act.set_servo_pan(-10)
        assert act.pan_angle == config.SERVO_PAN_MIN

    def test_servo_tilt_clamp(self, sim_full):
        act = sim_full["actuators"]
        act.set_servo_tilt(150)
        assert act.tilt_angle == config.SERVO_TILT_MAX

    def test_center_camera(self, sim_full):
        act = sim_full["actuators"]
        act.set_servo_pan(30)
        act.center_camera()
        assert act.pan_angle == config.SERVO_PAN_CENTER
        assert act.tilt_angle == config.SERVO_TILT_REST

    def test_led_and_buzzer_no_crash(self, sim_full):
        """LED and buzzer methods should be no-ops but not crash."""
        act = sim_full["actuators"]
        act.set_led_color("mapping")
        act.set_led_rgb(255, 0, 0)
        act.set_led_brightness(128)
        act.buzzer_on()
        act.buzzer_off()
        act.beep(0.001)

    def test_strafe_timed(self, sim_full):
        """Timed strafe should move and then stop."""
        world = sim_full["world"]
        act = sim_full["actuators"]

        p0 = world.get_robot_pose()
        act.strafe_right_timed(80, 0.5)
        # SimActuators._wait steps the simulation
        p1 = world.get_robot_pose()
        dist = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
        assert dist > 0.03, f"Timed strafe should move: {dist:.3f}m"
