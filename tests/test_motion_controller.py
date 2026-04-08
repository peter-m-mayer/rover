"""Tests for motion_controller.py -- PID and waypoint following."""

import math
import numpy as np
import pytest

from raspbot_slam.motion_controller import PIDController, MotionController
from raspbot_slam.actuators import Actuators


class TestPIDController:

    def test_proportional_response(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        output = pid.update(10.0)
        assert abs(output - 10.0) < 0.01

    def test_zero_error_zero_output(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        output = pid.update(0.0)
        assert abs(output) < 0.01

    def test_integral_accumulation(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        pid.update(1.0)
        pid.update(1.0)
        output = pid.update(1.0)
        # Integral of three 1.0 errors = 3.0
        assert abs(output - 3.0) < 0.01

    def test_anti_windup(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0, integral_limit=5.0)
        for _ in range(100):
            pid.update(10.0)
        # Integral should be clamped at 5.0
        output = pid.update(0.0)
        assert output <= 5.0

    def test_derivative_response(self):
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
        pid.update(0.0)
        output = pid.update(5.0)  # derivative = 5.0 - 0.0
        assert abs(output - 5.0) < 0.01

    def test_output_limit(self):
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit=50.0)
        output = pid.update(10.0)
        assert abs(output) <= 50.0

    def test_reset(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=1.0)
        pid.update(5.0)
        pid.update(5.0)
        pid.reset()
        output = pid.update(1.0)
        # After reset, integral=0 and prev_error=0
        # ki*1.0 + kd*(1.0 - 0) = 1.0 + 1.0 = 2.0
        assert abs(output - 2.0) < 0.01


class TestMotionController:

    def test_waypoint_reached(self):
        """When already at the waypoint, drive_to_waypoint returns True."""
        act = Actuators(bot=None)
        mc = MotionController(act)
        reached = mc.drive_to_waypoint((1.0, 2.0, 0.0), (1.0, 2.0))
        assert reached

    def test_waypoint_not_reached(self):
        """When far from waypoint, returns False."""
        act = Actuators(bot=None)
        mc = MotionController(act)
        reached = mc.drive_to_waypoint((0.0, 0.0, 0.0), (5.0, 5.0))
        assert not reached

    def test_heading_rotation_when_off_axis(self):
        """When heading error is large, robot should rotate in place."""
        act = Actuators(bot=None)
        mc = MotionController(act)
        # Robot at origin facing right (theta=0), target is behind (theta=pi)
        reached = mc.drive_to_waypoint((0, 0, 0), (-5, 0))
        assert not reached
        # Should be trying to rotate, not drive forward

    def test_rotate_to_heading_reached(self):
        act = Actuators(bot=None)
        mc = MotionController(act)
        # Already at target heading
        reached = mc.rotate_to_heading(1.0, 1.0)
        assert reached

    def test_rotate_to_heading_not_reached(self):
        act = Actuators(bot=None)
        mc = MotionController(act)
        reached = mc.rotate_to_heading(0.0, math.pi)
        assert not reached

    def test_stop(self):
        act = Actuators(bot=None)
        mc = MotionController(act)
        mc.drive_to_waypoint((0, 0, 0), (5, 5))
        mc.stop()  # should not error

    def test_normalize_angle(self):
        assert abs(MotionController._normalize_angle(0)) < 0.01
        assert abs(MotionController._normalize_angle(2 * math.pi)) < 0.01
        assert abs(MotionController._normalize_angle(-2 * math.pi)) < 0.01
        result = MotionController._normalize_angle(3 * math.pi)
        assert -math.pi <= result <= math.pi
