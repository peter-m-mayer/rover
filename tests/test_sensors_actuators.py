"""Tests for sensors.py and actuators.py -- mock hardware."""

import math
import numpy as np

from raspbot_slam.sensors import Sensors
from raspbot_slam.actuators import Actuators
from raspbot_slam import config


# =============================================================================
# Sensors (mock mode)
# =============================================================================

class TestSensorsMock:

    def test_ultrasonic_mock_returns_1000(self):
        s = Sensors(bot=None)
        assert s.read_ultrasonic_mm() == 1000

    def test_ultrasonic_m_conversion(self):
        s = Sensors(bot=None)
        assert s.read_ultrasonic_m() == 1.0

    def test_obstacle_not_ahead_mock(self):
        """Mock returns 1000mm, well above 200mm threshold."""
        s = Sensors(bot=None)
        assert not s.is_obstacle_ahead()

    def test_obstacle_ahead_custom_threshold(self):
        s = Sensors(bot=None)
        # Mock returns 1000mm. Threshold at 2000mm should trigger.
        assert s.is_obstacle_ahead(threshold_mm=2000)

    def test_line_tracker_mock(self):
        s = Sensors(bot=None)
        result = s.read_line_tracker()
        assert result == (False, False, False, False)
        assert len(result) == 4

    def test_ir_remote_mock(self):
        s = Sensors(bot=None)
        assert s.read_ir_remote() is None

    def test_enable_disable_ultrasonic(self):
        s = Sensors(bot=None)
        s.enable_ultrasonic()
        assert s._ultrasonic_enabled
        s.disable_ultrasonic()
        assert not s._ultrasonic_enabled


# =============================================================================
# Actuators (mock mode)
# =============================================================================

class TestActuatorsMock:

    def test_servo_pan_clamp(self):
        a = Actuators(bot=None)
        a.set_servo_pan(200)  # exceeds max 180
        assert a.pan_angle == config.SERVO_PAN_MAX

        a.set_servo_pan(-10)  # below min 0
        assert a.pan_angle == config.SERVO_PAN_MIN

    def test_servo_tilt_clamp(self):
        a = Actuators(bot=None)
        a.set_servo_tilt(150)  # exceeds max 110
        assert a.tilt_angle == config.SERVO_TILT_MAX

    def test_center_camera(self):
        a = Actuators(bot=None)
        a.set_servo_pan(30)
        a.set_servo_tilt(90)
        a.center_camera()
        assert a.pan_angle == config.SERVO_PAN_CENTER
        assert a.tilt_angle == config.SERVO_TILT_REST

    def test_set_deflection_forward(self):
        """Deflection at 90 degrees should be pure forward."""
        a = Actuators(bot=None)
        # Just verify it doesn't crash in mock mode
        a.set_deflection(100, 90.0)

    def test_set_deflection_right(self):
        """Deflection at 0 degrees should be pure right strafe."""
        a = Actuators(bot=None)
        a.set_deflection(100, 0.0)

    def test_stop_no_error(self):
        a = Actuators(bot=None)
        a.move_forward(100)
        a.stop()  # should not error

    def test_led_color_names(self):
        a = Actuators(bot=None)
        for color in ["mapping", "localizing", "lost", "scanning", "off"]:
            a.set_led_color(color)  # should not error

    def test_beep_no_error(self):
        """Beep should be a no-op with no hardware."""
        a = Actuators(bot=None)
        a.beep(0.001)  # tiny duration for test speed
