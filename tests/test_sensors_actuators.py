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


class _FlakyBus:
    """smbus stand-in: fails the first `fail` writes, then records."""

    def __init__(self, fail=0):
        self.fail, self.writes = fail, []

    def write_i2c_block_data(self, addr, reg, data):
        if self.fail > 0:
            self.fail -= 1
            raise OSError(121, "Remote I/O error")
        self.writes.append((addr, reg, list(data)))


class _FakeRaspbot:
    def __init__(self, fail=0):
        self._device, self._addr = _FlakyBus(fail), 0x2B


def test_motor_writes_retry_after_i2c_error():
    from raspbot_slam.actuators import Actuators
    bot = _FakeRaspbot(fail=2)
    act = Actuators(bot=bot)
    act.drive(-60)
    assert act.i2c_retries == 2
    assert [w[2] for w in bot._device.writes] == [[0, 1, 60], [1, 1, 60], [2, 1, 60], [3, 1, 60]]


def test_motor_write_raises_when_bus_stays_down():
    import pytest
    from raspbot_slam.actuators import Actuators, I2CWriteError
    act = Actuators(bot=_FakeRaspbot(fail=100))
    with pytest.raises(I2CWriteError):
        act.drive(50)


def test_stop_is_sent_twice_and_servo_uses_register_2():
    from raspbot_slam.actuators import Actuators
    bot = _FakeRaspbot()
    act = Actuators(bot=bot)
    act.stop()
    assert len(bot._device.writes) == 8 and all(w[2][2] == 0 for w in bot._device.writes)
    act.set_servo_pan(80)
    assert bot._device.writes[-1][1:] == (0x02, [1, 80])
