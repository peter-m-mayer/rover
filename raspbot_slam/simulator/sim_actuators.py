"""
Simulated actuators: motors, servos, LEDs via PyBullet physics.

Drop-in replacement for raspbot_slam.actuators.Actuators.
Translates motor commands into PyBullet forces on the rover body.
"""

import math
import time

from .. import config
from .sim_world import SimWorld


class SimActuators:
    """Simulated motor/servo/LED control backed by PyBullet.

    Implements the same interface as actuators.Actuators, but drives
    the simulated rover instead of I2C hardware.
    """

    def __init__(self, world: SimWorld, realtime: bool = False):
        """Initialize simulated actuators.

        Args:
            world: SimWorld instance.
            realtime: If True, sleep during timed operations for real-time
                      pacing. If False, advance simulation steps instead.
        """
        self._world = world
        self._realtime = realtime
        self._steps_per_second = int(1.0 / world._time_step)

    # =========================================================================
    # Servo Control
    # =========================================================================

    def set_servo_pan(self, angle_deg: float):
        angle_deg = max(config.SERVO_PAN_MIN, min(config.SERVO_PAN_MAX, angle_deg))
        self._world.set_servo_pan(angle_deg)

    def set_servo_tilt(self, angle_deg: float):
        angle_deg = max(config.SERVO_TILT_MIN, min(config.SERVO_TILT_MAX, angle_deg))
        self._world.set_servo_tilt(angle_deg)

    def center_camera(self):
        self.set_servo_pan(config.SERVO_PAN_CENTER)
        self.set_servo_tilt(config.SERVO_TILT_REST)

    @property
    def pan_angle(self) -> float:
        return self._world.pan_angle

    @property
    def tilt_angle(self) -> float:
        return self._world.tilt_angle

    # =========================================================================
    # Motor Control
    # =========================================================================

    def _set_motors(self, l1: int, l2: int, r1: int, r2: int):
        def clamp(v):
            return max(-config.MOTOR_SPEED_MAX, min(config.MOTOR_SPEED_MAX, int(v)))
        self._world.set_motor_speeds(clamp(l1), clamp(l2), clamp(r1), clamp(r2))

    def move_forward(self, speed: int):
        s = abs(speed)
        self._set_motors(s, s, s, s)

    def move_backward(self, speed: int):
        s = abs(speed)
        self._set_motors(-s, -s, -s, -s)

    def move_right(self, speed: int):
        s = abs(speed)
        self._set_motors(s, -s, -s, s)

    def move_left(self, speed: int):
        s = abs(speed)
        self._set_motors(-s, s, s, -s)

    def rotate_left(self, speed: int):
        s = abs(speed)
        self._set_motors(-s, -s, s, s)

    def rotate_right(self, speed: int):
        s = abs(speed)
        self._set_motors(s, s, -s, -s)

    def set_deflection(self, speed: int, angle_deg: float):
        rad = math.radians(angle_deg)
        vx = speed * math.cos(rad)
        vy = speed * math.sin(rad)
        l1 = vy + vx
        l2 = vy - vx
        r1 = vy - vx
        r2 = vy + vx
        self._set_motors(int(l1), int(l2), int(r1), int(r2))

    def stop(self):
        self._set_motors(0, 0, 0, 0)

    def strafe_right_timed(self, speed: int, duration_s: float):
        self.move_right(speed)
        self._wait(duration_s)
        self.stop()

    def strafe_left_timed(self, speed: int, duration_s: float):
        self.move_left(speed)
        self._wait(duration_s)
        self.stop()

    # =========================================================================
    # LEDs & Buzzer (visual feedback only in GUI mode)
    # =========================================================================

    def set_led_color(self, color_name: str):
        # In GUI mode, could change rover visual color. For now, no-op.
        pass

    def set_led_rgb(self, r: int, g: int, b: int, led_id: int = 0):
        pass

    def set_led_brightness(self, brightness: int):
        pass

    def buzzer_on(self):
        pass

    def buzzer_off(self):
        pass

    def beep(self, duration_s: float = 0.1):
        pass

    # =========================================================================
    # Time Management
    # =========================================================================

    def _wait(self, duration_s: float):
        """Wait by either sleeping (realtime) or stepping simulation."""
        if self._realtime:
            time.sleep(duration_s)
        else:
            n_steps = int(duration_s * self._steps_per_second)
            self._world.step(n_steps)
