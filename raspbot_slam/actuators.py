"""
Actuator interfaces for RASPBOT-V2.

Wraps servo control, motor control (via mecanum kinematics), and LED status.
Provides mock implementations for off-robot development.
"""

import math
import time

from . import config


class Actuators:
    """Motor, servo, and LED control for the RASPBOT-V2."""

    def __init__(self, bot=None):
        """Initialize actuator interface.

        Args:
            bot: Raspbot driver instance. If None, uses mock (print-only).
        """
        self._bot = bot
        self._pan_angle = config.SERVO_PAN_CENTER
        self._tilt_angle = config.SERVO_TILT_REST

    # =========================================================================
    # Servo Control
    # =========================================================================

    def set_servo_pan(self, angle_deg: float):
        """Set camera pan servo angle.

        Args:
            angle_deg: 0-180 degrees. 90 = center/forward.
        """
        angle_deg = max(config.SERVO_PAN_MIN, min(config.SERVO_PAN_MAX, angle_deg))
        self._pan_angle = angle_deg
        if self._bot is not None:
            self._bot.Ctrl_Servo(config.SERVO_PAN_ID, int(angle_deg))

    def set_servo_tilt(self, angle_deg: float):
        """Set camera tilt servo angle.

        Args:
            angle_deg: 0-110 degrees. 25 = rest (slightly down). Higher = more up.
        """
        angle_deg = max(config.SERVO_TILT_MIN, min(config.SERVO_TILT_MAX, angle_deg))
        self._tilt_angle = angle_deg
        if self._bot is not None:
            self._bot.Ctrl_Servo(config.SERVO_TILT_ID, int(angle_deg))

    def center_camera(self):
        """Return camera to center pan and rest tilt."""
        self.set_servo_pan(config.SERVO_PAN_CENTER)
        self.set_servo_tilt(config.SERVO_TILT_REST)

    @property
    def pan_angle(self) -> float:
        return self._pan_angle

    @property
    def tilt_angle(self) -> float:
        return self._tilt_angle

    # =========================================================================
    # Motor Control (Mecanum Kinematics)
    # =========================================================================

    def _set_motors(self, l1: int, l2: int, r1: int, r2: int):
        """Set individual motor speeds. Negative = backward.

        Args:
            l1, l2, r1, r2: Motor speeds in range -255..255.
        """
        def clamp(v):
            return max(-config.MOTOR_SPEED_MAX, min(config.MOTOR_SPEED_MAX, int(v)))

        if self._bot is not None:
            self._bot.Ctrl_Muto(config.MOTOR_L1_ID, clamp(l1))
            self._bot.Ctrl_Muto(config.MOTOR_L2_ID, clamp(l2))
            self._bot.Ctrl_Muto(config.MOTOR_R1_ID, clamp(r1))
            self._bot.Ctrl_Muto(config.MOTOR_R2_ID, clamp(r2))

    def move_forward(self, speed: int):
        """Drive forward. Speed 0-255."""
        s = abs(speed)
        self._set_motors(s, s, s, s)

    def move_backward(self, speed: int):
        """Drive backward. Speed 0-255."""
        s = abs(speed)
        self._set_motors(-s, -s, -s, -s)

    def move_right(self, speed: int):
        """Strafe right (no rotation). Speed 0-255."""
        s = abs(speed)
        self._set_motors(s, -s, -s, s)

    def move_left(self, speed: int):
        """Strafe left (no rotation). Speed 0-255."""
        s = abs(speed)
        self._set_motors(-s, s, s, -s)

    def rotate_left(self, speed: int):
        """Rotate counter-clockwise in place. Speed 0-255."""
        s = abs(speed)
        self._set_motors(-s, -s, s, s)

    def rotate_right(self, speed: int):
        """Rotate clockwise in place. Speed 0-255."""
        s = abs(speed)
        self._set_motors(s, s, -s, -s)

    def set_deflection(self, speed: int, angle_deg: float):
        """Move in arbitrary direction using mecanum kinematics.

        Args:
            speed: 0-255 magnitude.
            angle_deg: Direction in degrees. 0=right, 90=forward, 180=left, 270=backward.
        """
        rad = math.radians(angle_deg)
        vx = speed * math.cos(rad)
        vy = speed * math.sin(rad)
        l1 = vy + vx
        l2 = vy - vx
        r1 = vy - vx
        r2 = vy + vx
        self._set_motors(int(l1), int(l2), int(r1), int(r2))

    def drive(self, forward: float, turn: float = 0.0, strafe: float = 0.0):
        """Mecanum drive: combine forward, in-place turn, and lateral strafe.

        Args:
            forward: Forward speed (+) / backward (-).
            turn: Rotation command. + = clockwise / turn right, - = left.
            strafe: Lateral speed. + = strafe right, - = strafe left.

        Component magnitudes add on the wheels, so |forward|+|turn|+|strafe|
        may exceed MOTOR_SPEED_MAX; _set_motors clamps each wheel.
        """
        l1 = forward + strafe + turn
        l2 = forward - strafe + turn
        r1 = forward - strafe - turn
        r2 = forward + strafe - turn
        self._set_motors(l1, l2, r1, r2)

    def stop(self):
        """Stop all motors immediately."""
        self._set_motors(0, 0, 0, 0)

    def strafe_right_timed(self, speed: int, duration_s: float):
        """Strafe right for a fixed duration, then stop.

        Args:
            speed: Motor speed 0-255.
            duration_s: Time in seconds.
        """
        self.move_right(speed)
        time.sleep(duration_s)
        self.stop()

    def strafe_left_timed(self, speed: int, duration_s: float):
        """Strafe left for a fixed duration, then stop."""
        self.move_left(speed)
        time.sleep(duration_s)
        self.stop()

    # =========================================================================
    # LED Status Indication
    # =========================================================================

    # WS2812B preset color *indices*. The vendor firmware exposes a fixed
    # palette via Ctrl_WQ2812_ALL(state, color) where `color` is an index
    # (NOT an RGB triplet). Indices 0-3 verified against the vendor
    # color_detection.py demo; 4-6 follow the standard Yahboom ordering.
    _LED_INDEX = {
        "red": 0, "green": 1, "blue": 2, "yellow": 3,
        "purple": 4, "cyan": 5, "white": 6,
        # SLAM status aliases -> nearest preset
        "mapping": 2,      # blue
        "localizing": 1,   # green
        "lost": 0,         # red
        "scanning": 3,     # yellow
    }

    # RGB of each preset, for snapping a requested RGB to the nearest index.
    _INDEX_RGB = {
        0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0),
        4: (128, 0, 128), 5: (0, 255, 255), 6: (255, 255, 255),
    }

    def set_led_color(self, color_name: str):
        """Set all 14 LEDs to a named color.

        Args:
            color_name: One of 'off', 'red', 'green', 'blue', 'yellow',
                        'purple', 'cyan', 'white', or a SLAM status alias
                        ('mapping', 'localizing', 'lost', 'scanning').
        """
        if self._bot is None:
            return
        if not color_name or color_name == "off":
            self._bot.Ctrl_WQ2812_ALL(0, 0)   # state 0 = off
            return
        idx = self._LED_INDEX.get(color_name)
        if idx is None:
            self._bot.Ctrl_WQ2812_ALL(0, 0)
            return
        self._bot.Ctrl_WQ2812_ALL(1, idx)     # state 1 = on, at palette index

    def set_led_rgb(self, r: int, g: int, b: int, led_id: int = 0):
        """Approximate an RGB color with the nearest hardware preset.

        The MCU palette is index-based (no free RGB), so we snap to the
        closest preset by Euclidean distance.

        Args:
            r, g, b: Desired color 0-255.
            led_id: 0 = all LEDs, 1-14 = individual LED.
        """
        if self._bot is None:
            return
        if (r, g, b) == (0, 0, 0):
            if led_id == 0:
                self._bot.Ctrl_WQ2812_ALL(0, 0)
            else:
                self._bot.Ctrl_WQ2812_Alone(led_id, 0, 0)
            return
        idx = min(self._INDEX_RGB, key=lambda i: sum(
            (a - c) ** 2 for a, c in zip(self._INDEX_RGB[i], (r, g, b))))
        if led_id == 0:
            self._bot.Ctrl_WQ2812_ALL(1, idx)
        else:
            self._bot.Ctrl_WQ2812_Alone(led_id, 1, idx)

    def set_led_brightness(self, brightness: int):
        """Set LED brightness 0-255 across all channels (best effort)."""
        if self._bot is not None:
            b = max(0, min(255, int(brightness)))
            self._bot.Ctrl_WQ2812_brightness_ALL(b, b, b)

    # =========================================================================
    # Buzzer
    # =========================================================================

    def buzzer_on(self):
        """Turn buzzer on."""
        if self._bot is not None:
            self._bot.Ctrl_BEEP_Switch(1)

    def buzzer_off(self):
        """Turn buzzer off."""
        if self._bot is not None:
            self._bot.Ctrl_BEEP_Switch(0)

    def beep(self, duration_s: float = 0.1):
        """Short beep for feedback."""
        self.buzzer_on()
        time.sleep(duration_s)
        self.buzzer_off()
