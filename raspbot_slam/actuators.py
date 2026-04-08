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

    # WS2812B color presets matching Raspbot_Lib conventions
    _LED_COLORS = {
        "off":     (0, 0, 0),
        "red":     (255, 0, 0),
        "green":   (0, 255, 0),
        "blue":    (0, 0, 255),
        "yellow":  (255, 255, 0),
        "purple":  (128, 0, 128),
        "cyan":    (0, 255, 255),
        "white":   (255, 255, 255),
        "mapping": (0, 0, 128),      # dim blue
        "localizing": (0, 128, 0),   # dim green
        "lost":    (128, 0, 0),      # dim red
        "scanning": (128, 128, 0),   # dim yellow
    }

    def set_led_color(self, color_name: str):
        """Set all 14 LEDs to a named color.

        Args:
            color_name: One of 'off', 'red', 'green', 'blue', 'yellow',
                        'mapping', 'localizing', 'lost', 'scanning', etc.
        """
        r, g, b = self._LED_COLORS.get(color_name, (0, 0, 0))
        self.set_led_rgb(r, g, b)

    def set_led_rgb(self, r: int, g: int, b: int, led_id: int = 0):
        """Set LED color directly.

        Args:
            r, g, b: Color values 0-255.
            led_id: 0 = all LEDs, 1-14 = individual LED.
        """
        if self._bot is not None:
            self._bot.Ctrl_WS2812B(led_id, r, g, b)

    def set_led_brightness(self, brightness: int):
        """Set LED brightness.

        Args:
            brightness: 0-255.
        """
        if self._bot is not None:
            self._bot.Ctrl_WS2812B_Brig(brightness)

    # =========================================================================
    # Buzzer
    # =========================================================================

    def buzzer_on(self):
        """Turn buzzer on."""
        if self._bot is not None:
            self._bot.Ctrl_Buzzer(1)

    def buzzer_off(self):
        """Turn buzzer off."""
        if self._bot is not None:
            self._bot.Ctrl_Buzzer(0)

    def beep(self, duration_s: float = 0.1):
        """Short beep for feedback."""
        self.buzzer_on()
        time.sleep(duration_s)
        self.buzzer_off()
