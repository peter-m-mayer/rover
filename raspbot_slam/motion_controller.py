"""
PID-controlled waypoint following with mecanum kinematics.

Translates high-level waypoint commands into motor speeds, using
heading PID and cross-track PID for drift correction. Visual odometry
provides the feedback (not encoders -- there are none).
"""

import math
from typing import Tuple, Optional

from . import config
from .actuators import Actuators


class PIDController:
    """Simple positional PID with anti-windup."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float = 255.0, integral_limit: float = 500.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit

        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error: float) -> float:
        """Compute PID output for the given error.

        Args:
            error: Current error (setpoint - measured).

        Returns:
            Control output, clamped to [-output_limit, output_limit].
        """
        # Proportional
        p = self.kp * error

        # Integral with anti-windup
        self._integral += error
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral))
        i = self.ki * self._integral

        # Derivative
        d = self.kd * (error - self._prev_error)
        self._prev_error = error

        output = p + i + d
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


class MotionController:
    """Waypoint-following controller using PID and mecanum kinematics.

    Usage:
        mc = MotionController(actuators)
        while not mc.drive_to_waypoint(current_pose, target_xy):
            time.sleep(0.1)  # or wait for next VO update
    """

    def __init__(self, actuators: Actuators):
        self._actuators = actuators

        self._heading_pid = PIDController(
            config.HEADING_PID_P,
            config.HEADING_PID_I,
            config.HEADING_PID_D,
            output_limit=100.0,
        )
        self._crosstrack_pid = PIDController(
            config.CROSSTRACK_PID_P,
            config.CROSSTRACK_PID_I,
            config.CROSSTRACK_PID_D,
            output_limit=50.0,
        )
        self._nav_speed = config.NAV_SPEED

    def drive_to_waypoint(self, current_pose: Tuple[float, float, float],
                          target_xy: Tuple[float, float]) -> bool:
        """One control step toward a waypoint.

        Args:
            current_pose: (x, y, theta) in world frame. theta in radians.
            target_xy: (x, y) target position in world frame.

        Returns:
            True if waypoint is reached (within tolerance).
        """
        cx, cy, ctheta = current_pose
        tx, ty = target_xy

        dx = tx - cx
        dy = ty - cy
        distance = math.sqrt(dx**2 + dy**2)

        # Check arrival
        if distance < config.WAYPOINT_TOLERANCE_M:
            self._actuators.stop()
            self._heading_pid.reset()
            self._crosstrack_pid.reset()
            return True

        # Desired heading to target
        desired_heading = math.atan2(dy, dx)
        heading_error = self._normalize_angle(desired_heading - ctheta)

        # If heading error is significant, rotate in place first
        if abs(heading_error) > math.radians(15):
            rotation_speed = min(60, max(25, int(abs(heading_error) * 30)))
            if heading_error > 0:
                self._actuators.rotate_left(rotation_speed)
            else:
                self._actuators.rotate_right(rotation_speed)
            return False

        # Heading is roughly correct — drive forward with slight corrections
        forward_speed = min(self._nav_speed, max(25, int(distance * 150)))

        # Small heading correction via differential steering
        steer = self._heading_pid.update(heading_error)
        steer = max(-30, min(30, steer))

        # Convert to deflection: 90=forward, +steer=left, -steer=right
        move_angle = 90.0 + steer * 0.5
        self._actuators.set_deflection(forward_speed, move_angle)

        return False

    def rotate_to_heading(self, current_theta: float, target_theta: float,
                          tolerance_deg: float = 5.0) -> bool:
        """Rotate in place toward a target heading.

        Args:
            current_theta: Current heading in radians.
            target_theta: Target heading in radians.
            tolerance_deg: Acceptable error in degrees.

        Returns:
            True if heading is within tolerance.
        """
        error = self._normalize_angle(target_theta - current_theta)

        if abs(error) < math.radians(tolerance_deg):
            self._actuators.stop()
            return True

        rotation_speed = int(self._heading_pid.update(error))
        rotation_speed = max(20, min(80, abs(rotation_speed)))

        if error > 0:
            self._actuators.rotate_left(rotation_speed)
        else:
            self._actuators.rotate_right(rotation_speed)
        return False

    def stop(self):
        """Emergency stop."""
        self._actuators.stop()
        self._heading_pid.reset()
        self._crosstrack_pid.reset()

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
