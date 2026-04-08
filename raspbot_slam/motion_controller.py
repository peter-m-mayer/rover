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

        # If heading error is large, rotate in place first
        if abs(heading_error) > math.radians(30):
            rotation_speed = int(self._heading_pid.update(heading_error))
            if rotation_speed > 0:
                self._actuators.rotate_left(min(abs(rotation_speed), 60))
            else:
                self._actuators.rotate_right(min(abs(rotation_speed), 60))
            return False

        # Cross-track error: perpendicular distance from the line robot→target
        # Positive = target is to the left
        cross_track = distance * math.sin(heading_error)

        # Heading correction
        heading_correction = self._heading_pid.update(heading_error)

        # Cross-track correction (lateral component)
        lateral_correction = self._crosstrack_pid.update(cross_track)

        # Combine into mecanum motion
        # Forward speed proportional to distance (slow down near target)
        forward_speed = min(self._nav_speed, int(distance * 200))
        forward_speed = max(20, forward_speed)

        # Convert to deflection angle
        # angle=90 is pure forward, lateral correction shifts it
        move_angle = 90.0 + math.degrees(math.atan2(lateral_correction, forward_speed))
        move_angle = max(45.0, min(135.0, move_angle))

        # Apply heading correction as differential rotation
        speed = int(math.sqrt(forward_speed**2 + lateral_correction**2))
        speed = min(speed, self._nav_speed)

        self._actuators.set_deflection(speed, move_angle)

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
