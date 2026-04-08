"""
Simulated sensors: ultrasonic (via PyBullet ray-cast) and line tracker.

Drop-in replacement for raspbot_slam.sensors.Sensors.
"""

import math
from typing import Tuple, Optional
import numpy as np
import pybullet as p

from .. import config
from .sim_world import SimWorld


class SimSensors:
    """Simulated sensor interface using PyBullet ray-casting.

    The ultrasonic sensor is modeled as a single ray cast along the
    rover's forward direction, matching the real hardware's behavior.
    """

    def __init__(self, world: SimWorld, ultrasonic_noise_mm: float = None):
        """Initialize simulated sensors.

        Args:
            world: SimWorld instance.
            ultrasonic_noise_mm: Gaussian noise std on ultrasonic (default: config).
        """
        self._world = world
        self._noise_mm = ultrasonic_noise_mm or config.ULTRASONIC_NOISE_MM
        self._ultrasonic_enabled = True
        self._rng = np.random.RandomState(42)

        # Ultrasonic sensor parameters
        self._us_max_range = 4.0  # meters
        self._us_offset_x = 0.09  # sensor mounted at front of rover

    def enable_ultrasonic(self):
        self._ultrasonic_enabled = True

    def disable_ultrasonic(self):
        self._ultrasonic_enabled = False

    def read_ultrasonic_mm(self) -> int:
        """Simulated ultrasonic distance reading via ray-cast.

        Returns:
            Distance in millimeters. -1 if disabled or no hit.
        """
        if not self._ultrasonic_enabled:
            return -1

        # Get rover pose
        pos, orn = p.getBasePositionAndOrientation(self._world.rover_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # Ray origin: front of rover
        ray_from = [
            pos[0] + self._us_offset_x * math.cos(yaw),
            pos[1] + self._us_offset_x * math.sin(yaw),
            pos[2] + 0.05,  # slight offset up from floor
        ]

        # Ray end: forward direction, max range
        ray_to = [
            ray_from[0] + self._us_max_range * math.cos(yaw),
            ray_from[1] + self._us_max_range * math.sin(yaw),
            ray_from[2],
        ]

        result = p.rayTest(ray_from, ray_to)
        if result and result[0][0] != -1:
            hit_fraction = result[0][2]
            distance_m = hit_fraction * self._us_max_range

            # Add realistic noise
            noise_m = self._rng.normal(0, self._noise_mm / 1000.0)
            distance_m = max(0.01, distance_m + noise_m)

            return int(distance_m * 1000)
        else:
            # No hit within range
            return int(self._us_max_range * 1000)

    def read_ultrasonic_m(self) -> float:
        """Distance in meters."""
        mm = self.read_ultrasonic_mm()
        return mm / 1000.0 if mm > 0 else -1.0

    def is_obstacle_ahead(self, threshold_mm: int = None) -> bool:
        threshold = threshold_mm or config.OBSTACLE_STOP_MM
        dist = self.read_ultrasonic_mm()
        return 0 < dist < threshold

    def read_line_tracker(self) -> Tuple[bool, bool, bool, bool]:
        """Simulated line tracker via downward ray-casts.

        Casts 4 rays downward from the rover's underside. Detects floor
        vs raised obstacles directly below the rover.
        """
        pos, orn = p.getBasePositionAndOrientation(self._world.rover_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # 4 sensor positions (front-left, front-right, rear-left, rear-right)
        offsets = [
            (0.07, 0.05),   # front-left
            (0.07, -0.05),  # front-right
            (-0.07, 0.05),  # rear-left
            (-0.07, -0.05), # rear-right
        ]

        results = []
        for ox, oy in offsets:
            # Transform to world frame
            wx = pos[0] + ox * math.cos(yaw) - oy * math.sin(yaw)
            wy = pos[1] + ox * math.sin(yaw) + oy * math.cos(yaw)

            ray_from = [wx, wy, pos[2]]
            ray_to = [wx, wy, pos[2] - 0.15]  # 15cm downward

            hit = p.rayTest(ray_from, ray_to)
            # If hit distance < threshold, line detected
            if hit and hit[0][0] != -1 and hit[0][0] != self._world.rover_id:
                results.append(hit[0][2] < 0.8)  # close to floor
            else:
                results.append(False)

        return tuple(results)

    def read_ir_remote(self) -> Optional[int]:
        """No IR remote in simulation."""
        return None
