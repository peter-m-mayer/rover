"""
Sensor interfaces for RASPBOT-V2.

Wraps ultrasonic distance sensor and line tracking IR sensors.
Provides a mock implementation for off-robot development.
"""

import time
from typing import Tuple, Optional

from . import config


class Sensors:
    """Read-only sensor interface for the RASPBOT-V2."""

    def __init__(self, bot=None):
        """Initialize sensor interface.

        Args:
            bot: Raspbot driver instance. If None, uses mock readings.
        """
        self._bot = bot
        self._ultrasonic_enabled = False

    def enable_ultrasonic(self):
        """Power on the ultrasonic sensor. Must wait ~100ms before first read."""
        if self._bot is not None:
            self._bot.Ctrl_Ulatist_Switch(1)
            time.sleep(0.1)
        self._ultrasonic_enabled = True

    def disable_ultrasonic(self):
        """Power off the ultrasonic sensor to save power."""
        if self._bot is not None:
            self._bot.Ctrl_Ulatist_Switch(0)
        self._ultrasonic_enabled = False

    def read_ultrasonic_mm(self) -> int:
        """Read distance from ultrasonic sensor.

        Returns:
            Distance in millimeters. Returns -1 if sensor is off or read fails.
        """
        if self._bot is None:
            return 1000  # mock: 1 meter
        if not self._ultrasonic_enabled:
            self.enable_ultrasonic()
        try:
            high = self._bot.read_data_array(0x1B, 1)[0]
            low = self._bot.read_data_array(0x1A, 1)[0]
            return (high << 8) | low
        except Exception:
            return -1

    def read_ultrasonic_m(self) -> float:
        """Read distance from ultrasonic sensor in meters.

        Returns:
            Distance in meters. Returns -1.0 on failure.
        """
        mm = self.read_ultrasonic_mm()
        if mm < 0:
            return -1.0
        return mm / 1000.0

    def is_obstacle_ahead(self, threshold_mm: int = None) -> bool:
        """Check if an obstacle is within stopping distance.

        Args:
            threshold_mm: Distance threshold. Defaults to config.OBSTACLE_STOP_MM.
        """
        threshold = threshold_mm or config.OBSTACLE_STOP_MM
        dist = self.read_ultrasonic_mm()
        return 0 < dist < threshold

    def read_line_tracker(self) -> Tuple[bool, bool, bool, bool]:
        """Read the 4 line-tracking IR sensors.

        Returns:
            Tuple of 4 booleans (sensor1, sensor2, sensor3, sensor4).
            True = line detected (dark surface), False = no line (light surface).
        """
        if self._bot is None:
            return (False, False, False, False)
        try:
            data = self._bot.read_data_array(0x0A, 1)
            track = int(data[0])
            s1 = bool((track >> 3) & 0x01)
            s2 = bool((track >> 2) & 0x01)
            s3 = bool((track >> 1) & 0x01)
            s4 = bool(track & 0x01)
            return (s1, s2, s3, s4)
        except Exception:
            return (False, False, False, False)

    def read_ir_remote(self) -> Optional[int]:
        """Read IR remote key code.

        Returns:
            Key code integer, or None if no key pressed.
        """
        if self._bot is None:
            return None
        try:
            data = self._bot.read_data_array(0x0C, 1)
            code = int(data[0])
            return code if code != 0 else None
        except Exception:
            return None
