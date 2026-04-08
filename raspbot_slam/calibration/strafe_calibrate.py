"""
Strafe distance calibration: maps (motor_speed, duration) to actual displacement.

Since the rover has no wheel encoders, we calibrate strafe distance empirically
using the ultrasonic sensor and/or visual odometry as ground truth.

Usage (on the Pi):
    python -m raspbot_slam.calibration.strafe_calibrate
"""

import json
import os
import sys
import time
import numpy as np

from .. import config


def calibrate_strafe(bot=None):
    """Interactive strafe calibration.

    Places the rover perpendicular to a wall, strafes at various speeds
    and durations, and uses ultrasonic readings to measure actual displacement.

    Args:
        bot: Raspbot driver instance. If None, tries to import.
    """
    # Lazy import of hardware
    from ..actuators import Actuators
    from ..sensors import Sensors

    actuators = Actuators(bot)
    sensors = Sensors(bot)
    sensors.enable_ultrasonic()

    speeds = [40, 60, 80, 100]
    durations_ms = [200, 300, 500, 800]

    print("Strafe Distance Calibration")
    print("=" * 50)
    print("Position the rover PARALLEL to a wall on its RIGHT side,")
    print("about 30-50 cm from the wall. The ultrasonic sensor should")
    print("NOT point at the wall (it faces forward).")
    print()
    print("We'll measure displacement visually or by marking the floor.")
    print("After each strafe, enter the measured distance in mm.")
    print("Press Enter to skip, 'q' to quit.\n")

    calibrations = []

    for speed in speeds:
        for duration_ms in durations_ms:
            duration_s = duration_ms / 1000.0

            print(f"  Speed={speed}, Duration={duration_ms}ms")
            input("  Press Enter when ready to strafe RIGHT...")

            # Strafe right
            actuators.move_right(speed)
            time.sleep(duration_s)
            actuators.stop()
            time.sleep(0.3)

            # Ask for measured distance
            measured = input("  Measured distance (mm), Enter to skip, 'q' to quit: ").strip()

            if measured.lower() == 'q':
                break

            if measured:
                try:
                    dist_mm = float(measured)
                    calibrations.append({
                        "speed": speed,
                        "duration_ms": duration_ms,
                        "distance_mm": dist_mm,
                    })
                    print(f"  Recorded: speed={speed}, duration={duration_ms}ms, "
                          f"distance={dist_mm}mm")
                except ValueError:
                    print("  Skipped (invalid number)")

            # Return to original position
            print("  Strafing LEFT to return...")
            actuators.move_left(speed)
            time.sleep(duration_s)
            actuators.stop()
            time.sleep(0.5)
            print()

        if measured and measured.lower() == 'q':
            break

    actuators.stop()

    if not calibrations:
        print("No calibrations recorded.")
        return

    # Compute statistics for repeated measurements at same speed
    # Group by speed
    by_speed = {}
    for c in calibrations:
        s = c["speed"]
        if s not in by_speed:
            by_speed[s] = []
        by_speed[s].append(c)

    # Compute mm-per-ms rate for each speed
    print("\nCalibration Results:")
    print("-" * 40)
    for speed, entries in sorted(by_speed.items()):
        rates = [e["distance_mm"] / e["duration_ms"] for e in entries]
        mean_rate = np.mean(rates)
        std_rate = np.std(rates) if len(rates) > 1 else 0
        print(f"  Speed {speed}: {mean_rate:.3f} mm/ms (std={std_rate:.3f})")
        for e in entries:
            e["rate_mm_per_ms"] = e["distance_mm"] / e["duration_ms"]

    # Save
    cal_dir = os.path.dirname(__file__)
    cal_file = os.path.join(cal_dir, "strafe_calibration.json")

    data = {
        "floor_surface": "unknown",
        "calibrations": calibrations,
    }
    with open(cal_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {cal_file}")


if __name__ == "__main__":
    calibrate_strafe()
