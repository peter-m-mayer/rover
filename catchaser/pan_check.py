"""Determine the pan-servo direction sign (config.CHASE_PAN_SIGN) on hardware.

Pan tracking must steer the camera TOWARD the cat. Whether "increase the pan
angle" turns the camera left or right is a wiring/mounting detail that differs
between units, so we measure it: pan between two angles, see which way the
scene shifts in the image, and recommend the sign.

    python3 -m catchaser.pan_check      # on the robot; prints the recommended sign

Convention (matches the simulator): increasing the pan angle turns the camera
LEFT, so a static scene shifts RIGHT in the frame -> CHASE_PAN_SIGN = +1.
"""

import argparse
import sys
import time

import numpy as np

from raspbot_slam import config


def horizontal_shift(gray_a, gray_b) -> float:
    """Sub-pixel horizontal shift of the scene from a to b (+ = moved right)."""
    import cv2
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, _dy), _resp = cv2.phaseCorrelate(a * win, b * win)
    return float(dx)


def recommend_sign(shift_per_pan_increase: float) -> int:
    """+1 if increasing pan turns the camera left (scene shifts right)."""
    return 1 if shift_per_pan_increase > 0 else -1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pan-servo direction sign check")
    ap.add_argument("--delta", type=float, default=25.0,
                    help="degrees each side of center to pan (default 25)")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args(argv)

    import cv2
    from .hw import make_hardware
    bot, camera, sensors, actuators = make_hardware()
    if bot is None:
        print("[pan_check] not on hardware (mock mode) — run this on the robot.",
              file=sys.stderr)
        return 2

    lo = config.SERVO_PAN_CENTER - args.delta
    hi = config.SERVO_PAN_CENTER + args.delta

    def gray_at(angle):
        actuators.set_servo_pan(angle)
        time.sleep(0.5)                       # let the servo settle
        return cv2.cvtColor(camera.capture_color(), cv2.COLOR_BGR2GRAY)

    shifts = []
    try:
        for _ in range(args.samples):
            a = gray_at(lo)
            b = gray_at(hi)
            shifts.append(horizontal_shift(a, b))
        actuators.set_servo_pan(config.SERVO_PAN_CENTER)
    finally:
        camera.close()

    med = float(np.median(shifts))
    sign = recommend_sign(med)
    print(f"[pan_check] pan {lo:.0f} -> {hi:.0f} deg")
    print(f"[pan_check] scene shift (median): {med:+.1f}px   samples="
          f"{[round(s, 1) for s in shifts]}")
    print(f"[pan_check] => camera turns {'LEFT' if med > 0 else 'RIGHT'} "
          f"as pan angle increases")
    print(f"[pan_check] RECOMMENDED  CHASE_PAN_SIGN = {sign}   "
          f"(current config: {config.CHASE_PAN_SIGN})")
    if abs(med) < 2.0:
        print("[pan_check] WARNING: tiny shift — low texture or servo not moving; "
              "point at a detailed scene and retry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
