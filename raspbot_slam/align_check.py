"""
Camera-vs-body alignment check from a short straight drive.

    python3 -m raspbot_slam.align_check [--pan 74] [--seconds 1.5]

Drive straight (gyro heading hold), track the floor with the camera at the
given pan, and report the direction of travel *as the camera sees it*. With
the camera aligned that is 0 deg; a non-zero angle is how far the camera
points away from the body axis (+ = camera points right of travel).

Why: on 2026-10-11 the camera head jammed under a desk and the pan horn
slipped ~38 deg on its spline. Tags and gyro still agreed with each other, but
floor odometry then measured motion in a rotated frame and dead reckoning
drifted out of the room. Run this at the start of a session (it needs ~0.6 m
of clear floor ahead) or after any bump.
"""

import argparse
import math
import time

import numpy as np

from . import config
from .floor_odometry import FloorModel, FloorOdometry
from .heading import GyroHeading


def measure(rig, model, pan=None, seconds=1.5, speed=50):
    """Returns (angle_deg, forward_m, frames_used); angle + = camera points right of travel."""
    import cv2
    from .heading import wrap180
    pan = config.SERVO_PAN_CENTER if pan is None else pan
    rig.servo(pan - 10, config.SERVO_TILT_LEVEL, settle=0.3)     # approach from below (backlash)
    rig.servo(pan, config.SERVO_TILT_LEVEL, settle=0.6)
    fo = FloorOdometry(model)
    hd = rig.hd
    target = hd.heading_deg
    fo.step(rig.gray())
    acc, n, last_h = np.zeros(2), 0, hd.heading_deg
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            err = wrap180(hd.heading_deg - target)
            rig.act.drive(speed, turn=float(np.clip(2.5 * err, -30, 30)))
            with rig.cap_lock:                       # every frame: no gaps to track across
                ok, f = rig.cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            h = hd.heading_deg
            r = fo.step(g, math.radians(h - last_h))
            last_h = h
            if r is not None:
                acc += r[:2]
                n += 1
    finally:
        rig.act.stop()
    # travel seen at -38 deg (to the right) = camera points 38 deg LEFT of travel = -38
    return math.degrees(math.atan2(acc[1], acc[0])), float(acc[0]), n


def main(argv=None):
    from .calibration import auto_camera_cal as acc
    from .m1_run import load_intrinsics, ultrasonic_mm
    p = argparse.ArgumentParser(description="Camera-vs-body alignment from a straight drive.")
    p.add_argument("--pan", type=float, default=None)
    p.add_argument("--seconds", type=float, default=1.5)
    args = p.parse_args(argv)
    K, dist = load_intrinsics()
    model = FloorModel(K, dist, config.CAMERA_HEIGHT_M, config.CAMERA_PITCH_DEG)
    rig = acc.Rig("/tmp")
    try:
        clear = ultrasonic_mm(rig)
        if clear < 900:
            print(f"need ~0.9 m clear ahead (sonar {clear:.0f} mm): turn the robot toward open floor")
            return 1
        angle, fwd, n = measure(rig, model, args.pan, args.seconds)
        verdict = "OK" if abs(angle) <= 5 else "MISALIGNED"
        side = "right" if angle > 0 else "left"
        print(f"camera points {abs(angle):.1f} deg {side} of travel ({fwd:.2f} m, {n} frames): {verdict}")
        if verdict != "OK":
            print("  check the pan horn screw / re-seat the camera bracket straight ahead")
        return 0 if verdict == "OK" else 2
    finally:
        rig.act.stop()
        rig.hd.stop()
        rig.servo(config.SERVO_PAN_CENTER, config.SERVO_TILT_REST, settle=0.2)
        rig.cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
