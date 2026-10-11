"""
Reactive wander for data collection (AUTONOMY_PLAN.md M2): no map, no plan --
look around, drive at the most open direction, stop before anything, repeat.
Everything is recorded for offline mapping (offline/build_map.py).

    python3 -m raspbot_slam.wander --seconds 180

Safety: forward only (the camera and sonar both look forward); heading-held
hops of ~9 cm; stops when the sonar reads < --stop m or the camera free-space
scan says the floor ahead ends within --stop m; every leg is capped at --leg m.
"""

import argparse
import json
import math
import random
import time

import numpy as np

from . import config
from .calibration import auto_camera_cal as acc
from .floor_odometry import FloorModel
from .free_space import FreeSpaceScanner
from .heading import GyroHeading, drive_straight, turn_to, wrap180
from .m1_run import load_intrinsics, ultrasonic_mm
from .recorder import RunRecorder
from .tag_localizer import TagLocalizer, robot_heading


def camera_clear_m(fs, rig):
    """Median free distance straight ahead (centre +-8 deg) from the camera."""
    frame = rig.capture_color()
    res = fs.scan(frame)
    if res is None:
        return float("nan")
    b, r, _ = res
    centre = np.abs(np.degrees(b)) < 8
    return float(np.median(r[centre])) if centre.any() else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description="Reactive wander with recording.")
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--stop", type=float, default=0.45, help="stop distance (m)")
    p.add_argument("--leg", type=float, default=1.5, help="max distance per leg (m)")
    p.add_argument("--speed", type=int, default=50)
    args = p.parse_args(argv)

    K, dist = load_intrinsics()
    loc = TagLocalizer(K, dist)
    model = FloorModel(K, dist, config.CAMERA_HEIGHT_M, config.CAMERA_PITCH_DEG)
    fs = FreeSpaceScanner(model, max_range_m=1.6)
    rig = acc.Rig("/tmp", acc.TagTarget())
    rig.servo(config.SERVO_PAN_CENTER, config.SERVO_TILT_LEVEL, settle=0.4)
    rec = RunRecorder(camera=rig, imu=rig.imu, sensors=rig.sens, actuators=rig.act, label="wander")
    rec.bus_lock = rig.bus_lock
    hd = GyroHeading(rig.imu).start()
    t_end = time.monotonic() + args.seconds
    legs, last_dir = 0, None
    with rec:
        try:
            while time.monotonic() < t_end:
                # look around: 8 headings, sonar + camera, tag fix if any
                rec.mark("look")
                looks = []
                for k in range(8):
                    target = hd.heading_deg if k == 0 else looks[0][0] + 45 * k
                    if k:
                        turn_to(rig.act, hd, target)
                    time.sleep(0.3)
                    us = ultrasonic_mm(rig) / 1000
                    frame = rig.capture_color()
                    if us > 0.7:
                        fs.learn(frame)
                    cam = camera_clear_m(fs, rig)
                    est = loc.localize(rig.gray())
                    if est is not None:
                        est["heading"] = robot_heading(est["yaw"], rig.pan)
                        rec.mark("fix " + json.dumps(dict(est, ok=True, gyro=hd.heading_deg)))
                    h = hd.heading_deg
                    clear = np.nanmin([us if us == us else 3.0, cam if cam == cam else 3.0])
                    looks.append((h, clear))
                    rec.mark(f"scan {h:.1f} sonar {us:.3f} cam {cam:.3f}")
                # choose: open, and not straight back where we came from
                ok = [(h, c) for h, c in looks if c > args.stop + 0.3]
                if last_dir is not None:
                    ok = [(h, c) for h, c in ok if abs(wrap180(h - (last_dir + 180))) > 50] or ok
                if not ok:
                    print("boxed in: nothing open, stopping", flush=True)
                    break
                ok.sort(key=lambda hc: -hc[1])
                h, c = random.choice(ok[:3])
                turn_to(rig.act, hd, h)
                legs += 1
                rec.mark(f"move leg {legs}")
                print(f"leg {legs}: heading {h:.0f}, clear {c:.2f} m", flush=True)
                driven = 0.0
                while driven < args.leg and time.monotonic() < t_end:
                    us = ultrasonic_mm(rig, 3) / 1000
                    cam = camera_clear_m(fs, rig)
                    if (us == us and us < args.stop) or (cam == cam and cam < args.stop):
                        print(f"  stop: sonar {us:.2f} camera {cam:.2f}", flush=True)
                        break
                    drive_straight(rig.act, hd, args.speed, 0.5, target_deg=h)
                    time.sleep(0.3)
                    driven += 0.09
                rec.mark(f"end leg {legs}")
                last_dir = h
        except KeyboardInterrupt:
            print("interrupted")
        finally:
            rig.act.stop()
            hd.stop()
    rig.servo(config.SERVO_PAN_CENTER, config.SERVO_TILT_REST, settle=0.2)
    rig.cap.release()
    print(f"done: {legs} legs -> {rec.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
