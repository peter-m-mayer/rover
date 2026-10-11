"""
Self-test against the surveyed tags (AUTONOMY_PLAN.md M1/M2 scoring):
navigate by dead reckoning alone to a standing point in front of each tag,
then compare where dead reckoning thinks the robot is with a tag fix.

    python3 -m raspbot_slam.tag_tour [--route 3,2,1,0,4,3] [--standoff 0.8] [--correct]

Without --correct the dead-reckoning pose is never reset, so the errors at the
stops are the accumulated drift; with it, each fix resets the pose (per-leg
error). Every fix and dead-reckoning pose is logged as an event; the run is
recorded for offline mapping in room coordinates (offline/build_map.py).
"""

import argparse
import json
import math
import time

import numpy as np

from . import config
from .calibration import auto_camera_cal as acc
from .floor_odometry import FloorModel
from .free_space import FreeSpaceScanner
from .heading import GyroHeading, drive_straight, turn_to, wrap180
from .m1_run import load_intrinsics, ultrasonic_mm
from .online_odometry import OnlineOdometry
from .recorder import RunRecorder
from .tag_localizer import FACING, IN, TagLocalizer, load_survey, robot_heading
from .wander import camera_clear_m

HOP_S = 0.5


def standpoint(survey, tag_id, standoff, wall_margin=0.45):
    """Where to stand to look at a tag: `standoff` m out along its normal, kept
    `wall_margin` m inside the main room rectangle (tag 0 sits 11 cm from the
    alcove/mirror corner). Returns (point, heading that faces the tag)."""
    t = survey["markers"][str(tag_id)]
    n = np.array(FACING[t["facing"]][:2], float)
    tag = np.array([t["x"] * IN, t["y"] * IN])
    p = tag + standoff * n
    p[0] = np.clip(p[0], wall_margin, survey["xmax_in"] * IN - wall_margin)
    p[1] = np.clip(p[1], wall_margin, survey["ymax_in"] * IN - wall_margin)
    face = math.degrees(math.atan2(tag[1] - p[1], tag[0] - p[0]))
    return p, face


def centre_pan(rig):
    """Pan to centre approaching from below (the pan has 3-7 deg of backlash)."""
    c0 = config.SERVO_PAN_CENTER
    rig.servo(c0 - 10, config.SERVO_TILT_LEVEL, settle=0.3)
    rig.servo(c0, config.SERVO_TILT_LEVEL, settle=0.5)


def tag_fix(rig, loc, frames=5, turns=(0, 35, -70, 105, -140)):
    """Median tag pose with the pan CENTRED, turning the body (gyro) until a
    surveyed tag is in view. The pan isn't trusted for fixes: on 2026-10-11 the
    camera head jammed under a desk and the servo angle no longer said where
    the camera pointed. Returns the fix and the net body turn made (deg)."""
    from .heading import turn_to
    net = 0.0
    for t in turns:
        if t:
            turn_to(rig.act, rig.hd, rig.hd.heading_deg + t)
            net += t
        centre_pan(rig)
        ests = [e for e in (loc.localize(rig.gray()) for _ in range(frames)) if e is not None]
        if ests:
            med = {k: float(np.median([e[k] for e in ests])) for k in ("x", "y", "z", "yaw", "range_m")}
            med["heading"] = med["yaw"]                  # pan centred: body = camera
            med.update(tags=ests[0]["tags"], n=len(ests), turned=net)
            return med
    return None


def moved_since(odo, x0, y0):
    x, y, _ = odo.pose
    return math.hypot(x - x0, y - y0)


def go_to(rig, odo, fs, goal, stop_m=0.45, tol=0.15, max_hops=60):
    """Heading-held hops toward goal (room x, y); detour around obstacles.

    Stuck detection: a hop that should cover ~0.14 m but moves < 3 cm (floor
    odometry) twice in a row means the robot is pushing against something its
    sensors don't see (on 2026-10-11, a desk's low ring base for 48 s): back
    off and detour."""
    detours, stalls = 0, 0
    for hop in range(max_hops):
        x, y, h = odo.pose
        dx, dy = goal[0] - x, goal[1] - y
        dist = math.hypot(dx, dy)
        if dist < tol:
            return True
        bearing = math.degrees(math.atan2(dy, dx))
        if abs(wrap180(bearing - h)) > 8:
            turn_to(rig.act, rig.hd, rig.hd.heading_deg + wrap180(bearing - h))
        us = ultrasonic_mm(rig, 3) / 1000
        frame = rig.capture_color()
        if us > 0.5:                     # the patch 0.22-0.40 m ahead is floor: keep learning it
            fs.learn(frame)
        cam = camera_clear_m(fs, rig)
        blocked = min(v for v in (us, cam, 9.9) if v == v)
        if blocked < min(stop_m, dist + 0.1) or stalls >= 2:
            detours += 1
            if detours > 4:
                print(f"    blocked {detours} times: giving up on this leg", flush=True)
                return False
            if stalls >= 2:
                print("    stuck (driving but not moving): backing off", flush=True)
                rig.rec_mark("stuck")
                drive_straight(rig.act, rig.hd, -50, 0.8)
                stalls = 0
            side = 1 if detours % 2 else -1
            print(f"    obstacle at {blocked:.2f} m: detour {'left' if side > 0 else 'right'}", flush=True)
            turn_to(rig.act, rig.hd, rig.hd.heading_deg + side * 50)
            if ultrasonic_mm(rig, 3) > 700:
                for _ in range(3):
                    x0, y0, _ = odo.pose
                    drive_straight(rig.act, rig.hd, 50, HOP_S)
                    time.sleep(0.3)
                    if moved_since(odo, x0, y0) < 0.03:
                        print("    stuck during detour: backing off", flush=True)
                        drive_straight(rig.act, rig.hd, -50, 0.8)
                        break
            continue
        x0, y0, _ = odo.pose
        drive_straight(rig.act, rig.hd, 50, min(HOP_S, max(0.2, dist / 0.28)))
        time.sleep(0.3)
        stalls = stalls + 1 if moved_since(odo, x0, y0) < 0.03 else 0
    return False


def main(argv=None):
    p = argparse.ArgumentParser(description="Dead-reckoning self-test against the surveyed tags.")
    p.add_argument("--route", default="3,2,1,0,4,3")
    p.add_argument("--standoff", type=float, default=0.8)
    p.add_argument("--correct", action="store_true", help="reset the pose at every fix")
    args = p.parse_args(argv)

    K, dist = load_intrinsics()
    loc = TagLocalizer(K, dist)
    survey = load_survey("office")
    model = FloorModel(K, dist, config.CAMERA_HEIGHT_M, config.CAMERA_PITCH_DEG)
    fs = FreeSpaceScanner(model, max_range_m=1.6)
    rig = acc.Rig("/tmp", acc.TagTarget())
    rec = RunRecorder(camera=rig, imu=rig.imu, sensors=rig.sens, actuators=rig.act, label="tagtour")
    rec.bus_lock = rig.bus_lock
    rig.rec_mark = rec.mark
    results = []
    with rec:
        try:
            fix = tag_fix(rig, loc, turns=(0,) + (45,) * 7)      # full circle for the anchor
            if fix is None:
                print("no tag in view at the start: can't anchor")
                return 1
            rec.mark("fix " + json.dumps(dict(fix, label="start", ok=True)))
            print(f"start: x {fix['x']:.2f} y {fix['y']:.2f} heading {fix['heading']:+.0f} "
                  f"(tags {fix['tags']})", flush=True)
            if ultrasonic_mm(rig) > 500:
                fs.learn(rig.capture_color())
            odo = OnlineOdometry(rec, rig.hd, model).start(fix["x"], fix["y"], fix["heading"])
            t0 = time.monotonic()
            for leg, tag in enumerate(int(v) for v in args.route.split(",")):
                goal, face = standpoint(survey, tag, args.standoff)
                rec.mark(f"move leg {leg} to tag {tag}")
                print(f"leg {leg}: to tag {tag} standpoint ({goal[0]:.2f}, {goal[1]:.2f})", flush=True)
                reached = go_to(rig, odo, fs, goal)
                x, y, h = odo.pose
                turn_to(rig.act, rig.hd, rig.hd.heading_deg + wrap180(face - h))
                rec.mark(f"end leg {leg} to tag {tag}")
                time.sleep(0.4)
                x, y, h = odo.pose
                fix = tag_fix(rig, loc)
                row = {"leg": leg, "tag": tag, "reached": reached, "dr": [x, y, h],
                       "t": time.monotonic() - t0, "fix": fix}
                if fix is not None:
                    row["err_m"] = math.hypot(x - fix["x"], y - fix["y"])
                    row["err_deg"] = wrap180(h - fix["heading"])
                    print(f"  DR ({x:.2f}, {y:.2f}, {h:+.0f})  fix ({fix['x']:.2f}, {fix['y']:.2f}, "
                          f"{fix['heading']:+.0f}) tags {fix['tags']} -> error {row['err_m']:.2f} m, "
                          f"{row['err_deg']:+.1f} deg", flush=True)
                    if args.correct:
                        odo.set_pose(fix["x"], fix["y"], fix["heading"])
                else:
                    print(f"  DR ({x:.2f}, {y:.2f}, {h:+.0f}); no tag fix here", flush=True)
                rec.mark("tourstop " + json.dumps(row))
                results.append(row)
            odo.stop()
        except KeyboardInterrupt:
            print("interrupted")
        finally:
            rig.act.stop()
            rig.hd.stop()
    rig.servo(config.SERVO_PAN_CENTER, config.SERVO_TILT_REST, settle=0.2)
    rig.cap.release()
    with open(rec.path + "/tour.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"done -> {rec.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
