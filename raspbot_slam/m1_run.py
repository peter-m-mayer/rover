"""
M1 odometry test run (AUTONOMY_PLAN.md): scripted moves in front of a wall tag,
fully recorded, with a tag fix (ground-truth pose) at every stop.

    python3 -m raspbot_slam.m1_run [--tag 1] [--back 1.0] [--strafe 0.5]

Sequence (the robot starts anywhere it can see a surveyed tag within ~2 m):
  1. face the tag, fix
  2. safety look: turn 180, read the ultrasonic, turn back (refuse to reverse
     into anything closer than --back + 0.3 m)
  3. reverse --back m in hops, fix at each stop; drive forward back to start
  4. strafe left --strafe m, fix; back; right, fix; back
  5. spin 360 deg in place (gyro), fix

The run folder (datasets/slam_runs/<stamp>_m1) holds the raw streams; every
fix is an event "fix {json}" with the tag pose, so odometry methods can be
scored offline against it without moving the robot again.
"""

import argparse
import json
import math
import time

import numpy as np

from . import config
from .calibration import auto_camera_cal as acc
from .recorder import RunRecorder
from .tag_localizer import TagLocalizer, robot_heading


def load_intrinsics():
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration",
                        "camera_calibration.json")
    with open(path) as f:
        cal = json.load(f)
    return np.array(cal["camera_matrix"]), np.array(cal["dist_coeffs"])


class M1:
    def __init__(self, rig, rec, loc, tag_id):
        self.rig, self.rec, self.loc, self.tag_id = rig, rec, loc, tag_id
        self.fixes = []

    def fix(self, label, frames=5):
        """Median tag pose over a few frames, pan centred; logged as an event."""
        r = self.rig
        r.servo(config.SERVO_PAN_CENTER, r.tilt, settle=0.6)
        ests = []
        for _ in range(frames):
            est = self.loc.localize(r.gray())
            if est is not None:
                ests.append(est)
        if not ests:
            c, _ = acc.centre_in_image(r)         # pan to it, then fix off-centre
            for _ in range(frames):
                est = self.loc.localize(r.gray()) if c is not None else None
                if est is not None:
                    ests.append(est)
        if not ests:
            print(f"  fix [{label}]: no tag", flush=True)
            self.rec.mark(f"fix {json.dumps({'label': label, 'ok': False})}")
            return None
        med = {k: float(np.median([e[k] for e in ests])) for k in ("x", "y", "z", "yaw", "pitch")}
        med["heading"] = robot_heading(med["yaw"], r.pan)
        med.update(label=label, ok=True, pan=r.pan, n=len(ests), tags=ests[0]["tags"],
                   spread_m=float(np.std([e["x"] for e in ests]) + np.std([e["y"] for e in ests])))
        self.rec.mark("fix " + json.dumps(med))
        self.fixes.append(med)
        print(f"  fix [{label}]: x {med['x']:.3f} y {med['y']:.3f} heading {med['heading']:+.1f} "
              f"(tags {med['tags']}, n {len(ests)})", flush=True)
        return med

    def travel(self, label, forward=0.0, strafe=0.0, metres=0.5, speed=50, hop_s=0.5):
        """Hops in one direction until the tag fixes say `metres` were covered."""
        start = self.fixes[-1] if self.fixes and self.fixes[-1]["ok"] else self.fix(label + " start")
        if start is None:
            return
        self.rec.mark(f"move {label}")
        for hop in range(30):
            if forward > 0 and self.rig.clear_mm() < acc.STOP_MM + 50:
                print("  ultrasonic: stopping", flush=True)
                break
            self.rig.act.drive(speed * forward, turn=0, strafe=speed * strafe)
            time.sleep(hop_s)
            self.rig.act.stop()
            time.sleep(0.4)
            now = self.fix(f"{label} hop {hop}", frames=3)
            if now is None:
                print("  lost the tag: stopping this move", flush=True)
                break
            moved = math.hypot(now["x"] - start["x"], now["y"] - start["y"])
            if moved >= metres - 0.03:
                break
        self.rec.mark(f"end {label}")


def ultrasonic_mm(rig, n=5):
    vals = [rig.clear_mm() for _ in range(n)]
    vals = [v for v in vals if 20 < v < 4000]
    return float(np.median(vals)) if vals else float("nan")


def open_direction(rig, rec, step=30):
    """Turn a full circle in `step` deg increments, reading the ultrasonic;
    face the most open heading whose wall is still in sonar range (< 3.5 m)."""
    rig.servo(config.SERVO_PAN_CENTER, config.SERVO_TILT_LEVEL, settle=0.3)
    heading, scan = 0.0, []
    for _ in range(360 // step):
        d = ultrasonic_mm(rig)
        scan.append((heading, d))
        rec.mark(f"scan {heading:.1f} {d:.0f}")
        heading += rig.rotate(step, speed=60)
    ok = [(h, d) for h, d in scan if 1200 < d < 3500]
    if not ok:
        ok = [(h, d) for h, d in scan if d == d]
    best_h, best_d = max(ok, key=lambda hd: hd[1])
    turn = (best_h - heading + 180) % 360 - 180
    rig.rotate(turn, speed=60)
    print("  scan: " + ", ".join(f"{h:.0f}:{d / 1000:.2f}" for h, d in scan), flush=True)
    return best_d


def line_test(m, rig, rec, stop_m=0.45, hop_s=0.5, speed=50):
    """Forward in heading-held hops toward a wall: holding the heading keeps the
    sonar on the same spot, so its range drops by exactly the distance driven
    (ground truth). Then reverse the same number of hops, also heading-held."""
    from .heading import GyroHeading, drive_straight, turn_to
    d0 = open_direction(rig, rec)
    hd = GyroHeading(rig.imu).start()
    try:
        home = hd.heading_deg
        print(f"  facing open floor: wall at {d0 / 1000:.2f} m", flush=True)
        m.fix("line start")
        rec.mark(f"truth range_mm {ultrasonic_mm(rig):.0f} heading {hd.heading_deg:.2f}")
        rec.mark("move forward line")
        hops = 0
        for hops in range(1, 40):
            if ultrasonic_mm(rig, 3) < (stop_m + 0.15) * 1000:
                hops -= 1
                break
            drive_straight(rig.act, hd, speed, hop_s, target_deg=home)
            time.sleep(0.4)
            d = ultrasonic_mm(rig)
            rec.mark(f"truth range_mm {d:.0f} heading {hd.heading_deg:.2f}")
            print(f"  forward hop {hops}: wall at {d / 1000:.3f} m, heading {hd.heading_deg - home:+.1f}",
                  flush=True)
        rec.mark("end forward line")
        m.fix("line far end")
        rec.mark("move reverse line")
        for h in range(hops):
            drive_straight(rig.act, hd, -speed, hop_s, target_deg=home)
            time.sleep(0.4)
            d = ultrasonic_mm(rig)
            rec.mark(f"truth range_mm {d:.0f} heading {hd.heading_deg:.2f}")
            print(f"  reverse hop {h + 1}: wall at {d / 1000:.3f} m, heading {hd.heading_deg - home:+.1f}",
                  flush=True)
        rec.mark("end reverse line")
        m.fix("line back")
        rec.mark("move spin")
        err = turn_to(rig.act, hd, home + 180)
        err2 = turn_to(rig.act, hd, home + 360)
        rec.mark(f"end spin gyro {hd.heading_deg - home:.1f}")
        rec.mark(f"truth range_mm {ultrasonic_mm(rig):.0f} heading {hd.heading_deg:.2f}")
        print(f"  spin: heading error after 360 = {err2:+.1f} deg; wall at {ultrasonic_mm(rig) / 1000:.3f} m",
              flush=True)
    finally:
        hd.stop()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--tag", type=int, default=None, help="tag to work in front of (default: nearest)")
    p.add_argument("--back", type=float, default=1.0)
    p.add_argument("--strafe", type=float, default=0.5)
    p.add_argument("--mode", choices=("tag", "line"), default="tag",
                   help="tag: moves in front of a wall tag; line: drive at the most open "
                        "wall with the ultrasonic as ground truth")
    args = p.parse_args(argv)

    K, dist = load_intrinsics()
    loc = TagLocalizer(K, dist)
    target = acc.TagTarget([args.tag] if args.tag is not None else None)
    rig = acc.Rig("/tmp", target)
    rec = RunRecorder(camera=rig, imu=rig.imu, sensors=rig.sens, actuators=rig.act, label=f"m1{args.mode}")
    rec.bus_lock = rig.bus_lock
    m = M1(rig, rec, loc, args.tag)
    t0 = time.monotonic()
    with rec:
        if args.mode == "line":
            try:
                line_test(m, rig, rec)
            except KeyboardInterrupt:
                print("interrupted")
            finally:
                rig.act.stop()
            rig.servo(config.SERVO_PAN_CENTER, config.SERVO_TILT_REST, settle=0.2)
            rig.cap.release()
            print(f"done in {time.monotonic() - t0:.0f} s -> {rec.path}")
            return 0
        try:
            if not acc.search(rig) or not acc.face_board(rig):
                print("no tag to work in front of")
                return 1
            print(f"working in front of tag {target.locked}; run folder {rec.path}", flush=True)
            home = m.fix("home")
            if home is None:
                return 1

            # 2. look behind before reversing
            rig.servo(config.SERVO_PAN_CENTER, rig.tilt, settle=0.2)
            rec.mark("safety look")
            rig.rotate(180)
            time.sleep(0.3)
            behind = min(rig.clear_mm() for _ in range(3))
            rig.rotate(-180)
            acc.local_search(rig)
            acc.face_board(rig)
            m.fix("after safety look")
            back = args.back if behind > (args.back + 0.3) * 1000 else max(0.0, behind / 1000 - 0.4)
            print(f"  clear behind: {behind} mm -> reversing {back:.2f} m", flush=True)

            # 3. back and forth
            if back >= 0.2:
                m.travel("reverse", forward=-1, metres=back)
                m.travel("forward", forward=1, metres=back)
                acc.face_board(rig)
                m.fix("back home")

            # 4. strafes (camera stays on the tag)
            for name, side in (("strafe L", -1), ("strafe R back", 1), ("strafe R", 1), ("strafe L back", -1)):
                m.travel(name, strafe=side, metres=args.strafe, speed=65)
            acc.face_board(rig)
            m.fix("after strafes")

            # 5. spin
            rec.mark("move spin")
            rig.servo(config.SERVO_PAN_CENTER, rig.tilt, settle=0.2)
            turned = rig.rotate(350, speed=60, timeout=15)
            rec.mark(f"end spin gyro {turned:.1f}")
            acc.local_search(rig)
            acc.face_board(rig)
            m.fix("after spin")
        except KeyboardInterrupt:
            print("interrupted")
        finally:
            rig.act.stop()
    rig.servo(config.SERVO_PAN_CENTER, config.SERVO_TILT_REST, settle=0.2)
    rig.cap.release()
    with open(rec.path + "/fixes.json", "w") as f:
        json.dump(m.fixes, f, indent=1)
    print(f"done in {time.monotonic() - t0:.0f} s: {len(m.fixes)} fixes -> {rec.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
