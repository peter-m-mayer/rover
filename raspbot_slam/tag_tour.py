"""
Self-test against the surveyed tags (AUTONOMY_PLAN.md M1/M2 scoring; NAV plan v2):
navigate on the room map to a viewpoint of each tag, then compare where dead
reckoning thinks the robot is with a tag fix.

    python3 -m raspbot_slam.tag_tour [--route 3,2,1,0,4,3] [--correct]

v2 (2026-10-12): viewpoints come from the map (planner.tag_viewpoint: free,
reachable, line of sight) instead of a fixed 0.8 m in front of each tag, and
driving is nav_executor.Navigator (map + A* + contact handling) instead of
look-ahead-and-detour. Without --correct, fixes are checked (and logged) but
never move the pose, so the error at each stop is accumulated drift; the
heading watchdog still re-anchors a heading that disagrees with three fixes in
a row (logged as heading_reset). The run is recorded; the map is saved in the
run folder (room_map.npz, room_map.png).
"""

import argparse
import json
import math
import os
import time

import numpy as np

from . import config
from .calibration import auto_camera_cal as acc
from .floor_odometry import FloorModel
from .heading import turn_to, wrap180
from .m1_run import load_intrinsics
from .nav_executor import Navigator, install_stop_handlers
from .planner import tag_viewpoint
from .recorder import RunRecorder
from .tag_localizer import IN, TagLocalizer, load_survey


ANCHOR_FAR_RANGE_M = 2.0      # second circle: a start anchor may use a farther tag


def anchor_fix(nav, step=45):
    """Turn the body (pan centred) until a tag gives a fix; full circle at most,
    then a second circle accepting tags out to ANCHOR_FAR_RANGE_M (a run left
    the robot 1.6 m from tag 2 and the next one couldn't start, 2026-10-12)."""
    for max_range in (None, ANCHOR_FAR_RANGE_M):
        for k in range(360 // step):
            f = nav.fix(max_range=max_range)
            if f is not None:
                return f
            turn_to(nav.rig.act, nav.rig.hd, nav.rig.hd.heading_deg + step)
    return None


def fix_at_stop(nav, survey, tag, face, step_m=0.3, min_range_m=0.7):
    """Tag fix at a tour stop, trying harder than one look: as planned, then
    turned +-20 deg, then a hop closer (2026-10-12 baseline: tags 2 and 3 were
    seen at 1.57 m, just past TAG_MAX_RANGE_M, and tag 1 was out of view).
    Returns (fix or None, the dead-reckoning pose just before the fix)."""
    pose = nav.pose
    fix = nav.fix()
    for off in (20, -20):
        if fix is not None:
            break
        nav.turn_to_heading(face + off)
        pose = nav.pose
        fix = nav.fix()
    if fix is None:
        m = survey["markers"][str(tag)]
        tx, ty = m["x"] * IN, m["y"] * IN
        x, y, _ = nav.pose
        d = math.hypot(tx - x, ty - y)
        if d > min_range_m + step_m:
            nav.mark(f"no fix of tag {tag} at {d:.2f} m: moving {step_m} m closer")
            nav.go_to((x + step_m * (tx - x) / d, y + step_m * (ty - y) / d), tol=0.08, max_hops=6)
            nav.turn_to_heading(math.degrees(math.atan2(ty - y, tx - x)))
            pose = nav.pose
            fix = nav.fix()
    return fix, pose


def main(argv=None):
    p = argparse.ArgumentParser(description="Dead-reckoning self-test against the surveyed tags.")
    p.add_argument("--route", default="3,2,1,0,4,3")
    p.add_argument("--correct", action="store_true", help="apply tag fixes to the pose")
    p.add_argument("--prior", default=None, help="start from a saved map (offline/merge_runs.py output)")
    args = p.parse_args(argv)
    install_stop_handlers()

    K, dist = load_intrinsics()
    tagloc = TagLocalizer(K, dist)
    survey = load_survey("office")
    model = FloorModel(K, dist, config.CAMERA_HEIGHT_M, config.CAMERA_PITCH_DEG)
    rig = acc.Rig("/tmp", acc.TagTarget())
    rec = RunRecorder(camera=rig, imu=rig.imu, sensors=rig.sens, actuators=rig.act, label="tagtour")
    rec.bus_lock = rig.bus_lock
    nav = Navigator(rig, rec, model, survey, tagloc, correct=args.correct, prior=args.prior)
    results = []
    with rec:
        try:
            fix = anchor_fix(nav)
            if fix is None:
                print("no tag in view at the start: can't anchor")
                return 1
            rec.mark("fix " + json.dumps(dict(fix, label="start", ok=True)))
            print(f"start: x {fix['x']:.2f} y {fix['y']:.2f} heading {fix['heading']:+.0f} "
                  f"(tags {fix['tags']})", flush=True)
            nav.start(fix)
            t0 = time.monotonic()
            for leg, tag in enumerate(int(v) for v in args.route.split(",")):
                reached, vp = False, None
                for attempt in range(3):         # the map grows en route: re-pick an unreachable viewpoint
                    x, y, _ = nav.pose
                    nav.sense()
                    vp = tag_viewpoint(nav.map, survey, tag, (x, y))
                    if vp is None:
                        vp = tag_viewpoint(nav.map, survey, tag, (x, y), relaxed=True)
                    if vp is None:
                        break
                    goal, face, _ = vp
                    rec.mark(f"move leg {leg} to tag {tag}")
                    print(f"leg {leg}: to tag {tag} viewpoint ({goal[0]:.2f}, {goal[1]:.2f})", flush=True)
                    reached = nav.go_to(goal)
                    if reached or nav.why != "no path":
                        break
                if nav.why == "wedged":
                    rec.mark("tourstop " + json.dumps({"leg": leg, "tag": tag, "reached": False, "fix": None,
                                                        "why": "wedged"}))
                    print(f"leg {leg}: wedged; ending the tour (needs a hand)", flush=True)
                    break
                if vp is None:
                    print(f"leg {leg}: no reachable viewpoint for tag {tag}", flush=True)
                    rec.mark(f"tourstop " + json.dumps({"leg": leg, "tag": tag, "reached": False, "fix": None,
                                                         "why": "no viewpoint"}))
                    continue
                nav.turn_to_heading(face)
                rec.mark(f"end leg {leg} to tag {tag}")
                time.sleep(0.4)
                fix, (x, y, h) = fix_at_stop(nav, survey, tag, face)
                row = {"leg": leg, "tag": tag, "reached": reached, "dr": [x, y, h],
                       "t": time.monotonic() - t0, "fix": fix, "contacts": nav.contacts,
                       "why": None if reached else nav.why}
                if fix is not None:
                    row["err_m"] = math.hypot(x - fix["x"], y - fix["y"])
                    row["err_deg"] = wrap180(h - fix["heading"])
                    print(f"  DR ({x:.2f}, {y:.2f}, {h:+.0f})  fix ({fix['x']:.2f}, {fix['y']:.2f}, "
                          f"{fix['heading']:+.0f}) tags {fix['tags']} -> error {row['err_m']:.2f} m, "
                          f"{row['err_deg']:+.1f} deg [{fix['result']}]", flush=True)
                else:
                    print(f"  DR ({x:.2f}, {y:.2f}, {h:+.0f}); no tag fix here", flush=True)
                rec.mark("tourstop " + json.dumps(row))
                results.append(row)
        except KeyboardInterrupt as e:
            print(f"interrupted {e}")
        finally:
            rig.act.stop()
            nav.stop()
            rig.hd.stop()
    rig.servo(config.SERVO_PAN_CENTER, config.SERVO_TILT_REST, settle=0.2)
    rig.cap.release()
    with open(os.path.join(rec.path, "tour.json"), "w") as f:
        json.dump(results, f, indent=1)
    nav.save(rec.path)
    ok = [r for r in results if r.get("fix")]
    print(f"done -> {rec.path}: {sum(r['reached'] for r in results)}/{len(results)} reached, "
          f"{len(ok)} fixes, contacts {nav.contacts}, localizer {nav.loc.stats if nav.loc else {}}")
    try:
        from .offline.score_run import score, summary_line
        s = score(rec.path)
        with open(os.path.join(rec.path, "score.json"), "w") as f:
            json.dump(s, f, indent=1)
        print("score:", summary_line(s))
    except Exception as e:                       # scoring must never lose a run
        print("score failed:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
