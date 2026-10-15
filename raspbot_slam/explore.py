"""
Explore the room and map it (NAV plan N6; mission "learn the room", 2026-10-12).

    python3 -m raspbot_slam.explore [--minutes 10] [--prior maps/office_prior.npz] [--correct]

Start facing a tag. Loop until time runs out or nothing is left worth the drive:
  1. pick a goal: a seen-free, reachable cell that would show the most unseen
     floor within camera range, per metre of path (pick_goal)
  2. nav_executor.Navigator.go_to (map, A*, contacts, wedged detection)
  3. look all round: three body turns of 120 deg, each with a -40/0/+40 pan sweep
  4. when the position sigma passes FIX_SIGMA_M, detour to the nearest tag
     viewpoint for a fix (needs --correct to apply it)
Goals that fail are not retried nearby. The map, nav.json and score.json are
saved in the run folder; offline/merge_runs.py folds the run into the prior.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from . import config
from .planner import CostField, costs, tag_viewpoint
from .room_map import FREE_T

CAMERA_RANGE_M = 1.4          # free-space rays reach about this far
MIN_GAIN_M2 = 0.15            # stop when the best goal would show less unseen floor than this
FAILED_RADIUS_M = 0.4         # don't pick goals this close to one that failed
FIX_SIGMA_M = 0.15            # detour to a tag when the position sigma passes this
RETRY_SIGMA_M = 0.10          # after a failed detour, wait until sigma grows this much more


def gain_map(m, range_m=CAMERA_RANGE_M):
    """Unseen in-room floor (m^2) within range_m of each cell."""
    import cv2
    unseen = (~m.seen & ~m.wall).astype(np.float32)
    r = int(round(range_m / m.res))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = ((xx ** 2 + yy ** 2) <= r * r).astype(np.float32)
    return cv2.filter2D(unseen, -1, disc, borderType=cv2.BORDER_CONSTANT) * m.res ** 2


def pick_goal(m, start_xy, failed=(), relaxed=False, step=2):
    """(goal_xy, gain_m2, path_m) of the best place to look from next, or None."""
    cost, hard = costs(m, relaxed)
    field = CostField(cost, m.cell(*start_xy), hard=hard)
    if not field.ok:
        return None
    gain = gain_map(m)
    free = m.seen & (m.L < FREE_T) & np.isfinite(cost)
    dist = field.dist * m.res                                # cost-weighted metres
    ok = free & np.isfinite(dist)
    mask = np.zeros_like(ok)
    mask[::step, ::step] = ok[::step, ::step]
    js, is_ = np.nonzero(mask)
    if not len(js):
        return None
    xs = m.x0 + (is_ + 0.5) * m.res
    ys = m.y0 + (js + 0.5) * m.res
    g = gain[js, is_]
    d = dist[js, is_]
    score = g / (1.0 + d)
    for fx, fy in failed:
        score[np.hypot(xs - fx, ys - fy) < FAILED_RADIUS_M] = -1
    k = int(np.argmax(score))
    if score[k] <= 0 or g[k] < MIN_GAIN_M2:
        return None
    return (float(xs[k]), float(ys[k])), float(g[k]), float(d[k])


def nearest_tag_viewpoint(m, survey, start_xy, skip=()):
    best = None
    for tag in survey["markers"]:
        if int(tag) in skip:
            continue
        vp = tag_viewpoint(m, survey, int(tag), start_xy)
        if vp is None or vp[2] is None:
            continue
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(vp[2], vp[2][1:]))
        if best is None or length < best[0]:
            best = (length, int(tag), vp)
    return best


def look_around(nav, turns=3):
    for _ in range(turns):
        nav.sense(sweep=True)
        _, _, h = nav.pose
        nav.turn_to_heading(h + 360.0 / turns)


def main(argv=None):
    from .calibration import auto_camera_cal as acc
    from .floor_odometry import FloorModel
    from .m1_run import load_intrinsics
    from .nav_executor import Navigator, battery_gate, check_stop, install_stop_handlers
    from .recorder import RunRecorder
    from .tag_localizer import TagLocalizer, load_survey
    from .tag_tour import anchor_fix, fix_at_stop

    p = argparse.ArgumentParser(description="Explore and map the room.")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--prior", default=None, help="start from a saved map (offline/merge_runs.py output)")
    p.add_argument("--correct", action="store_true", help="apply tag fixes to the pose")
    p.add_argument("--no-battery-check", action="store_true",
                   help="skip the start hops that check motor speed (battery proxy)")
    args = p.parse_args(argv)
    install_stop_handlers()

    K, dist = load_intrinsics()
    tagloc = TagLocalizer(K, dist)
    survey = load_survey("office")
    model = FloorModel(K, dist, config.CAMERA_HEIGHT_M, config.CAMERA_PITCH_DEG)
    rig = acc.Rig("/tmp", acc.TagTarget())
    rec = RunRecorder(camera=rig, imu=rig.imu, sensors=rig.sens, actuators=rig.act, label="explore")
    rec.bus_lock = rig.bus_lock
    nav = Navigator(rig, rec, model, survey, tagloc, correct=args.correct, prior=args.prior)
    goals, failed = [], []
    # a failed detour must not repeat at once (2026-10-12: 202 detours to tag 3 in a row)
    detour_at, skip_tags = FIX_SIGMA_M, set()
    with rec:
        try:
            fix = anchor_fix(nav)
            if fix is None:
                print("no tag in view at the start: can't anchor")
                return 1
            rec.mark("fix " + json.dumps(dict(fix, label="start", ok=True)))
            nav.start(fix)
            if not args.no_battery_check and not battery_gate(nav):
                return 1
            print(f"start: ({fix['x']:.2f}, {fix['y']:.2f}) heading {fix['heading']:+.0f}", flush=True)
            t_end = time.monotonic() + args.minutes * 60
            look_around(nav)
            while time.monotonic() < t_end:
                check_stop()
                x, y, _ = nav.pose
                sigma = nav.loc.sigma_xy() if nav.loc is not None else 0.0
                if sigma > detour_at:
                    best = nearest_tag_viewpoint(nav.map, survey, (x, y), skip_tags)
                    if best is not None:
                        _, tag, (vp, face, _) = best
                        nav.mark(f"position sigma {sigma:.2f} m: detour to tag {tag}")
                        f = None
                        if nav.go_to(vp):
                            nav.turn_to_heading(face)
                            f, _ = fix_at_stop(nav, survey, tag, face)
                        elif nav.why == "wedged":
                            break
                        nav.mark(f"tag {tag} fix: {'none' if f is None else f.get('result')}")
                        if f is not None and str(f.get("result", "")).startswith(("used", "position reset")):
                            detour_at, skip_tags = FIX_SIGMA_M, set()
                        else:
                            detour_at = nav.loc.sigma_xy() + RETRY_SIGMA_M
                            skip_tags.add(tag)
                    else:
                        detour_at = sigma + RETRY_SIGMA_M
                        skip_tags = set()
                g = pick_goal(nav.map, (x, y), failed)
                if g is None:
                    g = pick_goal(nav.map, (x, y), failed, relaxed=True)
                if g is None:
                    nav.mark("nothing left worth exploring")
                    break
                goal, gain, dist_m = g
                nav.mark(f"explore goal ({goal[0]:.2f}, {goal[1]:.2f}): {gain:.2f} m2 unseen, path {dist_m:.1f}")
                ok = nav.go_to(goal)
                goals.append({"goal": goal, "gain_m2": gain, "reached": ok, "why": nav.why})
                rec.mark("exploregoal " + json.dumps(goals[-1]))
                if nav.why == "wedged":
                    print("wedged; ending (needs a hand)", flush=True)
                    break
                if not ok:
                    failed.append(goal)
                look_around(nav)
        except KeyboardInterrupt as e:
            print(f"interrupted {e}")
        finally:
            rig.act.stop()
            nav.stop()
            rig.hd.stop()
    rig.servo(config.SERVO_PAN_CENTER, config.SERVO_TILT_REST, settle=0.2)
    rig.cap.release()
    with open(os.path.join(rec.path, "explore.json"), "w") as f:
        json.dump(goals, f, indent=1)
    nav.save(rec.path)
    print(f"done -> {rec.path}: {sum(g['reached'] for g in goals)}/{len(goals)} goals reached, "
          f"contacts {nav.contacts}, localizer {nav.loc.stats if nav.loc else {}}")
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
