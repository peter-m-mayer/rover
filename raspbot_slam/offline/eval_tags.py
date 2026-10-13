"""
Score dead reckoning against the surveyed tags, offline (AUTONOMY_PLAN.md M1/M2).

    python -m raspbot_slam.offline.eval_tags datasets/slam_runs/<run> [--plot out.png] [--map]

1. Tag fixes: every frame with the pan centred goes through TagLocalizer (its
   gates drop far tags, out-of-room poses and disagreeing multi-tag solves);
   fixes within 1 s are grouped and their median taken.
2. Dead reckoning: gyro heading + floor-odometry translation, anchored at the
   first fix group and never corrected after that.
3. At each later fix group: position and heading error vs distance travelled.
   A fix group whose heading disagrees with the gyro by > 15 deg is an outlier
   (planar flip, camera knocked off-axis) and is reported but not scored.
With --map, the occupancy map is built in room coordinates (room_map.RoomMap:
camera free space at every pan, sonar cone, footprint) and drawn over the
surveyed walls. With --localizer, every fix also goes through
localizer.Localizer in scoring mode (gate, heading watchdog) and its decisions
are printed.
"""

import argparse
import json
import math
import os

import numpy as np

from .. import config
from ..floor_odometry import FloorModel, FloorOdometry
from ..free_space import FreeSpaceScanner
from ..recorder import RunLog
from ..localizer import Localizer
from ..room_map import RoomMap
from ..tag_localizer import IN, TagLocalizer
from .eval_odometry import gyro_yaw

PAN_SETTLE_S = 0.5            # servo travel + camera latency after a pan command


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def load_camera():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "calibration", "camera_calibration.json")) as f:
        cal = json.load(f)
    return np.array(cal["camera_matrix"]), np.array(cal["dist_coeffs"])


def replay(run_dir, rmap=None, localizer=False, correct=False, log=print):
    """Replay one run: dead reckoning anchored at the first tag fix, fix groups
    scored against it, and the room map built if rmap is given (a RoomMap to add
    to, so several runs can share one). With localizer=True the fixes also go
    through localizer.Localizer; correct=True lets it move the pose (as
    tag_tour --correct does live). Returns None without a tag fix, else a dict:
    anchor, raw_fixes, groups, path (rows t, x, y, th), rows (one per fix group
    after the anchor), stats (Localizer stats or None), gt, loc."""
    import cv2
    K, dist = load_camera()
    loc = TagLocalizer(K, dist)
    run = RunLog(run_dir)
    # the camera mount changes between sessions: use the run's own settings
    rc = run.meta.get("config", {})
    pan_c = rc.get("SERVO_PAN_CENTER", config.SERVO_PAN_CENTER)
    model = FloorModel(K, dist, rc.get("CAMERA_HEIGHT_M", config.CAMERA_HEIGHT_M),
                       rc.get("CAMERA_PITCH_DEG", config.CAMERA_PITCH_DEG))
    gt, gyaw, _ = gyro_yaw(run)
    pans = [(float(c["t"]), float(c["a"])) for c in run.cmds if c["kind"] == "pan"]
    pt = np.array([q[0] for q in pans]); pv = np.array([q[1] for q in pans])
    st, sd = run.sensors.get("t", np.array([])), run.sensors.get("us_mm", np.array([]))
    mapping = rmap is not None

    odo = FloorOdometry(model)
    fs = FreeSpaceScanner(model, max_range_m=1.6)

    class Odo:                                       # pose holder the Localizer can shift
        travelled_m = 0.0

        @property
        def pose(self):
            return pose[0], pose[1], math.degrees(pose[2])

        def shift(self, dx, dy, dh=0.0):
            global_shift[0] += dx; global_shift[1] += dy; global_shift[2] += math.radians(dh)
    global_shift = np.zeros(3)
    odo_shim = Odo()
    filt = None
    now = [0.0]

    raw_fixes, path = [], []
    anchor = None                                    # (x, y, yaw_offset) once anchored
    pose = np.zeros(3)                               # DR in room coordinates
    prev_t = None
    for idx, t in zip(run.frames["idx"], run.frames["t"]):
        img = cv2.imread(run.frame_path(int(idx)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = float(np.interp(t - config.CAMERA_LATENCY_S, gt, gyaw))                  # gyro_yaw is already in radians
        dth = None if prev_t is None else g - float(np.interp(prev_t - config.CAMERA_LATENCY_S, gt, gyaw))
        r = odo.step(gray, dth)
        k = np.searchsorted(pt, t) - 1
        pan = pv[max(0, k)] if len(pt) else pan_c
        # frames just after a pan command still show the old view (servo travel +
        # camera latency): a 2026-10-11 replay took a +25 deg view as centred
        settled = k < 0 or t - pt[k] > PAN_SETTLE_S
        est = None
        if pan == pan_c and settled:
            det = loc.detect(gray)
            est = loc.solve(det) if det else None
            if est:
                raw_fixes.append((t, est["x"], est["y"], est["yaw"], math.degrees(g), est["tags"], est["range_m"]))
        if anchor is None and est is not None:
            anchor = (t, est)
            pose = np.array([est["x"], est["y"], math.radians(est["yaw"])])
            yaw_off = math.radians(est["yaw"]) - g
            if localizer:
                filt = Localizer(odo_shim, correct=correct, log=lambda m: log(f"  [{now[0] - gt[0]:.1f}s] {m}"))
        elif anchor is not None:
            h_prev = pose[2]
            pose[2] = g + yaw_off + global_shift[2]
            if r is not None:
                c, s = math.cos(h_prev), math.sin(h_prev)
                pose[0] += c * r[0] - s * r[1]
                pose[1] += s * r[0] + c * r[1]
                odo_shim.travelled_m += math.hypot(r[0], r[1])
            pose[0] += global_shift[0]; pose[1] += global_shift[1]
            global_shift[:] = 0
            now[0] = t
            if filt is not None and est is not None:
                filt.add_fix(dict(est, heading=est["yaw"]))
                yaw_off += global_shift[2]; global_shift[2] = 0
                pose[2] = g + yaw_off
            path.append((t, *pose))
            if mapping and (fs.hist is not None or float(np.interp(t, st, sd)) > 500):
                if float(np.interp(t, st, sd)) > 500:
                    fs.learn(img)
                still = r is not None and math.hypot(r[0], r[1]) < 0.003 and (dth is None or abs(dth) < 0.01)
                if still and settled and len(path) % 3 == 0:
                    res = fs.scan(img)
                    if res is not None:
                        x, y, th = pose
                        rmap.add_camera_scan((x, y, th + math.radians(pan - pan_c)), *res)
                        if pan == pan_c:
                            us = float(np.interp(t, st, sd)) / 1000
                            rmap.add_sonar((x, y, th), us)
                rmap.add_footprint(pose[0], pose[1])
        prev_t = t
        now[0] = t
    if anchor is None:
        return None
    path = np.array(path)

    # group fixes within 1 s, median each group
    groups, cur = [], []
    for f in raw_fixes:
        if cur and f[0] - cur[-1][0] > 1.0:
            groups.append(cur); cur = []
        cur.append(f)
    if cur:
        groups.append(cur)
    rows = []
    for gset in groups:
        t = float(np.median([f[0] for f in gset]))
        if t <= anchor[0] + 0.5 or not len(path):
            continue
        fx, fy = np.median([f[1] for f in gset]), np.median([f[2] for f in gset])
        fyaw = math.degrees(math.atan2(np.median([math.sin(math.radians(f[3])) for f in gset]),
                                       np.median([math.cos(math.radians(f[3])) for f in gset])))
        i = min(np.searchsorted(path[:, 0], t), len(path) - 1)
        dx, dy, dh = path[i, 1], path[i, 2], math.degrees(path[i, 3])
        travel = float(np.sum(np.hypot(np.diff(path[: i + 1, 1]), np.diff(path[: i + 1, 2]))))
        err_deg = wrap(dh - fyaw)
        rows.append(dict(t=float(t - gt[0]), travel=travel, tags=list(gset[0][5]), fix=(float(fx), float(fy)),
                         dr=(float(dx), float(dy)), err_m=float(math.hypot(dx - fx, dy - fy)),
                         err_deg=float(err_deg), outlier=bool(abs(err_deg) > 15)))
    return dict(anchor=anchor, raw_fixes=raw_fixes, groups=groups, path=path, rows=rows,
                stats=dict(filt.stats) if filt else None, gt=gt, loc=loc)


def main(argv=None):
    import cv2
    p = argparse.ArgumentParser(description="Dead reckoning vs surveyed tags.")
    p.add_argument("run")
    p.add_argument("--plot", default=None)
    p.add_argument("--map", action="store_true")
    p.add_argument("--localizer", action="store_true",
                   help="run fixes through localizer.Localizer in scoring mode (gates, heading watchdog)")
    p.add_argument("--correct", action="store_true", help="with --localizer: let the fixes move the pose")
    args = p.parse_args(argv)

    K, dist = load_camera()
    rmap = RoomMap.from_survey(TagLocalizer(K, dist).survey) if args.map else None
    out = replay(args.run, rmap, localizer=args.localizer, correct=args.correct)
    if out is None:
        print("no usable tag fix in this run")
        return 1
    anchor, raw_fixes, groups, path, gt, loc = (out[k] for k in ("anchor", "raw_fixes", "groups", "path", "gt", "loc"))
    print(f"{len(raw_fixes)} fixes in {len(groups)} groups; anchored at t={anchor[0] - gt[0]:.1f}s "
          f"({anchor[1]['x']:.2f}, {anchor[1]['y']:.2f}, {anchor[1]['yaw']:+.0f}) from tags {anchor[1]['tags']}")
    print(f"\n{'t (s)':>7} {'travel m':>9} {'tags':>8} {'fix x,y':>13} {'DR x,y':>13} {'err m':>6} {'err deg':>8}")
    scored = []
    for r in out["rows"]:
        flag = "  OUTLIER (heading)" if r["outlier"] else ""
        print(f"{r['t']:7.1f} {r['travel']:9.2f} {str(r['tags']):>8} {r['fix'][0]:6.2f},{r['fix'][1]:5.2f} "
              f"{r['dr'][0]:6.2f},{r['dr'][1]:5.2f} {r['err_m']:6.2f} {r['err_deg']:+8.1f}{flag}")
        if not flag:
            scored.append((r["travel"], r["err_m"], r["err_deg"]))
    if scored:
        s = np.array(scored)
        print(f"\nscored {len(s)} fix groups over {s[:, 0].max():.1f} m travelled: "
              f"median error {np.median(s[:, 1]):.2f} m / {np.median(np.abs(s[:, 2])):.1f} deg, "
              f"max {s[:, 1].max():.2f} m; error per metre travelled {np.median(s[:, 1] / np.maximum(s[:, 0], 0.3)):.1%}")
    if out["stats"]:
        print("localizer:", out["stats"])

    # survey consistency: with one continuous gyro, (tag yaw - gyro) should be the
    # same constant for every tag; a tag whose offset differs is mis-surveyed (or
    # the gyro slipped between sightings -- compare the times)
    per = {}
    for f in raw_fixes:
        per.setdefault(tuple(f[5]), []).append((f[0], wrap(f[3] - f[4])))
    if len(per) > 1:
        print("\nper-tag heading offset (tag yaw - gyro), should agree across tags:")
        for k, v in sorted(per.items()):
            off = np.array([o for _, o in v])
            c = math.degrees(math.atan2(np.median(np.sin(np.radians(off))), np.median(np.cos(np.radians(off)))))
            print(f"  tags {str(list(k)):>8}: {c:+7.1f} deg  (n {len(v)}, t {v[0][0] - gt[0]:.0f}-{v[-1][0] - gt[0]:.0f} s)")

    if args.plot:
        sv = loc.survey
        X, Y = sv["xmax_in"] * IN, sv["ymax_in"] * IN
        a = sv["alcove"]
        ay0, ay1, ax0 = a["y0"] * IN, a["y1"] * IN, a["x0"] * IN
        outline = np.array([(0, 0), (X, 0), (X, Y), (0, Y), (0, ay1), (ax0, ay1), (ax0, ay0), (0, ay0), (0, 0)])
        scale = 100                                   # px per metre
        W, H = int((X + 2.2) * scale), int((Y + 1.0) * scale)
        canvas = np.full((H, W, 3), 255, np.uint8)
        to_px = lambda x, y: (int((x + 1.7) * scale), int(H - (y + 0.5) * scale))
        if args.map:
            res_ = rmap.res
            for mask, colour in ((rmap.free(), (235, 245, 235)), (rmap.occupied(contacts=False) & ~rmap.wall, (40, 40, 40))):
                for (j, i) in np.argwhere(mask):
                    x, y = rmap.centre(i, j)
                    cv2.rectangle(canvas, to_px(x - res_ / 2, y - res_ / 2), to_px(x + res_ / 2, y + res_ / 2), colour, -1)
        cv2.polylines(canvas, [np.array([to_px(x, y) for x, y in outline], np.int32)], False, (200, 120, 0), 2)
        mir = a.get("sliding_door_mirror")
        if mir:
            cv2.line(canvas, to_px(mir["x"] * IN, mir["y0"] * IN), to_px(mir["x"] * IN, mir["y1"] * IN),
                     (255, 160, 60), 5)                                                 # mirror door
        for k, m in sv["markers"].items():
            q = to_px(m["x"] * IN, m["y"] * IN)
            cv2.rectangle(canvas, (q[0] - 5, q[1] - 5), (q[0] + 5, q[1] + 5), (0, 0, 200), -1)
            cv2.putText(canvas, k, (q[0] + 7, q[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        pts = np.array([to_px(x, y) for _, x, y, _ in path], np.int32)
        cv2.polylines(canvas, [pts], False, (0, 0, 255), 1)
        for gset in groups:
            fx, fy = np.median([f[1] for f in gset]), np.median([f[2] for f in gset])
            cv2.circle(canvas, to_px(fx, fy), 4, (0, 170, 0), -1)
        cv2.putText(canvas, "red: dead reckoning (anchored at first fix)   green: tag fixes   blue: surveyed walls",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.imwrite(args.plot, canvas)
        print("plot:", args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
