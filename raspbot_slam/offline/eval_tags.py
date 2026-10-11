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
With --map, the occupancy map is built in room coordinates (camera free space
and sonar, as offline/build_map.py) and drawn over the surveyed walls.
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
from ..tag_localizer import IN, TagLocalizer
from .build_map import Grid
from .eval_odometry import gyro_yaw


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def main(argv=None):
    import cv2
    p = argparse.ArgumentParser(description="Dead reckoning vs surveyed tags.")
    p.add_argument("run")
    p.add_argument("--plot", default=None)
    p.add_argument("--map", action="store_true")
    args = p.parse_args(argv)

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "calibration", "camera_calibration.json")) as f:
        cal = json.load(f)
    K, dist = np.array(cal["camera_matrix"]), np.array(cal["dist_coeffs"])
    loc = TagLocalizer(K, dist)
    model = FloorModel(K, dist, config.CAMERA_HEIGHT_M, config.CAMERA_PITCH_DEG)
    run = RunLog(args.run)
    gt, gyaw, _ = gyro_yaw(run)
    pans = [(float(c["t"]), float(c["a"])) for c in run.cmds if c["kind"] == "pan"]
    pt = np.array([q[0] for q in pans]); pv = np.array([q[1] for q in pans])
    st, sd = run.sensors.get("t", np.array([])), run.sensors.get("us_mm", np.array([]))

    odo = FloorOdometry(model)
    fs = FreeSpaceScanner(model, max_range_m=1.6)
    grid = Grid(size_m=12.0, res_m=0.04)
    grid.origin = 0
    off = int(1.5 / grid.res)                       # room (0,0) sits 1.5 m in from the grid corner

    def gcell(x, y):
        return int(round(x / grid.res)) + off, int(round(y / grid.res)) + off

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
        pan = pv[max(0, np.searchsorted(pt, t) - 1)] if len(pt) else config.SERVO_PAN_CENTER
        est = None
        if pan == config.SERVO_PAN_CENTER:
            det = loc.detect(gray)
            est = loc.solve(det) if det else None
            if est:
                raw_fixes.append((t, est["x"], est["y"], est["yaw"], math.degrees(g), est["tags"], est["range_m"]))
        if anchor is None and est is not None:
            anchor = (t, est)
            pose = np.array([est["x"], est["y"], math.radians(est["yaw"])])
            yaw_off = math.radians(est["yaw"]) - g
        elif anchor is not None:
            h_prev = pose[2]
            pose[2] = g + yaw_off
            if r is not None:
                c, s = math.cos(h_prev), math.sin(h_prev)
                pose[0] += c * r[0] - s * r[1]
                pose[1] += s * r[0] + c * r[1]
            path.append((t, *pose))
            if args.map and fs.hist is not None or (args.map and float(np.interp(t, st, sd)) > 500):
                if float(np.interp(t, st, sd)) > 500:
                    fs.learn(img)
                still = r is not None and math.hypot(r[0], r[1]) < 0.003 and (dth is None or abs(dth) < 0.01)
                if still and pan == config.SERVO_PAN_CENTER and len(path) % 3 == 0:
                    res = fs.scan(img)
                    if res is not None:
                        x, y, th = pose
                        for bj, rj, hj in zip(*res):
                            if bj == bj:
                                a = th + bj
                                i0, j0 = gcell(x, y)
                                i1, j1 = gcell(x + rj * math.cos(a), y + rj * math.sin(a))
                                grid.ray(*(np.array([i0, j0, i1, j1]) * grid.res), bool(hj))
        prev_t = t
    if anchor is None:
        print("no usable tag fix in this run")
        return 1
    path = np.array(path)

    # group fixes within 1 s, median each group
    groups, cur = [], []
    for f in raw_fixes:
        if cur and f[0] - cur[-1][0] > 1.0:
            groups.append(cur); cur = []
        cur.append(f)
    if cur:
        groups.append(cur)
    print(f"{len(raw_fixes)} fixes in {len(groups)} groups; anchored at t={anchor[0] - gt[0]:.1f}s "
          f"({anchor[1]['x']:.2f}, {anchor[1]['y']:.2f}, {anchor[1]['yaw']:+.0f}) from tags {anchor[1]['tags']}")
    print(f"\n{'t (s)':>7} {'travel m':>9} {'tags':>8} {'fix x,y':>13} {'DR x,y':>13} {'err m':>6} {'err deg':>8}")
    scored = []
    for gset in groups:
        t = float(np.median([f[0] for f in gset]))
        if t <= anchor[0] + 0.5:
            continue
        fx, fy = np.median([f[1] for f in gset]), np.median([f[2] for f in gset])
        fyaw = math.degrees(math.atan2(np.median([math.sin(math.radians(f[3])) for f in gset]),
                                       np.median([math.cos(math.radians(f[3])) for f in gset])))
        i = min(np.searchsorted(path[:, 0], t), len(path) - 1)
        dx, dy, dh = path[i, 1], path[i, 2], math.degrees(path[i, 3])
        travel = float(np.sum(np.hypot(np.diff(path[: i + 1, 1]), np.diff(path[: i + 1, 2]))))
        err_m, err_deg = math.hypot(dx - fx, dy - fy), wrap(dh - fyaw)
        flag = "  OUTLIER (heading)" if abs(err_deg) > 15 else ""
        print(f"{t - gt[0]:7.1f} {travel:9.2f} {str(gset[0][5]):>8} {fx:6.2f},{fy:5.2f} {dx:6.2f},{dy:5.2f} "
              f"{err_m:6.2f} {err_deg:+8.1f}{flag}")
        if not flag:
            scored.append((travel, err_m, err_deg))
    if scored:
        s = np.array(scored)
        print(f"\nscored {len(s)} fix groups over {s[:, 0].max():.1f} m travelled: "
              f"median error {np.median(s[:, 1]):.2f} m / {np.median(np.abs(s[:, 2])):.1f} deg, "
              f"max {s[:, 1].max():.2f} m; error per metre travelled {np.median(s[:, 1] / np.maximum(s[:, 0], 0.3)):.1%}")

    if args.plot:
        sv = loc.survey
        X, Y = sv["xmax_in"] * IN, sv["ymax_in"] * IN
        a = sv["alcove"]
        outline = np.array([(0, 0), (X, 0), (X, Y), (0, Y), (0, a["y1"] * IN), (a["x0"] * IN, a["y1"] * IN),
                            (a["x0"] * IN, 0), (0, 0)])
        scale = 100                                   # px per metre
        W, H = int((X + 2.2) * scale), int((Y + 1.0) * scale)
        canvas = np.full((H, W, 3), 255, np.uint8)
        to_px = lambda x, y: (int((x + 1.7) * scale), int(H - (y + 0.5) * scale))
        if args.map:
            occ = grid.L > 0.8
            free = grid.L < -0.5
            for (j, i) in np.argwhere(free):
                x, y = (i - off) * grid.res, (j - off) * grid.res
                cv2.rectangle(canvas, to_px(x, y), to_px(x + grid.res, y + grid.res), (235, 245, 235), -1)
            for (j, i) in np.argwhere(occ):
                x, y = (i - off) * grid.res, (j - off) * grid.res
                cv2.rectangle(canvas, to_px(x, y), to_px(x + grid.res, y + grid.res), (40, 40, 40), -1)
        cv2.polylines(canvas, [np.array([to_px(x, y) for x, y in outline], np.int32)], False, (200, 120, 0), 2)
        cv2.line(canvas, to_px(0, 0), to_px(0, 36 * IN), (255, 160, 60), 5)            # mirror door
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
