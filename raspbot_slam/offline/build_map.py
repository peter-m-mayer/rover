"""
Build a 2-D occupancy map from a recorded run (AUTONOMY_PLAN.md M2, offline).

    python -m raspbot_slam.offline.build_map datasets/slam_runs/<run> [--out map.png]

Pose: floor odometry with the gyro supplying rotation (floor_odometry.py).
Evidence, in log-odds:
  camera  free-space scan (free_space.py) every --every frames while the
          robot is still: free along each column ray, occupied at its foot
  sonar   every reading: free along the beam axis, occupied at the range
Obstacle wins: a cell either sensor saw occupied stays occupied unless seen
free many times (overhangs are visible to the sonar only, floor-level clutter
to the camera only).
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
from .eval_odometry import gyro_yaw

L_FREE, L_OCC, L_MIN, L_MAX = -0.4, 1.2, -4.0, 4.0
SONAR_MIN_M, SONAR_MAX_M = 0.05, 3.0


class Grid:
    def __init__(self, size_m=8.0, res_m=0.03):
        self.res = res_m
        self.n = int(size_m / res_m)
        self.L = np.zeros((self.n, self.n), np.float32)
        self.origin = self.n // 2

    def cell(self, x, y):
        return int(round(x / self.res)) + self.origin, int(round(y / self.res)) + self.origin

    def ray(self, x0, y0, x1, y1, hit):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / (self.res * 0.7)))
        xs, ys = np.linspace(x0, x1, n + 1), np.linspace(y0, y1, n + 1)
        cells = {self.cell(x, y) for x, y in zip(xs[:-1], ys[:-1])}
        end = self.cell(x1, y1)
        cells.discard(end)
        for i, j in cells:
            if 0 <= i < self.n and 0 <= j < self.n:
                self.L[j, i] = max(L_MIN, self.L[j, i] + L_FREE)
        if hit and 0 <= end[0] < self.n and 0 <= end[1] < self.n:
            self.L[end[1], end[0]] = min(L_MAX, self.L[end[1], end[0]] + L_OCC)


def main(argv=None):
    import cv2
    p = argparse.ArgumentParser(description="Occupancy map from a recorded run.")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="map.png")
    p.add_argument("--every", type=int, default=3, help="camera scan every N still frames")
    p.add_argument("--max-range", type=float, default=1.6)
    args = p.parse_args(argv)

    cal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "calibration", "camera_calibration.json")
    with open(cal_path) as f:
        cal = json.load(f)
    model = FloorModel(cal["camera_matrix"], cal["dist_coeffs"], config.CAMERA_HEIGHT_M,
                       config.CAMERA_PITCH_DEG)
    grid = Grid()
    paths = []
    for run_dir in args.runs:
        run = RunLog(run_dir)
        odo = FloorOdometry(model)
        fs = FreeSpaceScanner(model, max_range_m=args.max_range)
        gt, gyaw, _ = gyro_yaw(run)
        st, sd = run.sensors.get("t", np.array([])), run.sensors.get("us_mm", np.array([]))
        poses, t_prev, still_count, last = [], None, 0, None
        for t, img in run.iter_frames():
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dth = None if t_prev is None else float(np.interp(t, gt, gyaw) - np.interp(t_prev, gt, gyaw))
            r = odo.step(gray, dth)
            x, y, th = odo.pose
            poses.append((t, x, y, th))
            moving = r is not None and math.hypot(r[0], r[1]) > 0.003 or (dth is not None and abs(dth) > 0.01)
            still_count = 0 if moving else still_count + 1
            # learn the floor when the sonar says the patch ahead is clear
            us = float(np.interp(t, st, sd)) if len(st) else 0
            if us > 700 and not moving:
                fs.learn(img)
            if still_count and still_count % args.every == 0 and fs.hist is not None:
                res = fs.scan(img)
                if res is not None:
                    b, rng, hit = res
                    for bj, rj, hj in zip(b, rng, hit):
                        if np.isnan(bj):
                            continue
                        a = th + bj
                        grid.ray(x, y, x + rj * math.cos(a), y + rj * math.sin(a), bool(hj))
            t_prev = t
        poses = np.array(poses)
        # sonar along the body axis (pan centred in these runs)
        for t, d in zip(st, sd):
            d = d / 1000.0
            if not SONAR_MIN_M < d < SONAR_MAX_M or t < poses[0, 0]:
                continue
            i = min(np.searchsorted(poses[:, 0], t), len(poses) - 1)
            _, x, y, th = poses[i]
            grid.ray(x, y, x + d * math.cos(th), y + d * math.sin(th), True)
        paths.append(poses)
        print(f"{run_dir}: {len(poses)} poses, floor failed {odo.stats['failed']}/{odo.stats['steps']}")

    prob = 1 / (1 + np.exp(-grid.L))
    img = np.full(grid.L.shape + (3,), 205, np.uint8)               # unknown: grey
    img[grid.L < -0.5] = (250, 250, 250)                            # free: white
    img[grid.L > 0.8] = (30, 30, 30)                                # occupied: black
    for poses in paths:
        pts = [grid.cell(x, y) for _, x, y, _ in poses]
        for (i0, j0), (i1, j1) in zip(pts, pts[1:]):
            cv2.line(img, (i0, j0), (i1, j1), (0, 0, 220), 1)
        cv2.circle(img, pts[0], 4, (0, 160, 0), -1)
    img = cv2.flip(img, 0)                                          # +y up
    occupied = np.argwhere(grid.L > 0.8)
    if len(occupied):
        lo = np.maximum(0, occupied.min(0) - 20)
        hi = np.minimum(grid.n, occupied.max(0) + 20)
        img = img[grid.n - hi[0]:grid.n - lo[0], lo[1]:hi[1]]
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, f"{grid.res * 100:.0f} cm cells; green = start, red = path", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(args.out, img)
    print("map:", args.out, f"({(grid.L < -0.5).sum() * grid.res ** 2:.2f} m^2 free, "
          f"{(grid.L > 0.8).sum()} occupied cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
