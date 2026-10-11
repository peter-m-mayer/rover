"""
Replay a recorded run through floor odometry and score it (AUTONOMY_PLAN.md M1).

    python -m raspbot_slam.offline.eval_odometry datasets/slam_runs/<run> [--pitch 1.0] [--plot out.png]

Per motion segment (events "move X" ... "end X") it reports:
  floor odometry: forward / left / turn
  gyro:           turn (bias from the first still second)
  ultrasonic:     change in range to the wall ahead (forward segments facing a wall)
  tag fixes:      pose change between the fixes around the segment, if any
"""

import argparse
import json
import math
import os

import numpy as np

from ..floor_odometry import FloorModel, FloorOdometry
from ..recorder import RunLog


def gyro_yaw(run, t_still=1.0):
    t, gz = run.imu["t"], run.imu["gz"]
    still = t < t[0] + t_still
    bias = float(np.median(gz[still])) if still.sum() > 10 else 0.0
    yaw = np.concatenate([[0.0], np.cumsum((gz[1:] - bias) * np.diff(t))])
    return t, yaw, bias


def segments(run):
    out, open_ = [], {}
    for t, lab in run.events:
        if lab.startswith("move "):
            open_[lab[5:]] = t
        elif lab.startswith("end "):
            name = lab[4:].split(" gyro")[0]
            if name in open_:
                out.append((name, open_.pop(name), t))
    return out


def fixes(run):
    res = []
    for t, lab in run.events:
        if lab.startswith("fix "):
            d = json.loads(lab[4:])
            if d.get("ok"):
                res.append((t, d))
    return res


def truth_ranges(run):
    out = []
    for t, lab in run.events:
        if lab.startswith("truth range_mm"):
            tok = lab.split()
            out.append((t, float(tok[tok.index("range_mm") + 1])))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Score floor odometry on a recorded run.")
    p.add_argument("run")
    p.add_argument("--pitch", type=float, default=None,
                   help="camera pitch (deg up) during the run (default config.CAMERA_PITCH_DEG)")
    p.add_argument("--height", type=float, default=None, help="camera height (m)")
    p.add_argument("--max-range", type=float, default=1.2, help="use floor points within this range (m)")
    p.add_argument("--calib", default=None, help="camera_calibration.json")
    p.add_argument("--plot", default=None)
    p.add_argument("--no-gyro", action="store_true", help="estimate rotation from the floor too")
    args = p.parse_args(argv)

    import cv2
    run = RunLog(args.run)
    calib = args.calib or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "calibration", "camera_calibration.json")
    with open(calib) as f:
        cal = json.load(f)
    from .. import config
    pitch = config.CAMERA_PITCH_DEG if args.pitch is None else args.pitch
    model = FloorModel(cal["camera_matrix"], cal["dist_coeffs"], args.height, pitch, args.max_range)
    odo = FloorOdometry(model)

    gt, gyaw, bias = gyro_yaw(run)
    traj = []                                  # t, x, y, theta, n_inliers
    t_prev = None
    for t, img in run.iter_frames(gray=True):
        dth = None
        if not args.no_gyro and t_prev is not None:
            dth = float(np.interp(t - config.CAMERA_LATENCY_S, gt, gyaw) - np.interp(t_prev - config.CAMERA_LATENCY_S, gt, gyaw))
        r = odo.step(img, dth)
        traj.append((t, *odo.pose, r[3] if r else 0))
        t_prev = t
    traj = np.array(traj)
    print(f"{len(traj)} frames, floor estimates failed on {odo.stats['failed']} of {odo.stats['steps']}; "
          f"gyro bias {math.degrees(bias):+.2f} deg/s")

    def at(t):
        i = min(np.searchsorted(traj[:, 0], t), len(traj) - 1)
        return traj[i]

    rng = truth_ranges(run)
    print(f"\n{'segment':<16}{'floor fwd':>10}{'floor left':>11}{'floor turn':>11}{'gyro turn':>10}{'sonar d':>9}")
    for name, t0, t1 in segments(run):
        a, b = at(t0), at(t1)
        c, s = math.cos(a[3]), math.sin(a[3])
        dx, dy = b[1] - a[1], b[2] - a[2]
        fwd, left = c * dx + s * dy, -s * dx + c * dy
        dth = math.degrees(b[3] - a[3])
        g = math.degrees(np.interp(t1, gt, gyaw) - np.interp(t0, gt, gyaw))
        r0 = [v for t, v in rng if t <= t0 + 0.05]
        r1 = [v for t, v in rng if t <= t1 + 0.05]
        sonar = (r0[-1] - r1[-1]) / 1000 if r0 and r1 else float("nan")
        print(f"{name:<16}{fwd:>10.3f}{left:>11.3f}{dth:>11.1f}{g:>10.1f}{sonar:>9.3f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(traj[:, 1], traj[:, 2], ".-", ms=2)
        ax[0].set_aspect("equal"); ax[0].set_title("floor odometry path (m)"); ax[0].grid(True)
        ax[1].plot(traj[:, 0] - traj[0, 0], np.degrees(traj[:, 3]), label="floor")
        ax[1].plot(gt - traj[0, 0], np.degrees(gyaw), label="gyro")
        ax[1].set_xlabel("s"); ax[1].set_ylabel("heading deg"); ax[1].legend(); ax[1].grid(True)
        for name, t0, t1 in segments(run):
            ax[1].axvspan(t0 - traj[0, 0], t1 - traj[0, 0], alpha=0.1)
        fig.tight_layout(); fig.savefig(args.plot, dpi=90)
        print("plot:", args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
