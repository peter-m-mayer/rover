"""
Refit the navigation motion and noise models from recorded runs, so the robot's
own driving updates the numbers it plans with.

    python -m raspbot_slam.offline.calibrate datasets/slam_runs/<run> [...] [--write]

Per run (eval_tags.replay, dead reckoning anchored at the first tag fix and
never corrected, so a tag jump can't land inside a hop):
  - hops: every forward heading-held drive in cmd.csv (all wheels the same
    sign, |speed| >= 40) with its duration t and the replayed displacement d
    from its start to 0.5 s after the stop. Fit d = v * (t - lag) by least
    squares, over hops not stopped by contact (d > 0.3 * the current model).
    -> NAV_SPEED_MPS, NAV_START_LAG_S (nav_executor hop length and stall test)
  - tag fix noise: the spread of single-frame fixes within each 1 s group
    -> LOC_SIG_FIX_XY_M (localizer measurement sigma)
  - dead-reckoning growth: uncorrected replay error at each fix group per
    metre travelled since the anchor, 80th percentile -> LOC_ODO_XY_PER_M
With --write, values that pass their sanity bounds and come from enough data
go into raspbot_slam/calibration/nav_calibration.json, which config.py reads
at import (so the Pi picks them up with git pull). Without it, prints only.
"""

import argparse
import json
import math
import os

import numpy as np

from .. import config
from ..recorder import RunLog
from .eval_tags import replay

CAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "calibration", "nav_calibration.json")
BOUNDS = {"NAV_SPEED_MPS": (0.15, 0.45), "NAV_START_LAG_S": (0.0, 0.3),
          "LOC_SIG_FIX_XY_M": (0.02, 0.2), "LOC_ODO_XY_PER_M": (0.01, 0.3)}
MIN_HOPS = 15


def hops(run):
    """[(t_start, t_stop, sign)] for each straight drive in cmd.csv."""
    out, start, sign = [], None, 0
    for c in run.cmds:
        if c["kind"] != "motors":
            continue
        t = float(c["t"])
        w = np.array([float(c[k]) for k in ("a", "b", "c", "d")])
        straight = np.all(w > 0) or np.all(w < 0)
        if straight and np.abs(w).mean() >= 40:
            if start is None:
                start, sign = t, int(np.sign(w[0]))
        elif start is not None:
            out.append((start, t, sign))
            start = None
    return out


def hop_samples(run_dir, out):
    """(duration s, forward displacement m) per forward hop, from a replay."""
    run = RunLog(run_dir)
    path = out["path"]
    res = []
    for t0, t1, sign in hops(run):
        if sign <= 0 or t0 < path[0, 0] or t1 + 0.5 > path[-1, 0]:
            continue
        i0, i1 = np.searchsorted(path[:, 0], [t0, t1 + 0.5])
        x0, y0, th0 = path[i0, 1:4]
        dx, dy = path[i1, 1] - x0, path[i1, 2] - y0
        res.append((t1 - t0, dx * math.cos(th0) + dy * math.sin(th0)))
    return res


def fit_hops(samples, v0, lag0):
    """Least squares d = v t - v lag over hops that moved (not contacts)."""
    s = np.array(samples, float)
    if len(s) == 0:
        return None
    moved = s[:, 1] > 0.3 * v0 * np.maximum(s[:, 0] - lag0, 0.05)
    s = s[moved]
    if len(s) < 3 or np.ptp(s[:, 0]) < 0.05:
        return None
    A = np.c_[s[:, 0], np.ones(len(s))]
    (v, c), *_ = np.linalg.lstsq(A, s[:, 1], rcond=None)
    pred = A @ np.array([v, c])
    rms = float(np.sqrt(np.mean((pred - s[:, 1]) ** 2)))
    return {"NAV_SPEED_MPS": float(v), "NAV_START_LAG_S": float(-c / v) if v > 0 else None,
            "n": int(len(s)), "rms_m": rms}


def fix_noise(groups):
    sd = [np.hypot(np.std([f[1] for f in g]), np.std([f[2] for f in g])) for g in groups if len(g) >= 3]
    return float(np.median(sd)) if sd else None


def main(argv=None):
    p = argparse.ArgumentParser(description="Refit hop, fix-noise and drift models from runs.")
    p.add_argument("runs", nargs="+")
    p.add_argument("--write", action="store_true", help="update calibration/nav_calibration.json")
    args = p.parse_args(argv)
    samples, groups, drift = [], [], []
    for r in args.runs:
        out = replay(r, None, log=lambda m: None)
        if out is None:
            print(f"{r}: no tag fix, skipped")
            continue
        hs = hop_samples(r, out)
        samples += hs
        groups += out["groups"]
        drift += [row["err_m"] / max(row["travel"], 0.5) for row in out["rows"] if not row["outlier"]]
        print(f"{r}: {len(hs)} hops, {len(out['groups'])} fix groups", flush=True)

    old = {"NAV_SPEED_MPS": config.NAV_SPEED_MPS, "NAV_START_LAG_S": config.NAV_START_LAG_S,
           "LOC_SIG_FIX_XY_M": config.LOC_SIG_FIX_XY_M, "LOC_ODO_XY_PER_M": config.LOC_ODO_XY_PER_M}
    new = {}
    fit = fit_hops(samples, old["NAV_SPEED_MPS"], old["NAV_START_LAG_S"])
    if fit:
        print(f"hops: v {fit['NAV_SPEED_MPS']:.3f} m/s, lag {fit['NAV_START_LAG_S']:.3f} s "
              f"(n {fit['n']}, rms {fit['rms_m'] * 100:.1f} cm)")
        if fit["n"] >= MIN_HOPS:
            new.update({k: fit[k] for k in ("NAV_SPEED_MPS", "NAV_START_LAG_S")})
        else:
            print(f"  only {fit['n']} hops (< {MIN_HOPS}): not adopted")
    sig = fix_noise(groups)
    if sig is not None:
        print(f"tag fix spread within a group: {sig * 100:.1f} cm (n {len(groups)} groups)")
        new["LOC_SIG_FIX_XY_M"] = max(sig, 0.03)        # floor: survey and pose errors don't average out
    if len(drift) >= 5:
        # a sigma has to cover the bumps (0.2-1.2 m slips), not just clean driving: 80th percentile
        d = float(np.percentile(drift, 80))
        print(f"uncorrected drift: median {np.median(drift):.1%}, 80th percentile {d:.1%} of distance "
              f"(n {len(drift)} fix groups)")
        new["LOC_ODO_XY_PER_M"] = d

    adopt = {}
    print(f"\n{'parameter':<20} {'current':>9} {'fitted':>9}  verdict")
    for k, v in new.items():
        lo, hi = BOUNDS[k]
        ok = v is not None and lo <= v <= hi
        if ok:
            adopt[k] = round(v, 4)
        print(f"{k:<20} {old[k]:9.4f} {v if v is not None else float('nan'):9.4f}  "
              f"{'adopt' if ok else f'out of bounds [{lo}, {hi}]: keep'}")
    if args.write and adopt:
        cur = {}
        if os.path.exists(CAL_PATH):
            with open(CAL_PATH) as f:
                cur = json.load(f)
        cur.update(adopt)
        cur["_runs"] = [os.path.basename(os.path.normpath(r)) for r in args.runs]
        with open(CAL_PATH, "w") as f:
            json.dump(cur, f, indent=1)
        print(f"wrote {CAL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
