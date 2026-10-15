"""
Score one navigation run automatically, so runs can be compared without reading logs.

    python -m raspbot_slam.offline.score_run datasets/slam_runs/<run> [--replay] [--no-append]

From the run folder (fast, no image processing):
  - tour stops and explore goals (events "tourstop {...}", "exploregoal {...}"):
    goals, reached, error at each stop
    against its tag fix (m, deg), why each failed leg gave up
  - executor troubles counted from the event log: contacts, blocks, stalled
    turns, relaxed plans, give-ups by reason, heading and position resets
  - distance driven (nav.json path, written by tag_tour / explore since 2026-10-12)
  - the saved room_map.npz: coverage (seen fraction of the surveyed floor) and
    wall offset (median distance of camera obstacle hits within 0.3 m of a
    surveyed wall to that wall: how well the map lines up with the survey)
With --replay, also replays the frames (eval_tags.replay, minutes per run):
dead reckoning with no correction against every tag fix group, the drift a
run would have without tags.

Writes score.json into the run folder and appends a row to
datasets/slam_runs/scores.csv (one row per run; a re-score replaces its row).
"""

import argparse
import csv
import json
import math
import os
import re

import numpy as np

SCORES_CSV = os.path.join("datasets", "slam_runs", "scores.csv")
FIELDS = ["run", "kind", "goals", "reached", "stop_err_med_m", "stop_err_max_m", "stop_err_med_deg",
          "contacts", "blocked", "turn_stalls", "relaxed", "gave_up", "wedged", "heading_resets",
          "position_resets", "distance_m", "coverage", "wall_offset_m", "replay_err_med_m", "replay_fixes",
          "seconds", "battery_mps", "battery_ratio"]

# event label -> counter (the executor's and localizer's own wording)
PATTERNS = [
    ("contacts", re.compile(r"^contact: hop moved")),
    ("blocked", re.compile(r"^blocked ahead")),
    ("turn_stalls", re.compile(r"^turn stalled")),
    ("relaxed", re.compile(r"^no normal path")),
    ("heading_resets", re.compile(r"heading_reset")),
    ("position_resets", re.compile(r"position_reset|position reset")),
    ("wedged", re.compile(r"^wedged")),
]
GAVE_UP = re.compile(r"(no path to|giving up on this goal|^wedged)")


def read_events(run_dir):
    p = os.path.join(run_dir, "events.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="") as f:
        return [(float(r["t"]), r["label"]) for r in csv.DictReader(f)]


def event_counts(events):
    out = {k: 0 for k, _ in PATTERNS}
    out["gave_up"] = 0
    stops = []
    for _, label in events:
        for k, rx in PATTERNS:
            if rx.search(label):
                out[k] += 1
        if GAVE_UP.search(label):
            out["gave_up"] += 1
        if label.startswith("battery "):                 # nav_executor.speed_check at the start
            try:
                b = json.loads(label[len("battery "):])
                out["battery_mps"], out["battery_ratio"] = b.get("v_mps"), b.get("ratio")
            except ValueError:
                pass
        for prefix in ("tourstop ", "exploregoal "):     # tag_tour stops, explore goals
            if label.startswith(prefix):
                try:
                    stops.append(json.loads(label[len(prefix):]))
                except ValueError:
                    pass
    return out, stops


def stop_scores(stops):
    reached = [s for s in stops if s.get("reached")]
    errs = [s["err_m"] for s in stops if s.get("err_m") is not None]
    degs = [abs(s["err_deg"]) for s in stops if s.get("err_deg") is not None]
    return {
        "goals": len(stops),
        "reached": len(reached),
        "stop_err_med_m": float(np.median(errs)) if errs else None,
        "stop_err_max_m": float(np.max(errs)) if errs else None,
        "stop_err_med_deg": float(np.median(degs)) if degs else None,
        "why": [s.get("why") for s in stops if not s.get("reached")],
    }


def map_scores(m, survey=None):
    """Coverage of the surveyed floor and the offset of camera hits from the surveyed walls."""
    import cv2
    floor = ~m.wall
    coverage = float((m.seen & floor).sum() / max(1, floor.sum()))
    dist = cv2.distanceTransform((~m.wall).astype(np.uint8), cv2.DIST_L2, 5) * m.res
    near = (m.hits >= 2) & ~m.wall & (dist < 0.3)
    wall_offset = float(np.median(dist[near])) if near.any() else None
    return {"coverage": coverage, "wall_offset_m": wall_offset}


def path_length(path):
    p = np.asarray(path, float)
    return float(np.sum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))) if len(p) > 1 else 0.0


def score(run_dir, replay=False, log=print):
    from ..room_map import RoomMap
    events = read_events(run_dir)
    counts, stops = event_counts(events)
    s = {"run": os.path.basename(os.path.normpath(run_dir))}
    s["kind"] = s["run"].split("_", 2)[-1] if s["run"].count("_") >= 2 else ""
    s.update(counts)
    s.update(stop_scores(stops))
    s["seconds"] = round(events[-1][0] - events[0][0], 1) if len(events) > 1 else None
    nav_p = os.path.join(run_dir, "nav.json")
    if os.path.exists(nav_p):
        with open(nav_p) as f:
            nav = json.load(f)
        s["distance_m"] = path_length(nav.get("path", []))
    map_p = os.path.join(run_dir, "room_map.npz")
    if os.path.exists(map_p):
        s.update(map_scores(RoomMap.load(map_p)))
    if replay:
        from .eval_tags import replay as do_replay
        out = do_replay(run_dir, None, localizer=True, correct=False, log=lambda m: None)
        if out is not None:
            good = [r for r in out["rows"] if not r["outlier"]]
            s["replay_err_med_m"] = float(np.median([r["err_m"] for r in good])) if good else None
            s["replay_fixes"] = len(good)
            if "distance_m" not in s and len(out["path"]):
                s["distance_m"] = path_length(out["path"][:, 1:3])
            if out["stats"]:
                s["heading_resets"] = max(s["heading_resets"], out["stats"]["heading_resets"])
    return s


def summary_line(s):
    def f(v, fmt):
        return "-" if v is None else format(v, fmt)
    return (f"{s['run']}: reached {s['reached']}/{s['goals']}, stop error median {f(s['stop_err_med_m'], '.2f')} m "
            f"(max {f(s['stop_err_max_m'], '.2f')}), contacts {s['contacts']}, blocked {s['blocked']}, "
            f"turn stalls {s['turn_stalls']}, gave up {s['gave_up']} {s.get('why') or ''}, "
            f"distance {f(s.get('distance_m'), '.1f')} m, coverage {f(s.get('coverage'), '.0%')}, "
            f"wall offset {f(s.get('wall_offset_m'), '.2f')} m, battery {f(s.get('battery_ratio'), '.0%')}")


def append_csv(s, path=SCORES_CSV):
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["run"] != s["run"]]
    rows.append({k: ("" if s.get(k) is None else s.get(k)) for k in FIELDS})
    rows.sort(key=lambda r: r["run"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Score a navigation run.")
    p.add_argument("runs", nargs="+")
    p.add_argument("--replay", action="store_true", help="also replay frames for uncorrected drift (slow)")
    p.add_argument("--no-append", action="store_true", help="don't add to datasets/slam_runs/scores.csv")
    args = p.parse_args(argv)
    for run in args.runs:
        s = score(run, replay=args.replay)
        with open(os.path.join(run, "score.json"), "w") as f:
            json.dump(s, f, indent=1)
        if not args.no_append:
            append_csv(s)
        print(summary_line(s), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
