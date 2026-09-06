"""Aggregate triage statistics across harvested datasets.

Reads each dataset's review_status.json (written by catchaser.review) and
reports, per session and combined:

  * detector precision — of the frames where the detector *claimed* a cat
    (non-empty original label), how many the human confirmed as Good (TP) vs
    corrected to No-cat (FP) or Fix (box wrong). This measures the detector,
    not the human's clicks on already-empty frames.
  * final dataset composition — positives (non-empty labels) vs negatives.

    python3 -m catchaser.dataset_stats                 # all of datasets/
    python3 -m catchaser.dataset_stats datasets/0056   # one or more explicit dirs

Pure file I/O + a tiny bit of arithmetic; no hardware, no heavy deps.
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List


@dataclass
class SessionStats:
    name: str
    frames: int = 0
    claimed: int = 0        # detector produced a box (non-empty orig label)
    true_pos: int = 0       # confirmed Good
    false_pos: int = 0      # corrected to No-cat
    fix: int = 0            # cat present, box wrong
    positives: int = 0      # non-empty final labels
    negatives: int = 0      # empty final labels

    @property
    def precision(self) -> float:
        return self.true_pos / self.claimed if self.claimed else 0.0


def _count_positive_labels(label_dir: str) -> int:
    n = 0
    for f in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(f) as fh:
            if fh.read().strip():
                n += 1
    return n


def session_stats(dataset: str) -> SessionStats:
    """Compute stats for one dataset directory."""
    s = SessionStats(name=os.path.basename(os.path.normpath(dataset)))
    s.frames = len(glob.glob(os.path.join(dataset, "images", "*.jpg")))

    status_path = os.path.join(dataset, "review_status.json")
    if os.path.exists(status_path):
        with open(status_path) as f:
            status = json.load(f)
        for _frame, v in status.items():
            decision = v.get("decision")
            claimed = bool((v.get("orig") or "").strip())
            if claimed:
                s.claimed += 1
                if decision == "good":
                    s.true_pos += 1
                elif decision == "nocat":
                    s.false_pos += 1
                elif decision == "fix":
                    s.fix += 1

    s.positives = _count_positive_labels(os.path.join(dataset, "labels"))
    s.negatives = s.frames - s.positives
    return s


def combine(sessions: List[SessionStats], name: str = "COMBINED") -> SessionStats:
    t = SessionStats(name=name)
    for s in sessions:
        t.frames += s.frames
        t.claimed += s.claimed
        t.true_pos += s.true_pos
        t.false_pos += s.false_pos
        t.fix += s.fix
        t.positives += s.positives
        t.negatives += s.negatives
    return t


_ROW = "%-18s %6s %8s %5s %5s %5s %7s %5s %5s"


def format_report(sessions: List[SessionStats], target_positives: int = 500) -> str:
    lines = [_ROW % ("session", "frames", "claimed", "TP", "FP", "fix",
                     "prec", "pos", "neg")]
    for s in sessions:
        lines.append(_ROW % (s.name, s.frames, s.claimed, s.true_pos,
                             s.false_pos, s.fix, f"{s.precision*100:.1f}%",
                             s.positives, s.negatives))
    total = combine(sessions)
    lines.append("-" * 72)
    lines.append(_ROW % (total.name, total.frames, total.claimed, total.true_pos,
                         total.false_pos, total.fix,
                         f"{total.precision*100:.1f}%", total.positives,
                         total.negatives))
    pct = total.positives / target_positives * 100 if target_positives else 0
    lines.append("")
    lines.append(f"training progress: {total.positives} positives of "
                 f"~{target_positives} target ({pct:.0f}%)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dataset triage statistics")
    ap.add_argument("datasets", nargs="*",
                    help="dataset dirs (default: all of datasets/*/)")
    ap.add_argument("--target", type=int, default=500,
                    help="positive-count target for the fine-tune (default 500)")
    args = ap.parse_args(argv)

    dirs = args.datasets or sorted(
        d for d in glob.glob("datasets/*") if os.path.isdir(d))
    if not dirs:
        print("no datasets found (run catchaser.chase --save-dir first)",
              file=sys.stderr)
        return 1

    sessions = [session_stats(d) for d in dirs]
    print(format_report(sessions, target_positives=args.target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
