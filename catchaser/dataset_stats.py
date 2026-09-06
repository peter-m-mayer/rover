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
    true_pos: int = 0       # detector boxed, confirmed Good
    false_pos: int = 0      # detector boxed, corrected to No-cat
    fix: int = 0            # detector boxed, cat present but box wrong
    true_neg: int = 0       # detector empty, confirmed no cat
    false_neg: int = 0      # detector empty, but a cat was there (missed)
    positives: int = 0      # non-empty final labels
    negatives: int = 0      # empty final labels

    # --- confusion-matrix view (classification: "is a cat present?") ---------
    # A box-fix is a correctly-classified cat with a bad box, so it counts as a
    # true positive here; the imperfect box is a separate localization concern.
    @property
    def cm_tp(self) -> int:
        return self.true_pos + self.fix

    @property
    def cm_fp(self) -> int:
        return self.false_pos

    @property
    def cm_fn(self) -> int:
        return self.false_neg

    @property
    def cm_tn(self) -> int:
        return self.true_neg

    @property
    def precision(self) -> float:
        d = self.cm_tp + self.cm_fp
        return self.cm_tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.cm_tp + self.cm_fn
        return self.cm_tp / d if d else 0.0


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
                # Detector drew a box.
                s.claimed += 1
                if decision == "good":
                    s.true_pos += 1
                elif decision == "nocat":
                    s.false_pos += 1
                elif decision == "fix":
                    s.fix += 1
            else:
                # Detector drew NO box. 'nocat' confirms it was empty (TN);
                # 'missed' is the explicit false-negative action. 'good'/'fix'
                # on a no-box frame are the OLD trap (pre-fix data) where the
                # reviewer affirmed a cat but the empty label was kept — also
                # a missed cat (verified 2026-09-07 by eye). All count as FN.
                if decision == "nocat":
                    s.true_neg += 1
                elif decision in ("missed", "good", "fix"):
                    s.false_neg += 1

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
        t.true_neg += s.true_neg
        t.false_neg += s.false_neg
        t.positives += s.positives
        t.negatives += s.negatives
    return t


def format_matrix(total: SessionStats) -> str:
    """Render the classic 2x2 confusion matrix + precision/recall/F1."""
    tp, fp, fn, tn = total.cm_tp, total.cm_fp, total.cm_fn, total.cm_tn
    p, r = total.precision, total.recall
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    lines = [
        "",
        "confusion matrix (positive = cat present):",
        "                     actual: CAT    actual: NO CAT",
        f"  predicted: CAT     TP = {tp:<8d}   FP = {fp:d}",
        f"  predicted: NO CAT  FN = {fn:<8d}   TN = {tn:d}",
        "",
        f"  precision = {p*100:5.1f}%   recall = {r*100:5.1f}%   "
        f"F1 = {f1*100:5.1f}%   accuracy = {acc*100:5.1f}%",
    ]
    if total.fix:
        lines.append(f"  note: {total.fix} detection(s) had an imperfect box "
                     f"(localization, counted TP here).")
    if total.false_neg:
        lines.append("  note: misses (FN) are currently MISLABELED as negatives "
                     "in labels/ — re-box before fine-tuning.")
    return "\n".join(lines)


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
    lines.append(format_matrix(total))
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
