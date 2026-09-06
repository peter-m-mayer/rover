"""Tests for the dataset triage statistics tool."""

import json
import os

from catchaser.dataset_stats import combine, format_report, session_stats


def make_dataset(root, frames):
    """frames: list of (stem, orig_label, decision, final_label)."""
    os.makedirs(os.path.join(root, "images"))
    os.makedirs(os.path.join(root, "labels"))
    status = {}
    for stem, orig, decision, final in frames:
        open(os.path.join(root, "images", stem + ".jpg"), "wb").close()
        with open(os.path.join(root, "labels", stem + ".txt"), "w") as f:
            f.write(final)
        status[stem + ".jpg"] = {"decision": decision, "orig": orig}
    with open(os.path.join(root, "review_status.json"), "w") as f:
        json.dump(status, f)


BOX = "0 0.5 0.5 0.3 0.3\n"


class TestSessionStats:
    def test_precision_counts_only_detector_claims(self, tmp_path):
        ds = str(tmp_path / "s")
        make_dataset(ds, [
            ("a", BOX, "good", BOX),    # TP
            ("b", BOX, "good", BOX),    # TP
            ("c", BOX, "nocat", ""),    # FP: detector claimed, human rejected
            ("d", BOX, "fix", BOX),     # fix
            ("e", "", "nocat", ""),     # empty orig -> NOT a detector claim
        ])
        s = session_stats(ds)
        assert s.frames == 5
        assert s.claimed == 4          # 'e' excluded — detector claimed nothing
        assert s.true_pos == 2
        assert s.false_pos == 1
        assert s.fix == 1
        assert s.precision == 0.5      # 2 / 4

    def test_positive_negative_composition(self, tmp_path):
        ds = str(tmp_path / "s")
        make_dataset(ds, [
            ("a", BOX, "good", BOX),
            ("b", "", "nocat", ""),
            ("c", BOX, "nocat", ""),   # rejected -> now negative
        ])
        s = session_stats(ds)
        assert s.positives == 1
        assert s.negatives == 2

    def test_no_status_file_is_safe(self, tmp_path):
        ds = str(tmp_path / "s")
        os.makedirs(os.path.join(ds, "images"))
        os.makedirs(os.path.join(ds, "labels"))
        open(os.path.join(ds, "images", "a.jpg"), "wb").close()
        with open(os.path.join(ds, "labels", "a.txt"), "w") as f:
            f.write(BOX)
        s = session_stats(ds)
        assert s.frames == 1 and s.positives == 1 and s.claimed == 0


class TestCombine:
    def test_combine_sums_and_reprecision(self, tmp_path):
        d1 = str(tmp_path / "s1")
        d2 = str(tmp_path / "s2")
        make_dataset(d1, [("a", BOX, "good", BOX), ("b", BOX, "nocat", "")])
        make_dataset(d2, [("c", BOX, "good", BOX), ("d", BOX, "good", BOX)])
        total = combine([session_stats(d1), session_stats(d2)])
        assert total.claimed == 4
        assert total.true_pos == 3
        assert total.precision == 0.75


class TestReport:
    def test_report_has_combined_and_progress(self, tmp_path):
        ds = str(tmp_path / "s")
        make_dataset(ds, [("a", BOX, "good", BOX)])
        out = format_report([session_stats(ds)], target_positives=500)
        assert "COMBINED" in out
        assert "100.0%" in out
        assert "1 positives of ~500 target" in out
