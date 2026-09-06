"""Tests for the label review & triage tool (no hardware)."""

import json
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("flask")

from catchaser.review import ReviewStore, make_app  # noqa: E402


def _dataset(tmp_path, n=3, with_box=True):
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    for k in range(n):
        img = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(ds / "images" / f"f{k}.jpg"), img)
        txt = "0 0.5 0.5 0.3 0.3\n" if with_box else ""
        (ds / "labels" / f"f{k}.txt").write_text(txt)
    return str(ds)


class TestReviewStore:
    def test_lists_frames_all_pending(self, tmp_path):
        s = ReviewStore(_dataset(tmp_path))
        assert len(s.frames) == 3
        assert s.counts()["pending"] == 3

    def test_good_keeps_label(self, tmp_path):
        s = ReviewStore(_dataset(tmp_path))
        s.decide("f0.jpg", "good")
        assert s.decision("f0.jpg") == "good"
        assert s._read_label("f0.jpg").strip() == "0 0.5 0.5 0.3 0.3"
        assert s.counts()["good"] == 1

    def test_nocat_empties_label(self, tmp_path):
        s = ReviewStore(_dataset(tmp_path))
        s.decide("f1.jpg", "nocat")
        assert s._read_label("f1.jpg").strip() == ""      # forced negative
        assert s.counts()["nocat"] == 1

    def test_undo_is_lossless(self, tmp_path):
        s = ReviewStore(_dataset(tmp_path))
        s.decide("f2.jpg", "nocat")
        assert s._read_label("f2.jpg").strip() == ""
        s.decide("f2.jpg", "undo")
        assert s._read_label("f2.jpg").strip() == "0 0.5 0.5 0.3 0.3"   # restored
        assert s.decision("f2.jpg") is None

    def test_fix_quarantines_copies(self, tmp_path):
        ds = _dataset(tmp_path)
        s = ReviewStore(ds)
        s.decide("f0.jpg", "fix")
        assert os.path.exists(os.path.join(ds, "review_fix", "images", "f0.jpg"))
        assert os.path.exists(os.path.join(ds, "review_fix", "labels", "f0.txt"))
        # changing the mind removes the quarantine copy
        s.decide("f0.jpg", "good")
        assert not os.path.exists(os.path.join(ds, "review_fix", "images", "f0.jpg"))

    def test_status_persists_across_reload(self, tmp_path):
        ds = _dataset(tmp_path)
        ReviewStore(ds).decide("f0.jpg", "good")
        s2 = ReviewStore(ds)             # fresh load reads review_status.json
        assert s2.decision("f0.jpg") == "good"

    def test_bad_decision_rejected(self, tmp_path):
        s = ReviewStore(_dataset(tmp_path))
        with pytest.raises(ValueError):
            s.decide("f0.jpg", "dog")

    def test_is_boxed_reflects_detector_output(self, tmp_path):
        boxed = ReviewStore(_dataset(tmp_path, with_box=True))
        empty = ReviewStore(_dataset(tmp_path / "e", with_box=False))
        assert boxed.is_boxed("f0.jpg") is True
        assert empty.is_boxed("f0.jpg") is False

    def test_missed_quarantines_for_boxing(self, tmp_path):
        ds = _dataset(tmp_path, with_box=False)   # detector drew nothing
        s = ReviewStore(ds)
        s.decide("f0.jpg", "missed")
        assert s.decision("f0.jpg") == "missed"
        assert s.counts()["missed"] == 1
        assert os.path.exists(os.path.join(ds, "review_fix", "images", "f0.jpg"))

    def test_fix_traps_reopens_good_on_empty(self, tmp_path):
        ds = _dataset(tmp_path, with_box=False)
        s = ReviewStore(ds)
        s.decide("f0.jpg", "good")        # the old trap: cat kept as negative
        s.decide("f1.jpg", "nocat")       # legitimately empty — must NOT reopen
        assert s.resurface_traps() == 1
        assert s.decision("f0.jpg") is None      # reopened
        assert s.decision("f1.jpg") == "nocat"   # untouched


class TestReviewApp:
    def test_list_and_decide(self, tmp_path):
        s = ReviewStore(_dataset(tmp_path))
        client = make_app(s).test_client()
        lst = client.get("/api/list").get_json()
        assert len(lst["items"]) == 3
        r = client.post("/api/decide", json={"frame": "f0.jpg", "decision": "nocat"}).get_json()
        assert r["decision"] == "nocat" and r["counts"]["nocat"] == 1

    def test_img_endpoint(self, tmp_path):
        s = ReviewStore(_dataset(tmp_path))
        client = make_app(s).test_client()
        resp = client.get("/img/f0.jpg")
        assert resp.status_code == 200 and resp.mimetype == "image/jpeg"

    def test_decide_bad_frame_400(self, tmp_path):
        s = ReviewStore(_dataset(tmp_path))
        client = make_app(s).test_client()
        assert client.post("/api/decide",
                           json={"frame": "nope.jpg", "decision": "good"}).status_code == 400
