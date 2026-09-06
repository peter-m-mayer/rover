"""Tests for the dataset harvester (chase --save-dir output)."""

import os

import numpy as np
import pytest

from catchaser.dataset import save_sample, yolo_lines
from catchaser.detector import COCO_CAT, Detection


def det(x1, y1, x2, y2, conf=0.9):
    return Detection(x1, y1, x2, y2, confidence=conf, class_id=COCO_CAT)


class TestYoloLines:
    def test_centered_box(self):
        # 100x100 box centered in a 640x480 frame.
        d = det(270, 190, 370, 290)
        line = yolo_lines([d], 640, 480).strip()
        cls, cx, cy, w, h = line.split()
        assert cls == "0"                        # dataset-local class, not COCO 15
        assert float(cx) == pytest.approx(0.5, abs=1e-5)
        assert float(cy) == pytest.approx(0.5, abs=1e-5)
        assert float(w) == pytest.approx(100 / 640, abs=1e-5)
        assert float(h) == pytest.approx(100 / 480, abs=1e-5)

    def test_empty_detections_empty_content(self):
        assert yolo_lines([], 640, 480) == ""

    def test_multiple_detections_multiple_lines(self):
        out = yolo_lines([det(0, 0, 10, 10), det(50, 50, 90, 90)], 640, 480)
        assert len(out.strip().splitlines()) == 2

    def test_out_of_frame_coords_clamped(self):
        d = det(-20, -20, 700, 500)              # box larger than the frame
        _, cx, cy, w, h = yolo_lines([d], 640, 480).strip().split()
        for v in (cx, cy, w, h):
            assert 0.0 <= float(v) <= 1.0


class TestSaveSample:
    def _frame(self):
        return np.full((480, 640, 3), 128, dtype=np.uint8)

    def test_writes_review_compatible_layout(self, tmp_path):
        root = str(tmp_path / "ds")
        save_sample(root, "f0001", self._frame(), [det(100, 100, 200, 200)])
        assert os.path.exists(os.path.join(root, "images", "f0001.jpg"))
        assert os.path.exists(os.path.join(root, "labels", "f0001.txt"))
        assert os.path.exists(os.path.join(root, "preview", "f0001.jpg"))
        with open(os.path.join(root, "labels", "f0001.txt")) as f:
            assert f.read().startswith("0 ")

    def test_negative_frame_gets_empty_label(self, tmp_path):
        root = str(tmp_path / "ds")
        save_sample(root, "f0002", self._frame(), [])
        with open(os.path.join(root, "labels", "f0002.txt")) as f:
            assert f.read() == ""

    def test_clean_image_has_no_burned_in_box(self, tmp_path):
        import cv2
        root = str(tmp_path / "ds")
        save_sample(root, "f0003", self._frame(), [det(100, 100, 200, 200)])
        img = cv2.imread(os.path.join(root, "images", "f0003.jpg"))
        # A pure green box line would show as bright green pixels; the clean
        # image must stay uniform gray (allow jpg noise).
        assert int(img[:, :, 1].max()) - int(img[:, :, 1].min()) < 30
        prev = cv2.imread(os.path.join(root, "preview", "f0003.jpg"))
        assert int(prev[:, :, 1].max()) > 200   # preview does have the box

    def test_review_store_consumes_harvest(self, tmp_path):
        pytest.importorskip("flask")
        from catchaser.review import ReviewStore
        root = str(tmp_path / "ds")
        save_sample(root, "f0004", self._frame(), [det(100, 100, 200, 200)])
        save_sample(root, "f0005", self._frame(), [])
        store = ReviewStore(root)
        assert set(store.frames) == {"f0004.jpg", "f0005.jpg"}
        store.decide("f0004.jpg", "good")
        assert store._read_label("f0004.jpg").startswith("0 ")
