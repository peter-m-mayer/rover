"""Tests for the ONNX cat detector (runs anywhere with onnxruntime; no robot)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")

from catchaser.detector import (  # noqa: E402
    COCO_CAT, CatDetector, Detection, best_target, letterbox,
)


class TestLetterbox:
    def test_square_output_and_roundtrip(self):
        img = np.zeros((300, 451, 3), dtype=np.uint8)
        out, scale, (pad_x, pad_y) = letterbox(img, 320)
        assert out.shape == (320, 320, 3)
        # A point at original (451, 300) maps inside the padded image
        # and back again: orig = (padded - pad) / scale.
        px, py = 451 * scale + pad_x, 300 * scale + pad_y
        assert px <= 320 + 1e-6 and py <= 320 + 1e-6
        assert abs((px - pad_x) / scale - 451) < 1.0
        assert abs((py - pad_y) / scale - 300) < 1.0

    def test_padding_is_gray(self):
        img = np.zeros((100, 320, 3), dtype=np.uint8)
        out, _, _ = letterbox(img, 320)
        assert out[0, 0, 0] == 114  # top padding row


class TestDetectionGeometry:
    def test_centroid_and_area(self):
        d = Detection(10, 20, 110, 220, 0.9, COCO_CAT)
        assert d.cx == 60 and d.cy == 120
        assert d.width == 100 and d.height == 200
        assert d.area == 20000

    def test_best_target_empty(self):
        assert best_target([]) is None


class TestCatDetector:
    @pytest.fixture(scope="class")
    def detector(self):
        return CatDetector()

    def test_detects_real_cat(self, detector):
        skimage_data = pytest.importorskip("skimage.data")
        frame = cv2.cvtColor(skimage_data.chelsea(), cv2.COLOR_RGB2BGR)
        dets = detector.detect(frame)
        assert len(dets) >= 1
        top = best_target(dets)
        assert top.class_id == COCO_CAT
        assert top.confidence > 0.5
        # Centroid roughly mid-frame for this near-full-frame cat photo
        h, w = frame.shape[:2]
        assert 0.2 * w < top.cx < 0.8 * w

    def test_no_cat_in_gradient(self, detector):
        neg = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
        neg = cv2.cvtColor(neg, cv2.COLOR_GRAY2BGR)
        assert detector.detect(neg) == []

    def test_reports_latency(self, detector):
        neg = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(neg)
        assert detector.last_inference_ms > 0
