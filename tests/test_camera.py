"""Tests for camera.py -- calibration and undistortion math."""

import json
import os
import tempfile
import numpy as np
import pytest

from raspbot_slam.camera import Camera


def test_default_intrinsics():
    cam = Camera(device=None)
    K = cam.K
    assert K.shape == (3, 3)
    assert K[0, 0] > 0  # fx
    assert K[1, 1] > 0  # fy
    assert K[2, 2] == 1.0


def test_set_calibration():
    cam = Camera(device=None)
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.array([0.1, -0.2, 0, 0, 0.05], dtype=np.float64)
    cam.set_calibration(K, dist)

    assert np.allclose(cam.K, K)
    assert np.allclose(cam.dist_coeffs, dist)
    assert cam.focal_length_px == 600.0
    assert cam.principal_point == (320.0, 240.0)


def test_save_load_calibration():
    cam = Camera(device=None)
    K = np.array([[550, 0, 315], [0, 552, 238], [0, 0, 1]], dtype=np.float64)
    dist = np.array([0.05, -0.1, 0.001, -0.001, 0.02], dtype=np.float64)
    cam.set_calibration(K, dist)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cal.json")
        cam.save_calibration(path)

        # Load into a new camera
        cam2 = Camera(device=None, calibration_file=path)
        assert np.allclose(cam2.K, K, atol=1e-6)
        assert np.allclose(cam2.dist_coeffs, dist, atol=1e-6)


def test_undistort_points(mock_camera):
    """Undistortion with zero distortion should return normalized coordinates."""
    # Points at the principal point should map to (0, 0)
    pts = np.array([[320.0, 240.0]], dtype=np.float64)
    result = mock_camera.undistort_points(pts)
    assert result.shape == (1, 2)
    assert abs(result[0, 0]) < 0.01
    assert abs(result[0, 1]) < 0.01


def test_undistort_points_batch(mock_camera):
    """Multiple points undistorted at once."""
    pts = np.array([[0, 0], [320, 240], [640, 480]], dtype=np.float64)
    result = mock_camera.undistort_points(pts)
    assert result.shape == (3, 2)


def test_focal_length_average(mock_camera):
    assert mock_camera.focal_length_px == 500.0


def test_width_height():
    cam = Camera(device=None, width=320, height=240)
    assert cam.width == 320
    assert cam.height == 240


class TestAutoBrightness:
    """Auto-brightness must brighten dark scenes with GAIN first (keeping the
    shutter short), and only lengthen exposure once gain is maxed."""

    def _cam(self):
        c = Camera()
        # stub the hardware write so no v4l2/subprocess runs in the test
        c._apply_exposure = lambda: None
        c.enable_auto_brightness(target=125, exp_short=78, exp_max=220,
                                 gain_max=8, every=1)
        return c

    def test_dark_scene_raises_gain_first(self):
        c = self._cam()
        dark = np.full((48, 64, 3), 40, np.uint8)   # mean 40 << target
        for _ in range(4):
            c.auto_brightness(dark)
        assert c._gain > 1                # gain climbed
        assert c._exposure == 78          # exposure still at the short floor

    def test_gain_caps_then_exposure_lengthens(self):
        c = self._cam()
        dark = np.zeros((48, 64, 3), np.uint8)      # pitch black
        for _ in range(20):
            c.auto_brightness(dark)
        assert c._gain == 8               # gain hit the ceiling
        assert c._exposure > 78           # then exposure had to grow

    def test_bright_scene_shortens_exposure_first(self):
        c = self._cam()
        c.set_manual_exposure(200, 6)     # start long+high
        bright = np.full((48, 64, 3), 240, np.uint8)
        c.auto_brightness(bright)
        assert c._exposure < 200          # shutter shortened before touching gain

    def test_deadband_no_change(self):
        c = self._cam()
        c.set_manual_exposure(78, 4)
        ok = np.full((48, 64, 3), 125, np.uint8)    # exactly on target
        c.auto_brightness(ok)
        assert (c._exposure, c._gain) == (78, 4)
