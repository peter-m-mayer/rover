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
