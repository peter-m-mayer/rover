"""
Shared fixtures for the raspbot_slam test suite.

Provides synthetic images, camera calibrations, ORB features,
and pre-built maps -- all without hardware.
"""

import math
import numpy as np
import pytest

try:
    import cv2
except ImportError:
    pytest.skip("OpenCV required for tests", allow_module_level=True)

from raspbot_slam.camera import Camera
from raspbot_slam.sensors import Sensors
from raspbot_slam.actuators import Actuators
from raspbot_slam.feature_extractor import FeatureExtractor
from raspbot_slam.map_manager import MapManager, OccupancyGrid
from raspbot_slam.state_estimator import EKFSLAM


# =============================================================================
# Synthetic Image Generation
# =============================================================================

@pytest.fixture
def synthetic_textured_image():
    """A 640x480 grayscale image with rich texture for ORB features."""
    img = np.zeros((480, 640), dtype=np.uint8)
    rng = np.random.RandomState(42)

    # Checkerboard background
    for y in range(0, 480, 40):
        for x in range(0, 640, 40):
            if ((x // 40) + (y // 40)) % 2 == 0:
                img[y:y+40, x:x+40] = 200

    # Random shapes for feature-rich texture
    for _ in range(30):
        cx = rng.randint(50, 590)
        cy = rng.randint(50, 430)
        r = rng.randint(5, 30)
        cv2.circle(img, (cx, cy), r, int(rng.randint(50, 255)), -1)

    for _ in range(20):
        x1, y1 = rng.randint(0, 640), rng.randint(0, 480)
        x2, y2 = rng.randint(0, 640), rng.randint(0, 480)
        cv2.line(img, (x1, y1), (x2, y2), int(rng.randint(50, 255)), 2)

    return img


@pytest.fixture
def synthetic_stereo_pair():
    """Two 640x480 images simulating a 5cm lateral shift.

    Left image has features; right image is shifted ~16px right
    (simulating objects at ~1.5m with f=500px, baseline=0.05m).
    """
    rng = np.random.RandomState(123)
    left = np.zeros((480, 640), dtype=np.uint8)

    # Create distinct blobs at known positions
    blob_positions = [(100, 100), (300, 200), (500, 150),
                      (150, 350), (400, 400), (550, 300)]
    for (bx, by) in blob_positions:
        cv2.circle(left, (bx, by), 15, 200, -1)
        cv2.circle(left, (bx, by), 8, 100, -1)

    # Add texture
    for _ in range(50):
        cx = rng.randint(20, 620)
        cy = rng.randint(20, 460)
        cv2.circle(left, (cx, cy), rng.randint(3, 10),
                   int(rng.randint(80, 220)), -1)

    # Right image: shift left image by ~16 pixels (objects at ~1.5m)
    shift_px = 16
    right = np.zeros_like(left)
    right[:, :640-shift_px] = left[:, shift_px:]

    return left, right, shift_px


@pytest.fixture
def two_frame_sequence():
    """Two frames with a small simulated forward motion (translation in Z).

    The second frame is a slightly zoomed version of the first
    (simulating forward motion toward the scene).
    """
    rng = np.random.RandomState(77)
    frame1 = np.zeros((480, 640), dtype=np.uint8)

    # Rich texture
    for y in range(0, 480, 30):
        for x in range(0, 640, 30):
            if ((x // 30) + (y // 30)) % 2 == 0:
                frame1[y:y+30, x:x+30] = rng.randint(150, 220)
            else:
                frame1[y:y+30, x:x+30] = rng.randint(30, 80)

    for _ in range(40):
        cx, cy = rng.randint(50, 590), rng.randint(50, 430)
        cv2.circle(frame1, (cx, cy), rng.randint(5, 20),
                   int(rng.randint(60, 240)), -1)

    # Frame 2: slight zoom (simulates moving forward)
    scale = 1.03
    h, w = frame1.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    frame2 = cv2.warpAffine(frame1, M, (w, h))

    return frame1, frame2


# =============================================================================
# Camera and Hardware Mocks
# =============================================================================

@pytest.fixture
def mock_camera():
    """Camera with known calibration, no hardware."""
    cam = Camera(device=None)
    K = np.array([[500.0, 0, 320.0],
                  [0, 500.0, 240.0],
                  [0, 0, 1.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    cam.set_calibration(K, dist)
    return cam


@pytest.fixture
def mock_sensors():
    """Sensors with bot=None (mock mode)."""
    return Sensors(bot=None)


@pytest.fixture
def mock_actuators():
    """Actuators with bot=None (mock mode)."""
    return Actuators(bot=None)


@pytest.fixture
def feature_extractor():
    """FeatureExtractor with default config."""
    return FeatureExtractor()


# =============================================================================
# Pre-built Map Fixtures
# =============================================================================

@pytest.fixture
def simple_room_grid():
    """A 100x100 occupancy grid with a 4m x 4m room (walls + free interior).

    Grid resolution 0.05m, origin at center.
    Room from (-2,-2) to (2,2) meters.
    """
    grid = OccupancyGrid(size_cells=100, resolution_m=0.05)
    grid.origin_offset = 50  # center

    # Mark interior as free (leave a border of unknown outside the walls)
    for y in range(20, 80):
        for x in range(20, 80):
            grid.update_cell(x, y, -2.0)  # strongly free

    # Mark walls as occupied
    for i in range(20, 80):
        grid.update_cell(i, 20, 3.0)   # bottom wall
        grid.update_cell(i, 79, 3.0)   # top wall
        grid.update_cell(20, i, 3.0)   # left wall
        grid.update_cell(79, i, 3.0)   # right wall

    # Cells outside walls (0-19, 80-99) remain unknown (default 0.0)
    # Free cells just inside walls (21-78) will have unknown neighbors through walls
    return grid


@pytest.fixture
def simple_map(simple_room_grid):
    """MapManager with a simple room and some landmarks."""
    mm = MapManager()
    mm.occupancy = simple_room_grid

    rng = np.random.RandomState(99)
    # Add landmarks along the walls
    wall_positions = [
        (0.5, -1.8, 1.0), (-0.5, -1.8, 1.0),   # bottom wall
        (0.5, 1.8, 1.0), (-0.5, 1.8, 1.0),       # top wall
        (-1.8, 0.5, 1.0), (-1.8, -0.5, 1.0),     # left wall
        (1.8, 0.5, 1.0), (1.8, -0.5, 1.0),       # right wall
    ]
    for pos in wall_positions:
        desc = rng.randint(0, 256, size=32).astype(np.uint8)
        mm.add_landmark(np.array(pos), desc)

    return mm


@pytest.fixture
def ekf():
    """EKF initialized at origin."""
    return EKFSLAM(initial_pose=(0, 0, 0))
