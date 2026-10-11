"""Floor odometry: render a textured floor from known poses, recover the motion."""

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from raspbot_slam.floor_odometry import FloorModel, FloorOdometry, rigid_2d, robust_translation

K = np.array([[473.0, 0, 320], [0, 473.0, 240], [0, 0, 1]])
DIST = np.array([-0.30, 0.10, 0, 0, 0])
H, PITCH = 0.107, 0.3

rng = np.random.default_rng(3)
TEX_RES = 0.002                                   # m per texel
TEX = cv2.GaussianBlur((rng.random((1500, 1500)) * 255).astype(np.uint8), (0, 0), 1.5)


def render(model, x, y, th):
    """Camera image of the textured floor with the camera at floor pose (x, y, th)."""
    h, w = 480, 640
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.c_[uu.ravel(), vv.ravel()].astype(np.float64)
    f = model.to_floor(pts)                        # floor coords in the camera's frame
    c, s = math.cos(th), math.sin(th)
    wx = x + c * f[:, 0] - s * f[:, 1]
    wy = y + s * f[:, 0] + c * f[:, 1]
    mx = (wx / TEX_RES + 750).astype(np.float32).reshape(h, w)
    my = (wy / TEX_RES + 750).astype(np.float32).reshape(h, w)
    img = cv2.remap(TEX, mx, my, cv2.INTER_LINEAR, borderValue=0)
    img[np.isnan(f[:, 0]).reshape(h, w)] = 90      # wall / far area: flat grey
    return img


def test_floor_projection_geometry():
    m = FloorModel(K, np.zeros(5), H, 0.0)
    # a pixel 473 px below the principal point looks 45 deg down: floor at distance h
    xy = m.to_floor([[320, 240 + 473]])
    assert xy[0] == pytest.approx([H, 0.0], abs=1e-6)
    # pixels left of centre map to +Y (left)
    assert m.to_floor([[100, 400]])[0, 1] > 0
    assert np.isnan(m.to_floor([[320, 100]])[0, 0])            # above the horizon


def test_rigid_and_translation_fits():
    b = rng.random((30, 2))
    th = 0.1
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    a = b @ R.T + [0.03, -0.01]
    R2, t2 = rigid_2d(a, b)
    assert np.allclose(R2, R) and np.allclose(t2, [0.03, -0.01])
    dx, dy, _, inl = robust_translation(a, b, th)
    assert (dx, dy) == pytest.approx((0.03, -0.01), abs=1e-9) and inl.all()


@pytest.mark.parametrize("motion", [(0.04, 0.0, 0.0), (-0.03, 0.0, 0.0), (0.02, 0.01, 0.05)])
@pytest.mark.parametrize("use_gyro", [False, True])
def test_recovers_motion(motion, use_gyro):
    model = FloorModel(K, DIST, H, PITCH)
    odo = FloorOdometry(model)
    odo.step(render(model, 0.0, 0.0, 0.0))
    dx, dy, dth = motion
    r = odo.step(render(model, dx, dy, dth), dtheta=dth if use_gyro else None)
    assert r is not None
    assert r[0] == pytest.approx(dx, abs=0.002)
    assert r[1] == pytest.approx(dy, abs=0.002)
    assert r[2] == pytest.approx(dth, abs=0.003)
