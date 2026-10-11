import numpy as np

from catchaser.camera_zero import foe_from_flow


def _synthetic(foe, rng, n=150, outliers=20):
    pts = rng.uniform([0, 0], [640, 480], size=(n, 2))
    depth_scale = rng.uniform(0.02, 0.15, size=(n, 1))   # varied depths
    flow = (pts - foe) * depth_scale + rng.normal(0, 0.2, size=(n, 2))
    flow[:outliers] = rng.normal(0, 8, size=(outliers, 2))  # moving cat etc.
    return pts, flow


def test_foe_recovered_with_outliers():
    rng = np.random.default_rng(1)
    foe = np.array([371.0, 205.0])
    pts, flow = _synthetic(foe, rng)
    e, frac = foe_from_flow(pts, flow)
    assert e is not None and frac > 0.6
    assert np.allclose(e, foe, atol=3.0)


def test_contraction_gives_same_point():
    rng = np.random.default_rng(2)
    foe = np.array([290.0, 260.0])
    pts, flow = _synthetic(foe, rng)
    e, _ = foe_from_flow(pts, -flow)          # driving backward
    assert np.allclose(e, foe, atol=3.0)


def test_too_few_points():
    e, frac = foe_from_flow(np.zeros((5, 2)), np.ones((5, 2)))
    assert e is None and frac == 0.0


def test_foe_recovered_despite_yaw():
    import math
    from catchaser.camera_zero import foe_with_yaw, yaw_flow
    rng = np.random.default_rng(3)
    foe = np.array([350.0, 250.0])
    pts, flow = _synthetic(foe, rng, outliers=10)
    flow = flow + yaw_flow(pts, math.radians(1.2), 425.0, (320, 240))
    e, yaw, frac = foe_with_yaw(pts, flow, 425.0, (320, 240))
    assert abs(yaw - 1.2) < 0.15
    assert np.allclose(e, foe, atol=8.0)
