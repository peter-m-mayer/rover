"""Tests for optical-flow yaw estimation (real cv2, synthetic frames)."""

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from catchaser.flow import FlowRotationEstimator, turn_by_flow, yaw_from_shift


def _textured(w=640, h=480, seed=1):
    rng = np.random.RandomState(seed)
    img = np.full((h, w), 128, np.uint8)
    for _ in range(400):
        x, y = rng.randint(5, w - 5), rng.randint(5, h - 5)
        cv2.circle(img, (x, y), rng.randint(2, 6), int(rng.randint(0, 255)), -1)
    return img


def test_yaw_from_shift_sign_and_scale():
    assert yaw_from_shift(500, 500) == pytest.approx(1.0)   # du=fx -> 1 rad
    assert yaw_from_shift(-250, 500) == pytest.approx(-0.5)  # left shift = CW


class TestEstimator:
    def test_primes_then_measures_known_shift(self):
        fx = 500.0
        est = FlowRotationEstimator(fx)
        base = _textured()
        assert est.update(base) is None            # first frame primes
        shifted = np.roll(base, 20, axis=1)        # scene moved RIGHT 20px = CCW
        dpsi = est.update(shifted)
        assert dpsi is not None
        assert dpsi == pytest.approx(20 / fx, abs=0.01)   # ~ +20/fx rad, CCW

    def test_left_shift_is_negative_yaw(self):
        fx = 500.0
        est = FlowRotationEstimator(fx)
        base = _textured(seed=2)
        est.update(base)
        dpsi = est.update(np.roll(base, -15, axis=1))     # scene left = CW
        assert dpsi < 0

    def test_blank_scene_returns_none(self):
        est = FlowRotationEstimator(500.0)
        blank = np.full((480, 640), 128, np.uint8)
        est.update(blank)
        assert est.update(blank) is None           # no features to track


class TestClosedLoopTurn:
    """Fake camera whose scene shifts in response to the commanded turn — a
    self-contained closed-loop test of the control law + sign, no hardware."""

    class FakeRobot:
        def __init__(self, fx=500.0, px_per_unit=0.30):
            self.fx = fx
            self.k = px_per_unit
            self.total_px = 0.0        # cumulative scene shift
            self._turn = 0.0
            self._base = _textured(seed=7)

        def drive(self, forward, turn=0.0, strafe=0.0):
            self._turn = turn         # CW (turn>0) shifts scene LEFT

        def stop(self):
            self._turn = 0.0

        def capture(self):
            # apply the last commanded turn as scene motion, then render
            self.total_px += -self._turn * self.k     # turn>0 (CW) -> left (-)
            return np.roll(self._base, int(round(self.total_px)), axis=1)

    def test_converges_to_target_ccw(self):
        r = self.FakeRobot()
        measured = turn_by_flow(r, r.capture, r.fx, target_deg=30,
                                tol_deg=3, max_s=1e9,
                                sleep=lambda s: None, now=_fake_clock())
        assert measured == pytest.approx(30, abs=5)
        assert r._turn == 0.0                        # stopped at the end

    def test_converges_to_target_cw(self):
        r = self.FakeRobot()
        measured = turn_by_flow(r, r.capture, r.fx, target_deg=-30,
                                tol_deg=3, max_s=1e9,
                                sleep=lambda s: None, now=_fake_clock())
        assert measured == pytest.approx(-30, abs=5)


def _fake_clock():
    t = [0.0]

    def now():
        t[0] += 0.05
        return t[0]
    return now
