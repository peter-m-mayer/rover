"""Camera optical-flow yaw estimation — measure ACTUAL rotation, closed-loop.

The rover has no wheel encoders, so a commanded rotation is only a suggestion
(worse on carpet, where mecanum wheels scrub). This estimates how far the robot
*actually* yawed from how far the scene slid across the camera: rotating the
body CCW by dpsi shifts background features to the RIGHT by about fx*dpsi, so

    dpsi (CCW, rad) ~= median_horizontal_feature_flow_px / fx

The median rejects a moving cat (a minority of outlier features). Sign matches
the rest of the stack: body turn > 0 = clockwise = right = yaw decreasing.

    est = FlowRotationEstimator(camera.focal_length_px)
    est.update(gray0)                 # prime
    dpsi = est.update(gray1)          # CCW yaw radians since gray0 (or None)

    turn_by_flow(actuators, camera.capture, fx, target_deg=90)   # closed-loop turn
"""

import math
import time

import numpy as np


def yaw_from_shift(du_px: float, fx: float) -> float:
    """CCW yaw (radians) for a rightward scene shift of du_px (small-angle)."""
    return du_px / float(fx)


class FlowRotationEstimator:
    """Sparse Lucas-Kanade flow -> robust median horizontal shift -> yaw."""

    def __init__(self, fx: float, min_features: int = 8, max_features: int = 200,
                 quality: float = 0.01, min_distance: int = 7):
        self.fx = float(fx)
        self.min_features = min_features
        self.max_features = max_features
        self.quality = quality
        self.min_distance = min_distance
        self._prev = None
        self._pts = None
        self.last_n = 0          # tracked features last update (confidence)

    def reset(self):
        self._prev = None
        self._pts = None
        self.last_n = 0

    def _detect(self, gray):
        import cv2
        return cv2.goodFeaturesToTrack(gray, self.max_features, self.quality,
                                       self.min_distance)

    def update(self, gray):
        """Return CCW yaw delta (rad) since the previous frame, or None if the
        estimate is unreliable (too few tracked features)."""
        import cv2
        if self._prev is None:
            self._prev = gray
            self._pts = self._detect(gray)
            return None
        if self._pts is None or len(self._pts) < self.min_features:
            self._pts = self._detect(self._prev)
        if self._pts is None or len(self._pts) < self.min_features:
            self._prev = gray
            self.last_n = 0
            return None
        nxt, st, _err = cv2.calcOpticalFlowPyrLK(self._prev, gray, self._pts, None)
        self._prev = gray
        if nxt is None or st is None:
            self._pts = None
            self.last_n = 0
            return None
        st = st.reshape(-1)
        old = self._pts.reshape(-1, 2)[st == 1]
        new = nxt.reshape(-1, 2)[st == 1]
        self.last_n = len(new)
        if self.last_n < self.min_features:
            self._pts = None
            return None
        du = new[:, 0] - old[:, 0]
        self._pts = new.reshape(-1, 1, 2)          # re-seed with tracked points
        return yaw_from_shift(float(np.median(du)), self.fx)


def turn_by_flow(actuators, capture_gray, fx, target_deg, *,
                 speed: float = 70, min_speed: float = 45, kp: float = 3.0,
                 tol_deg: float = 4.0, max_s: float = 6.0, period_s: float = 0.08,
                 estimator=None, sleep=time.sleep, now=time.monotonic):
    """Rotate in place until the *measured* yaw reaches target_deg (CCW +).

    Closed-loop on optical flow instead of trusting the motor command. Returns
    the measured yaw actually achieved, in degrees.
    """
    est = estimator or FlowRotationEstimator(fx)
    est.reset()
    target = math.radians(target_deg)
    acc = 0.0
    est.update(capture_gray())                     # prime with the first frame
    t0 = now()
    try:
        while True:
            remaining_deg = math.degrees(target - acc)
            if abs(remaining_deg) <= tol_deg:
                break
            if now() - t0 > max_s:
                break
            mag = _clamp(abs(remaining_deg) * kp, min_speed, speed)
            # remaining > 0 => need more CCW => turn negative (turn>0 is CW).
            turn = -math.copysign(mag, remaining_deg)
            actuators.drive(0.0, turn, 0.0)
            sleep(period_s)
            d = est.update(capture_gray())
            if d is not None:
                acc += d
    finally:
        actuators.stop()
    return math.degrees(acc)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))
