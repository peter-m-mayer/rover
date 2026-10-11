"""
Metric visual odometry from the floor (AUTONOMY_PLAN.md M1; review §4B option 1).

The camera's height above the floor and its pitch are known, so every pixel
below the horizon maps to one point on the floor, in metres. Track carpet
texture between two frames, map both ends of each track onto the floor, and
the rigid 2-D transform between the two point sets *is* the camera's motion
(forward, left, turn) -- with real scale, no monocular ambiguity.

Robot/camera floor frame: X forward, Y left, angles CCW positive (same as the
gyro z axis). Positions are of the camera's optical centre projected on the
floor; for in-place turns that point moves on a small circle, which is real
motion and is reported as such.
"""

import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from . import config


class FloorModel:
    """Pixel <-> floor point for a camera at `height_m`, pitched `pitch_deg` up."""

    def __init__(self, K, dist, height_m=None, pitch_deg=0.0, max_range_m=1.2):
        self.K = np.asarray(K, float)
        self.dist = np.asarray(dist, float).ravel()
        self.h = height_m if height_m is not None else config.CAMERA_HEIGHT_M
        self.max_range = max_range_m
        p = math.radians(pitch_deg)
        # camera axes (x right, y down, z fwd) -> floor frame (X fwd, Y left, Z up), then pitch up
        base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        pitch = np.array([[math.cos(p), 0, -math.sin(p)], [0, 1, 0], [math.sin(p), 0, math.cos(p)]])
        self.R = pitch @ base

    def horizon_row(self):
        """Image row of the horizon (undistorted, optical-axis column)."""
        p = math.asin(self.R[2, 2])                # elevation of the optical axis
        return self.K[1, 2] - self.K[1, 1] * math.tan(p)

    def to_floor(self, pts):
        """(N, 2) pixels -> (N, 2) floor points (m); NaN rows above the horizon / too far."""
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2))
        n = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        rays = np.c_[n, np.ones(len(n))] @ self.R.T
        out = np.full((len(n), 2), np.nan)
        down = rays[:, 2] < -1e-6
        t = -self.h / rays[down, 2]
        xy = rays[down, :2] * t[:, None]
        ok = np.hypot(xy[:, 0], xy[:, 1]) <= self.max_range
        idx = np.flatnonzero(down)[ok]
        out[idx] = xy[ok]
        return out

    def floor_mask(self, shape, margin_px=40):
        """Mask of image rows that see floor within range (below the horizon + margin)."""
        h, w = shape[:2]
        rows = np.arange(h)
        cols = np.full(h, self.K[0, 2])
        xy = self.to_floor(np.c_[cols, rows])
        good = ~np.isnan(xy[:, 0])
        mask = np.zeros((h, w), np.uint8)
        first = int(np.argmax(good)) if good.any() else h
        mask[min(h, first + margin_px):, :] = 255
        return mask


def rigid_2d(a, b):
    """Least-squares rotation + translation with a ~= R b + t (no scale)."""
    ca, cb = a.mean(0), b.mean(0)
    H = (b - cb).T @ (a - ca)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    return R, ca - R @ cb


def ransac_rigid_2d(a, b, iters=200, tol=0.006, rng=None):
    """Robust rigid fit; returns (dx, dy, dtheta, inlier_mask) or None."""
    if len(a) < 6:
        return None
    rng = rng or np.random.default_rng(0)
    best = None
    for _ in range(iters):
        i = rng.choice(len(a), 2, replace=False)
        if np.linalg.norm(b[i[0]] - b[i[1]]) < 0.03:
            continue
        R, t = rigid_2d(a[i], b[i])
        inl = np.linalg.norm(a - (b @ R.T + t), axis=1) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 6:
        return None
    R, t = rigid_2d(a[best], b[best])
    inl = np.linalg.norm(a - (b @ R.T + t), axis=1) < tol
    R, t = rigid_2d(a[inl], b[inl])
    return float(t[0]), float(t[1]), math.atan2(R[1, 0], R[0, 0]), inl


def robust_translation(a, b, dtheta, tol=0.006):
    """Translation with a ~= R(dtheta) b + t for a KNOWN rotation (e.g. from the
    gyro): a 2-parameter problem, far better conditioned than the full fit."""
    if len(a) < 4:
        return None
    c, s = math.cos(dtheta), math.sin(dtheta)
    cand = a - b @ np.array([[c, s], [-s, c]])       # rows: a_i - R b_i
    t = np.median(cand, axis=0)
    inl = np.linalg.norm(cand - t, axis=1) < tol
    if inl.sum() < 4:
        return None
    t = cand[inl].mean(0)
    inl = np.linalg.norm(cand - t, axis=1) < tol
    return float(t[0]), float(t[1]), dtheta, inl


class FloorOdometry:
    """Frame-to-frame floor tracking; call step(gray) per frame."""

    def __init__(self, model, max_corners=300, min_inliers=12):
        self.model = model
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.prev = None
        self.mask = None
        self.pose = np.zeros(3)          # x, y, theta in the start frame
        self.stats = {"steps": 0, "failed": 0}

    def step(self, gray, dtheta=None):
        """Returns (dx, dy, dtheta, n_inliers) for this frame, or None (no estimate).

        dtheta: the rotation since the previous frame from the gyro (rad,
        CCW +). Given it, only the translation is estimated from the floor."""
        if self.mask is None:
            self.mask = self.model.floor_mask(gray.shape)
        prev, self.prev = self.prev, gray
        if prev is None:
            return None
        self.stats["steps"] += 1
        p0 = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 8, mask=self.mask, blockSize=7)
        if p0 is None or len(p0) < self.min_inliers:
            self.stats["failed"] += 1
            if dtheta is not None:
                self.pose[2] += dtheta
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, winSize=(21, 21), maxLevel=3)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - p0).reshape(-1, 2), axis=1) < 0.7)
        f0 = self.model.to_floor(p0.reshape(-1, 2)[good])
        f1 = self.model.to_floor(p1.reshape(-1, 2)[good])
        ok = ~np.isnan(f0[:, 0]) & ~np.isnan(f1[:, 0])
        # point seen at f1 now was at f0 before: f0 = R f1 + t, (R, t) = motion prev -> now
        if dtheta is None:
            fit = ransac_rigid_2d(f0[ok], f1[ok])
        else:
            fit = robust_translation(f0[ok], f1[ok], dtheta)
        if fit is None or fit[3].sum() < self.min_inliers:
            self.stats["failed"] += 1
            if dtheta is not None:                 # keep the heading even without translation
                self.pose[2] += dtheta
            return None
        dx, dy, dth, inl = fit
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        self.pose += (c * dx - s * dy, s * dx + c * dy, dth)
        return dx, dy, dth, int(inl.sum())
