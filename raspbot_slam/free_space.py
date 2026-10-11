"""
Camera "pseudo-laser": distance to the nearest obstacle in each image column,
from where the floor stops looking like floor (appearance-based obstacle
detection, Ulrich & Nourbakhsh 2000). AUTONOMY_PLAN.md M2.

Floor model = colour (H, S, V histogram) + texture (local contrast range),
learned from the floor directly ahead (~0.22-0.40 m) *only when the sonar says
that patch is clear*, then kept for later frames. Colour alone can't tell this
off-white carpet from white baseboards; texture can (speckled pile vs smooth
paint). Scanning each column up from the bottom, the first run of non-floor
pixels is the foot of an obstacle; FloorModel turns that row into a distance.
Columns that stay floor up to the range limit report "free to max range".

Colour is OFF by default (prob_frac=0): with the USB camera's auto exposure and
white balance, the same carpet measured S 16 / V 157 facing the window and
S 40 / V 109 facing away (2026-10-11), so a colour model learned in one
direction rejects the floor in another. Texture alone matched the sonar on
close walls (0.20-0.26 m vs 0.19-0.26 m). Re-enable colour (prob_frac ~0.08)
once exposure and white balance are locked for mapping.

Limits: a floor-coloured, floor-textured obstacle is invisible; mirrors show
"floor" through the glass (keep surveyed mirrors as walls); strong glare
washes the texture out (those columns come back as obstacles, which is the
safe failure).
"""

import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .floor_odometry import FloorModel

BINS = np.array([12, 6, 2])     # H, S, V: brightness barely counts (vignetting, shadows)


class FreeSpaceScanner:
    def __init__(self, model: FloorModel, col_step=8, ref_range=(0.22, 0.40),
                 ref_half_width_m=0.10, run_px=8, prob_frac=0.0, max_range_m=1.6,
                 skip_bottom_px=8, tex_win=7):
        self.m = model
        self.col_step = col_step
        self.ref_range = ref_range
        self.ref_half_width = ref_half_width_m
        self.run_px = run_px
        self.prob_frac = prob_frac
        self.max_range = max_range_m
        self.skip_bottom = skip_bottom_px
        self.tex_win = tex_win
        self.hist = None             # learned floor colour histogram (normalised)
        self.tex_lo = None           # learned floor texture band
        self.tex_hi = None
        self._grid = None

    # ---- geometry ---------------------------------------------------------------
    def _floor_grid(self, shape):
        """Per-pixel floor (X, Y) for the sampled columns, cached."""
        if self._grid is None or self._grid[0] != shape[:2]:
            h, w = shape[:2]
            cols = np.arange(self.col_step // 2, w, self.col_step)
            vv, uu = np.meshgrid(np.arange(h), cols, indexing="ij")
            m = FloorModel(self.m.K, self.m.dist, self.m.h, 0.0, max_range_m=50.0)
            m.R = self.m.R
            xy = m.to_floor(np.c_[uu.ravel(), vv.ravel()]).reshape(h, len(cols), 2)
            self._grid = (shape[:2], cols, xy)
        return self._grid[1], self._grid[2]

    def reference_mask(self, shape):
        _, xy = self._floor_grid(shape)
        rng = np.hypot(xy[..., 0], xy[..., 1])
        return ((rng >= self.ref_range[0]) & (rng <= self.ref_range[1])
                & (np.abs(xy[..., 1]) <= self.ref_half_width))

    # ---- features -----------------------------------------------------------------
    def _features(self, bgr):
        cols, _ = self._floor_grid(bgr.shape)
        blur = cv2.GaussianBlur(bgr, (5, 5), 0)
        hsv = np.ascontiguousarray(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)[:, cols])
        q = (hsv.astype(np.int32) * BINS // np.array([180, 256, 256])).clip(0, BINS - 1)
        idx = (q[..., 0] * BINS[1] + q[..., 1]) * BINS[2] + q[..., 2]
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = (self.tex_win, self.tex_win)
        mean = cv2.blur(g, k)
        tex = np.sqrt(np.maximum(cv2.blur(g * g, k) - mean * mean, 0))[:, cols]
        return idx, tex

    def learn(self, bgr):
        """Update the floor model from the reference patch (call only when the
        sonar confirms the patch is clear floor). Returns True if it looked usable."""
        idx, tex = self._features(bgr)
        ref = self.reference_mask(bgr.shape)
        if ref.sum() < 50:
            return False
        hist = np.bincount(idx[ref], minlength=int(BINS.prod())).astype(np.float32)
        hist /= hist.sum()
        lo, hi = np.percentile(tex[ref], [3, 97])
        if self.hist is None:
            self.hist, self.tex_lo, self.tex_hi = hist, lo, hi
        else:                                          # slow blend: lighting drifts
            self.hist = 0.7 * self.hist + 0.3 * hist
            self.tex_lo = 0.7 * self.tex_lo + 0.3 * lo
            self.tex_hi = 0.7 * self.tex_hi + 0.3 * hi
        return True

    def floor_likelihood(self, bgr):
        """Boolean floor-like map at the sampled columns (needs a learned model)."""
        idx, tex = self._features(bgr)
        h = self.hist / max(self.hist.max(), 1e-9)
        colour_ok = h[idx] >= self.prob_frac
        # lower bound only: smooth paint is what must be rejected, and a still
        # camera sees far more carpet texture than a moving (blurred) one
        tex_ok = tex >= 0.6 * self.tex_lo
        like = (colour_ok & tex_ok).astype(np.float32)
        # vertical majority vote: single odd pixels in carpet pile aren't obstacles
        like = cv2.blur(like, (1, 9))
        return like >= 0.5

    # ---- the scan -------------------------------------------------------------------
    def scan(self, bgr, learn=False):
        """Returns (bearing_rad, range_m, hit) per sampled column, or None if no
        floor model has been learned yet.

        bearing: CCW-positive from the camera's forward axis on the floor.
        hit: True where an obstacle foot was found within max_range; False =
        floor all the way to max_range (range_m = max_range)."""
        if learn:
            self.learn(bgr)
        if self.hist is None:
            return None
        cols, xy = self._floor_grid(bgr.shape)
        floor_like = self.floor_likelihood(bgr)
        h = bgr.shape[0]
        rng = np.hypot(xy[..., 0], xy[..., 1])
        bearings = np.full(len(cols), np.nan)
        ranges = np.full(len(cols), self.max_range)
        hits = np.zeros(len(cols), bool)
        for j in range(len(cols)):
            rows = np.flatnonzero(~np.isnan(rng[:, j]))
            if len(rows) == 0:
                continue
            inrange = rows[rng[rows, j] <= self.max_range]
            far = inrange.min() if len(inrange) else rows.max()
            bearings[j] = math.atan2(xy[far, j, 1], xy[far, j, 0])
            bad = 0
            for v in range(h - 1 - self.skip_bottom, rows.min() - 1, -1):    # near -> far
                if rng[v, j] > self.max_range:
                    break
                if floor_like[v, j]:
                    bad = 0
                    continue
                bad += 1
                if bad >= self.run_px:
                    foot = v + self.run_px - 1                 # lowest non-floor row of the run
                    ranges[j] = rng[foot, j]
                    bearings[j] = math.atan2(xy[foot, j, 1], xy[foot, j, 0])
                    hits[j] = True
                    break
        # 5-column median: an isolated odd column (a stain, a shadow) is not a wall
        k = 2
        padded = np.pad(ranges, k, mode="edge")
        ranges = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1), axis=1)
        hits = ranges < self.max_range - 1e-9
        return bearings, ranges, hits
