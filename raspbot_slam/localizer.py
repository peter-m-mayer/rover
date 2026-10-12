"""
Pose estimate in room coordinates (NAV plan step 4): dead reckoning corrected
by surveyed-tag fixes, with uncertainty, gating and a heading watchdog.

    loc = Localizer(odo)               # odo: OnlineOdometry (gyro + floor)
    loc.pose, loc.P                    # (x, y, heading_deg), 3x3 covariance (m, m, rad)
    loc.add_fix(fix)                   # TagLocalizer result -> "used" / "gated" / reason
    loc.correct = False                # scoring mode: fixes checked, never applied

Uncertainty grows with distance travelled (floor odometry: 3%/m clean, much
worse after bumps) and with turning. A fix is applied as a Kalman update when
it passes the Mahalanobis gate. Heading watchdog (2026-10-11: a tour's gyro
heading ended ~170 deg away from three tag-1 fixes in a row): if the last
`watch_n` fixes all disagree with the heading by more than `watch_deg` in the
same direction, the heading is re-anchored to the fixes and the event is
reported, whether or not `correct` is set.
"""

import math

import numpy as np

from .heading import wrap180

SIG_FIX_XY = 0.06            # m, a gated tag fix within ~1.5 m
SIG_FIX_TH = math.radians(3.0)
ODO_XY_PER_M = 0.05          # m of position sigma per m travelled
ODO_TH_PER_RAD = 0.02        # rad of heading sigma per rad turned
GATE_CHI2 = 11.3             # 3 dof, p = 0.01


class Localizer:
    def __init__(self, odo, correct=True, watch_n=3, watch_deg=15.0, log=None):
        self.odo = odo
        self.correct = correct
        self.watch_n, self.watch_deg = watch_n, watch_deg
        self.log = log or (lambda msg: None)
        self.P = np.diag([0.02 ** 2, 0.02 ** 2, math.radians(2) ** 2])
        self._last = (odo.travelled_m, odo.pose[2])
        self.recent = []                 # signed heading disagreements of the latest fixes (deg)
        self.rejected = []               # positions of consecutive gated fixes
        self.stats = {"used": 0, "gated": 0, "heading_resets": 0, "position_resets": 0}

    @property
    def pose(self):
        return self.odo.pose

    def predict(self):
        """Grow P by the motion since the last call (cheap; call before using P)."""
        d, h = self.odo.travelled_m, self.odo.pose[2]
        dd = max(0.0, d - self._last[0])
        dh = abs(math.radians(wrap180(h - self._last[1])))
        self._last = (d, h)
        q = (ODO_XY_PER_M * dd) ** 2 + 1e-6
        self.P = self.P + np.diag([q, q, (ODO_TH_PER_RAD * dh) ** 2 + 1e-7])

    def innovation(self, fix):
        x, y, h = self.pose
        return np.array([fix["x"] - x, fix["y"] - y, math.radians(wrap180(fix["heading"] - h))])

    def add_fix(self, fix):
        """fix: dict with x, y, heading (deg, robot body), range_m, n (frames)."""
        self.predict()
        nu = self.innovation(fix)
        # heading watchdog first: a gyro gone wrong must not be gated forever
        self.recent = (self.recent + [math.degrees(nu[2])])[-self.watch_n:]
        if (len(self.recent) == self.watch_n and all(abs(e) > self.watch_deg for e in self.recent)
                and len(set(np.sign(self.recent))) == 1):
            dh = float(np.median(self.recent))
            self.odo.shift(0.0, 0.0, dh)
            self.P[2, 2] = SIG_FIX_TH ** 2
            self.stats["heading_resets"] += 1
            self.recent = []
            self.log(f"heading_reset {dh:+.1f}")
            nu = self.innovation(fix)
        rng = fix.get("range_m", 1.0)
        R = np.diag([(SIG_FIX_XY * max(1.0, rng)) ** 2] * 2 + [SIG_FIX_TH ** 2])
        S = self.P + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        if d2 > GATE_CHI2:
            self.stats["gated"] += 1
            # lost: several gated fixes that agree with each other beat dead reckoning
            self.rejected = (self.rejected + [(fix["x"], fix["y"])])[-self.watch_n:]
            pts = np.array(self.rejected)
            if self.correct and len(pts) == self.watch_n and np.max(np.linalg.norm(pts - np.median(pts, 0), axis=1)) < 0.2:
                mx, my = np.median(pts, 0)
                x, y, _ = self.pose
                self.odo.shift(mx - x, my - y)
                self.P[0, 0] = self.P[1, 1] = (SIG_FIX_XY * max(1.0, rng)) ** 2
                self.stats["position_resets"] += 1
                self.rejected = []
                self.log(f"position_reset ({mx:.2f}, {my:.2f}) from ({x:.2f}, {y:.2f})")
                return "position reset"
            return f"gated (d2 {d2:.1f})"
        self.rejected = []
        if not self.correct:
            return f"ok (d2 {d2:.1f}, not applied)"
        K = self.P @ np.linalg.inv(S)
        dx = K @ nu
        self.odo.shift(dx[0], dx[1], math.degrees(dx[2]))
        self.P = (np.eye(3) - K) @ self.P
        self.stats["used"] += 1
        return f"used (d2 {d2:.1f})"

    def sigma_xy(self):
        self.predict()
        return math.sqrt(max(self.P[0, 0], self.P[1, 1]))
