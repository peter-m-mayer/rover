"""
Live dead reckoning on the robot: gyro heading + floor-odometry translation,
fed from a RunRecorder's frames (it owns the camera) in a background thread.

    odo = OnlineOdometry(rec, hd, model).start(x0, y0, heading0_deg)
    odo.pose          # (x, y, heading_deg) in room coordinates
    odo.set_pose(...) # e.g. after a tag fix (only if you *want* to correct)
    odo.travelled_m   # path length so far (for growing the pose uncertainty)

Frames arrive config.CAMERA_LATENCY_S after capture (75 ms, 2026-10-11), so the
rotation fed to floor odometry is the gyro heading at capture time, looked up
in a short heading history, not the heading when the frame is processed.

The floor point tracked is the camera's; it sits a few cm ahead of the turning
centre, which floor odometry reports as (real) motion during in-place turns.
"""

import bisect
import collections
import math
import threading
import time

from . import config
from .floor_odometry import FloorOdometry


class HeadingHistory:
    """(t, heading_deg) samples; heading at an earlier time by interpolation."""

    def __init__(self, seconds=2.0):
        self.seconds = seconds
        self.t = collections.deque()
        self.h = collections.deque()

    def add(self, t, h):
        self.t.append(t)
        self.h.append(h)
        while self.t and self.t[0] < t - self.seconds:
            self.t.popleft()
            self.h.popleft()

    def at(self, t):
        if not self.t:
            return None
        ts = list(self.t)
        i = bisect.bisect_left(ts, t)
        if i <= 0:
            return self.h[0]
        if i >= len(ts):
            return self.h[-1]
        t0, t1 = ts[i - 1], ts[i]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return self.h[i - 1] + w * (self.h[i] - self.h[i - 1])


class OnlineOdometry:
    def __init__(self, rec, hd, model, latency_s=None):
        self.rec, self.hd = rec, hd
        self.fo = FloorOdometry(model)
        self.latency = config.CAMERA_LATENCY_S if latency_s is None else latency_s
        self.hist = HeadingHistory()
        self._lock = threading.Lock()
        self._x = self._y = 0.0
        self._h0 = 0.0                  # room heading = gyro heading + _h0
        self._stop = threading.Event()
        self._thread = None
        self.frames = 0
        self.travelled_m = 0.0
        self.last_step = None           # (t, dx, dy) of the latest floor estimate

    def start(self, x0=0.0, y0=0.0, heading0_deg=0.0):
        self.set_pose(x0, y0, heading0_deg)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def set_pose(self, x, y, heading_deg):
        with self._lock:
            self._x, self._y = x, y
            self._h0 = heading_deg - self.hd.heading_deg

    def shift(self, dx, dy, dheading_deg=0.0):
        """Apply a correction (e.g. a filter update) without losing motion in flight."""
        with self._lock:
            self._x += dx
            self._y += dy
            self._h0 += dheading_deg

    @property
    def pose(self):
        with self._lock:
            return self._x, self._y, self.hd.heading_deg + self._h0

    def _loop(self):
        import cv2
        last_t, last_h = None, None
        while not self._stop.is_set():
            now = time.monotonic()
            self.hist.add(now, self.hd.heading_deg)
            item = self.rec._latest
            if item is None or item[0] == last_t:
                time.sleep(0.005)
                continue
            t, frame = item
            h = self.hist.at(t - self.latency)
            if h is None:
                h = self.hd.heading_deg
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dth = None if last_h is None else math.radians(h - last_h)
            r = self.fo.step(gray, dth)
            self.frames += 1
            if r is not None:
                dx, dy = r[0], r[1]
                with self._lock:
                    # the motion is in the robot frame at the previous heading
                    a = math.radians(last_h + self._h0) if last_h is not None else 0.0
                    self._x += math.cos(a) * dx - math.sin(a) * dy
                    self._y += math.sin(a) * dx + math.cos(a) * dy
                    self.travelled_m += math.hypot(dx, dy)
                    self.last_step = (t, dx, dy)
            last_t, last_h = t, h
