"""
Live dead reckoning on the robot: gyro heading + floor-odometry translation,
fed from a RunRecorder's frames (it owns the camera) in a background thread.

    odo = OnlineOdometry(rec, hd, model).start(x0, y0, heading0_deg)
    odo.pose          # (x, y, heading_deg) in room coordinates
    odo.set_pose(...) # e.g. after a tag fix (only if you *want* to correct)

The floor point tracked is the camera's; it sits a few cm ahead of the turning
centre, which floor odometry reports as (real) motion during in-place turns.
"""

import math
import threading
import time

import numpy as np

from .floor_odometry import FloorOdometry


class OnlineOdometry:
    def __init__(self, rec, hd, model):
        self.rec, self.hd = rec, hd
        self.fo = FloorOdometry(model)
        self._lock = threading.Lock()
        self._x = self._y = 0.0
        self._h0 = 0.0                  # room heading = gyro heading + _h0
        self._stop = threading.Event()
        self._thread = None
        self.frames = 0

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

    @property
    def pose(self):
        with self._lock:
            return self._x, self._y, self.hd.heading_deg + self._h0

    def _loop(self):
        import cv2
        last_t, last_h = None, None
        while not self._stop.is_set():
            item = self.rec._latest
            if item is None or item[0] == last_t:
                time.sleep(0.01)
                continue
            t, frame = item
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
            last_t, last_h = t, h
