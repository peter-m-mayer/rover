"""
Gyro heading tracker and heading-hold driving.

On carpet the mecanum chassis does not drive straight: on 2026-10-11 a
"straight" 1 m drive turned 17 deg and the reverse 36 deg (gyro). The gyro is
good (agrees with camera and tag headings to <1 deg over a full turn), so it
closes the loop on heading.

    hd = GyroHeading(imu).start()      # background integration, ~200 Hz
    hd.heading_deg                     # CCW-positive since start (or reset)
    drive_straight(act, hd, speed=50, seconds=0.5)   # holds the current heading
"""

import math
import threading
import time

import numpy as np


class GyroHeading:
    def __init__(self, imu, rate_hz=200.0, axis=2):
        self.imu = imu
        self.period = 1.0 / rate_hz
        self.axis = axis
        self.bias = 0.0
        self._heading = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def calibrate(self, seconds=1.0):
        """Gyro bias from a still period (robot must not move)."""
        vals, t_end = [], time.monotonic() + seconds
        while time.monotonic() < t_end:
            vals.append(self.imu.gyro[self.axis])
            time.sleep(0.005)
        self.bias = float(np.median(vals))
        return self.bias

    def start(self, calibrate=True):
        if calibrate:
            self.calibrate()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    def _loop(self):
        t_prev = time.monotonic()
        while not self._stop.is_set():
            try:
                rate = self.imu.gyro[self.axis] - self.bias
            except OSError:
                rate = None
            now = time.monotonic()
            if rate is not None:
                with self._lock:
                    self._heading += rate * (now - t_prev)
            t_prev = now
            self._stop.wait(self.period)

    @property
    def heading_deg(self):
        with self._lock:
            return math.degrees(self._heading)

    def reset(self, deg=0.0):
        with self._lock:
            self._heading = math.radians(deg)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def drive_straight(act, hd, speed, seconds, target_deg=None, kp=2.5, max_turn=30, strafe=0.0):
    """Drive at `speed` (negative = reverse) for `seconds`, steering with the
    gyro to hold `target_deg` (default: the heading at the start)."""
    target = hd.heading_deg if target_deg is None else target_deg
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            err = wrap180(hd.heading_deg - target)       # + = drifted left (CCW)
            turn = float(np.clip(kp * err, -max_turn, max_turn))   # drive(): turn + = CW
            act.drive(speed, turn=turn, strafe=strafe)
            time.sleep(0.02)
    finally:
        act.stop()
    return wrap180(hd.heading_deg - target)


def turn_to(act, hd, target_deg, speed=60, tol=2.0, timeout=8.0, coast_deg=6.0):
    """Turn in place to an absolute gyro heading; returns the final error (deg)."""
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        err = wrap180(target_deg - hd.heading_deg)       # + = need to go CCW
        if abs(err) <= tol:
            break
        if abs(err) <= coast_deg:                        # nudge
            act.drive(0, turn=-np.sign(err) * 45)
            time.sleep(0.06)
            act.stop()
            time.sleep(0.25)
            continue
        act.drive(0, turn=-speed if err > 0 else speed)
        while time.monotonic() < t_end and abs(wrap180(target_deg - hd.heading_deg)) > coast_deg:
            time.sleep(0.01)
        act.stop()
        time.sleep(0.4)
    act.stop()
    return wrap180(target_deg - hd.heading_deg)
