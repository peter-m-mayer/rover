"""
Record-and-replay logger for hardware runs (AUTONOMY_PLAN.md, M0).

Every run becomes one folder that holds everything needed to re-run
odometry / mapping offline on the PC:

    datasets/slam_runs/<YYYYmmdd_HHMMSS>[_label]/
        meta.json      config snapshot (servo centres, IMU axis map, camera K,
                       printed target sizes), git commit, start time
        frames/        000000.jpg ...          (camera, ~10 Hz)
        frames.csv     idx, t                  (t = time.monotonic() seconds)
        imu.csv        t, ax..az (m/s^2), gx..gz (rad/s), mx..mz (uT)
                       -- SENSOR frame, raw (no axis map / calibration), so
                       replay can apply any calibration later
        sensors.csv    t, us_mm, line (4-bit mask)
        cmd.csv        t, kind, a, b, c, d     (motors l1 l2 r1 r2 | pan | tilt)
        power.csv      t, throttled, ext5v_v
        events.csv     t, label                (mark() from the driving script)

All streams share one clock (time.monotonic). The I2C bus to the Yahboom MCU
is shared by motor commands and sensor reads, so both go through one lock;
the IMU is on its own bus (i2c-3) and needs none.

Recording (passive, e.g. the static-noise test):
    python3 -m raspbot_slam.recorder --seconds 60 --label static

From a driving script:
    with RunRecorder(camera=cam, imu=imu, sensors=s, actuators=act) as rec:
        rec.mark("square start")
        act.drive(60)          # logged automatically
        frame = rec.latest_frame()

Replay:
    run = RunLog("datasets/slam_runs/20261011_153000_square")
    for t, img in run.iter_frames(): ...
    run.imu["gz"], run.cmds, run.events
"""

import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from . import config

DEFAULT_ROOT = os.path.join("datasets", "slam_runs")
IMU_FIELDS = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz")


def _git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        dirty = subprocess.call(["git", "-C", here, "diff", "--quiet"],
                                stderr=subprocess.DEVNULL, timeout=2) != 0
        return out + ("+dirty" if dirty else "")
    except Exception:
        return "?"


def read_power():
    """(throttled_hex, ext5v_volts) from vcgencmd; ('?', nan) off-Pi."""
    try:
        thr = subprocess.check_output(["vcgencmd", "get_throttled"],
                                      timeout=2).decode().strip().split("=")[1]
    except Exception:
        thr = "?"
    volts = float("nan")
    try:
        out = subprocess.check_output(["vcgencmd", "pmic_read_adc", "EXT5V_V"],
                                      timeout=2).decode()
        volts = float(out.strip().split("=")[1].rstrip("V"))
    except Exception:
        pass
    return thr, volts


def config_snapshot(camera=None):
    """Everything replay needs to interpret the raw streams."""
    snap = {k: getattr(config, k) for k in dir(config)
            if k.isupper() and k.startswith(("SERVO_", "IMU_", "CAMERA_", "CHECKER_",
                                              "ARUCO_", "MOTOR_"))}
    if camera is not None:
        try:
            snap["camera_K"] = np.asarray(camera.K).tolist()
            snap["camera_dist"] = np.asarray(camera.dist_coeffs).ravel().tolist()
        except Exception:
            pass
    return json.loads(json.dumps(snap, default=list))


class _Csv:
    """Thread-safe append-only CSV with periodic flush."""

    def __init__(self, path, header):
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(header)
        self._lock = threading.Lock()
        self._n = 0

    def row(self, *values):
        with self._lock:
            self._w.writerow(values)
            self._n += 1
            if self._n % 50 == 0:
                self._f.flush()

    def close(self):
        with self._lock:
            self._f.close()


class RunRecorder:
    """Logs camera, IMU, sensors, motor/servo commands and power for one run.

    Any of camera / imu / sensors / actuators may be None (that stream is
    simply not recorded). `imu` is a raw driver with .acceleration/.gyro/
    .magnetic properties (lsm9ds1.LSM9DS1), not the fused imu.IMU class.
    """

    def __init__(self, camera=None, imu=None, sensors=None, actuators=None,
                 root=DEFAULT_ROOT, label=None, frame_hz=10.0, imu_hz=100.0,
                 sensor_hz=10.0, power_hz=1.0, jpeg_quality=90):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(root, stamp + (f"_{label}" if label else ""))
        self.camera, self.imu, self.sensors, self.actuators = camera, imu, sensors, actuators
        self.rates = {"frame": frame_hz, "imu": imu_hz, "sensor": sensor_hz, "power": power_hz}
        self.jpeg_quality = jpeg_quality
        self.bus_lock = threading.RLock()   # Yahboom MCU I2C (motors + sensors)
        self.counts = {"frames": 0, "imu": 0, "sensors": 0, "cmds": 0, "errors": 0}
        self._stop = threading.Event()
        self._threads = []
        self._latest = None
        self._latest_lock = threading.Lock()
        self._unwrap = []
        self.t0 = None

    # ---- lifecycle ----------------------------------------------------------
    def start(self):
        os.makedirs(os.path.join(self.path, "frames"), exist_ok=True)
        self.t0 = time.monotonic()
        meta = {"started": datetime.now().isoformat(timespec="seconds"),
                "t0_monotonic": self.t0, "git": _git_commit(), "rates_hz": self.rates,
                "imu_frame": "sensor (raw, before IMU_AXIS_MAP and calibration)",
                "config": config_snapshot(self.camera)}
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self._frames = _Csv(os.path.join(self.path, "frames.csv"), ("idx", "t"))
        self._imu = _Csv(os.path.join(self.path, "imu.csv"), ("t",) + IMU_FIELDS)
        self._sens = _Csv(os.path.join(self.path, "sensors.csv"), ("t", "us_mm", "line"))
        self._cmd = _Csv(os.path.join(self.path, "cmd.csv"), ("t", "kind", "a", "b", "c", "d"))
        self._pow = _Csv(os.path.join(self.path, "power.csv"), ("t", "throttled", "ext5v_v"))
        self._ev = _Csv(os.path.join(self.path, "events.csv"), ("t", "label"))
        if self.actuators is not None:
            self._wrap_actuators()
            self._log_cmd("pan", self.actuators.pan_angle)
            self._log_cmd("tilt", self.actuators.tilt_angle)
        if self.sensors is not None:
            with self.bus_lock:
                self.sensors.enable_ultrasonic()
        self.mark("start")
        loops = [(self.camera, self._frame_loop), (self.imu, self._imu_loop),
                 (self.sensors, self._sensor_loop), (True, self._power_loop)]
        for source, fn in loops:
            if source is not None:
                t = threading.Thread(target=fn, daemon=True)
                t.start()
                self._threads.append(t)
        return self

    def stop(self):
        if self.t0 is None:
            return
        self.mark("stop")
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3)
        for undo in self._unwrap:
            undo()
        for c in (self._frames, self._imu, self._sens, self._cmd, self._pow, self._ev):
            c.close()
        summary = dict(self.counts, seconds=round(time.monotonic() - self.t0, 2))
        with open(os.path.join(self.path, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        self.t0 = None
        return summary

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- API for driving scripts --------------------------------------------
    def mark(self, label):
        """Annotate the timeline (e.g. 'square corner 2')."""
        self._ev.row(f"{time.monotonic():.4f}", label)

    def latest_frame(self):
        """Most recent colour frame (the recorder owns the camera)."""
        with self._latest_lock:
            return None if self._latest is None else self._latest[1]

    # ---- actuator logging -----------------------------------------------------
    def _log_cmd(self, kind, a, b="", c="", d=""):
        self._cmd.row(f"{time.monotonic():.4f}", kind, a, b, c, d)
        self.counts["cmds"] += 1

    def _wrap_actuators(self):
        act = self.actuators
        names = ("_set_motors", "set_servo_pan", "set_servo_tilt")
        originals = {n: getattr(act, n) for n in names}

        def set_motors(l1, l2, r1, r2):
            with self.bus_lock:
                originals["_set_motors"](l1, l2, r1, r2)
            self._log_cmd("motors", int(l1), int(l2), int(r1), int(r2))

        def servo(name, kind):
            def fn(angle_deg):
                with self.bus_lock:
                    originals[name](angle_deg)
                self._log_cmd(kind, getattr(act, kind + "_angle"))
            return fn

        act._set_motors = set_motors
        act.set_servo_pan = servo("set_servo_pan", "pan")
        act.set_servo_tilt = servo("set_servo_tilt", "tilt")

        def undo():
            for n in names:
                act.__dict__.pop(n, None)
        self._unwrap.append(undo)

    # ---- sampling threads -----------------------------------------------------
    def _paced(self, hz, body):
        period = 1.0 / hz
        nxt = time.monotonic()
        while not self._stop.is_set():
            try:
                body()
            except Exception:
                self.counts["errors"] += 1
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                nxt = time.monotonic()   # fell behind: don't burst to catch up

    def _frame_loop(self):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality] if cv2 else []

        def grab():
            frame = self.camera.capture_color()
            t = time.monotonic()
            idx = self.counts["frames"]
            if cv2 is not None:
                cv2.imwrite(os.path.join(self.path, "frames", f"{idx:06d}.jpg"), frame, params)
            self._frames.row(idx, f"{t:.4f}")
            self.counts["frames"] += 1
            with self._latest_lock:
                self._latest = (t, frame)
        self._paced(self.rates["frame"], grab)

    def _imu_loop(self):
        def sample():
            a, g, m = self.imu.acceleration, self.imu.gyro, self.imu.magnetic
            t = time.monotonic()
            self._imu.row(f"{t:.4f}", *(f"{v:.5f}" for v in (*a, *g, *m)))
            self.counts["imu"] += 1
        self._paced(self.rates["imu"], sample)

    def _sensor_loop(self):
        def sample():
            with self.bus_lock:
                us = self.sensors.read_ultrasonic_mm()
                line = self.sensors.read_line_tracker()
            mask = sum(int(b) << (3 - i) for i, b in enumerate(line))
            self._sens.row(f"{time.monotonic():.4f}", us, mask)
            self.counts["sensors"] += 1
        self._paced(self.rates["sensor"], sample)

    def _power_loop(self):
        def sample():
            thr, volts = read_power()
            self._pow.row(f"{time.monotonic():.4f}", thr, f"{volts:.3f}")
        self._paced(self.rates["power"], sample)


class RunLog:
    """Read a recorded run back for offline processing."""

    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.frames = self._table("frames.csv")
        self.imu = self._table("imu.csv")
        self.sensors = self._table("sensors.csv")
        self.power = self._table("power.csv", numeric=False)
        self.cmds = self._rows("cmd.csv")
        self.events = [(float(r["t"]), r["label"]) for r in self._rows("events.csv")]

    def _rows(self, name):
        p = os.path.join(self.path, name)
        if not os.path.exists(p):
            return []
        with open(p, newline="") as f:
            return list(csv.DictReader(f))

    def _table(self, name, numeric=True):
        rows = self._rows(name)
        if not rows:
            return {}
        cols = rows[0].keys()
        if not numeric:
            return {c: [r[c] for r in rows] for c in cols}
        return {c: np.array([float(r[c]) for r in rows]) for c in cols}

    @property
    def t0(self):
        return self.meta["t0_monotonic"]

    def frame_path(self, idx):
        return os.path.join(self.path, "frames", f"{int(idx):06d}.jpg")

    def iter_frames(self, gray=False):
        """Yield (t, image) in order."""
        flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        for idx, t in zip(self.frames.get("idx", []), self.frames.get("t", [])):
            img = cv2.imread(self.frame_path(idx), flag)
            if img is not None:
                yield float(t), img

    def motor_segments(self):
        """[(t_start, t_end, (l1, l2, r1, r2))] -- each motor command held until the next."""
        m = [(float(r["t"]), tuple(int(r[k]) for k in "abcd"))
             for r in self.cmds if r["kind"] == "motors"]
        end = self.events[-1][0] if self.events else (m[-1][0] if m else 0.0)
        return [(t, (m[i + 1][0] if i + 1 < len(m) else end), w) for i, (t, w) in enumerate(m)]

    def event_time(self, label):
        for t, lab in self.events:
            if lab == label:
                return t
        raise KeyError(label)


def main(argv=None):
    import argparse
    from .camera import Camera
    from .sensors import Sensors
    from .actuators import Actuators

    p = argparse.ArgumentParser(description="Record a passive run (nothing moves).")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--label", default=None)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--no-camera", action="store_true")
    args = p.parse_args(argv)

    from catchaser.hw import connect_bot
    bot = connect_bot()
    imu = None
    try:
        from .lsm9ds1 import LSM9DS1
        imu = LSM9DS1(bus=config.IMU_I2C_BUS)
    except Exception as exc:
        print(f"[recorder] no IMU: {exc}")
    cam = None if args.no_camera else Camera()
    rec = RunRecorder(camera=cam, imu=imu, sensors=Sensors(bot=bot),
                      actuators=Actuators(bot=bot), root=args.root, label=args.label)
    print(f"[recorder] {rec.path}: recording {args.seconds:g} s (Ctrl+C to stop early)")
    try:
        with rec:
            time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.close()
    with open(os.path.join(rec.path, "summary.json")) as f:
        print("[recorder] done:", f.read().replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
