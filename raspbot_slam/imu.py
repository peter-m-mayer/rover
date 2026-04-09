"""
ICM-20948 9-DOF IMU interface with orientation estimation.

Provides gyroscope (angular velocity), accelerometer (linear acceleration),
and magnetometer (compass heading) data. Includes a complementary filter
for fusing gyro + accel into a stable orientation estimate, and automatic
gyro bias calibration on startup.

This is an OPTIONAL upgrade -- the SLAM system works without it, but gains:
- 100 Hz heading from gyro (vs 10 Hz from VO)
- Dead reckoning between camera frames
- Absolute heading from magnetometer (no drift)

Hardware: Adafruit ICM-20948 breakout on I2C bus, address 0x69.
Install: sudo pip3 install adafruit-circuitpython-icm20x

Usage:
    imu = IMU()               # auto-detects hardware
    imu.calibrate_gyro()      # hold still for 2 seconds
    while running:
        imu.update()
        heading = imu.heading  # radians, absolute (mag-corrected)
        omega = imu.gyro_z     # rad/s, yaw rate
        ax, ay = imu.accel_xy  # m/s², gravity-compensated, body frame
"""

import math
import time
import threading
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np

from . import config

# Try to import the real hardware library
try:
    import board
    import adafruit_icm20x
    _HAS_HARDWARE = True
except ImportError:
    _HAS_HARDWARE = False


# =============================================================================
# Configuration (added to config.py namespace)
# =============================================================================

# IMU config defaults (can be overridden in config.py)
IMU_I2C_ADDRESS = 0x69
IMU_GYRO_RATE_HZ = 100
IMU_ACCEL_RATE_HZ = 50
IMU_MAG_RATE_HZ = 10
IMU_COMPLEMENTARY_ALPHA = 0.98      # gyro weight in complementary filter
IMU_GYRO_BIAS_SAMPLES = 200         # samples for bias calibration
IMU_MAG_DECLINATION_DEG = 0.0       # local magnetic declination (set for your location)

# Noise parameters for EKF
IMU_GYRO_NOISE_RAD = 0.001          # gyro noise (rad/s) -- very small
IMU_ACCEL_NOISE_M = 0.05            # accelerometer noise (m/s²)
IMU_MAG_NOISE_RAD = 0.05            # magnetometer heading noise (rad)


@dataclass
class IMUReading:
    """Single timestamped IMU reading with all 9 axes."""
    timestamp: float                 # time.time()
    accel: Tuple[float, float, float]   # (ax, ay, az) in m/s²
    gyro: Tuple[float, float, float]    # (gx, gy, gz) in rad/s
    mag: Tuple[float, float, float]     # (mx, my, mz) in uT
    temperature: float = 0.0


class IMU:
    """ICM-20948 9-DOF IMU with orientation estimation.

    Works as an optional upgrade -- check `is_available` before use.
    Falls back gracefully when hardware is absent.
    """

    def __init__(self, mock_mode: bool = False):
        """Initialize IMU.

        Args:
            mock_mode: If True, use simulated data (for testing without hardware).
        """
        self._mock_mode = mock_mode
        self._icm = None
        self._available = False
        self._running = False
        self._thread = None

        # Calibration
        self._gyro_bias = np.zeros(3)   # rad/s offset to subtract
        self._mag_offset = np.zeros(3)  # hard-iron offset
        self._mag_scale = np.ones(3)    # soft-iron scale
        self._calibrated = False

        # State
        self._heading = 0.0             # fused heading (rad, CCW from +X)
        self._pitch = 0.0               # pitch (rad)
        self._roll = 0.0                # roll (rad)
        self._gyro_raw = np.zeros(3)    # latest gyro reading (bias-corrected)
        self._accel_raw = np.zeros(3)   # latest accel reading
        self._mag_raw = np.zeros(3)     # latest mag reading
        self._mag_heading = 0.0         # magnetometer-only heading
        self._last_update_time = 0.0
        self._dt = 0.01                 # estimated dt between updates

        # Lock for thread safety
        self._lock = threading.Lock()

        if not mock_mode:
            self._try_init_hardware()

    def _try_init_hardware(self):
        """Attempt to initialize the ICM-20948 over I2C."""
        if not _HAS_HARDWARE:
            return
        try:
            i2c = board.I2C()
            self._icm = adafruit_icm20x.ICM20948(i2c)
            self._available = True
        except Exception:
            self._icm = None
            self._available = False

    @property
    def is_available(self) -> bool:
        """True if IMU hardware is detected and initialized."""
        return self._available or self._mock_mode

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    # =========================================================================
    # Calibration
    # =========================================================================

    def calibrate_gyro(self, n_samples: int = None, verbose: bool = True):
        """Calibrate gyro bias. Robot must be STATIONARY during this.

        Averages n_samples readings to estimate the zero-rate offset.
        This offset is subtracted from all subsequent gyro readings.

        Args:
            n_samples: Number of samples (default: IMU_GYRO_BIAS_SAMPLES).
            verbose: Print progress.
        """
        n = n_samples or IMU_GYRO_BIAS_SAMPLES

        if verbose:
            print("IMU: Calibrating gyro -- hold robot still...")

        samples = np.zeros((n, 3))
        for i in range(n):
            reading = self._read_raw()
            samples[i] = reading.gyro
            if not self._mock_mode:
                time.sleep(1.0 / IMU_GYRO_RATE_HZ)

        self._gyro_bias = np.mean(samples, axis=0)
        gyro_std = np.std(samples, axis=0)

        if verbose:
            print(f"  Gyro bias: ({self._gyro_bias[0]:.4f}, "
                  f"{self._gyro_bias[1]:.4f}, {self._gyro_bias[2]:.4f}) rad/s")
            print(f"  Gyro std:  ({gyro_std[0]:.4f}, "
                  f"{gyro_std[1]:.4f}, {gyro_std[2]:.4f}) rad/s")

        self._calibrated = True

    def calibrate_magnetometer(self, duration_s: float = 15.0, verbose: bool = True):
        """Calibrate magnetometer hard/soft iron offsets.

        Rotate the robot slowly through 360 degrees during calibration.
        Collects min/max for each axis to compute offsets and scales.

        Args:
            duration_s: Collection time in seconds.
            verbose: Print progress.
        """
        if verbose:
            print(f"IMU: Calibrating magnetometer -- rotate robot slowly for {duration_s:.0f}s...")

        samples = []
        t_start = time.time()
        while time.time() - t_start < duration_s:
            reading = self._read_raw()
            samples.append(reading.mag)
            time.sleep(0.05)

        samples = np.array(samples)
        if len(samples) < 20:
            if verbose:
                print("  Not enough samples for magnetometer calibration.")
            return

        # Hard-iron offset: center of the min/max range
        mag_min = np.min(samples, axis=0)
        mag_max = np.max(samples, axis=0)
        self._mag_offset = (mag_max + mag_min) / 2.0

        # Soft-iron scale: normalize each axis range
        mag_range = mag_max - mag_min
        avg_range = np.mean(mag_range)
        self._mag_scale = avg_range / np.maximum(mag_range, 1e-6)

        if verbose:
            print(f"  Hard-iron offset: ({self._mag_offset[0]:.1f}, "
                  f"{self._mag_offset[1]:.1f}, {self._mag_offset[2]:.1f}) uT")
            print(f"  Soft-iron scale:  ({self._mag_scale[0]:.3f}, "
                  f"{self._mag_scale[1]:.3f}, {self._mag_scale[2]:.3f})")

    # =========================================================================
    # Reading
    # =========================================================================

    def update(self):
        """Read sensors and update orientation estimate.

        Call this at the IMU update rate (100 Hz) or as fast as possible.
        Thread-safe.
        """
        now = time.time()
        if self._last_update_time > 0:
            self._dt = now - self._last_update_time
        self._last_update_time = now

        reading = self._read_raw()

        with self._lock:
            # Bias-corrected gyro
            self._gyro_raw = np.array(reading.gyro) - self._gyro_bias

            # Raw accel
            self._accel_raw = np.array(reading.accel)

            # Calibrated magnetometer
            mag = np.array(reading.mag)
            self._mag_raw = (mag - self._mag_offset) * self._mag_scale

            # Compute magnetometer heading (atan2 of X, Y in horizontal plane)
            # Assumes IMU is roughly level (valid for ground robot)
            self._mag_heading = math.atan2(self._mag_raw[1], self._mag_raw[0])
            self._mag_heading += math.radians(IMU_MAG_DECLINATION_DEG)
            self._mag_heading = self._normalize_angle(self._mag_heading)

            # Complementary filter: fuse gyro rate with magnetometer heading
            # Gyro provides high-frequency response, mag provides low-frequency correction
            alpha = IMU_COMPLEMENTARY_ALPHA
            gyro_heading = self._heading + self._gyro_raw[2] * self._dt
            self._heading = alpha * gyro_heading + (1 - alpha) * self._mag_heading
            self._heading = self._normalize_angle(self._heading)

            # Pitch and roll from accelerometer (for tilt compensation)
            ax, ay, az = self._accel_raw
            self._pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
            self._roll = math.atan2(ay, az)

    def _read_raw(self) -> IMUReading:
        """Read raw sensor data from hardware or mock."""
        if self._mock_mode:
            return self._mock_reading()

        if self._icm is None:
            return IMUReading(
                timestamp=time.time(),
                accel=(0, 0, 9.81),
                gyro=(0, 0, 0),
                mag=(0, 0, 0),
            )

        try:
            accel = self._icm.acceleration   # (ax, ay, az) in m/s²
            gyro = self._icm.gyro            # (gx, gy, gz) in rad/s
            mag = self._icm.magnetic         # (mx, my, mz) in uT
            return IMUReading(
                timestamp=time.time(),
                accel=accel,
                gyro=gyro,
                mag=mag,
            )
        except Exception:
            return IMUReading(
                timestamp=time.time(),
                accel=(0, 0, 9.81),
                gyro=(0, 0, 0),
                mag=(0, 0, 0),
            )

    # =========================================================================
    # Properties (thread-safe reads of latest state)
    # =========================================================================

    @property
    def heading(self) -> float:
        """Fused heading in radians (CCW from +X, mag-corrected)."""
        with self._lock:
            return self._heading

    @property
    def gyro_z(self) -> float:
        """Yaw rate in rad/s (bias-corrected)."""
        with self._lock:
            return float(self._gyro_raw[2])

    @property
    def gyro_xyz(self) -> Tuple[float, float, float]:
        """Full gyro reading (gx, gy, gz) in rad/s, bias-corrected."""
        with self._lock:
            return tuple(self._gyro_raw)

    @property
    def accel_xy(self) -> Tuple[float, float]:
        """Horizontal acceleration (ax, ay) in m/s², body frame.
        Gravity component removed assuming the robot is on a flat floor."""
        with self._lock:
            # Remove gravity (assume flat: gravity is pure Z)
            ax = self._accel_raw[0]
            ay = self._accel_raw[1]
            return (float(ax), float(ay))

    @property
    def accel_xyz(self) -> Tuple[float, float, float]:
        """Full accelerometer reading (ax, ay, az) in m/s²."""
        with self._lock:
            return tuple(self._accel_raw)

    @property
    def mag_heading(self) -> float:
        """Raw magnetometer heading in radians (before complementary filter)."""
        with self._lock:
            return self._mag_heading

    @property
    def pitch(self) -> float:
        """Pitch angle in radians (from accelerometer)."""
        with self._lock:
            return self._pitch

    @property
    def roll(self) -> float:
        """Roll angle in radians (from accelerometer)."""
        with self._lock:
            return self._roll

    @property
    def dt(self) -> float:
        """Time delta since last update() call."""
        return self._dt

    # =========================================================================
    # Background Thread
    # =========================================================================

    def start_background(self, rate_hz: int = None):
        """Start reading IMU in a background thread.

        Args:
            rate_hz: Update rate (default: IMU_GYRO_RATE_HZ).
        """
        if self._running:
            return
        rate = rate_hz or IMU_GYRO_RATE_HZ
        self._running = True
        self._thread = threading.Thread(
            target=self._background_loop, args=(rate,), daemon=True)
        self._thread.start()

    def stop_background(self):
        """Stop the background reading thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _background_loop(self, rate_hz: int):
        """Background thread: read IMU at fixed rate."""
        interval = 1.0 / rate_hz
        while self._running:
            self.update()
            time.sleep(interval)

    # =========================================================================
    # EKF Integration Helpers
    # =========================================================================

    def get_prediction_delta(self, dt: float = None) -> Tuple[float, float, float]:
        """Get IMU-based motion prediction for the EKF.

        Returns (dx, dy, dtheta) estimated from gyro and accelerometer
        over the given time interval.

        This replaces VO as the primary prediction source when the IMU is
        available. VO then becomes an observation that corrects drift.

        Args:
            dt: Time interval. If None, uses self.dt.

        Returns:
            (dx, dy, dtheta) in robot frame (meters, meters, radians).
        """
        if dt is None:
            dt = self._dt

        with self._lock:
            # Heading change from gyro (very accurate for short dt)
            dtheta = self._gyro_raw[2] * dt

            # Position change from accelerometer (noisy, but fills gaps between VO)
            # Double-integrate: dx = 0.5 * ax * dt^2 (first-order approximation)
            # This is only useful for short intervals -- accel drift is severe
            # over longer periods. VO corrects this every ~100ms.
            ax, ay = self.accel_xy
            dx = 0.5 * ax * dt * dt
            dy = 0.5 * ay * dt * dt

        return (dx, dy, dtheta)

    def get_heading_observation(self) -> Tuple[float, float]:
        """Get magnetometer heading as an EKF observation.

        Returns:
            (heading_rad, noise_rad) for EKF update step.
        """
        return (self.heading, IMU_MAG_NOISE_RAD)

    # =========================================================================
    # Mock / Simulation
    # =========================================================================

    def _mock_reading(self) -> IMUReading:
        """Generate a mock IMU reading (for testing without hardware)."""
        return IMUReading(
            timestamp=time.time(),
            accel=(0.0, 0.0, 9.81),  # stationary, gravity in Z
            gyro=(0.0, 0.0, 0.0),     # no rotation
            mag=(20.0, 0.0, 40.0),    # arbitrary field
        )

    def set_mock_data(self, accel=None, gyro=None, mag=None):
        """Inject mock sensor data (for simulation/testing).

        Args:
            accel: (ax, ay, az) in m/s².
            gyro: (gx, gy, gz) in rad/s.
            mag: (mx, my, mz) in uT.
        """
        with self._lock:
            if accel is not None:
                self._accel_raw = np.array(accel)
            if gyro is not None:
                self._gyro_raw = np.array(gyro) - self._gyro_bias
            if mag is not None:
                self._mag_raw = np.array(mag)

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop_background()
