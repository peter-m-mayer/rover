"""
Simulated IMU: generates realistic gyro/accel/mag data from PyBullet ground truth.

Drop-in for raspbot_slam.imu.IMU -- same interface, backed by physics simulation.
Adds configurable noise, bias, and drift to match real sensor behavior.
"""

import math
import time
from typing import Tuple
import numpy as np
import pybullet as p

from ..imu import IMU, IMUReading, IMU_GYRO_NOISE_RAD, IMU_ACCEL_NOISE_M, IMU_MAG_NOISE_RAD


class SimIMU(IMU):
    """Simulated ICM-20948 IMU backed by PyBullet ground-truth pose.

    Generates sensor readings by differentiating the GT pose (for gyro),
    using gravity + body acceleration (for accel), and a simulated
    magnetic field (for magnetometer). Configurable noise on all axes.

    Usage:
        world = SimWorld(...)
        imu = SimIMU(world)
        imu.calibrate_gyro()  # instant in sim
        imu.update()
        heading = imu.heading
    """

    def __init__(self, world, gyro_noise: float = None, accel_noise: float = None,
                 mag_noise: float = None, gyro_bias: Tuple[float, float, float] = None):
        """Initialize simulated IMU.

        Args:
            world: SimWorld instance.
            gyro_noise: Noise std in rad/s (default: IMU_GYRO_NOISE_RAD).
            accel_noise: Noise std in m/s^2 (default: IMU_ACCEL_NOISE_M).
            mag_noise: Noise std in uT (default: ~2.5 uT).
            gyro_bias: Simulated gyro bias (gx, gy, gz) in rad/s.
        """
        super().__init__(mock_mode=True)  # skip hardware init

        self._world = world
        self._available = True
        self._calibrated = False

        # Noise levels
        self._gyro_noise = gyro_noise if gyro_noise is not None else IMU_GYRO_NOISE_RAD
        self._accel_noise = accel_noise if accel_noise is not None else IMU_ACCEL_NOISE_M
        self._mag_noise = mag_noise if mag_noise is not None else 2.5  # uT

        # Simulated bias (to be calibrated out)
        self._sim_gyro_bias = np.array(gyro_bias or (0.002, -0.001, 0.003))

        # Simulated magnetic field (approximate Earth's field, NED frame)
        # Typical: ~20 uT north, ~0 uT east, ~40 uT down (mid-latitude)
        self._earth_mag_field = np.array([20.0, 0.0, 40.0])

        # Previous state for velocity/acceleration computation
        self._prev_pos = None
        self._prev_vel = None
        self._prev_yaw = None
        self._prev_time = None
        self._rng = np.random.RandomState(42)

    def calibrate_gyro(self, n_samples=None, verbose=True):
        """In simulation, calibration is instant -- we know the true bias."""
        self._gyro_bias = self._sim_gyro_bias.copy()
        self._calibrated = True
        if verbose:
            print(f"SimIMU: Gyro bias calibrated: ({self._gyro_bias[0]:.4f}, "
                  f"{self._gyro_bias[1]:.4f}, {self._gyro_bias[2]:.4f}) rad/s")

    def calibrate_magnetometer(self, duration_s=None, verbose=True):
        """In simulation, mag calibration is instant."""
        self._mag_offset = np.zeros(3)
        self._mag_scale = np.ones(3)
        if verbose:
            print("SimIMU: Magnetometer calibrated (no offset in simulation)")

    def _read_raw(self) -> IMUReading:
        """Generate IMU reading from PyBullet ground truth + noise."""
        now = time.time()
        pos, orn = p.getBasePositionAndOrientation(self._world.rover_id)
        vel, ang_vel = p.getBaseVelocity(self._world.rover_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # =====================================================================
        # Gyroscope: angular velocity + bias + noise
        # =====================================================================
        # PyBullet ang_vel is in world frame; gyro measures in body frame
        # For a ground robot, gz_body ≈ ang_vel_z_world
        gyro = np.array([
            ang_vel[0],  # roll rate
            ang_vel[1],  # pitch rate
            ang_vel[2],  # yaw rate (the one we care about)
        ])
        gyro += self._sim_gyro_bias
        gyro += self._rng.normal(0, self._gyro_noise, 3)

        # =====================================================================
        # Accelerometer: gravity + linear acceleration + noise
        # =====================================================================
        # On a flat floor, accel ≈ (0, 0, 9.81) + body acceleration
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        # Linear acceleration from velocity difference
        if self._prev_vel is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0:
                # World frame acceleration
                ax_world = (vel[0] - self._prev_vel[0]) / dt
                ay_world = (vel[1] - self._prev_vel[1]) / dt
                # Transform to body frame
                ax_body = ax_world * cos_y + ay_world * sin_y
                ay_body = -ax_world * sin_y + ay_world * cos_y
            else:
                ax_body, ay_body = 0, 0
        else:
            ax_body, ay_body = 0, 0

        accel = np.array([
            ax_body,       # forward acceleration
            ay_body,       # lateral acceleration
            9.81,          # gravity (Z-up)
        ])
        accel += self._rng.normal(0, self._accel_noise, 3)

        # =====================================================================
        # Magnetometer: Earth's field rotated by robot heading + noise
        # =====================================================================
        # Rotate Earth's field into body frame
        mx = self._earth_mag_field[0] * cos_y + self._earth_mag_field[1] * sin_y
        my = -self._earth_mag_field[0] * sin_y + self._earth_mag_field[1] * cos_y
        mz = self._earth_mag_field[2]

        mag = np.array([mx, my, mz])
        mag += self._rng.normal(0, self._mag_noise, 3)

        # Save state for next iteration
        self._prev_vel = vel
        self._prev_time = now

        return IMUReading(
            timestamp=now,
            accel=tuple(accel),
            gyro=tuple(gyro),
            mag=tuple(mag),
        )
