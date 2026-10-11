"""
LSM9DS1 9-DOF IMU driver (accel + gyro + magnetometer) over plain smbus2.

Hardware: Adafruit LSM9DS1 STEMMA QT breakout (product 4634) on its own I2C
bus — on this robot, Pi 5 i2c3 on GPIO22 (SDA, header pin 15) / GPIO23 (SCL,
pin 16), enabled by `dtoverlay=i2c3-pi5,pins_22_23,baudrate=400000` in
/boot/firmware/config.txt. See docs/IMU_LSM9DS1.html for the wiring.

The chip shows up as TWO I2C devices: accel/gyro at 0x6B (0x6A if SDO_AG is
pulled low) and magnetometer at 0x1E (0x1C). Both are probed.

Exposes the same three properties as adafruit_icm20x.ICM20948, so the IMU
class in imu.py can use either part interchangeably:

    acceleration -> (ax, ay, az) m/s^2
    gyro         -> (gx, gy, gz) rad/s
    magnetic     -> (mx, my, mz) uT

All three are in the ACCEL/GYRO chip frame (the X/Y arrows printed on the
Adafruit board). The magnetometer die's X/Y axes are rotated relative to the
accel/gyro frame (datasheet Fig. 1); MAG_TO_AG re-expresses the mag reading
in the accel/gyro frame. catchaser.imu_check verifies this on hardware.
"""

import math
import struct
import time

# Accel/gyro registers
_WHO_AM_I_AG = 0x0F           # -> 0x68
_CTRL_REG1_G = 0x10
_OUT_TEMP_L = 0x15
_OUT_X_L_G = 0x18
_CTRL_REG6_XL = 0x20
_CTRL_REG8 = 0x22
_OUT_X_L_XL = 0x28
# Magnetometer registers
_WHO_AM_I_M = 0x0F            # -> 0x3D
_CTRL_REG1_M = 0x20
_CTRL_REG2_M = 0x21
_CTRL_REG3_M = 0x22
_CTRL_REG4_M = 0x23
_CTRL_REG5_M = 0x24
_OUT_X_L_M = 0x28
_MAG_AUTOINC = 0x80           # mag needs the sub-address MSB set to burst-read

AG_ADDRESSES = (0x6B, 0x6A)
MAG_ADDRESSES = (0x1E, 0x1C)

# Settings: gyro 238 Hz / +-500 dps, accel 238 Hz / +-4 g, mag 80 Hz / +-4 gauss.
_GYRO_CFG = 0b100_01_000      # ODR_G=238 Hz, FS_G=500 dps
_ACCEL_CFG = 0b100_10_000     # ODR_XL=238 Hz, FS_XL=+-4 g
GYRO_DPS_PER_LSB = 0.0175     # 17.50 mdps/LSB at 500 dps
ACCEL_G_PER_LSB = 0.000122    # 0.122 mg/LSB at +-4 g
MAG_UT_PER_LSB = 0.014        # 0.14 mgauss/LSB at +-4 gauss = 0.014 uT
G = 9.80665

# Mag axes -> accel/gyro axes (per-axis sign). The mag die is rotated 180 deg
# about Z relative to the accel/gyro die. Measured 2026-10-11 (imu_check --mag):
# flipping X alone (a mirror image) made the compass turn -1.01x the gyro.
MAG_TO_AG = (-1.0, -1.0, 1.0)


def _open_bus(bus):
    if not isinstance(bus, int):
        return bus                       # already an SMBus-like object (tests)
    from smbus2 import SMBus
    return SMBus(bus)


class LSM9DS1:
    """Minimal polling driver. Raises OSError/RuntimeError if not found."""

    def __init__(self, bus=3, ag_address=None, mag_address=None):
        self._bus = _open_bus(bus)
        self.ag_address = ag_address or self._probe(AG_ADDRESSES, _WHO_AM_I_AG, 0x68)
        self.mag_address = mag_address or self._probe(MAG_ADDRESSES, _WHO_AM_I_M, 0x3D)
        self._configure()

    def _probe(self, addresses, reg, expect):
        for addr in addresses:
            try:
                if self._bus.read_byte_data(addr, reg) == expect:
                    return addr
            except OSError:
                continue
        raise RuntimeError(f"LSM9DS1 not found (tried {[hex(a) for a in addresses]}, "
                           f"WHO_AM_I {hex(expect)})")

    def _configure(self):
        ag, m = self.ag_address, self.mag_address
        self._bus.write_byte_data(ag, _CTRL_REG8, 0x05)     # SW_RESET + IF_ADD_INC
        time.sleep(0.02)
        self._bus.write_byte_data(ag, _CTRL_REG8, 0x44)     # BDU + IF_ADD_INC
        self._bus.write_byte_data(ag, _CTRL_REG1_G, _GYRO_CFG)
        self._bus.write_byte_data(ag, _CTRL_REG6_XL, _ACCEL_CFG)
        self._bus.write_byte_data(m, _CTRL_REG2_M, 0x0C)    # REBOOT + SOFT_RST
        time.sleep(0.02)
        self._bus.write_byte_data(m, _CTRL_REG1_M, 0xFC)    # temp comp, XY ultra-high, 80 Hz
        self._bus.write_byte_data(m, _CTRL_REG2_M, 0x00)    # +-4 gauss
        self._bus.write_byte_data(m, _CTRL_REG3_M, 0x00)    # continuous conversion
        self._bus.write_byte_data(m, _CTRL_REG4_M, 0x0C)    # Z ultra-high performance
        self._bus.write_byte_data(m, _CTRL_REG5_M, 0x40)    # BDU
        time.sleep(0.05)

    def _xyz(self, addr, reg):
        raw = bytes(self._bus.read_i2c_block_data(addr, reg, 6))
        return struct.unpack("<hhh", raw)

    @property
    def gyro(self):
        k = math.radians(GYRO_DPS_PER_LSB)
        return tuple(v * k for v in self._xyz(self.ag_address, _OUT_X_L_G))

    @property
    def acceleration(self):
        k = ACCEL_G_PER_LSB * G
        return tuple(v * k for v in self._xyz(self.ag_address, _OUT_X_L_XL))

    @property
    def magnetic(self):
        raw = self._xyz(self.mag_address, _OUT_X_L_M | _MAG_AUTOINC)
        return tuple(s * v * MAG_UT_PER_LSB for s, v in zip(MAG_TO_AG, raw))

    @property
    def temperature(self):
        lo, hi = self._bus.read_i2c_block_data(self.ag_address, _OUT_TEMP_L, 2)
        return 25.0 + struct.unpack("<h", bytes((lo, hi)))[0] / 16.0
