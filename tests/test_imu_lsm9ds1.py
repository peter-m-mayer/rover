import math
import struct

import numpy as np
import pytest

from raspbot_slam import lsm9ds1
from raspbot_slam.imu import axis_map_matrix, to_robot_frame
from catchaser.imu_check import axis_of, build_axis_map


class FakeBus:
    """Register-level stand-in for smbus2.SMBus with an LSM9DS1 on it."""

    def __init__(self, ag=0x6B, mag=0x1E):
        self.regs = {ag: {0x0F: 0x68}, mag: {0x0F: 0x3D}}
        self.writes = []

    def read_byte_data(self, addr, reg):
        if addr not in self.regs:
            raise OSError(121, "Remote I/O error")
        return self.regs[addr].get(reg, 0)

    def write_byte_data(self, addr, reg, val):
        self.writes.append((addr, reg, val))

    def set_xyz(self, addr, reg, xyz):
        for i, b in enumerate(struct.pack("<hhh", *xyz)):
            self.regs[addr][reg + i] = b

    def read_i2c_block_data(self, addr, reg, n):
        reg &= 0x7F                       # mag burst-read flag
        return [self.regs[addr].get(reg + i, 0) for i in range(n)]


def test_probe_and_configure():
    bus = FakeBus()
    dev = lsm9ds1.LSM9DS1(bus=bus)
    assert (dev.ag_address, dev.mag_address) == (0x6B, 0x1E)
    assert (0x6B, 0x10, 0b10001000) in bus.writes      # gyro 238 Hz, 500 dps
    assert (0x1E, 0x22, 0x00) in bus.writes            # mag continuous


def test_alternate_addresses():
    dev = lsm9ds1.LSM9DS1(bus=FakeBus(ag=0x6A, mag=0x1C))
    assert (dev.ag_address, dev.mag_address) == (0x6A, 0x1C)


def test_missing_chip_raises():
    bus = FakeBus()
    bus.regs = {}
    with pytest.raises(RuntimeError):
        lsm9ds1.LSM9DS1(bus=bus)


def test_unit_conversion():
    bus = FakeBus()
    dev = lsm9ds1.LSM9DS1(bus=bus)
    bus.set_xyz(0x6B, 0x28, (0, 0, 8197))               # ~1 g on z
    bus.set_xyz(0x6B, 0x18, (0, 0, 5714))               # ~100 dps about z
    bus.set_xyz(0x1E, 0x28, (1000, 2000, -3000))
    ax, ay, az = dev.acceleration
    assert az == pytest.approx(9.807, abs=0.01)
    assert dev.gyro[2] == pytest.approx(math.radians(100.0), rel=1e-3)
    mx, my, mz = dev.magnetic
    assert (mx, my, mz) == pytest.approx((-14.0, -28.0, -42.0))  # mag X, Y flipped (180 deg about Z)


def test_axis_map_rotation():
    m = axis_map_matrix(("+y", "-x", "+z"))
    assert to_robot_frame((1.0, 2.0, 3.0), ("+y", "-x", "+z")) == pytest.approx((2.0, -1.0, 3.0))
    assert np.linalg.det(m) == pytest.approx(1.0)


def test_axis_map_rejects_reflection():
    with pytest.raises(ValueError):
        axis_map_matrix(("+y", "+x", "+z"))


def test_build_axis_map_examples():
    # Board flat, component side up, printed Y arrow pointing forward.
    assert build_axis_map((2, 1), (1, 1)) == ("+y", "-x", "+z")
    # Board upside down (z down), X arrow pointing backward.
    amap = build_axis_map((2, -1), (0, -1))
    assert amap[0] == "-x" and amap[2] == "-z"
    axis_map_matrix(amap)                               # must be a rotation


def test_axis_of_skips_excluded():
    assert axis_of((0.1, -0.4, 9.8), exclude=(2,)) == (1, -1)
