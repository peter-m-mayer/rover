"""IMU bring-up and calibration for the LSM9DS1 (or ICM-20948).

Run the stages in order after wiring (docs/IMU_LSM9DS1.html). Each stage prints
PASS/FAIL with the numbers behind it.

    python3 -m catchaser.imu_check            # 1+2: detect, then 3 s still test
    python3 -m catchaser.imu_check --axes     # 3: measure the sensor->robot axis
                                              #    map (robot spins + nudges
                                              #    forward: on the floor, clear)
    python3 -m catchaser.imu_check --mag      # 4: magnetometer calibration
                                              #    (robot spins 2 turns in place)
    python3 -m catchaser.imu_check --axes --write   # also save IMU_AXIS_MAP
    python3 -m catchaser.imu_check --stream   # live robot-frame numbers

Stage 3 needs no knowledge of how the board was mounted:
  up      = the axis gravity reads +1 g on while still,
  yaw     = a commanded CCW spin must show up as + rotation about "up"
            (checks the gyro sign and that the board is rigidly mounted),
  forward = the horizontal axis that jumps first when the robot launches
            forward (the accelerometer feels the push),
  left    = up x forward (right-handed).
"""

import argparse
import math
import os
import re
import sys
import time

import numpy as np

from raspbot_slam import config

AXES = "xyz"


def spec(axis, sign):
    return ("+" if sign > 0 else "-") + AXES[axis]


def axis_of(v, exclude=()):
    """(index, sign) of the dominant component of v, skipping `exclude`."""
    v = np.array(v, float)
    for i in exclude:
        v[i] = 0.0
    i = int(np.argmax(np.abs(v)))
    return i, (1 if v[i] > 0 else -1)


def build_axis_map(up, forward):
    """Robot-frame rows (forward, left, up) as sensor-axis specs, from the
    sensor (index, sign) of up and forward. left = up x forward."""
    e = np.eye(3)
    u = up[1] * e[up[0]]
    f = forward[1] * e[forward[0]]
    left = np.cross(u, f)
    li = int(np.argmax(np.abs(left)))
    return (spec(*forward), spec(li, left[li]), spec(*up))


def _raw(dev):
    return np.array(dev.acceleration), np.array(dev.gyro), np.array(dev.magnetic)


def sample(dev, secs, rate=100.0):
    acc, gyr, mag, ts = [], [], [], []
    t0 = time.time()
    while time.time() - t0 < secs:
        a, g, m = _raw(dev)
        acc.append(a); gyr.append(g); mag.append(m); ts.append(time.time())
        time.sleep(max(0.0, 1.0 / rate - 0.002))
    return np.array(acc), np.array(gyr), np.array(mag), np.array(ts)


def detect():
    bus = getattr(config, "IMU_I2C_BUS", 3)
    if not os.path.exists(f"/dev/i2c-{bus}"):
        print(f"FAIL  /dev/i2c-{bus} missing. Add to /boot/firmware/config.txt:\n"
              f"        dtoverlay=i2c3-pi5,pins_22_23,baudrate=400000\n"
              f"      then reboot (docs/IMU_LSM9DS1.html, 'What I do next').")
        return None
    from raspbot_slam.lsm9ds1 import LSM9DS1
    try:
        dev = LSM9DS1(bus=bus)
    except Exception as e:                      # noqa: BLE001 - report anything
        print(f"FAIL  no LSM9DS1 on /dev/i2c-{bus}: {e}\n"
              f"      check wiring: `i2cdetect -y {bus}` should show 1e and 6b.")
        return None
    print(f"PASS  LSM9DS1 on /dev/i2c-{bus}: accel/gyro {hex(dev.ag_address)}, "
          f"mag {hex(dev.mag_address)}, die temp {dev.temperature:.1f} C")
    return dev


def still_test(dev, secs=3.0):
    acc, gyr, mag, ts = sample(dev, secs)
    rate = (len(ts) - 1) / (ts[-1] - ts[0])
    g_norm = np.linalg.norm(acc.mean(0))
    bias = np.degrees(gyr.mean(0))
    noise = np.degrees(gyr.std(0))
    up = axis_of(acc.mean(0))
    ok_g = abs(g_norm - 9.81) < 1.0      # LSM9DS1 zero-g/scale tolerance: ~5% seen
    ok_noise = noise.max() < 0.5
    print(f"{'PASS' if ok_g else 'FAIL'}  |accel| = {g_norm:.2f} m/s^2 (expect 9.81); "
          f"gravity on sensor {spec(*up)}  (rate {rate:.0f} Hz)")
    print(f"{'PASS' if ok_noise else 'FAIL'}  gyro bias {np.round(bias, 2)} deg/s, "
          f"noise {np.round(noise, 3)} deg/s rms")
    print(f"INFO  |mag| = {np.linalg.norm(mag.mean(0)):.1f} uT uncalibrated "
          f"(Earth ~45-50 uT; much more = magnetic parts nearby)")
    if noise.max() > 0.2:
        print("WARN  gyro noise high: board loose, or motors running?")
    return gyr.mean(0), up


def _spin(actuators, turn, secs):
    try:
        actuators.drive(0.0, turn=turn)
        time.sleep(secs)
    finally:
        actuators.stop()


def measure_axes(dev, actuators, gyro_bias, up):
    # Yaw: command a CCW (left) spin; turn < 0 = counter-clockwise.
    t0 = time.time()
    actuators.drive(0.0, turn=-55)
    acc, gyr, mag, ts = [], [], [], []
    try:
        while time.time() - t0 < 1.2:
            a, g, m = _raw(dev)
            gyr.append(g - gyro_bias); ts.append(time.time())
            time.sleep(0.008)
    finally:
        actuators.stop()
    time.sleep(0.8)
    gyr, ts = np.array(gyr), np.array(ts)
    rot = (gyr[1:] * np.diff(ts)[:, None]).sum(0)          # integrated, rad
    yaw_axis = axis_of(rot)
    ccw_deg = math.degrees(rot[up[0]] * up[1])
    ok = yaw_axis[0] == up[0] and ccw_deg > 0
    print(f"{'PASS' if ok else 'FAIL'}  CCW spin: {ccw_deg:+.1f} deg about 'up' "
          f"(sensor {spec(*yaw_axis)} dominant; integrated {np.round(np.degrees(rot), 1)})")
    if not ok:
        print("      the spin did not show as + rotation about the gravity axis: "
              "board not rigid, or a driver axis sign is wrong.")

    # Forward: launch forward, look at the first ~0.25 s of horizontal accel.
    still = np.array([dev.acceleration for _ in range(20)]).mean(0)
    acc, ts = [], []
    t0 = time.time()
    actuators.drive(70)
    try:
        while time.time() - t0 < 0.25:
            acc.append(np.array(dev.acceleration) - still)
            time.sleep(0.004)
    finally:
        actuators.stop()
    time.sleep(0.5)
    push = np.array(acc).mean(0)
    fwd = axis_of(push, exclude=(up[0],))
    horiz = np.delete(push, up[0])
    clear = np.abs(horiz).max() > 2.5 * np.abs(horiz).min() + 0.05
    print(f"{'PASS' if clear else 'WARN'}  launch push {np.round(push, 2)} m/s^2 "
          f"-> forward = sensor {spec(*fwd)}"
          + ("" if clear else "  (ambiguous: board at ~45 deg? re-run or pass --forward)"))
    # Mecanum launch can also yaw; the push still dominates on the true axis.
    return fwd, ok


def calibrate_mag(dev, actuators, gyro_bias, axis_map, turns=2.0, timeout=25.0):
    from raspbot_slam.imu import axis_map_matrix
    R = axis_map_matrix(axis_map)
    mags, yaw, ts = [], 0.0, []
    yaws = []
    t_last = time.time()
    t0 = t_last
    actuators.drive(0.0, turn=-45)                          # CCW
    try:
        while abs(yaw) < math.radians(360 * turns) and time.time() - t0 < timeout:
            a, g, m = _raw(dev)
            now = time.time()
            yaw += (R @ (g - gyro_bias))[2] * (now - t_last)
            t_last = now
            mags.append(R @ m)
            yaws.append(yaw)
            time.sleep(0.01)
    finally:
        actuators.stop()
    mags = np.array(mags)
    print(f"INFO  spun {math.degrees(yaw):.0f} deg (gyro) with {len(mags)} mag samples")
    if abs(yaw) < math.radians(300):
        print("FAIL  less than one full turn — wheels slipping or blocked")
        return None
    lo, hi = mags.min(0), mags.max(0)
    offset = (lo + hi) / 2
    rng = hi - lo
    scale = np.ones(3)
    scale[:2] = rng[:2].mean() / np.maximum(rng[:2], 1e-6)
    c = (mags - offset) * scale
    heading = np.unwrap(np.arctan2(-c[:, 1], c[:, 0]))      # same as imu.py
    ratio = (heading[-1] - heading[0]) / (yaws[-1] - yaws[0])
    radius = np.hypot(c[:, 0], c[:, 1])
    print(f"INFO  hard-iron offset {np.round(offset, 1)} uT, horizontal field "
          f"{radius.mean():.1f} uT (+-{radius.std():.1f})")
    ok = 0.8 < ratio < 1.2
    print(f"{'PASS' if ok else 'FAIL'}  compass turned {ratio:+.2f}x the gyro "
          f"(expect +1.00)")
    if ratio < -0.5:
        print("      compass runs BACKWARDS: flip the sign of Y in "
              "lsm9ds1.MAG_TO_AG (mag die axes), then re-run --mag")
    if radius.std() > 0.25 * radius.mean():
        print("WARN  field circle very uneven: motor/battery magnets dominate. "
              "Mount the IMU farther from motors, or rely on gyro only.")
    return offset, scale, ok


def write_axis_map(axis_map, path=None):
    path = path or os.path.join(os.path.dirname(config.__file__), "config.py")
    src = open(path, encoding="utf-8").read()
    new = f'IMU_AXIS_MAP = ("{axis_map[0]}", "{axis_map[1]}", "{axis_map[2]}")'
    src = re.sub(r"^IMU_AXIS_MAP = \([^)]*\)", new, src, count=1, flags=re.M)
    open(path, "w", encoding="utf-8", newline="\n").write(src)
    print(f"wrote {new} to {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IMU bring-up / calibration")
    ap.add_argument("--axes", action="store_true", help="measure axis map (robot moves)")
    ap.add_argument("--forward", default=None,
                    help="skip the launch test: sensor axis pointing forward, e.g. +y")
    ap.add_argument("--mag", action="store_true", help="mag calibration (robot spins)")
    ap.add_argument("--write", action="store_true", help="save axis map to config.py")
    ap.add_argument("--stream", action="store_true", help="print robot-frame data")
    args = ap.parse_args(argv)

    dev = detect()
    if dev is None:
        return 1
    bias, up = still_test(dev)

    if args.stream:
        from raspbot_slam.imu import IMU
        imu = IMU()
        imu.calibrate_gyro(verbose=False)
        try:
            while True:
                imu.update()
                ax, ay, az = imu.accel_xyz
                print(f"\rgyro_z {math.degrees(imu.gyro_z):+7.2f} deg/s  heading "
                      f"{math.degrees(imu.heading):+7.1f}  mag {math.degrees(imu.mag_heading):+7.1f}"
                      f"  accel {ax:+5.2f} {ay:+5.2f} {az:+5.2f}", end="", flush=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print()
        return 0

    if not (args.axes or args.mag):
        return 0
    from catchaser.hw import make_hardware
    bot, camera, sensors, actuators = make_hardware()
    camera.close()
    if bot is None:
        print("FAIL  no robot driver (mock mode)")
        return 1
    axis_map = getattr(config, "IMU_AXIS_MAP", ("+x", "+y", "+z"))
    try:
        if args.axes:
            if args.forward:
                fwd = (AXES.index(args.forward[1]), -1 if args.forward[0] == "-" else 1)
            else:
                fwd, _ = measure_axes(dev, actuators, bias, up)
            axis_map = build_axis_map(up, fwd)
            print(f"RESULT IMU_AXIS_MAP = {axis_map}   (robot x fwd, y left, z up)")
            if args.write:
                write_axis_map(axis_map)
        if args.mag:
            res = calibrate_mag(dev, actuators, bias, axis_map)
            if res is not None:
                from raspbot_slam.imu import IMU
                imu = IMU(mock_mode=True)
                imu.model = "LSM9DS1"
                imu._mag_offset, imu._mag_scale = res[0], res[1]
                imu.save_calibration(axis_map=list(axis_map), compass_ok=bool(res[2]))
                print(f"saved {IMU.calibration_path()}")
    finally:
        actuators.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
