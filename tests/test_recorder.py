"""Record-and-replay logger: streams land on disk and read back intact."""

import time

import numpy as np

from raspbot_slam.actuators import Actuators
from raspbot_slam.recorder import RunLog, RunRecorder


class FakeCamera:
    K = np.eye(3)
    dist_coeffs = np.zeros(5)

    def __init__(self):
        self.n = 0

    def capture_color(self):
        self.n += 1
        img = np.zeros((48, 64, 3), np.uint8)
        img[:, : self.n % 64] = 200
        return img


class FakeIMU:
    acceleration = (0.0, 0.0, 9.81)
    gyro = (0.0, 0.0, 0.1)
    magnetic = (20.0, 0.0, -40.0)


class FakeSensors:
    def enable_ultrasonic(self):
        pass

    def read_ultrasonic_mm(self):
        return 812

    def read_line_tracker(self):
        return (True, False, False, True)


def test_round_trip(tmp_path):
    act = Actuators(bot=None)
    rec = RunRecorder(camera=FakeCamera(), imu=FakeIMU(), sensors=FakeSensors(),
                      actuators=act, root=str(tmp_path), label="t",
                      frame_hz=40, imu_hz=200, sensor_hz=40)
    with rec:
        rec.mark("go")
        act.drive(60, 0, 0)
        time.sleep(0.25)
        act.set_servo_pan(80)
        act.stop()
        time.sleep(0.05)
        assert rec.latest_frame() is not None

    run = RunLog(rec.path)
    assert len(run.frames["t"]) >= 3
    assert len(list(run.iter_frames())) == len(run.frames["t"])
    assert len(run.imu["gz"]) >= 20
    assert np.allclose(run.imu["az"], 9.81) and np.allclose(run.imu["gz"], 0.1)
    assert set(run.sensors["us_mm"]) == {812.0}
    assert set(run.sensors["line"]) == {9.0}          # 1001b
    kinds = [c["kind"] for c in run.cmds]
    assert kinds[:2] == ["pan", "tilt"] and "motors" in kinds
    segs = run.motor_segments()
    assert segs[0][2] == (60, 60, 60, 60) and segs[-1][2] == (0, 0, 0, 0)
    assert segs[0][1] > segs[0][0] + 0.2
    assert [lab for _, lab in run.events] == ["start", "go", "stop"]
    assert run.meta["config"]["ARUCO_MARKER_SIZE_M"] > 0
    # timestamps share one monotonic clock and are ordered
    assert np.all(np.diff(run.imu["t"]) >= 0)   # >= : Windows monotonic ticks ~16 ms
    assert run.event_time("start") <= run.frames["t"][0]


def test_actuators_unwrapped_after_stop(tmp_path):
    act = Actuators(bot=None)
    rec = RunRecorder(actuators=act, root=str(tmp_path))
    with rec:
        assert "_set_motors" in act.__dict__
    assert "_set_motors" not in act.__dict__
    act.drive(10)    # still works, no longer logged
