"""Tests for rcbot — the web drive controller (logic + endpoints, no hardware)."""

import pytest

from catchaser.rc import RCController
from raspbot_slam import config


class RecordingActuators:
    def __init__(self):
        self.cmds = []
        self.pan = None
        self.tilt = None
        self.stops = 0

    def drive(self, forward, turn=0.0, strafe=0.0):
        self.cmds.append((forward, turn, strafe))

    def stop(self):
        self.stops += 1

    def set_servo_pan(self, a):
        self.pan = a

    def set_servo_tilt(self, a):
        self.tilt = a


class FakeSensors:
    def read_ultrasonic_mm(self):
        return 512


def rc(**kw):
    return RCController(RecordingActuators(), FakeSensors(), **kw)


class TestDrive:
    def test_forward_scales_by_speed(self):
        c = rc(speed=100)
        f, t, s = c.drive(1.0, 0.0, 0.0)
        assert (f, t, s) == (100.0, 0.0, 0.0)
        assert c.act.cmds[-1] == (100.0, 0.0, 0.0)

    def test_strafe_and_turn_signs(self):
        c = rc(speed=100)
        _, _, s = c.drive(0, 1.0, 0)      # +strafe = right
        assert s == 100.0
        _, t, _ = c.drive(0, 0, 1.0)      # +turn = clockwise/right
        assert t == 100.0

    def test_combined_vector_capped(self):
        c = rc(speed=config.CHASE_MAX_SPEED)
        f, t, s = c.drive(1.0, 1.0, 1.0)  # would be 3x cap before scaling
        assert abs(f) + abs(t) + abs(s) <= config.CHASE_MAX_SPEED + 1e-6
        assert f == t == s                # equal inputs stay equal after scaling

    def test_diagonal_preserved(self):
        c = rc(speed=80)
        f, t, s = c.drive(1.0, 1.0, 0.0)  # forward + strafe = diagonal
        assert f > 0 and s > 0 and t == 0


class TestSpeed:
    def test_nudge_and_clamp(self):
        c = rc(speed=90)
        assert c.nudge_speed(10) == 100
        assert c.set_speed(5) == 20                      # floor
        assert c.set_speed(9999) == config.MOTOR_SPEED_MAX


class TestServos:
    def test_pan_tilt_clamp_to_limits(self):
        c = rc()
        c.pan = config.SERVO_PAN_MAX - 2
        assert c.pan_by(50) == config.SERVO_PAN_MAX      # clamped
        c.tilt = config.SERVO_TILT_MIN + 2
        assert c.tilt_by(-50) == config.SERVO_TILT_MIN
        assert c.act.pan == config.SERVO_PAN_MAX

    def test_sethome_then_home_round_trips(self):
        c = rc()
        c.pan_by(15)
        c.tilt_by(10)
        hp, ht = c.set_home()
        c.pan_by(-30)                                    # wander off
        c.tilt_by(-20)
        p, t = c.go_home()
        assert (p, t) == (hp, ht)
        assert c.act.pan == hp and c.act.tilt == ht


class TestFrameGrabber:
    class _Cam:
        def __init__(self, fail_first=0):
            self.calls = 0
            self.fail_first = fail_first
            self.closed = 0

        def capture_color(self):
            self.calls += 1
            if self.calls <= self.fail_first:
                raise RuntimeError("transient camera hiccup")
            return f"frame{self.calls}"

        def close(self):
            self.closed += 1

    def _pump(self, g, n=40):
        # run the loop body n times synchronously instead of threading
        for _ in range(n):
            try:
                fr = g._camera.capture_color()
                if fr is not None:
                    g._latest = fr
                g._fails = 0
            except Exception:
                g._fails += 1
                if g._fails >= 3:
                    g._camera.close()
                    g._fails = 0

    def test_serves_latest_frame(self):
        from catchaser.rc import FrameGrabber
        cam = self._Cam()
        g = FrameGrabber(cam)
        self._pump(g, 3)
        assert g.latest() == "frame3"

    def test_survives_transient_failures_and_reopens(self):
        from catchaser.rc import FrameGrabber
        cam = self._Cam(fail_first=3)      # first 3 reads throw
        g = FrameGrabber(cam)
        self._pump(g, 4)                   # 3 fails -> reopen, then 1 good read
        assert cam.closed == 1             # reopened after 3 straight fails
        assert g.latest() == "frame4"      # recovered to a good frame


class TestApp:
    def _client(self):
        pytest.importorskip("flask")
        from catchaser.rc import make_app
        c = rc(speed=90)
        return make_app(c).test_client(), c

    def test_drive_endpoint(self):
        client, c = self._client()
        r = client.post("/api/drive", json={"forward": 1.0}).get_json()
        assert r["forward"] == 90.0

    def test_stop_endpoint(self):
        client, c = self._client()
        client.post("/api/stop")
        assert c.act.stops >= 1

    def test_servo_and_home_endpoints(self):
        client, c = self._client()
        client.post("/api/servo", json={"pan": 6, "tilt": -6})
        r = client.post("/api/sethome").get_json()
        assert r["home"] is True
        st = client.get("/api/state").get_json()
        assert st["distance"] == 512 and "speed" in st

    def test_speed_endpoint(self):
        client, c = self._client()
        r = client.post("/api/speed", json={"delta": 10}).get_json()
        assert r["speed"] == 100

    def test_stream_503_without_camera(self):
        client, c = self._client()          # made without a frame_source
        assert client.get("/stream.mjpg").status_code == 503
