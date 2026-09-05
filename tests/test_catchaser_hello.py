"""Tests for the catchaser hello-world smoke test (mock mode, no hardware)."""

import pytest

from catchaser import hw
from catchaser.hello import main


class TestConnectBot:
    def test_returns_none_without_vendor_lib(self):
        # Vendor driver is not installed on dev machines.
        assert hw.connect_bot() is None

    def test_make_hardware_mock(self):
        bot, camera, sensors, actuators = hw.make_hardware(mock=True)
        assert bot is None
        # Mock-mode interfaces must be constructible and callable.
        assert sensors.read_ultrasonic_mm() == 1000  # documented mock value
        actuators.move_forward(60)
        actuators.stop()


class TestHelloMain:
    def test_mock_full_run_passes(self, capsys):
        rc = main(["--mock", "--drive-s", "0.01", "--spin-s", "0.01"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "MOCK mode" in out
        assert "ALL GOOD" in out

    def test_no_motion_flag(self, capsys):
        rc = main(["--mock", "--no-motion"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "SKIP" in out

    def test_obstacle_gates_forward_drive(self, capsys):
        # Mock ultrasonic reports 1000 mm; demand more clearance than that.
        rc = main(["--mock", "--min-clear-mm", "2000",
                   "--drive-s", "0.01", "--spin-s", "0.01"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "forward skipped" in out
        assert "spin" in out  # spin still runs
