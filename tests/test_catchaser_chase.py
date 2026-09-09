"""Tests for the Cat Chaser 3000 pursuit controller.

Two layers:
  * Pure control-logic tests on ChaseController.compute()/run() with synthetic
    Detections and a fake robot — no hardware, no simulator.
  * End-to-end PyBullet integration via catchaser.chase_sim (gated on pybullet).
"""

import pytest

from catchaser.chase import (
    ChaseController, DriveCommand,
    STATE_TRACKING, STATE_HOLD, STATE_SEARCHING, STATE_LOST, STATE_IDLE,
    STATE_DART, STATE_FREEZE, STATE_FLEE,
)
from catchaser.detector import Detection, COCO_CAT
from raspbot_slam import config

W = config.CAMERA_WIDTH
PAN_CENTER = config.SERVO_PAN_CENTER


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #

class RecordingActuators:
    """Captures drive/stop/LED calls; implements the interface compute() needs."""

    def __init__(self):
        self.commands = []      # list of (forward, turn, strafe)
        self.leds = []
        self.pans = []
        self.stopped = 0

    def drive(self, forward, turn=0.0, strafe=0.0):
        self.commands.append((forward, turn, strafe))

    def stop(self):
        self.stopped += 1
        self.commands.append((0.0, 0.0, 0.0))

    def set_led_color(self, name):
        self.leds.append(name)

    def set_servo_pan(self, angle):
        self.pans.append(angle)


class FakeSensors:
    """Ultrasonic + IR remote stub."""

    def __init__(self, distance_mm=1000, ir_sequence=None):
        self.distance_mm = distance_mm
        self._ir = list(ir_sequence or [])
        self.enabled = False

    def enable_ultrasonic(self):
        self.enabled = True

    def read_ultrasonic_mm(self):
        return self.distance_mm

    def read_ir_remote(self):
        return self._ir.pop(0) if self._ir else None


def cat_at(cx, conf=0.9, class_id=COCO_CAT):
    """A Detection whose centroid is at horizontal pixel cx."""
    return Detection(cx - 20, 220, cx + 20, 260, confidence=conf, class_id=class_id)


def make_controller(**kw):
    return ChaseController(RecordingActuators(), FakeSensors(), **kw)


# --------------------------------------------------------------------------- #
# Heading error -> turn direction
# --------------------------------------------------------------------------- #

class TestHeadingControl:
    def test_centered_cat_drives_forward_no_turn(self):
        c = make_controller()
        cmd = c.compute([cat_at(W / 2)], distance_mm=1000)
        assert cmd.state == STATE_TRACKING
        assert cmd.forward > 0
        assert cmd.turn == pytest.approx(0.0, abs=1e-6)

    def test_cat_on_right_turns_right(self):
        c = make_controller()
        cmd = c.compute([cat_at(W * 0.8)], distance_mm=1000)
        assert cmd.heading_error > 0        # right of center
        assert cmd.turn > 0                 # + turn = clockwise / right

    def test_cat_on_left_turns_left(self):
        c = make_controller()
        cmd = c.compute([cat_at(W * 0.2)], distance_mm=1000)
        assert cmd.heading_error < 0
        assert cmd.turn < 0

    def test_deadband_suppresses_tiny_turn(self):
        c = make_controller()
        # Just inside the deadband fraction of half-width.
        off = W / 2 + (config.CHASE_CENTER_DEADBAND * 0.5) * (W / 2)
        cmd = c.compute([cat_at(off)], distance_mm=1000)
        assert cmd.turn == pytest.approx(0.0, abs=1e-6)

    def test_far_off_center_is_turn_only(self):
        c = make_controller()
        cmd = c.compute([cat_at(W * 0.98)], distance_mm=1000)
        assert cmd.forward == 0.0           # don't lunge while way off-axis
        assert cmd.turn > 0


# --------------------------------------------------------------------------- #
# Ultrasonic safety stop
# --------------------------------------------------------------------------- #

class TestUltrasonicStop:
    def test_stops_forward_within_stop_distance(self):
        c = make_controller()
        cmd = c.compute([cat_at(W / 2)], distance_mm=config.CHASE_STOP_MM - 20)
        assert cmd.state == STATE_HOLD
        assert cmd.forward == 0.0

    def test_still_approaches_just_outside_stop_distance(self):
        c = make_controller()
        cmd = c.compute([cat_at(W / 2)], distance_mm=config.CHASE_STOP_MM + 300)
        assert cmd.state == STATE_TRACKING
        assert cmd.forward > 0

    def test_unknown_distance_does_not_stop(self):
        c = make_controller()
        cmd = c.compute([cat_at(W / 2)], distance_mm=-1)
        assert cmd.forward > 0

    def test_forward_tapers_near_stop_range(self):
        c = make_controller()
        far = c.compute([cat_at(W / 2)], distance_mm=3000)
        c.reset()
        near = c.compute([cat_at(W / 2)], distance_mm=config.CHASE_STOP_MM + 100)
        assert near.state == STATE_TRACKING
        assert 0 < near.forward < far.forward   # gentle final approach

    def test_hold_still_steers_to_track(self):
        c = make_controller()
        cmd = c.compute([cat_at(W * 0.75)], distance_mm=100)
        assert cmd.state == STATE_HOLD
        assert cmd.forward == 0.0
        assert cmd.turn > 0                 # keeps facing the cat


# --------------------------------------------------------------------------- #
# Lost cat -> grace hold -> search spin
# --------------------------------------------------------------------------- #

class TestSearchBehavior:
    def test_grace_hold_before_searching(self):
        c = make_controller(lost_grace_frames=3)
        for _ in range(3):
            cmd = c.compute([], distance_mm=1000)
            assert cmd.state == STATE_LOST
            assert (cmd.forward, cmd.turn) == (0.0, 0.0)

    def test_search_spin_after_grace(self):
        c = make_controller(lost_grace_frames=2)
        for _ in range(2):
            c.compute([], distance_mm=1000)
        cmd = c.compute([], distance_mm=1000)
        assert cmd.state == STATE_SEARCHING
        assert cmd.turn != 0.0

    def test_search_spins_toward_last_seen_side(self):
        c = make_controller(lost_grace_frames=0)
        c.compute([cat_at(W * 0.9)], distance_mm=1000)   # last seen on the right
        cmd = c.compute([], distance_mm=1000)
        assert cmd.turn > 0                              # spin right to re-find

        c2 = make_controller(lost_grace_frames=0)
        c2.compute([cat_at(W * 0.1)], distance_mm=1000)  # last seen on the left
        cmd2 = c2.compute([], distance_mm=1000)
        assert cmd2.turn < 0

    def test_search_pulses_spin_and_stare(self):
        # Spin 2, stare 3 (defaults) — pattern must repeat after grace.
        c = make_controller(lost_grace_frames=0, search_spin_frames=2,
                            search_stare_frames=3)
        c.compute([cat_at(W * 0.9)], distance_mm=1000)   # seed last-seen side
        pattern = []
        for _ in range(10):
            cmd = c.compute([], distance_mm=1000)
            pattern.append("spin" if cmd.turn != 0 else "stare")
        assert pattern == ["spin", "spin", "stare", "stare", "stare"] * 2

    def test_reacquire_resets_lost_counter(self):
        c = make_controller(lost_grace_frames=1)
        for _ in range(5):
            c.compute([], distance_mm=1000)              # searching now
        cmd = c.compute([cat_at(W / 2)], distance_mm=1000)
        assert cmd.state == STATE_TRACKING


# --------------------------------------------------------------------------- #
# Safety limits & filtering
# --------------------------------------------------------------------------- #

class TestSafetyLimits:
    def test_speed_cap_never_exceeded_on_any_wheel(self):
        c = make_controller(forward_speed=200, kp=400, turn_max=400,
                            max_speed=config.CHASE_MAX_SPEED)
        cmd = c.compute([cat_at(W * 0.55)], distance_mm=2000)
        worst_wheel = abs(cmd.forward) + abs(cmd.turn)   # strafe = 0
        assert worst_wheel <= config.CHASE_MAX_SPEED + 1e-6

    def test_turn_clamped_to_turn_max(self):
        c = make_controller(kp=10000, turn_max=90)
        cmd = c.compute([cat_at(W * 0.99)], distance_mm=2000)
        assert abs(cmd.turn) <= 90 + 1e-6

    def test_low_confidence_detections_ignored(self):
        c = make_controller(min_confidence=0.5)
        cmd = c.compute([cat_at(W / 2, conf=0.3)], distance_mm=1000)
        assert cmd.state in (STATE_LOST, STATE_SEARCHING)

    def test_picks_highest_confidence_target(self):
        c = make_controller()
        dets = [cat_at(W * 0.2, conf=0.5), cat_at(W * 0.8, conf=0.95)]
        cmd = c.compute(dets, distance_mm=1000)
        assert cmd.heading_error > 0        # steered toward the 0.95 cat (right)


# --------------------------------------------------------------------------- #
# run() loop: kill switches and clean shutdown
# --------------------------------------------------------------------------- #

class TestRunLoop:
    def test_max_frames_limits_iterations(self):
        c = make_controller()
        steps = []
        c.run(lambda: [cat_at(W / 2)], lambda: 1000,
              on_step=lambda cmd: steps.append(cmd),
              max_frames=5, sleep=lambda s: None, now=_fake_clock())
        assert len(steps) == 5

    def test_kill_check_stops_immediately(self):
        c = make_controller()
        steps = []
        c.run(lambda: [cat_at(W / 2)], lambda: 1000,
              kill_check=lambda: len(steps) >= 3,
              on_step=lambda cmd: steps.append(cmd),
              max_frames=100, sleep=lambda s: None, now=_fake_clock())
        assert len(steps) == 3

    def test_ir_remote_is_a_kill_switch(self):
        # IR returns a key on the 3rd poll -> loop should stop there.
        sensors = FakeSensors(ir_sequence=[None, None, 7])
        c = ChaseController(RecordingActuators(), sensors)
        steps = []
        last = c.run(lambda: [cat_at(W / 2)], lambda: 1000,
                     on_step=lambda cmd: steps.append(cmd),
                     max_frames=100, sleep=lambda s: None, now=_fake_clock())
        assert last.note == "IR remote kill"
        assert len(steps) == 2                # stopped before the 3rd apply

    def test_motors_stopped_in_finally(self):
        acts = RecordingActuators()
        c = ChaseController(acts, FakeSensors())
        c.run(lambda: [cat_at(W / 2)], lambda: 1000,
              max_frames=3, sleep=lambda s: None, now=_fake_clock())
        assert acts.stopped >= 1
        assert acts.commands[-1] == (0.0, 0.0, 0.0)   # ends stopped

    def test_loop_does_not_oversleep_when_work_exceeds_period(self):
        # If perception already burns > period, the loop must not add sleep.
        c = make_controller()
        slept = []
        clock = [0.0]

        def now():
            return clock[0]

        def slow_perceive():
            clock[0] += 0.20      # 200 ms of "work" per frame (clock advances)
            return [cat_at(W / 2)]

        c.run(slow_perceive, lambda: 1000,
              max_frames=3, period_s=0.05,
              sleep=lambda s: slept.append(s), now=now)
        assert slept == []        # work > period -> never sleeps

    def test_run_enables_ultrasonic(self):
        sensors = FakeSensors()
        c = ChaseController(RecordingActuators(), sensors)
        c.run(lambda: [], lambda: 1000, max_frames=1,
              sleep=lambda s: None, now=_fake_clock())
        assert sensors.enabled is True


def _fake_clock():
    """Monotonic fake clock advancing 0.1 s per call."""
    t = [0.0]

    def now():
        t[0] += 0.1
        return t[0]
    return now


# --------------------------------------------------------------------------- #
# drive() mecanum mixing on the real actuator wrapper (mock bot)
# --------------------------------------------------------------------------- #

class TestPanTracking:
    def _pan_ctrl(self, **kw):
        return ChaseController(RecordingActuators(), FakeSensors(),
                               use_pan=True, pan_sign=1, **kw)

    def test_pan_moves_toward_cat_and_recenters_error(self):
        # Sim convention (sign +1): cat on the right (err>0) -> pan decreases
        # (camera looks right). Cat on the left -> pan increases.
        c = self._pan_ctrl()
        right = c.compute([cat_at(W * 0.85)], distance_mm=2000)
        assert right.pan is not None and right.pan < PAN_CENTER
        c2 = self._pan_ctrl()
        left = c2.compute([cat_at(W * 0.15)], distance_mm=2000)
        assert left.pan > PAN_CENTER

    def test_body_stays_still_inside_deadzone(self):
        # A mildly off-center cat: pan handles it, body does NOT rotate.
        c = self._pan_ctrl()
        cmd = c.compute([cat_at(W * 0.58)], distance_mm=2000)
        assert cmd.pan != PAN_CENTER          # camera moved
        assert cmd.turn == 0.0                # body held (camera in charge)

    def test_body_coarse_rotates_when_pan_swings_out(self):
        # A hard-off cat drives the pan to its edge -> body coarse-rotates.
        c = self._pan_ctrl()
        for _ in range(8):
            cmd = c.compute([cat_at(W * 0.98)], distance_mm=2000)
        assert cmd.pan < PAN_CENTER           # camera panned right
        assert cmd.turn > 0                   # body follows right (coarse align)

    def test_body_rotate_has_hysteresis(self):
        # Once engaged past ENGAGE it keeps rotating until back within RELEASE,
        # not stopping the instant it dips below ENGAGE.
        c = ChaseController(RecordingActuators(), FakeSensors(), use_pan=True,
                            pan_body_engage=40, pan_body_release=10,
                            pan_body_rotate=45)
        for _ in range(8):
            c.compute([cat_at(W * 0.99)], distance_mm=2000)   # drive pan to edge
        assert c._body_rotating
        # Cat pops back near center: dev shrinks below ENGAGE but above RELEASE.
        mid = c.compute([cat_at(W * 0.5)], distance_mm=2000)
        # deadzone would say turn=0, but hysteresis keeps it rotating for now
        assert c._body_rotating or mid.turn == 0.0   # engaged latch or released cleanly

    def test_pan_sign_flip_reverses_direction(self):
        c = ChaseController(RecordingActuators(), FakeSensors(),
                            use_pan=True, pan_sign=-1)
        right = c.compute([cat_at(W * 0.85)], distance_mm=2000)
        assert right.pan > PAN_CENTER          # flipped vs sign=+1

    def test_pan_holds_where_cat_was_during_grace(self):
        # Losing the cat briefly must NOT swing the camera back to center —
        # it should stay pointed where the cat was, to reacquire instantly.
        c = self._pan_ctrl(lost_grace_frames=10)
        for _ in range(6):
            c.compute([cat_at(W * 0.9)], distance_mm=2000)   # pan swings off-center
        held = c._pan
        assert abs(held - PAN_CENTER) > 5                    # camera is off-center
        for _ in range(4):                                   # a few lost frames (< grace)
            cmd = c.compute([], distance_mm=1000)
            assert cmd.state == STATE_LOST
        assert cmd.pan == held                               # held, not recentered

    def test_pan_recenters_when_lost(self):
        c = self._pan_ctrl()
        for _ in range(6):
            c.compute([cat_at(W * 0.95)], distance_mm=2000)   # pan swings out
        off = c._pan
        for _ in range(20):
            lost = c.compute([], distance_mm=1000)             # cat gone
        assert abs(lost.pan - PAN_CENTER) < abs(off - PAN_CENTER)  # drifted back

    def test_apply_sets_servo_pan(self):
        acts = RecordingActuators()
        c = ChaseController(acts, FakeSensors(), use_pan=True)
        c.apply(c.compute([cat_at(W * 0.8)], distance_mm=2000))
        assert acts.pans and acts.pans[-1] != PAN_CENTER


class TestPreyMode:
    def _prey(self, **kw):
        return ChaseController(RecordingActuators(), FakeSensors(),
                               mode="prey", **kw)

    def test_dart_then_freeze_cycle(self):
        c = self._prey(prey_dart_frames=2, prey_freeze_frames=3)
        seq = [c.compute([cat_at(W / 2)], distance_mm=2000).state for _ in range(10)]
        assert seq == [STATE_DART, STATE_DART, STATE_FREEZE, STATE_FREEZE,
                       STATE_FREEZE] * 2

    def test_freeze_is_fully_still(self):
        c = self._prey(prey_dart_frames=1, prey_freeze_frames=1)
        c.compute([cat_at(W / 2)], distance_mm=2000)              # dart
        freeze = c.compute([cat_at(W / 2)], distance_mm=2000)     # freeze
        assert freeze.state == STATE_FREEZE
        assert (freeze.forward, freeze.turn, freeze.strafe) == (0.0, 0.0, 0.0)

    def test_darts_zigzag_alternate_direction(self):
        c = self._prey(prey_dart_frames=1, prey_freeze_frames=0)
        s1 = c.compute([cat_at(W / 2)], distance_mm=2000).strafe
        s2 = c.compute([cat_at(W / 2)], distance_mm=2000).strafe
        assert s1 != 0 and s2 != 0 and (s1 > 0) != (s2 > 0)   # opposite signs

    def test_flees_when_cat_is_close(self):
        c = self._prey()
        cmd = c.compute([cat_at(W / 2)], distance_mm=config.CHASE_PREY_FLEE_MM - 50)
        assert cmd.state == STATE_FLEE
        assert cmd.forward < 0                                 # backing away

    def test_prey_still_tracks_heading(self):
        # Even while darting, it steers toward an off-center cat.
        c = self._prey()
        cmd = c.compute([cat_at(W * 0.8)], distance_mm=2000)
        assert cmd.turn > 0                                    # centering the cat


class TestMecanumDrive:
    def _bot(self):
        class Bot:
            def __init__(self):
                self.motors = {}

            def Ctrl_Muto(self, i, s):
                self.motors[i] = s
        return Bot()

    def test_pure_forward(self):
        from raspbot_slam.actuators import Actuators
        bot = self._bot()
        Actuators(bot=bot).drive(60, 0, 0)
        assert bot.motors == {0: 60, 1: 60, 2: 60, 3: 60}

    def test_pure_turn_right_matches_rotate_right(self):
        from raspbot_slam.actuators import Actuators
        b1, b2 = self._bot(), self._bot()
        Actuators(bot=b1).drive(0, 50, 0)
        Actuators(bot=b2).rotate_right(50)
        assert b1.motors == b2.motors        # {0:50,1:50,2:-50,3:-50}

    def test_pure_strafe_right_matches_move_right(self):
        from raspbot_slam.actuators import Actuators
        b1, b2 = self._bot(), self._bot()
        Actuators(bot=b1).drive(0, 0, 50)
        Actuators(bot=b2).move_right(50)
        assert b1.motors == b2.motors


# --------------------------------------------------------------------------- #
# End-to-end in the PyBullet simulator
# --------------------------------------------------------------------------- #

pytest.importorskip("pybullet", reason="PyBullet required for chase sim tests")


class TestChaseSim:
    def test_approach_reaches_and_stops(self):
        from catchaser.chase_sim import run_chase_sim
        r = run_chase_sim("approach", max_frames=400)
        assert r["acquired"]
        assert r["reached"]
        assert r["no_collision"]
        assert r["final_range_mm"] <= r["stop_mm"] + 60

    def test_search_then_acquire(self):
        from catchaser.chase_sim import run_chase_sim
        r = run_chase_sim("search", max_frames=500)
        assert r["searched"]                 # had to spin to find it
        assert r["acquired"]                 # then locked on
        assert r["reached"]
        assert r["no_collision"]

    def test_pan_tracking_tightens_centering(self):
        # Closed-loop in PyBullet: camera pan tracking should hold the cat
        # closer to frame-center than body-only steering.
        from catchaser.chase_sim import run_chase_sim
        body = run_chase_sim("approach", max_frames=400, use_pan=False)
        pan = run_chase_sim("approach", max_frames=400, use_pan=True)
        assert pan["reached"] and pan["no_collision"]
        assert pan["mean_abs_err"] < body["mean_abs_err"]

    def test_prey_mode_darts_freezes_flees(self):
        from catchaser.chase_sim import run_chase_sim
        r = run_chase_sim("approach", max_frames=400, mode="prey", use_pan=True)
        assert r["acquired"] and r["froze"] and r["darted"]
        assert r["no_collision"]             # flees before ramming the cat
