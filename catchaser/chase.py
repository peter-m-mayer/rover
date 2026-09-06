"""Cat Chaser 3000 — pursuit controller.

Closes the loop from perception to motion:

    detector centroid  ->  normalized heading error  ->  heading PID
                       ->  mecanum drive (forward + turn)

Behaviors:
  * TRACKING  — a cat is visible: steer to center it, drive forward, speed
                scaled down as the cat drifts off-center (turn-in-place when
                far off).
  * LOST      — cat just disappeared: hold still for a few grace frames in
                case it flickers back (avoids twitchy spin on a dropped frame).
  * SEARCHING — cat stayed lost: spin in place toward the side it was last
                seen until it reappears.
  * HOLD      — ultrasonic says we're within CHASE_STOP_MM: stop advancing
                (keep gently steering to stay pointed at the cat).

Safety:
  * Ultrasonic emergency stop at CHASE_STOP_MM (200 mm) — never rams the cat.
  * Speed cap: no single wheel command exceeds CHASE_MAX_SPEED.
  * Kill switch: Ctrl+C, an optional kill_check callback, and (on hardware)
    any IR-remote key press all stop the motors immediately.
  * Motors are stopped in a finally block no matter how the loop exits.

The control math lives in ChaseController.compute(), which is pure (given
its internal PID state) and takes plain inputs — so it is unit-testable
without any hardware or simulator. run() wires it to real or simulated I/O.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from raspbot_slam import config

from .detector import Detection


# Control-loop states (also drive LED status colors).
STATE_IDLE = "IDLE"
STATE_TRACKING = "TRACKING"
STATE_LOST = "LOST"
STATE_SEARCHING = "SEARCHING"
STATE_HOLD = "HOLD"

_STATE_LED = {
    STATE_IDLE: "off",
    STATE_TRACKING: "green",
    STATE_LOST: "yellow",
    STATE_SEARCHING: "red",
    STATE_HOLD: "cyan",
}


@dataclass
class DriveCommand:
    """One control decision: what to send to the wheels, plus telemetry."""
    forward: float
    turn: float
    state: str
    heading_error: float = 0.0
    distance_mm: int = -1
    note: str = ""


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ChaseController:
    """Turns detections + range into mecanum drive commands (with PID)."""

    def __init__(self, actuators, sensors=None, *,
                 image_width: int = config.CAMERA_WIDTH,
                 kp: float = config.CHASE_HEADING_KP,
                 ki: float = config.CHASE_HEADING_KI,
                 kd: float = config.CHASE_HEADING_KD,
                 forward_speed: float = config.CHASE_FORWARD_SPEED,
                 max_speed: float = config.CHASE_MAX_SPEED,
                 turn_max: float = config.CHASE_TURN_MAX,
                 stop_mm: int = config.CHASE_STOP_MM,
                 search_speed: float = config.CHASE_SEARCH_SPIN_SPEED,
                 lost_grace_frames: int = config.CHASE_LOST_GRACE_FRAMES,
                 center_deadband: float = config.CHASE_CENTER_DEADBAND,
                 turn_only_error: float = config.CHASE_TURN_ONLY_ERROR,
                 min_confidence: float = config.CHASE_MIN_CONFIDENCE):
        self.actuators = actuators
        self.sensors = sensors
        self.image_width = image_width
        self.kp, self.ki, self.kd = kp, ki, kd
        self.forward_speed = forward_speed
        self.max_speed = max_speed
        self.turn_max = turn_max
        self.stop_mm = stop_mm
        self.search_speed = search_speed
        self.lost_grace_frames = lost_grace_frames
        self.center_deadband = center_deadband
        self.turn_only_error = turn_only_error
        self.min_confidence = min_confidence
        self.reset()

    def reset(self):
        """Clear PID state and search memory."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._frames_lost = 0
        self._last_seen_sign = 1.0   # default: search to the right first

    # ------------------------------------------------------------------ core
    def compute(self, detections: List[Detection], distance_mm: int,
                dt: float = 0.1) -> DriveCommand:
        """Decide the next drive command.

        Args:
            detections: candidate detections for this frame (any classes;
                filtered here by confidence, then best_target picks one).
            distance_mm: forward ultrasonic reading in mm; <=0 means unknown.
            dt: seconds since the previous compute() (for the PID D/I terms).

        Returns:
            DriveCommand with forward/turn in motor units and a state label.
        """
        candidates = [d for d in detections if d.confidence >= self.min_confidence]
        # Chase the most confident cat. (Don't rely on input ordering — only
        # detector.detect() pre-sorts; other perception sources may not.)
        target = max(candidates, key=lambda d: d.confidence) if candidates else None

        if target is None:
            return self._handle_lost()

        # --- cat visible: steer to center it -----------------------------
        self._frames_lost = 0
        err = (target.cx - self.image_width / 2.0) / (self.image_width / 2.0)
        err = _clamp(err, -1.0, 1.0)
        self._last_seen_sign = 1.0 if err >= 0 else -1.0

        # Deadband: treat a nearly-centered cat as centered (no jitter).
        err_eff = 0.0 if abs(err) < self.center_deadband else err

        # Heading PID -> turn command.
        self._integral += err_eff * dt
        deriv = (err_eff - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = err_eff
        turn = self.kp * err_eff + self.ki * self._integral + self.kd * deriv
        turn = _clamp(turn, -self.turn_max, self.turn_max)

        # Ultrasonic emergency stop: close enough — hold, keep steering only.
        if 0 < distance_mm <= self.stop_mm:
            self._integral = 0.0
            return DriveCommand(0.0, turn, STATE_HOLD, err, distance_mm,
                                "ultrasonic stop")

        # Forward speed: full when centered, tapering to 0 as the cat drifts;
        # pure turn-in-place once it's past turn_only_error.
        if abs(err) >= self.turn_only_error:
            forward = 0.0
        else:
            forward = self.forward_speed * (1.0 - abs(err) / self.turn_only_error)

        forward, turn = self._cap(forward, turn)
        return DriveCommand(forward, turn, STATE_TRACKING, err, distance_mm)

    def _handle_lost(self) -> DriveCommand:
        """No cat this frame: grace-hold briefly, then search-spin."""
        self._frames_lost += 1
        self._integral = 0.0
        self._prev_error = 0.0
        if self._frames_lost <= self.lost_grace_frames:
            return DriveCommand(0.0, 0.0, STATE_LOST, note="grace hold")
        turn = self._last_seen_sign * self.search_speed
        return DriveCommand(0.0, turn, STATE_SEARCHING, note="search spin")

    def _cap(self, forward, turn):
        """Scale (forward, turn) so no wheel exceeds max_speed.

        With strafe=0 the worst-case wheel magnitude is |forward| + |turn|.
        """
        peak = abs(forward) + abs(turn)
        if peak > self.max_speed and peak > 0:
            k = self.max_speed / peak
            forward *= k
            turn *= k
        return forward, turn

    # -------------------------------------------------------------- actuation
    def apply(self, cmd: DriveCommand):
        """Send a command to the wheels and reflect state on the LEDs."""
        self.actuators.drive(cmd.forward, cmd.turn)
        try:
            self.actuators.set_led_color(_STATE_LED.get(cmd.state, "off"))
        except Exception:
            pass  # LEDs are non-essential; never let them break the loop

    # -------------------------------------------------------------- main loop
    def run(self, perceive: Callable[[], List[Detection]],
            read_distance: Callable[[], int], *,
            kill_check: Optional[Callable[[], bool]] = None,
            on_step: Optional[Callable[[DriveCommand], None]] = None,
            max_frames: Optional[int] = None,
            max_runtime_s: Optional[float] = None,
            period_s: Optional[float] = None,
            sleep: Callable[[float], None] = time.sleep,
            now: Callable[[], float] = time.monotonic) -> DriveCommand:
        """Run the chase loop until a kill condition or limit is hit.

        Args:
            perceive: returns this frame's detections (e.g. detector.detect(frame)).
            read_distance: returns forward ultrasonic distance in mm.
            kill_check: optional; return True to stop (software kill switch).
            on_step: optional callback given each DriveCommand (telemetry/logging).
            max_frames: stop after this many iterations (safety / testing).
            max_runtime_s: stop after this many seconds (safety).
            period_s: target loop period; defaults to 1/CHASE_LOOP_HZ.
            sleep / now: injectable clock (tests use a fake, sim uses stepping).

        Returns:
            The last DriveCommand issued.
        """
        period = period_s if period_s is not None else 1.0 / config.CHASE_LOOP_HZ
        if self.sensors is not None:
            try:
                self.sensors.enable_ultrasonic()
            except Exception:
                pass

        self.reset()
        last = DriveCommand(0.0, 0.0, STATE_IDLE)
        frame = 0
        t_start = now()
        t_prev = t_start
        try:
            while True:
                if kill_check is not None and kill_check():
                    last = DriveCommand(0.0, 0.0, STATE_IDLE, note="kill switch")
                    break
                if self._ir_kill():
                    last = DriveCommand(0.0, 0.0, STATE_IDLE, note="IR remote kill")
                    break
                if max_frames is not None and frame >= max_frames:
                    break
                t = now()
                if max_runtime_s is not None and (t - t_start) >= max_runtime_s:
                    break

                dt = t - t_prev
                if dt <= 0:
                    dt = period
                t_prev = t

                detections = perceive()
                distance_mm = read_distance()
                cmd = self.compute(detections, distance_mm, dt=dt)
                self.apply(cmd)
                last = cmd
                if on_step is not None:
                    on_step(cmd)

                frame += 1
                sleep(period)
        finally:
            self._safe_stop()
        return last

    def _ir_kill(self) -> bool:
        """Hardware kill switch: any IR-remote key press stops the chase."""
        if self.sensors is None:
            return False
        try:
            return self.sensors.read_ir_remote() is not None
        except Exception:
            return False

    def _safe_stop(self):
        try:
            self.actuators.stop()
        except Exception:
            pass
        try:
            self.actuators.set_led_color("off")
        except Exception:
            pass


def run_hardware_chase(camera, detector, sensors, actuators, **kwargs) -> DriveCommand:
    """Convenience wiring for the real robot (or any Camera/Sensors/Actuators).

    Builds the perceive/read_distance callables from the hardware interfaces
    and runs the chase loop. Extra kwargs pass through to ChaseController.run.
    """
    controller = ChaseController(actuators, sensors)

    def perceive() -> List[Detection]:
        return detector.detect(camera.capture_color())

    try:
        return controller.run(perceive, sensors.read_ultrasonic_mm, **kwargs)
    finally:
        try:
            camera.close()
        except Exception:
            pass
