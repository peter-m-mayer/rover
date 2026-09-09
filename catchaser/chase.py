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

import argparse
import sys
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
STATE_DART = "DART"          # prey mode: darting toward/around the cat
STATE_FREEZE = "FREEZE"      # prey mode: frozen still (pounce bait)
STATE_FLEE = "FLEE"          # prey mode: retreating from a close cat

_STATE_LED = {
    STATE_IDLE: "off",
    STATE_TRACKING: "green",
    STATE_LOST: "yellow",
    STATE_SEARCHING: "red",
    STATE_HOLD: "cyan",
    STATE_DART: "green",
    STATE_FREEZE: "white",
    STATE_FLEE: "purple",
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
    strafe: float = 0.0
    pan: Optional[float] = None   # camera pan angle to command, or None


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
                 search_spin_frames: int = config.CHASE_SEARCH_SPIN_FRAMES,
                 search_stare_frames: int = config.CHASE_SEARCH_STARE_FRAMES,
                 lost_grace_frames: int = config.CHASE_LOST_GRACE_FRAMES,
                 center_deadband: float = config.CHASE_CENTER_DEADBAND,
                 turn_only_error: float = config.CHASE_TURN_ONLY_ERROR,
                 min_confidence: float = config.CHASE_MIN_CONFIDENCE,
                 mode: str = "chase",
                 use_pan: bool = config.CHASE_PAN_ENABLED,
                 pan_gain: float = config.CHASE_PAN_GAIN,
                 pan_sign: int = config.CHASE_PAN_SIGN,
                 pan_turn_only_deg: float = config.CHASE_PAN_TURN_ONLY_DEG,
                 pan_body_engage: float = config.CHASE_PAN_BODY_ENGAGE_DEG,
                 pan_body_release: float = config.CHASE_PAN_BODY_RELEASE_DEG,
                 pan_body_rotate: float = config.CHASE_PAN_BODY_ROTATE,
                 pan_search_rotate: float = config.CHASE_PAN_SEARCH_ROTATE,
                 prey_dart_frames: int = config.CHASE_PREY_DART_FRAMES,
                 prey_freeze_frames: int = config.CHASE_PREY_FREEZE_FRAMES,
                 prey_dart_speed: float = config.CHASE_PREY_DART_SPEED,
                 prey_strafe: float = config.CHASE_PREY_STRAFE,
                 prey_flee_mm: int = config.CHASE_PREY_FLEE_MM,
                 prey_flee_speed: float = config.CHASE_PREY_FLEE_SPEED):
        self.actuators = actuators
        self.sensors = sensors
        self.image_width = image_width
        self.kp, self.ki, self.kd = kp, ki, kd
        self.forward_speed = forward_speed
        self.max_speed = max_speed
        self.turn_max = turn_max
        self.stop_mm = stop_mm
        self.search_speed = search_speed
        self.search_spin_frames = max(1, int(search_spin_frames))
        self.search_stare_frames = max(0, int(search_stare_frames))
        self.lost_grace_frames = lost_grace_frames
        self.center_deadband = center_deadband
        self.turn_only_error = turn_only_error
        self.min_confidence = min_confidence
        self.mode = mode
        self.use_pan = use_pan
        self.pan_gain = pan_gain
        self.pan_sign = pan_sign
        self.pan_turn_only_deg = pan_turn_only_deg
        self.pan_body_engage = pan_body_engage
        self.pan_body_release = pan_body_release
        self.pan_body_rotate = pan_body_rotate
        self.pan_search_rotate = pan_search_rotate
        self.prey_dart_frames = max(1, int(prey_dart_frames))
        self.prey_freeze_frames = max(0, int(prey_freeze_frames))
        self.prey_dart_speed = prey_dart_speed
        self.prey_strafe = prey_strafe
        self.prey_flee_mm = prey_flee_mm
        self.prey_flee_speed = prey_flee_speed
        self.reset()

    def reset(self):
        """Clear PID state, search memory, pan angle, prey cycle."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._have_prev = False      # no D-term until a second tracked frame
        self._frames_lost = 0
        self._last_seen_sign = 1.0   # default: search to the right first
        self._pan = float(config.SERVO_PAN_CENTER)
        self._prey_i = 0
        self._body_rotating = False   # coarse-align hysteresis latch (pan mode)

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

        # --- cat visible -------------------------------------------------
        self._frames_lost = 0
        err = (target.cx - self.image_width / 2.0) / (self.image_width / 2.0)
        err = _clamp(err, -1.0, 1.0)
        self._last_seen_sign = 1.0 if err >= 0 else -1.0
        # Deadband: treat a nearly-centered cat as centered (no jitter).
        err_eff = 0.0 if abs(err) < self.center_deadband else err

        # Keep the cat centered (camera pan and/or body turn) ...
        turn, pan = self._heading(err_eff, dt)
        # ... then decide how to move (approach, or dart/freeze/flee).
        if self.mode == "prey":
            return self._prey_policy(err, turn, pan, distance_mm)
        return self._chase_policy(err, turn, pan, distance_mm)

    # ---------------------------------------------------------------- heading
    def _heading(self, err_eff, dt):
        """Center the cat. Returns (body_turn, pan_angle_or_None).

        Pan mode: the camera servo tracks the centroid (fast, low-blur) and the
        body turn just follows the pan back toward center. Body mode: the
        heading PID drives the body turn directly (pan stays None).
        """
        if self.use_pan:
            # Fine tracking: the camera pans to keep the cat centered.
            self._pan = _clamp(
                self._pan - self.pan_sign * self.pan_gain * err_eff,
                config.SERVO_PAN_MIN, config.SERVO_PAN_MAX)
            dev = self._pan - config.SERVO_PAN_CENTER
            # Coarse hand-off: the body sits still (camera does the work) until
            # the pan swings out near the FOV edge, then slow-rotates to bring
            # the camera back toward center — with hysteresis so it doesn't
            # chatter around the threshold.
            if self._body_rotating:
                if abs(dev) <= self.pan_body_release:
                    self._body_rotating = False
            elif abs(dev) >= self.pan_body_engage:
                self._body_rotating = True
            if self._body_rotating:
                # dev>0 => pan past center = camera left => body turns left.
                turn = -self.pan_sign * (1.0 if dev > 0 else -1.0) * self.pan_body_rotate
                turn = _clamp(turn, -self.turn_max, self.turn_max)
            else:
                turn = 0.0
            return turn, self._pan

        # Body-only PID. The D-term needs two consecutive tracked frames — on a
        # fresh acquisition prev_error is meaningless and kicks toward overshoot.
        self._integral += err_eff * dt
        deriv = (err_eff - self._prev_error) / dt if (self._have_prev and dt > 0) else 0.0
        self._prev_error = err_eff
        self._have_prev = True
        turn = self.kp * err_eff + self.ki * self._integral + self.kd * deriv
        return _clamp(turn, -self.turn_max, self.turn_max), None

    def _off_axis_fraction(self, err):
        """How far off-center are we, in [0, inf): 1.0 = the turn-only limit."""
        if self.use_pan:
            return abs(self._pan - config.SERVO_PAN_CENTER) / self.pan_turn_only_deg
        return abs(err) / self.turn_only_error

    def _forward_toward(self, err, distance_mm):
        """Approach speed with off-axis + close-range tapers (0 if too far off)."""
        off = self._off_axis_fraction(err)
        if off >= 1.0:
            return 0.0
        forward = self.forward_speed * (1.0 - off)
        # Close-range taper: ease off as the ultrasonic closes on the stop band,
        # so the final approach is gentle instead of charging the 200 mm wall.
        if forward > 0 and distance_mm > 0:
            frac = (distance_mm - self.stop_mm) / float(config.CHASE_APPROACH_TAPER_MM)
            forward *= _clamp(frac, config.CHASE_APPROACH_MIN_FACTOR, 1.0)
        return forward

    # ----------------------------------------------------------- motion policy
    def _chase_policy(self, err, turn, pan, distance_mm):
        """Steady pursuit: approach, taper, ultrasonic HOLD at stop distance."""
        if 0 < distance_mm <= self.stop_mm:
            self._integral = 0.0
            return DriveCommand(0.0, turn, STATE_HOLD, err, distance_mm,
                                "ultrasonic stop", pan=pan)
        forward = self._forward_toward(err, distance_mm)
        forward, turn = self._cap(forward, turn)
        return DriveCommand(forward, turn, STATE_TRACKING, err, distance_mm, pan=pan)

    def _prey_policy(self, err, turn, pan, distance_mm):
        """Prey behavior: dart + zig-zag, FREEZE (pounce bait), FLEE when close."""
        self._prey_i += 1
        cycle = self.prey_dart_frames + self.prey_freeze_frames
        zig = 1.0 if ((self._prey_i - 1) // cycle) % 2 == 0 else -1.0

        # FLEE: the cat (or a wall) is close -> retreat, still facing it. A
        # fleeing "prey" is the single most engaging move for a cat.
        if 0 < distance_mm < self.prey_flee_mm:
            f, t, s = self._cap3(-self.prey_flee_speed, turn, self.prey_strafe * zig)
            return DriveCommand(f, t, STATE_FLEE, err, distance_mm, "flee",
                                strafe=s, pan=pan)

        phase = (self._prey_i - 1) % cycle
        if phase < self.prey_dart_frames:
            f, t, s = self._cap3(self.prey_dart_speed, turn, self.prey_strafe * zig)
            return DriveCommand(f, t, STATE_DART, err, distance_mm, "dart",
                                strafe=s, pan=pan)
        # FREEZE: fully still (the pounce bait). Camera keeps watching via pan.
        return DriveCommand(0.0, 0.0, STATE_FREEZE, err, distance_mm,
                            "freeze (pounce bait)", pan=pan)

    def _handle_lost(self) -> DriveCommand:
        """No cat this frame: grace-hold, then pulsed search (spin-and-stare).

        Continuous spinning outruns the detector: at ~6 Hz loop rate the
        robot sweeps a large share of the FOV between frames and motion blur
        smears the cat during capture. So the search alternates short spin
        bursts with stationary "stare" frames where detection gets a sharp,
        stable image. In pan mode the camera drifts back to center while lost.
        """
        self._frames_lost += 1
        self._integral = 0.0
        self._prev_error = 0.0
        self._have_prev = False
        self._body_rotating = False
        # Grace period: the cat probably just blurred/occluded for a frame or
        # two. HOLD everything — critically, keep the camera pointed where the
        # cat was (do NOT recenter the pan), so it reacquires the instant the
        # cat reappears instead of having looked away.
        if self._frames_lost <= self.lost_grace_frames:
            pan = self._pan if self.use_pan else None
            return DriveCommand(0.0, 0.0, STATE_LOST, note="grace hold", pan=pan)
        pan = None
        if self.use_pan:
            # Now actually searching: drift the camera back to center so it
            # looks where the body is rotating.
            self._pan += _clamp(config.SERVO_PAN_CENTER - self._pan, -8.0, 8.0)
            pan = self._pan
        if self.use_pan:
            # Slow, CONTINUOUS rotate toward the last-seen side to reacquire —
            # no stop-and-go; the camera + detector tolerate the mild blur.
            turn = self._last_seen_sign * self.pan_search_rotate
            return DriveCommand(0.0, turn, STATE_SEARCHING, note="slow search", pan=pan)
        # Body-only mode keeps the pulsed spin-and-stare (blur-safe without pan).
        cycle_pos = (self._frames_lost - self.lost_grace_frames - 1) % (
            self.search_spin_frames + self.search_stare_frames)
        if cycle_pos < self.search_spin_frames:
            turn = self._last_seen_sign * self.search_speed
            return DriveCommand(0.0, turn, STATE_SEARCHING, note="search spin", pan=pan)
        return DriveCommand(0.0, 0.0, STATE_SEARCHING, note="search stare", pan=pan)

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

    def _cap3(self, forward, turn, strafe):
        """Scale (forward, turn, strafe) so no wheel exceeds max_speed.

        Worst-case wheel magnitude in the mecanum mix is |f| + |t| + |s|.
        """
        peak = abs(forward) + abs(turn) + abs(strafe)
        if peak > self.max_speed and peak > 0:
            k = self.max_speed / peak
            forward *= k
            turn *= k
            strafe *= k
        return forward, turn, strafe

    # -------------------------------------------------------------- actuation
    def apply(self, cmd: DriveCommand):
        """Send a command to the wheels + camera, and reflect state on the LEDs."""
        self.actuators.drive(cmd.forward, cmd.turn, cmd.strafe)
        if cmd.pan is not None:
            try:
                self.actuators.set_servo_pan(cmd.pan)
            except Exception:
                pass
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
            try:
                self.sensors.enable_ir()   # kill switch needs the receiver on
            except AttributeError:
                pass  # sim/mock sensors may not have IR

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
                # Sleep only the *remainder* of the period. Perception already
                # costs ~50-70 ms on the Pi; sleeping a full period on top of
                # that throttled the loop to ~6 Hz (half the detector's rate)
                # and added tracking latency. Now the detector is the limiter.
                remaining = period - (now() - t)
                if remaining > 0:
                    sleep(remaining)
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


def main(argv=None) -> int:
    """Run the chase loop on the real robot.

        python3 -m catchaser.chase                      # full chase
        python3 -m catchaser.chase --max-runtime 30     # timed session
        python3 -m catchaser.chase --forward-speed 0    # steer-only (bench test)

    Stop with Ctrl+C or any IR-remote key. Motors always stop on exit.
    """
    parser = argparse.ArgumentParser(description="Cat Chaser 3000 — chase loop")
    parser.add_argument("--max-runtime", type=float, default=120.0,
                        help="stop after this many seconds (default 120; safety)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after this many frames")
    parser.add_argument("--forward-speed", type=float,
                        default=config.CHASE_FORWARD_SPEED,
                        help="base approach speed (0 = steer/spin only)")
    parser.add_argument("--max-speed", type=float, default=config.CHASE_MAX_SPEED,
                        help="per-wheel speed cap")
    parser.add_argument("--search-speed", type=float,
                        default=config.CHASE_SEARCH_SPIN_SPEED,
                        help="search-spin speed (0 = hold still when cat lost)")
    parser.add_argument("--stop-mm", type=int, default=config.CHASE_STOP_MM,
                        help="ultrasonic hold distance in mm")
    parser.add_argument("--min-confidence", type=float,
                        default=config.CHASE_MIN_CONFIDENCE)
    parser.add_argument("--pan", action="store_true",
                        help="camera pan-servo tracking (body follows the pan)")
    parser.add_argument("--prey", action="store_true",
                        help="prey/play mode: dart, freeze, and flee instead of "
                             "steady pursuit (more engaging for the cat)")
    parser.add_argument("--fast-shutter", action=argparse.BooleanOptionalAction,
                        default=config.CHASE_CAM_FAST_SHUTTER,
                        help="short camera exposure + gain to cut motion blur "
                             "(default ON; --no-fast-shutter for auto exposure)")
    parser.add_argument("--quiet", action="store_true",
                        help="only print state changes, not every frame")
    parser.add_argument("--save-dir", default=None,
                        help="harvest a training dataset here: clean frames + "
                             "YOLO weak labels + annotated previews. "
                             "Detections at most 1/s, negatives 1/10s, cap 300. "
                             "Review afterwards with: python3 -m catchaser.review <dir>")
    args = parser.parse_args(argv)

    from .detector import CatDetector
    from .hw import make_hardware

    bot, camera, sensors, actuators = make_hardware()
    if bot is None:
        print("[chase] vendor driver not found — refusing to run the chase "
              "loop in mock mode (nothing would move). Run on the robot.")
        return 2

    if args.fast_shutter:
        camera.enable_auto_brightness(
            target=config.CHASE_CAM_TARGET_BRIGHTNESS,
            exp_short=config.CHASE_CAM_FAST_EXPOSURE,
            exp_max=config.CHASE_CAM_EXP_MAX, gain_max=config.CHASE_CAM_GAIN_MAX)
        print(f"[chase] fast shutter + auto-brightness "
              f"(exposure>={config.CHASE_CAM_FAST_EXPOSURE}, "
              f"target~{config.CHASE_CAM_TARGET_BRIGHTNESS})")

    detector = CatDetector()
    controller = ChaseController(
        actuators, sensors,
        forward_speed=args.forward_speed, max_speed=args.max_speed,
        search_speed=args.search_speed, stop_mm=args.stop_mm,
        min_confidence=args.min_confidence,
        mode="prey" if args.prey else "chase", use_pan=args.pan)

    print(f"[chase] starting: mode={'prey' if args.prey else 'chase'} "
          f"pan={'on' if args.pan else 'off'} forward={args.forward_speed} "
          f"max={args.max_speed} stop={args.stop_mm}mm runtime={args.max_runtime}s")
    print("[chase] kill: Ctrl+C or any IR-remote key")

    frame_n = [0]
    last_state = [None]

    def on_step(cmd):
        frame_n[0] += 1
        changed = cmd.state != last_state[0]
        last_state[0] = cmd.state
        if changed or not args.quiet:
            pan = f" pan={cmd.pan:5.1f}" if cmd.pan is not None else ""
            strafe = f" str={cmd.strafe:+5.1f}" if cmd.strafe else ""
            print(f"  f{frame_n[0]:04d} {cmd.state:9s} "
                  f"err={cmd.heading_error:+.2f} fwd={cmd.forward:5.1f} "
                  f"turn={cmd.turn:+6.1f}{strafe}{pan} us={cmd.distance_mm:5d}mm "
                  f"det={detector.last_inference_ms:5.1f}ms"
                  f"{'  <-- ' + cmd.note if cmd.note else ''}")

    saver = {"last_det": 0.0, "last_neg": 0.0, "n": 0}
    run_tag = time.strftime("%Y%m%d_%H%M%S")

    def maybe_harvest(frame, dets):
        """Rate-limited dataset capture: detections 1/s, negatives 1/10s."""
        if not args.save_dir or saver["n"] >= 300:
            return
        t = time.monotonic()
        if dets:
            if t - saver["last_det"] < 1.0:
                return
            saver["last_det"] = t
        else:
            # Occasional empty frames = clean negatives / background images.
            if t - saver["last_neg"] < 10.0:
                return
            saver["last_neg"] = t
        saver["n"] += 1
        from .dataset import save_sample
        save_sample(args.save_dir, f"{run_tag}_{saver['n']:04d}", frame, dets)

    def perceive():
        frame = camera.capture_color()
        camera.auto_brightness(frame)      # adaptive exposure/gain (no-op if off)
        dets = detector.detect(frame)
        maybe_harvest(frame, dets)
        return dets

    try:
        last = controller.run(
            perceive, sensors.read_ultrasonic_mm,
            on_step=on_step,
            max_frames=args.max_frames,
            max_runtime_s=args.max_runtime)
        print(f"[chase] done after {frame_n[0]} frames "
              f"(exit: {last.note or 'runtime/frame limit'})")
    except KeyboardInterrupt:
        print("\n[chase] Ctrl+C — motors stopped.")
    finally:
        try:
            camera.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
