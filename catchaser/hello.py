"""Cat Chaser 3000 — "hello world" hardware smoke test.

Exercises every subsystem the cat chaser will depend on, in order of
increasing risk, and prints a PASS/FAIL summary:

  1. I2C driver connect        (vendor Raspbot_Lib)
  2. LED + buzzer              (visible/audible proof of I2C writes)
  3. Camera capture            (saves hello_frame.jpg)
  4. Ultrasonic read           (also gates the motion test)
  5. Motion: forward ~1 s, spin ~1 s, stop

Run on the Pi:
    python -m catchaser.hello

Safe-by-default behavior:
  - If the ultrasonic sensor reports an obstacle closer than --min-clear-mm,
    the forward drive is skipped (the spin still runs — it needs no clearance).
  - Motors are always stopped in a finally block, even on Ctrl+C or crash.
  - --no-motion runs everything except the drive/spin.
  - On a dev machine (no vendor driver) it runs in mock mode automatically.

Usage:
    python -m catchaser.hello [--mock] [--no-motion] [--speed 60]
                              [--drive-s 1.0] [--spin-s 1.0]
                              [--out hello_frame.jpg] [--min-clear-mm 400]
"""

import argparse
import sys
import time

from .hw import make_hardware


def _step(results: dict, name: str, fn):
    """Run one smoke-test step, record PASS/FAIL, never raise."""
    try:
        detail = fn()
        results[name] = ("PASS", detail if detail else "")
    except Exception as exc:
        results[name] = ("FAIL", str(exc))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Raspbot hello-world smoke test")
    parser.add_argument("--mock", action="store_true", help="force mock mode (no hardware)")
    parser.add_argument("--no-motion", action="store_true", help="skip drive and spin")
    parser.add_argument("--speed", type=int, default=60, help="motor speed 0-255 (default 60)")
    parser.add_argument("--drive-s", type=float, default=1.0, help="forward drive duration (s)")
    parser.add_argument("--spin-s", type=float, default=1.0, help="spin duration (s)")
    parser.add_argument("--out", default="hello_frame.jpg", help="saved camera frame path")
    parser.add_argument("--min-clear-mm", type=int, default=400,
                        help="skip forward drive if obstacle closer than this")
    args = parser.parse_args(argv)

    results = {}

    # --- 1. Hardware connect -------------------------------------------------
    bot, camera, sensors, actuators = make_hardware(mock=args.mock)
    mode = "HARDWARE" if bot is not None else "MOCK"
    results["driver connect"] = ("PASS", mode) if (bot is not None or args.mock) else (
        "WARN", "vendor driver not found -> mock mode (fine on a dev machine, "
                "a problem if you are running this on the Pi)")
    print(f"[hello] running in {mode} mode")

    try:
        # --- 2. LED + buzzer -------------------------------------------------
        def led_buzzer():
            actuators.set_led_color("green")
            actuators.beep(0.15)
            return "LEDs green + beep"
        _step(results, "led+buzzer", led_buzzer)

        # --- 3. Camera capture ----------------------------------------------
        def camera_capture():
            if args.mock:
                return "skipped in mock mode (no camera assumed)"
            import cv2
            frame = camera.capture_color()
            h, w = frame.shape[:2]
            cv2.imwrite(args.out, frame)
            return f"{w}x{h} frame -> {args.out}"
        _step(results, "camera", camera_capture)

        # --- 4. Ultrasonic ----------------------------------------------------
        clearance_mm = None

        def ultrasonic():
            nonlocal clearance_mm
            clearance_mm = sensors.read_ultrasonic_mm()
            if clearance_mm < 0:
                raise RuntimeError("ultrasonic read failed (-1)")
            return f"{clearance_mm} mm ahead"
        _step(results, "ultrasonic", ultrasonic)

        # --- 5. Motion --------------------------------------------------------
        if args.no_motion:
            results["motion"] = ("SKIP", "--no-motion")
        else:
            def motion():
                notes = []
                if clearance_mm is not None and 0 <= clearance_mm < args.min_clear_mm:
                    notes.append(f"forward skipped ({clearance_mm} mm < "
                                 f"{args.min_clear_mm} mm clearance)")
                else:
                    actuators.move_forward(args.speed)
                    time.sleep(args.drive_s)
                    actuators.stop()
                    time.sleep(0.2)
                    notes.append(f"forward {args.drive_s}s @ {args.speed}")
                actuators.rotate_right(args.speed)
                time.sleep(args.spin_s)
                actuators.stop()
                notes.append(f"spin {args.spin_s}s @ {args.speed}")
                return "; ".join(notes)
            _step(results, "motion", motion)

    finally:
        # Whatever happened: motors off, LEDs off.
        try:
            actuators.stop()
            actuators.set_led_color("off")
        except Exception:
            pass
        try:
            camera.close()
        except Exception:
            pass

    # --- Summary -------------------------------------------------------------
    print("\n=== Cat Chaser 3000 hello-world summary ===")
    worst = 0
    for name, (status, detail) in results.items():
        print(f"  {status:5s}  {name:15s} {detail}")
        if status == "FAIL":
            worst = 1
    print("===========================================")
    print("ALL GOOD — the rover stack is alive." if worst == 0
          else "Some steps FAILED — see above.")
    return worst


if __name__ == "__main__":
    sys.exit(main())
