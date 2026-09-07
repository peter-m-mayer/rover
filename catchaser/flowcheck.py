"""Validate optical-flow yaw on the robot: closed-loop turn vs open-loop.

    python3 -m catchaser.flowcheck --deg 90        # closed-loop: turn until flow says 90 CCW
    python3 -m catchaser.flowcheck --deg -90       # 90 CW
    python3 -m catchaser.flowcheck --open 1.5      # open-loop spin 1.5s, report measured yaw

Wheels can be on the floor (needs a textured scene in view). Point at a
feature-rich part of the room, not a blank wall.
"""

import argparse
import sys
import time

from .flow import FlowRotationEstimator, turn_by_flow


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Optical-flow rotation check")
    ap.add_argument("--deg", type=float, help="closed-loop: turn to this yaw (CCW +)")
    ap.add_argument("--open", type=float, metavar="SECS",
                    help="open-loop: spin this many seconds, measure actual yaw")
    ap.add_argument("--speed", type=int, default=70)
    args = ap.parse_args(argv)

    from .hw import make_hardware
    bot, camera, sensors, actuators = make_hardware()
    if bot is None:
        print("[flowcheck] not on hardware — run on the robot.", file=sys.stderr)
        return 2
    fx = camera.focal_length_px
    print(f"[flowcheck] focal length fx={fx:.0f}px")

    try:
        if args.open is not None:
            est = FlowRotationEstimator(fx)
            est.update(camera.capture())
            actuators.drive(0.0, args.speed, 0.0)     # spin CW (turn>0)
            acc = 0.0
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.open:
                time.sleep(0.08)
                d = est.update(camera.capture())
                if d is not None:
                    acc += d
            actuators.stop()
            import math
            print(f"[flowcheck] open-loop {args.open}s @ speed {args.speed} (CW): "
                  f"measured {math.degrees(acc):+.1f} deg  (features "
                  f"{est.last_n})")
        else:
            target = args.deg if args.deg is not None else 90.0
            print(f"[flowcheck] closed-loop turn to {target:+.0f} deg ...")
            measured = turn_by_flow(actuators, camera.capture, fx, target,
                                    speed=args.speed)
            print(f"[flowcheck] measured {measured:+.1f} deg (target {target:+.0f})")
    finally:
        actuators.stop()
        camera.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
