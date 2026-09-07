"""Wheels-off diagnostic: motor mapping + mecanum strafe-vs-rotate.

Run on the robot with the WHEELS OFF THE GROUND and watch each wheel.

    python3 -m catchaser.wheelcheck --describe        # just print the patterns, no motion
    python3 -m catchaser.wheelcheck --each            # spin motors 0,1,2,3 one at a time
    python3 -m catchaser.wheelcheck --move strafe_right
    python3 -m catchaser.wheelcheck --all             # every primitive in sequence

Motor IDs (per config): 0=L1 front-left, 1=L2 rear-left, 2=R1 front-right,
3=R2 rear-right. '+' = that wheel drives the robot forward.

The two moves people confuse:
  ROTATE RIGHT (spin CW in place): left wheels forward, right wheels backward.
  STRAFE RIGHT (slide sideways):   front-left + rear-right forward, the other
                                   two backward — the mecanum rollers turn that
                                   diagonal into pure lateral motion.
If strafe doesn't translate sideways (feels like a rotation or a shuffle), the
mecanum wheels are almost certainly mounted wrong: viewed from above, the
roller axes must form an 'X'. Two wheels swapped gives an 'O' and strafe fails.
"""

import argparse
import sys
import time

from raspbot_slam import config

# name -> (l1, l2, r1, r2) sign pattern (the mecanum mix for that primitive)
PATTERNS = {
    "forward":      (+1, +1, +1, +1),
    "back":         (-1, -1, -1, -1),
    "strafe_left":  (-1, +1, +1, -1),
    "strafe_right": (+1, -1, -1, +1),
    "rotate_left":  (-1, -1, +1, +1),   # CCW / left
    "rotate_right": (+1, +1, -1, -1),   # CW / right
}
MOTOR_NAMES = {0: "L1 front-left", 1: "L2 rear-left",
               2: "R1 front-right", 3: "R2 rear-right"}


def describe():
    print("motor IDs:", ", ".join(f"{i}={n}" for i, n in MOTOR_NAMES.items()))
    print(f"{'move':14s}  L1  L2  R1  R2")
    for name, (l1, l2, r1, r2) in PATTERNS.items():
        sgn = lambda v: f"{'+' if v > 0 else '-'}"
        print(f"{name:14s}  {sgn(l1):>2} {sgn(l2):>2} {sgn(r1):>2} {sgn(r2):>2}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Wheels-off motor/mecanum diagnostic")
    ap.add_argument("--describe", action="store_true", help="print patterns, no motion")
    ap.add_argument("--each", action="store_true", help="spin each motor forward, one at a time")
    ap.add_argument("--move", choices=sorted(PATTERNS), help="run one primitive")
    ap.add_argument("--all", action="store_true", help="run every primitive in sequence")
    ap.add_argument("--speed", type=int, default=80)
    ap.add_argument("--secs", type=float, default=1.3)
    args = ap.parse_args(argv)

    if args.describe:
        describe()
        return 0

    from .hw import make_hardware
    bot, camera, sensors, actuators = make_hardware()
    if bot is None:
        print("[wheelcheck] not on hardware — run on the robot.", file=sys.stderr)
        return 2

    print("*** WHEELS OFF THE GROUND — starting in 3 s ***")
    time.sleep(3)
    s = args.speed
    try:
        if args.each:
            for mid in (0, 1, 2, 3):
                print(f"motor {mid} = {MOTOR_NAMES[mid]}: driving FORWARD "
                      f"(this wheel should spin so its top goes toward the front)")
                bot.Ctrl_Muto(mid, s)
                time.sleep(args.secs)
                bot.Ctrl_Muto(mid, 0)
                time.sleep(0.6)

        moves = list(PATTERNS) if args.all else ([args.move] if args.move else [])
        for name in moves:
            l1, l2, r1, r2 = (v * s for v in PATTERNS[name])
            print(f"{name}: L1={l1:+d} L2={l2:+d} R1={r1:+d} R2={r2:+d}")
            actuators._set_motors(l1, l2, r1, r2)
            time.sleep(args.secs)
            actuators.stop()
            time.sleep(0.7)

        if not (args.each or moves):
            describe()
            print("\n(nothing to run — use --each, --move NAME, or --all)")
    finally:
        actuators.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
