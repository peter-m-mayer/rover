"""Map a camera servo axis on hardware: sign, gain (image deg per command deg),
usable travel, and hold stability — from the camera alone.

Steps the servo through a range of commands. At each step it holds, grabs a
burst of frames, and measures (a) the scene shift versus the previous step by
phase correlation (-> how far the camera actually turned) and (b) the
frame-to-frame jitter while holding (-> chatter / hunting). Wheels never move.

    python3 -m catchaser.servo_map --axis pan  --lo 20 --hi 160 --step 5
    python3 -m catchaser.servo_map --axis tilt --lo 30 --hi 130 --step 5

Image shift -> camera angle uses fx from the camera model (atan(dx/fx)), so
"gain" is real camera degrees per commanded degree (1.0 = ideal servo).
A step that produces ~0 shift while neighbours move = an end stop or dead zone.
"""

import argparse
import json
import math
import sys
import time

import numpy as np

from raspbot_slam import config


def _shift(a, b):
    import cv2
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a * win, b * win)
    return float(dx), float(dy), float(resp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Camera servo mapper")
    ap.add_argument("--axis", choices=["pan", "tilt"], required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--other", type=int, default=None,
                    help="hold the other axis at this command (default: its rest)")
    ap.add_argument("--settle", type=float, default=0.6)
    ap.add_argument("--hold-frames", type=int, default=6)
    ap.add_argument("--json", default=None, help="write results here")
    args = ap.parse_args(argv)

    import cv2
    from .hw import make_hardware
    bot, camera, sensors, actuators = make_hardware()
    if bot is None:
        print("[servo_map] mock mode — run on the robot.", file=sys.stderr)
        return 2
    sid = config.SERVO_PAN_ID if args.axis == "pan" else config.SERVO_TILT_ID
    oid = config.SERVO_TILT_ID if args.axis == "pan" else config.SERVO_PAN_ID
    other = args.other if args.other is not None else (
        config.SERVO_TILT_REST if args.axis == "pan" else config.SERVO_PAN_CENTER)
    fx = float(getattr(camera, "focal_length_px", 0) or
               (config.CAMERA_WIDTH / 2) / math.tan(math.radians(65 / 2)))

    def gray():
        return cv2.cvtColor(camera.capture_color(), cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Raw Ctrl_Servo, bypassing the config clamps (we are mapping the clamps).
    bot.Ctrl_Servo(oid, int(other))
    rows = []
    try:
        bot.Ctrl_Servo(sid, args.lo)
        time.sleep(1.2)
        for _ in range(3):
            camera.capture_color()            # flush stale buffered frames
        prev = gray()
        for cmd in range(args.lo, args.hi + 1, args.step):
            bot.Ctrl_Servo(sid, cmd)
            time.sleep(args.settle)
            for _ in range(2):
                camera.capture_color()
            burst = [gray() for _ in range(args.hold_frames)]
            ref = burst[0]
            dx, dy, resp = _shift(prev, ref)
            jit = [math.hypot(*_shift(ref, f)[:2]) for f in burst[1:]]
            along = dx if args.axis == "pan" else dy
            ang = math.degrees(math.atan2(along, fx))
            rows.append(dict(cmd=cmd, dx=round(dx, 2), dy=round(dy, 2),
                             cam_deg=round(ang, 2), resp=round(resp, 3),
                             jitter_px=round(max(jit) if jit else 0.0, 2)))
            r = rows[-1]
            print(f"cmd {cmd:4d}  dx {r['dx']:+7.2f}  dy {r['dy']:+7.2f}  "
                  f"step {r['cam_deg']:+6.2f} deg  jitter {r['jitter_px']:5.2f}px  "
                  f"resp {r['resp']:.2f}", flush=True)
            prev = ref
    finally:
        bot.Ctrl_Servo(config.SERVO_PAN_ID, int(config.SERVO_PAN_CENTER))
        bot.Ctrl_Servo(config.SERVO_TILT_ID, int(config.SERVO_TILT_REST))
        camera.close()

    steps = [r["cam_deg"] for r in rows[1:]]
    if steps:
        gain = np.median(steps) / args.step
        print(f"[servo_map] fx={fx:.0f}px  median gain {gain:+.2f} camera-deg "
              f"per command-deg  (sign {'+' if gain > 0 else '-'})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(axis=args.axis, fx=fx, rows=rows), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
