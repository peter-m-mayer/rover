"""Auto-zero the camera pan/tilt: find the servo commands that point the
camera along the robot's actual direction of travel ("straight ahead").

Idea: while the robot drives straight, the static scene streams outward from
one image point, the focus of expansion (FOE). The FOE IS the direction of
travel. If the camera points straight ahead the FOE sits at the image center;
otherwise its pixel offset says exactly how far off the camera is, on both
axes (pan from FOE x, tilt-to-level from FOE y). No eyeballing, no fx needed.

Method "lines" (default, static, no wheel motion): set the robot on a floor
with straight lines parallel to its chassis (plank seams, or a ruler/level
laid alongside it). Those lines meet at a vanishing point (VP) that IS the
chassis's straight-ahead direction (x) and the horizon (y). The servos are
stepped, always approaching from below (servo slack), until the VP sits at
the image center. First run 2026-10-10: 32 of 36 floor segments agreed.

Method "drive" (wheels move — clear ~1.5 m ahead, robot on the floor). Less
reliable on this robot: the chassis yaws several degrees per hop on carpet:
  1. Hold pan/tilt at the current config center and make a few short
     stop-and-go hops forward, then the same backward, keeping only the STILL
     frames before/after each hop (frames grabbed while moving are dominated
     by chassis bounce). Ultrasonic blocks a forward hop if anything is within
     --min-clear-mm.
  2. Estimate the FOE from sparse LK flow (RANSAC line intersection) for each
     hop. Backward motion gives a focus of *contraction* at the same point,
     but any slow body yaw biases the two legs in OPPOSITE directions, so
     their mean cancels that bias to first order.
  3. Steer the servos (robot parked) until the scene point at the FOE lands
     on the image center, measuring the image shift by phase correlation.
     Closed-loop, so the servo's uneven gain and slack don't matter; the
     final approach is always from the same side (consistent backlash).

    python3 -m catchaser.camera_zero                 # floor-line VP, prints result
    python3 -m catchaser.camera_zero --method drive  # FOE hops instead
    python3 -m catchaser.camera_zero --write         # ...and update config.py
    python3 -m catchaser.camera_zero --analyze DIR   # re-analyze saved frames

Output: SERVO_PAN_CENTER (camera aims along travel) and SERVO_TILT_LEVEL
(optical axis parallel to the direction of travel = level on a flat floor).
"""

import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

from raspbot_slam import config


# --------------------------------------------------------------------------
# FOE estimation (pure functions, unit-testable off-robot)
# --------------------------------------------------------------------------

def foe_from_flow(pts, flow, iters=400, inlier_rad=0.06, rng=None):
    """Focus of expansion from point/flow pairs by RANSAC line intersection.

    Each flow vector defines a line through its point; for pure translation
    all lines pass through the FOE. Residual = sine of the angle between the
    flow and the ray from the candidate FOE to the point (an angular test, so
    short far-away flow vectors aren't penalized for their length). Returns
    (foe_xy, inlier_frac) or (None, 0).
    """
    pts = np.asarray(pts, float)
    flow = np.asarray(flow, float)
    mag = np.linalg.norm(flow, axis=1)
    keep = mag > 0.5
    pts, flow, mag = pts[keep], flow[keep], mag[keep]
    if len(pts) < 12:
        return None, 0.0
    n = np.stack([-flow[:, 1], flow[:, 0]], 1) / mag[:, None]   # line normals
    c = np.sum(n * pts, 1)                                      # n.p = c
    rng = rng or np.random.default_rng(0)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        A = n[[i, j]]
        if abs(np.linalg.det(A)) < 0.2:          # near-parallel lines
            continue
        e = np.linalg.solve(A, c[[i, j]])
        inl = _angular_resid(n, c, pts, e) < inlier_rad
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 8:
        return None, 0.0
    e, *_ = np.linalg.lstsq(n[best], c[best], rcond=None)
    return e, float(best.mean())


def yaw_flow(pts, psi, fx, center):
    """Image flow of a pure camera yaw by psi rad (+ = scene moves right),
    exact pinhole small-angle form: wide lenses make it non-uniform."""
    x = (pts[:, 0] - center[0]) / fx
    y = (pts[:, 1] - center[1]) / fx
    return np.stack([fx * psi * (1 + x * x), fx * psi * x * y], 1)


def foe_with_yaw(pts, flow, fx, center, max_deg=3.0, step_deg=0.05):
    """FOE when the body also yawed a little during the hop.

    Grid-search the yaw: for each candidate, remove its flow and run the
    RANSAC FOE fit; keep the yaw whose de-rotated flow is most consistent
    with pure translation. Separable because translation flow scales with
    inverse depth (near floor vs far wall) while rotation flow does not.
    Returns (foe_xy, yaw_deg, inlier_frac)."""
    pts = np.asarray(pts, float)
    flow = np.asarray(flow, float)
    best = (None, 0.0, -1.0)
    for d in np.arange(-max_deg, max_deg + 1e-9, step_deg):
        e, frac = foe_from_flow(pts, flow - yaw_flow(pts, math.radians(d), fx, center),
                                iters=120)
        if e is not None and frac > best[2]:
            best = (e, d, frac)
    if best[0] is None:
        return None, 0.0, 0.0
    d = best[1]
    e, frac = foe_from_flow(pts, flow - yaw_flow(pts, math.radians(d), fx, center))
    return e, d, frac


def _angular_resid(n, c, pts, e):
    return np.abs(n @ e - c) / np.maximum(np.linalg.norm(pts - e, axis=1), 1.0)


def vanishing_point(gray, floor_frac=0.52, min_len=35, tol=0.02, iters=2000,
                    rng=None):
    """Vanishing point of the floor lines (plank seams, a ruler/level laid
    along the chassis). Segments from the lower part of the image (below
    `floor_frac` of the height), near-horizontal ones dropped (cross seams);
    RANSAC on the angle between each segment and the ray to the candidate
    point, length-weighted. Returns (vp_xy, n_inliers, n_segments)."""
    import cv2
    h, w = gray.shape
    segs = cv2.createLineSegmentDetector().detect(gray)[0]
    if segs is None:
        return None, 0, 0
    keep = []
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_len or min(y1, y2) < h * floor_frac:
            continue
        if math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1))) < 8:
            continue
        keep.append((x1, y1, x2, y2, length))
    if len(keep) < 4:
        return None, 0, len(keep)
    k = np.array(keep, float)
    ones = np.ones(len(k))
    lines = np.cross(np.c_[k[:, :2], ones], np.c_[k[:, 2:4], ones])
    lines /= np.linalg.norm(lines[:, :2], axis=1)[:, None]
    mid = (k[:, :2] + k[:, 2:4]) / 2
    d = k[:, 2:4] - k[:, :2]
    d /= np.linalg.norm(d, axis=1)[:, None]
    rng = rng or np.random.default_rng(0)
    best_score, best_inl = -1.0, None
    for _ in range(iters):
        i, j = rng.choice(len(k), 2, replace=False)
        v = np.cross(lines[i], lines[j])
        if abs(v[2]) < 1e-9:
            continue
        tv = v[:2] / v[2] - mid
        tv /= np.maximum(np.linalg.norm(tv, axis=1), 1e-9)[:, None]
        inl = np.abs(d[:, 0] * tv[:, 1] - d[:, 1] * tv[:, 0]) < tol
        score = k[inl, 4].sum()
        if score > best_score:
            best_score, best_inl = score, inl
    if best_inl is None or best_inl.sum() < 3:
        return None, 0, len(k)
    wts = k[best_inl, 4]
    vp = np.linalg.lstsq(lines[best_inl, :2] * wts[:, None],
                         -lines[best_inl, 2] * wts, rcond=None)[0]
    return vp, int(best_inl.sum()), len(k)


def image_shift(a, b):
    """Phase-correlation shift of the scene from a to b, in pixels."""
    import cv2
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a * win, b * win)
    return float(dx), float(dy)


# --------------------------------------------------------------------------
# Hardware steps
# --------------------------------------------------------------------------

def _gray(camera):
    import cv2
    return cv2.cvtColor(camera.capture_color(), cv2.COLOR_BGR2GRAY)


def hop(camera, actuators, sensors, forward, secs, min_clear_mm):
    """One stop-and-go hop: still frame, drive `secs`, stop, still frame.

    Only the two STILL frames are used: frames grabbed while moving are
    dominated by chassis bounce/sway, which swamps the translation flow.
    Returns (before, after) grayscale, or None if blocked ahead."""
    if forward > 0:
        mm = sensors.read_ultrasonic_mm()
        if mm and 0 < mm < min_clear_mm:
            print(f"[camera_zero] obstacle at {mm} mm — hop skipped")
            return None
    before = _settled_gray(camera, 0.5).astype(np.uint8)
    ramp = min(0.25, secs / 3)                # soft start/stop: less yaw jerk
    try:
        t0 = time.time()
        while (t := time.time() - t0) < secs:
            k = min(1.0, t / ramp, (secs - t) / ramp)
            actuators.drive(forward * max(0.3, k))
            time.sleep(0.03)
    finally:
        actuators.stop()
    after = _settled_gray(camera, 0.8).astype(np.uint8)
    return before, after


def yaw_homography(theta, fx, center):
    """Image homography of a pure camera yaw by theta rad (pinhole K)."""
    K = np.array([[fx, 0, center[0]], [0, fx, center[1]], [0, 0, 1.0]])
    c, si = math.cos(theta), math.sin(theta)
    R = np.array([[c, 0, si], [0, 1, 0], [-si, 0, c]])
    return K @ R @ np.linalg.inv(K)


def coarse_yaw(a, b, fx, center, top_frac=0.45):
    """Body yaw between two stills from the far (upper) part of the scene,
    where translation parallax is small. Returns theta such that warping b
    by yaw_homography(theta) re-aligns it with a."""
    import cv2
    h, w = a.shape
    n = int(h * top_frac)
    dx, _ = image_shift(a[:n], b[:n])
    best = (0.0, abs(dx))
    for th in (math.atan2(dx, fx), -math.atan2(dx, fx)):
        bw = cv2.warpPerspective(b, yaw_homography(th, fx, center), (w, h))
        r = abs(image_shift(a[:n], bw[:n])[0])
        if r < best[1]:
            best = (th, r)
    return best[0]


def foe_from_pair(a, b, fx, center_frac=0.8):
    """FOE from one still-to-still pair.

    The chassis yaws on every hop (mecanum on carpet), often by several
    degrees, which slides the whole scene sideways and wrecks LK tracking
    on the carpet texture. So: estimate that yaw from the far scene, warp b
    to undo it, track the residual (mostly translation) flow, then fit the
    FOE with a fine residual-yaw search. Features are restricted to the
    central part of the image (lens distortion bends flow near the edges).
    Returns (foe_xy, total_yaw_deg, inlier_frac)."""
    import cv2
    h, w = a.shape
    center = (w / 2.0, h / 2.0)
    th = coarse_yaw(a, b, fx, center)
    b = cv2.warpPerspective(b, yaw_homography(th, fx, center), (w, h))
    mask = np.zeros_like(a)
    mh, mw = int(h * (1 - center_frac) / 2), int(w * (1 - center_frac) / 2)
    mask[mh:h - mh, mw:w - mw] = 255
    p0 = cv2.goodFeaturesToTrack(a, 400, 0.01, 7, mask=mask)
    if p0 is None:
        return None, math.degrees(th), 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(31, 31),
                                         maxLevel=4)
    ok = st.ravel() == 1
    p0, p1 = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
    e, d, frac = foe_with_yaw(p0, p1 - p0, fx, center)
    return e, math.degrees(th) + d, frac


def _settled_gray(camera, wait=0.7):
    time.sleep(wait)
    for _ in range(3):
        camera.capture_color()
    return _gray(camera).astype(np.float32)


def steer_to(camera, bot, sid, cmd_home, target_px, axis, sign_px_per_cmd,
             backoff=6, tol_px=3.0, max_iter=14):
    """Move servo `sid` until the scene has shifted by target_px along `axis`
    (0=x, 1=y) relative to the view at cmd_home. Returns (command, shift).

    Backs off `backoff` commands below the target side first, so the final
    approach always comes from the same direction (consistent backlash).
    sign_px_per_cmd: rough signed px per command for the first guess; later
    steps use the measured response (secant)."""
    bot.Ctrl_Servo(sid, int(cmd_home))
    home = _settled_gray(camera)
    cmd = round(cmd_home + target_px / sign_px_per_cmd) - backoff
    bot.Ctrl_Servo(sid, int(cmd))
    ref = _settled_gray(camera)
    shift = image_shift(home, ref)[axis]       # where we are vs home, measured
    k = sign_px_per_cmd
    for _ in range(max_iter):
        err = target_px - shift
        if abs(err) < tol_px:
            break
        step = float(np.clip(err / k, -12, 12))
        if abs(step) < 1:
            step = math.copysign(1, step)
        new = cmd + step
        bot.Ctrl_Servo(sid, int(round(new)))
        time.sleep(0.6)
        for _ in range(2):
            camera.capture_color()
        new_shift = image_shift(home, _gray(camera).astype(np.float32))[axis]
        moved = new_shift - shift
        if abs(moved) > 1.0 and np.sign(moved) == np.sign(k):
            k = 0.5 * k + 0.5 * (moved / (round(new) - round(cmd) or step))
        cmd, shift = round(new), new_shift
    return int(cmd), shift


def _vp_at(camera, bot, pan, tilt, approach=8, samples=2):
    """VP with the servos settled at (pan, tilt), each approached from BELOW
    (servo slack: always land on the same side)."""
    bot.Ctrl_Servo(config.SERVO_PAN_ID, int(pan - approach))
    bot.Ctrl_Servo(config.SERVO_TILT_ID, int(tilt - approach))
    time.sleep(0.5)
    bot.Ctrl_Servo(config.SERVO_PAN_ID, int(pan))
    bot.Ctrl_Servo(config.SERVO_TILT_ID, int(tilt))
    vps = []
    for _ in range(samples):
        vp, n, _ = vanishing_point(_settled_gray(camera, 0.6).astype(np.uint8))
        if vp is not None and n >= 4:
            vps.append(vp)
    return np.median(vps, 0) if vps else None


def zero_by_floor_lines(camera, bot, pan0, tilt0, px_per_pan=5.0,
                        px_per_tilt=7.4, max_iter=8):
    """Step pan/tilt until the floor-line VP sits at the image center.
    Increasing pan turns the camera left = VP shifts right; increasing tilt
    looks up = VP shifts down. Returns (pan, tilt, vp_offset_px) or None."""
    w, h = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
    pan, tilt = int(pan0), int(tilt0)
    seen = set()
    off = None
    for _ in range(max_iter):
        vp = _vp_at(camera, bot, pan, tilt)
        if vp is None:
            print("[camera_zero] no vanishing point — need floor lines "
                  "parallel to the chassis in view")
            return None
        off = vp - np.array([w / 2.0, h / 2.0])
        print(f"[camera_zero]   pan {pan} tilt {tilt}: VP offset "
              f"{off[0]:+.1f}, {off[1]:+.1f} px")
        dp = int(round(-off[0] / px_per_pan))
        dt = int(round(-off[1] / px_per_tilt))
        seen.add((pan, tilt))
        if (dp == 0 and dt == 0) or (pan + dp, tilt + dt) in seen:
            break
        pan, tilt = pan + dp, tilt + dt
    # The servo gain is uneven (pan moved 3 px for one command, 20 px for
    # the next), so the step guess can overshoot: finish by checking the
    # integer neighbours on each axis and keeping the most centered one.
    for axis in (0, 1):
        best = (abs(off[axis]), pan, tilt, off)
        for step in (-1, 1):
            p, t = (pan + step, tilt) if axis == 0 else (pan, tilt + step)
            vp = _vp_at(camera, bot, p, t)
            if vp is None:
                continue
            o = vp - np.array([w / 2.0, h / 2.0])
            print(f"[camera_zero]   pan {p} tilt {t}: VP offset "
                  f"{o[0]:+.1f}, {o[1]:+.1f} px")
            if abs(o[axis]) < best[0]:
                best = (abs(o[axis]), p, t, o)
        _, pan, tilt, off = best
    return pan, tilt, off


def write_config(pan_center, tilt_level, path=None):
    path = path or os.path.join(os.path.dirname(config.__file__), "config.py")
    src = open(path).read()
    src = re.sub(r"^SERVO_PAN_CENTER = \d+.*$",
                 f"SERVO_PAN_CENTER = {pan_center}             "
                 f"# auto-zeroed (catchaser.camera_zero, {time.strftime('%Y-%m-%d')})",
                 src, count=1, flags=re.M)
    line = (f"SERVO_TILT_LEVEL = {tilt_level}             "
            f"# optical axis parallel to travel (camera_zero)")
    if re.search(r"^SERVO_TILT_LEVEL = ", src, re.M):
        src = re.sub(r"^SERVO_TILT_LEVEL = .*$", line, src, count=1, flags=re.M)
    else:
        src = re.sub(r"^(SERVO_TILT_REST = .*)$", line + "\n" + r"\1",
                     src, count=1, flags=re.M)
    open(path, "w").write(src)
    print(f"[camera_zero] wrote {path}")


def analyze(fwd_pairs, back_pairs, w, h, fx):
    """Per-hop FOE; forward and backward medians; their mean cancels a
    consistent yaw/crab bias (it shifts the two legs in opposite directions)."""
    legs = {}
    for name, pairs in (("forward", fwd_pairs), ("backward", back_pairs)):
        foes = []
        for a, b in pairs:
            e, yaw, frac = foe_from_pair(a, b, fx)
            if e is not None and frac > 0.4:
                foes.append(e)
            print(f"[camera_zero]   {name} hop: "
                  f"{'none' if e is None else np.round(e, 1)}  "
                  f"yaw {yaw:+.2f} deg  inliers {frac:.2f}")
        if foes:
            legs[name] = np.median(np.array(foes), 0)
            spread = np.std(np.array(foes), 0) if len(foes) > 1 else np.zeros(2)
            print(f"[camera_zero] {name}: {np.round(legs[name], 1)}  "
                  f"(n={len(foes)}, spread {np.round(spread, 1)} px)")
    if not legs:
        return None
    e = np.mean(list(legs.values()), 0)
    if len(legs) == 2:
        d = legs["forward"] - legs["backward"]
        print(f"[camera_zero] fwd-back disagreement {np.round(d, 1)} px "
              f"(drift/yaw bias; the mean cancels it)")
    off = e - np.array([w / 2.0, h / 2.0])
    print(f"[camera_zero] travel direction: {off[0]:+.1f} px right, "
          f"{off[1]:+.1f} px down of image center")
    return off


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Auto-zero camera pan/tilt by FOE")
    ap.add_argument("--method", choices=["lines", "drive"], default="lines",
                    help="lines: static, floor lines parallel to the chassis "
                         "(plank seams / a ruler) -> vanishing point (default); "
                         "drive: hop forward/back and use the FOE")
    ap.add_argument("--repeat", type=int, default=3,
                    help="lines: re-measure the result this many times")
    ap.add_argument("--speed", type=int, default=50)
    ap.add_argument("--secs", type=float, default=0.7, help="drive time per hop")
    ap.add_argument("--hops", type=int, default=3, help="hops each direction")
    ap.add_argument("--fx", type=float, default=425.0,
                    help="focal length px (only shapes the yaw-flow model; "
                         "tilt servo maps ~7.4 px/deg -> ~425)")
    ap.add_argument("--min-clear-mm", type=int, default=450)
    ap.add_argument("--pan", type=int, default=None, help="start pan (default config)")
    ap.add_argument("--tilt", type=int, default=None, help="start tilt (default rest)")
    ap.add_argument("--save", default="/tmp/camera_zero", help="frame dump dir")
    ap.add_argument("--analyze", default=None, help="re-analyze a dump dir only")
    ap.add_argument("--no-steer", action="store_true", help="measure only")
    ap.add_argument("--write", action="store_true", help="update config.py")
    args = ap.parse_args(argv)

    import cv2
    if args.analyze:
        meta = json.load(open(os.path.join(args.analyze, "meta.json")))
        rd = lambda n: cv2.imread(os.path.join(args.analyze, n), 0)
        pairs = {leg: [(rd(f"{leg}{i}_a.png"), rd(f"{leg}{i}_b.png"))
                       for i in range(meta[leg])] for leg in ("fwd", "back")}
        h, w = pairs["fwd"][0][0].shape
        print(f"[camera_zero] pan {meta['pan']} tilt {meta['tilt']}")
        analyze(pairs["fwd"], pairs["back"], w, h, args.fx)
        return 0

    from .hw import make_hardware
    bot, camera, sensors, actuators = make_hardware()
    if bot is None:
        print("[camera_zero] mock mode — run on the robot.", file=sys.stderr)
        return 2
    pan0 = args.pan if args.pan is not None else config.SERVO_PAN_CENTER
    tilt0 = args.tilt if args.tilt is not None else getattr(
        config, "SERVO_TILT_LEVEL", config.SERVO_TILT_REST)
    try:
        if args.method == "lines":
            res = zero_by_floor_lines(camera, bot, pan0, tilt0)
            if res is None:
                return 1
            pan_c, tilt_l, _ = res
            center = np.array([config.CAMERA_WIDTH / 2, config.CAMERA_HEIGHT / 2])
            offs = [_vp_at(camera, bot, pan_c, tilt_l) for _ in range(args.repeat)]
            offs = np.array([o - center for o in offs if o is not None])
            if len(offs):
                print(f"[camera_zero] repeat check at pan {pan_c} tilt {tilt_l}: "
                      f"offset mean {np.round(offs.mean(0), 1)} px, spread "
                      f"{np.round(offs.std(0), 1)} px")
            print(f"[camera_zero] RESULT  SERVO_PAN_CENTER = {pan_c}   "
                  f"SERVO_TILT_LEVEL = {tilt_l}   (was pan "
                  f"{config.SERVO_PAN_CENTER}, tilt level "
                  f"{getattr(config, 'SERVO_TILT_LEVEL', None)})")
            if args.write:
                write_config(pan_c, tilt_l)
            return 0
        sensors.enable_ultrasonic()
        bot.Ctrl_Servo(config.SERVO_PAN_ID, pan0)
        bot.Ctrl_Servo(config.SERVO_TILT_ID, tilt0)
        time.sleep(1.0)
        mm = sensors.read_ultrasonic_mm()
        print(f"[camera_zero] pan {pan0} tilt {tilt0}; clear ahead: {mm} mm")
        if mm and 0 < mm < args.min_clear_mm + 300:
            print("[camera_zero] not enough room ahead — aborting.")
            return 1
        pairs = {"fwd": [], "back": []}
        for leg, sgn in (("fwd", 1), ("back", -1)):
            print(f"[camera_zero] {args.hops} {leg} hops...")
            for _ in range(args.hops):
                p = hop(camera, actuators, sensors, sgn * args.speed,
                        args.secs, args.min_clear_mm)
                if p is not None:
                    pairs[leg].append(p)
        os.makedirs(args.save, exist_ok=True)
        for leg, pl in pairs.items():
            for i, (a, b) in enumerate(pl):
                cv2.imwrite(os.path.join(args.save, f"{leg}{i}_a.png"), a)
                cv2.imwrite(os.path.join(args.save, f"{leg}{i}_b.png"), b)
        json.dump(dict(pan=pan0, tilt=tilt0, speed=args.speed,
                       fwd=len(pairs["fwd"]), back=len(pairs["back"])),
                  open(os.path.join(args.save, "meta.json"), "w"))
        h, w = pairs["fwd"][0][0].shape
        off = analyze(pairs["fwd"], pairs["back"], w, h, args.fx)
        if off is None:
            print("[camera_zero] no FOE found (too little texture/motion?)")
            return 1
        if args.no_steer:
            return 0
        # Put the FOE scene point at image center: the scene must shift by
        # -off. Pan: increasing pan turns the camera left = scene shifts
        # right (+x), ~+5 px/cmd. Tilt: increasing tilt = scene shifts down
        # (+y), ~+7 px/cmd. Approach from below (same side) for backlash.
        pan_c, sx = steer_to(camera, bot, config.SERVO_PAN_ID, pan0,
                             -off[0], 0, 5.0)
        tilt_l, sy = steer_to(camera, bot, config.SERVO_TILT_ID, tilt0,
                              -off[1], 1, 7.0)
        print(f"[camera_zero] residual: pan {(-off[0]) - sx:+.1f} px, "
              f"tilt {(-off[1]) - sy:+.1f} px")
        print(f"[camera_zero] RESULT  SERVO_PAN_CENTER = {pan_c}   "
              f"SERVO_TILT_LEVEL = {tilt_l}   (was pan {config.SERVO_PAN_CENTER})")
        if args.write:
            write_config(pan_c, tilt_l)
    finally:
        actuators.stop()
        bot.Ctrl_Servo(config.SERVO_PAN_ID, int(config.SERVO_PAN_CENTER))
        bot.Ctrl_Servo(config.SERVO_TILT_ID, int(config.SERVO_TILT_REST))
        camera.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
