"""
Fully autonomous camera calibration: the robot moves its own camera around a
fixed checkerboard (equivalent to waving the board in front of a fixed camera).

    python3 -m raspbot_slam.calibration.auto_camera_cal

Steps
  1. search   sweep pan/tilt, then turn the body in 45 deg gyro-measured steps,
              until the board is in view
  2. face     turn the body until the board is straight ahead (pan centred)
  3. stations at three distances (driving in short hops, distance from solvePnP
              on the board), use pan/tilt to put the board in each part of the
              image -- centre, corners, edges -- and capture every new pose
  4. orbit    at the middle distance, strafe sideways while turning to keep the
              board in view, for oblique (foreshortened) views on both sides
  5. solve    camera_calibrate.finish(): RMS, K, distortion, coverage -> json

Needs ~1 m of clear floor in front of the board. Stops if the ultrasonic sees
anything closer than 0.2 m. The board stays where it is.
"""

import math
import os
import threading
import time

import cv2
import numpy as np

from .. import config
from . import camera_calibrate as cc

BOARD = tuple(config.CHECKER_INNER_CORNERS)
W, H = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
FX_GUESS = 425.0                    # camera_zero estimate; only used for control
K_GUESS = np.array([[FX_GUESS, 0, W / 2], [0, FX_GUESS, H / 2], [0, 0, 1]])
# No CALIB_CB_FAST_CHECK: on this board/camera it rejected every view the full
# detector accepts (measured 2026-10-11: 0/6 vs 6/6 at the same poses).
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
class CheckerTarget:
    """The printed checkerboard: 54 corners per view, full lens model."""
    name = "checkerboard"
    grid = BOARD
    calib_flags = 0
    K0 = None
    near_m = (0.35, 0.55, 0.85)

    def __init__(self):
        self.objp = cc.board_points(BOARD, config.CHECKER_SQUARE_MM)

    def detect(self, g):
        # SB detector: the classic one mis-assigned corners when squares were
        # ~11 px (board at ~1 m), giving 7 px residuals on a whole station
        found, c = cv2.findChessboardCornersSB(g, BOARD, flags=cv2.CALIB_CB_EXHAUSTIVE)
        return c.reshape(-1, 2) if found else None

    def refine(self, g, c):
        found, c2 = cv2.findChessboardCornersSB(g, BOARD, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        return (c2 if found else c).reshape(-1, 1, 2).astype(np.float32)

    def grid_order(self, c):
        return c


class TagTarget:
    """One wall ArUco tag: 4 corners per view, so a reduced lens model
    (k1, k2; no k3 or tangential terms). Interim until the board is back up."""
    name = "aruco"
    grid = (2, 2)
    # f, k1, k2 only: square pixels, principal point at the centre. 4 points per
    # view can't support the full model. Start from f ~ 480 px, measured from
    # gyro turns vs. image shift (2026-10-11).
    calib_flags = (cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_ASPECT_RATIO
                   | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_USE_INTRINSIC_GUESS)
    K0 = [[480.0, 0, W / 2], [0, 480.0, H / 2], [0, 0, 1]]
    near_m = (0.30, 0.45, 0.70)

    def __init__(self, ids=None):
        s = config.ARUCO_MARKER_SIZE_M * 1000
        # ArUco corner order: TL, TR, BR, BL (y down, like the image)
        self.objp = np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]], np.float32)
        self.ids = set(ids) if ids else set(config.ARUCO_MARKER_IDS)
        self.locked = None          # stick to the first tag found...
        self.min_side_px = 40       # ...that is at least this big (~1.5 m away)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT)), params)

    def detect(self, g):
        corners, ids, _ = self.det.detectMarkers(g)
        if ids is None:
            return None
        best = None
        for c, i in zip(corners, ids.ravel()):
            i = int(i)
            if (self.locked is not None and i != self.locked) or i not in self.ids:
                continue
            c = c.reshape(4, 2)
            if best is None or cv2.contourArea(c) > cv2.contourArea(best[1]):
                best = (i, c)
        if best is None:
            return None
        if self.locked is None:
            side = np.sqrt(cv2.contourArea(best[1]))
            if side < self.min_side_px:
                return None
            self.locked = best[0]
            print(f"  target: tag ID {best[0]}", flush=True)
        return best[1]

    def refine(self, g, c):
        return cv2.cornerSubPix(g, c.reshape(-1, 1, 2).astype(np.float32), (5, 5), (-1, -1), cc.CRITERIA)

    def grid_order(self, c):
        return c[[0, 1, 3, 2]]      # TL TR BR BL -> row-major TL TR BL BR for the descriptor


MARGIN = 12                         # px: every corner must be this far inside the frame
STOP_MM = 200


class Rig:
    def __init__(self, save_dir, target=None):
        from catchaser.hw import connect_bot
        from raspbot_slam.actuators import Actuators
        from raspbot_slam.sensors import Sensors
        from raspbot_slam.lsm9ds1 import LSM9DS1
        self.bot = connect_bot()
        if self.bot is None:
            raise SystemExit("no robot driver: run this on the Pi")
        self.act = Actuators(bot=self.bot)
        self.sens = Sensors(bot=self.bot)
        self.sens.enable_ultrasonic()
        self.imu = LSM9DS1(bus=config.IMU_I2C_BUS)
        self.cap = cv2.VideoCapture(config.CAMERA_DEVICE)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap_lock = threading.Lock()     # a RunRecorder may share the camera
        self.bus_lock = threading.RLock()    # ...and the MCU I2C bus (pass to the recorder)
        self.pan = config.SERVO_PAN_CENTER
        self.tilt = config.SERVO_TILT_LEVEL
        self.servo(self.pan, self.tilt, settle=0.6)
        self.gz_bias = self._gyro_bias()
        from raspbot_slam.heading import GyroHeading
        self.hd = GyroHeading(self.imu).start(calibrate=False)
        self.hd.bias = self.gz_bias
        self.target = target or CheckerTarget()
        self.objp = self.target.objp
        self.save_dir = save_dir
        self.views, self.descs = [], []
        self.home_tilt = config.SERVO_TILT_LEVEL     # tilt at which the board was found
        self.coast_deg = 8.0                         # body keeps turning after stop (learned)

    # ---- sensing ----------------------------------------------------------
    def _gyro_bias(self, seconds=1.0):
        vals, t_end = [], time.monotonic() + seconds
        while time.monotonic() < t_end:
            vals.append(self.imu.gyro[2])
            time.sleep(0.01)
        return float(np.mean(vals))

    def gray(self):
        with self.cap_lock:
            for _ in range(3):              # drop buffered frames
                self.cap.grab()
            ok, f = self.cap.read()
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if ok else None

    def capture_color(self):
        """Camera interface for RunRecorder (shares the device with gray())."""
        with self.cap_lock:
            ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return f

    @property
    def K(self):
        return K_GUESS

    @property
    def dist_coeffs(self):
        return np.zeros(5)

    def board(self, tries=4, strict=False):
        """Target corners (N, 2) and the frame.

        strict: two consecutive frames must agree (sharp, still) -- used for
        captures. Otherwise the first detection is enough -- used for steering,
        where a far tag only shows up in some frames."""
        prev = None
        for _ in range(tries + 1):
            g = self.gray()
            if g is None:
                continue
            c = self.target.detect(g)
            if c is None:
                prev = None
                continue
            if not strict:
                return c, g
            if prev is not None and np.mean(np.linalg.norm(c - prev, axis=1)) < 1.5:
                return c, g
            prev = c
        return None, None

    def pnp(self, c):
        """(distance m, board yaw deg seen from the camera, bearing deg) -- rough, K guessed."""
        ok, rv, tv = cv2.solvePnP(self.objp, c.reshape(-1, 1, 2).astype(np.float64), K_GUESS, None)
        R, _ = cv2.Rodrigues(rv)
        centre = R @ self.objp.mean(0) + tv.ravel()
        normal = R[:, 2] if R[2, 2] < 0 else -R[:, 2]     # the side facing the camera
        yaw = math.degrees(math.atan2(normal[0], -normal[2]))
        return np.linalg.norm(centre) / 1000.0, yaw, math.degrees(math.atan2(centre[0], centre[2]))

    def snapshot(self, name):
        g = self.gray()
        if g is not None:
            cv2.imwrite(os.path.join(self.save_dir, f"debug_{name}.jpg"), g)

    def clear_mm(self):
        with self.bus_lock:
            d = self.sens.read_ultrasonic_mm()
        return d if d > 0 else 9999

    # ---- moving -------------------------------------------------------------
    def servo(self, pan, tilt, settle=0.5):
        self.pan = float(np.clip(pan, config.SERVO_PAN_MIN, config.SERVO_PAN_MAX))
        self.tilt = float(np.clip(tilt, config.SERVO_TILT_MIN, config.SERVO_TILT_MAX))
        self.act.set_servo_pan(self.pan)
        self.act.set_servo_tilt(self.tilt)
        time.sleep(settle)

    def rotate(self, deg, speed=55, timeout=6.0):
        """Turn the body by `deg` (CCW +), measured by the gyro."""
        target = math.radians(deg)
        early = math.radians(min(self.coast_deg, 0.6 * abs(deg)))
        turned, t_prev, t_end = 0.0, time.monotonic(), time.monotonic() + timeout
        self.act.drive(0, turn=-speed if deg > 0 else speed)     # drive(): turn + = CW
        try:
            while abs(turned) < abs(target) - early and time.monotonic() < t_end:
                time.sleep(0.01)
                now = time.monotonic()
                turned += (self.imu.gyro[2] - self.gz_bias) * (now - t_prev)
                t_prev = now
        finally:
            self.act.stop()
        at_stop = abs(turned)
        t_stop = time.monotonic() + 0.4                 # coast
        while time.monotonic() < t_stop:
            time.sleep(0.01)
            now = time.monotonic()
            turned += (self.imu.gyro[2] - self.gz_bias) * (now - t_prev)
            t_prev = now
        coast = math.degrees(abs(turned) - at_stop)
        if abs(deg) >= 10:
            self.coast_deg = float(np.clip(0.5 * self.coast_deg + 0.5 * coast, 0, 20))
        return math.degrees(turned)

    def hop(self, forward=0.0, strafe=0.0, seconds=0.18, speed=50):
        from raspbot_slam.heading import drive_straight
        if forward > 0 and self.clear_mm() < STOP_MM:
            return False
        if strafe == 0:          # heading-held: carpet veers 17-36 deg/m open-loop
            drive_straight(self.act, self.hd, speed * forward, seconds)
        else:
            self.act.drive(speed * forward, turn=0, strafe=speed * strafe)
            time.sleep(seconds)
            self.act.stop()
        time.sleep(0.35)
        return True

    # ---- capture --------------------------------------------------------------
    def inside(self, c):
        return (c[:, 0].min() > MARGIN and c[:, 0].max() < W - MARGIN
                and c[:, 1].min() > MARGIN and c[:, 1].max() < H - MARGIN)

    def capture(self, c, g, why):
        if not self.inside(c):
            return False
        d = cc.pose_descriptor(self.target.grid_order(c), self.target.grid, (W, H))
        if self.descs and min(np.linalg.norm(d - e) for e in self.descs) < 0.10:
            return False
        c = self.target.refine(g, c)
        cv2.imwrite(os.path.join(self.save_dir, f"view_{len(self.views):02d}.png"), g)
        self.views.append(c)
        self.descs.append(d)
        row, col = cc._region(d)
        print(f"  captured {len(self.views):2d} [{why}]: {row}-{col}, size {d[2]:.0%}, "
              f"tilt L/R {d[3]:+.2f} U/D {d[4]:+.2f}", flush=True)
        return True


def search(rig):
    if search_once(rig):
        return True
    if getattr(rig.target, "min_side_px", 0) > 20:     # no near tag: accept a far one
        rig.target.min_side_px = 20
        print("search: no tag within ~1.5 m, accepting farther ones", flush=True)
        return search_once(rig)
    return False


def search_once(rig):
    c0 = config.SERVO_PAN_CENTER
    pans = [c0, c0 - 20, c0 + 20, c0 - 40, c0 + 40, c0 - 60, c0 + 60]
    tilts = [config.SERVO_TILT_LEVEL - 8, config.SERVO_TILT_LEVEL + 6]
    for step in range(9):
        for tilt in tilts:
            for pan in pans:
                rig.servo(pan, tilt)
                c, _ = rig.board()
                if c is not None:
                    rig.home_tilt = tilt
                    print(f"search: board found at pan {pan:.0f} tilt {tilt:.0f} after {step} body turns", flush=True)
                    return True
        if step < 8:
            print(f"search: not in view, turning 45 deg ({step + 1}/8)", flush=True)
            rig.servo(c0, config.SERVO_TILT_LEVEL, settle=0.2)
            rig.rotate(45)
    return False


def centre_in_image(rig, target=(0.5, 0.5), tol_px=30, iters=8):
    """Pan/tilt until the board centroid sits at `target` (fractions of W, H)."""
    gain = [FX_GUESS * math.pi / 180] * 2      # px per servo degree, refined as we go
    c, g = rig.board()
    for _ in range(iters):
        if c is None:
            return None, None
        err = np.array(target) * (W, H) - c.mean(0)
        if np.all(np.abs(err) < tol_px):
            return c, g
        # pan > centre looks left -> board moves right (+x); tilt up -> board moves down (+y)
        dpan = float(np.clip(err[0] / gain[0], -15, 15))
        dtilt = float(np.clip(err[1] / gain[1], -12, 12))
        before, last = c.mean(0), (rig.pan, rig.tilt)
        rig.servo(rig.pan + dpan, rig.tilt + dtilt)
        c, g = rig.board()
        if c is None:                      # overshot out of view: step back, halve the gain guess
            rig.servo(*last)
            gain = [2 * v for v in gain]
            c, g = rig.board()
            continue
        if c is not None:
            moved = c.mean(0) - before
            for i, dv in ((0, dpan), (1, dtilt)):
                if abs(dv) >= 3 and moved[i] * dv > 0:
                    gain[i] = 0.5 * gain[i] + 0.5 * abs(moved[i] / dv)
    return (c, g) if c is not None else (None, None)


def local_search(rig):
    """Re-find the board with the camera only (no driving)."""
    c0 = config.SERVO_PAN_CENTER
    for tilt in (rig.home_tilt, rig.home_tilt - 8, rig.home_tilt + 8, rig.home_tilt - 16):
        for pan in (c0, c0 - 15, c0 + 15, c0 - 30, c0 + 30):
            rig.servo(pan, tilt)
            c, g = rig.board()
            if c is not None:
                rig.home_tilt = tilt
                return c, g
    return None, None


def face_board(rig):
    """Turn the body so the board is straight ahead with the pan centred."""
    for _ in range(4):
        c, _ = centre_in_image(rig)
        if c is None:
            print("    face: board lost", flush=True)
            return False
        _, _, bearing = rig.pnp(c)
        body = (rig.pan - config.SERVO_PAN_CENTER) - bearing      # deg left of the body axis
        print(f"    face: board {body:+.0f} deg from body axis", flush=True)
        if abs(body) < 8:          # the pan fine-centres the rest
            return True
        rig.servo(config.SERVO_PAN_CENTER, rig.tilt, settle=0.2)
        rig.rotate(body * 0.9)
        c, _ = rig.board()
        if c is None:
            c, _ = local_search(rig)
            if c is None and not search(rig):
                return False
    return True


def goto_distance(rig, target_m, tol=0.05, max_hops=30):
    d = None
    for hop in range(max_hops):
        c, _ = centre_in_image(rig, tol_px=60)
        if c is None:
            print(f"    approach hop {hop}: board lost after centring, local search", flush=True)
            rig.snapshot(f"lost_{target_m:.2f}_{hop}")
            c, _ = local_search(rig)
            if c is None:
                return None
            c, _ = centre_in_image(rig, tol_px=60)
            if c is None:
                return None
        d, _, bearing = rig.pnp(c)
        print(f"    approach hop {hop}: {d:.2f} m, pan {rig.pan:.0f} tilt {rig.tilt:.0f}, "
              f"board {(rig.pan - config.SERVO_PAN_CENTER) - bearing:+.0f} deg from body axis", flush=True)
        if abs(d - target_m) < tol:
            return d
        if abs((rig.pan - config.SERVO_PAN_CENTER) - bearing) > 12:   # drifted: face it again
            face_board(rig)
            continue
        # ~0.28 m/s at speed 50; under ~0.2 s the wheels barely break free on carpet
        step = float(np.clip(abs(d - target_m) / 0.28, 0.2, 0.8))
        if not rig.hop(forward=1.0 if d > target_m else -1.0, seconds=step):
            print("  ultrasonic stop: too close, not driving further", flush=True)
            return d
    return d


TARGETS = [(0.5, 0.5), (0.27, 0.27), (0.73, 0.27), (0.27, 0.73), (0.73, 0.73),
           (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)]


def sweep_station(rig, label, targets=TARGETS):
    for tgt in targets:
        c, g = centre_in_image(rig, tgt, tol_px=45)
        if c is not None:
            c, g = rig.board(strict=True)
        if c is None:
            print(f"    target {tgt}: lost the board", flush=True)
            c, g = local_search(rig)
            continue
        if not rig.capture(c, g, label):
            off = c.mean(0) / (W, H)
            why = "board cut off by the frame edge" if not rig.inside(c) else "pose too similar"
            print(f"    target {tgt}: reached ({off[0]:.2f}, {off[1]:.2f}) but {why}", flush=True)
    rig.servo(config.SERVO_PAN_CENTER, rig.home_tilt)


SPEED_MPS = 0.28          # speed 50, heading-held, carpet (sonar line runs 2026-10-11)


def circle_move(rig, dist_m, a_from, a_to):
    """Move around the board on a circle of radius dist_m, from viewing angle
    a_from to a_to (deg; + = to the robot's right of the board's normal), using
    only a turn and a heading-held forward drive (strafing barely works on
    carpet). Ends facing the board."""
    from raspbot_slam.heading import drive_straight, turn_to
    p0 = dist_m * np.array([math.sin(math.radians(a_from)), -math.cos(math.radians(a_from))])
    p1 = dist_m * np.array([math.sin(math.radians(a_to)), -math.cos(math.radians(a_to))])
    u = -p0 / np.linalg.norm(p0)                      # currently facing the board
    v = p1 - p0
    phi = math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], u @ v))   # CCW +
    chord = float(np.linalg.norm(v))
    rig.servo(config.SERVO_PAN_CENTER, rig.tilt, settle=0.2)
    turn_to(rig.act, rig.hd, rig.hd.heading_deg + phi)
    if rig.clear_mm() < (chord + 0.2) * 1000:
        print(f"  circle: blocked ({rig.clear_mm()} mm ahead), skipping {a_from:+.0f} -> {a_to:+.0f}",
              flush=True)
        turn_to(rig.act, rig.hd, rig.hd.heading_deg - phi)
        return False
    drive_straight(rig.act, rig.hd, 50, chord / SPEED_MPS)
    time.sleep(0.3)
    c, _ = local_search(rig)
    if c is None and not search(rig):
        return False
    return face_board(rig)


def oblique_views(rig, dist_m, angles=(30, 45)):
    """Angled views on both sides: what pins down the focal length."""
    for side in (1, -1):
        prev = 0.0
        for a in angles:
            if not circle_move(rig, dist_m, prev, side * a):
                break
            prev = side * a
            goto_distance(rig, dist_m, tol=0.08)
            c, _ = rig.board()
            if c is not None:
                _, yaw, _ = rig.pnp(c)
                print(f"  oblique {side * a:+d}: board seen at {yaw:+.0f} deg", flush=True)
            sweep_station(rig, f"oblique {side * a:+d}", TARGETS[:5])
        if prev:
            circle_move(rig, dist_m, prev, 0.0)


def orbit(rig, side, marks=(15, 28, 40), dist_m=0.5, max_hops=16):
    """Strafe to `side` (-1 left, +1 right) in bursts, turning after each to keep
    facing the target, so the robot walks an arc around it. Capture a set of
    views each time the oblique angle passes the next mark."""
    marks = list(marks)
    for hop in range(max_hops):
        rig.hop(strafe=side, seconds=0.5, speed=70)
        if not face_board(rig):
            c, _ = local_search(rig)
            if c is None or not face_board(rig):
                print("  orbit: lost the target", flush=True)
                return
        c, _ = rig.board()
        if c is None:
            continue
        d, yaw, _ = rig.pnp(c)
        print(f"  orbit {'R' if side > 0 else 'L'} hop {hop}: {d:.2f} m, target seen at {yaw:+.0f} deg",
              flush=True)
        if marks and abs(yaw) >= marks[0]:
            sweep_station(rig, f"orbit {yaw:+.0f}", TARGETS[:5])
            while marks and abs(yaw) >= marks[0]:
                marks.pop(0)
            if not marks:
                return
        if abs(d - dist_m) > 0.12:
            goto_distance(rig, dist_m, tol=0.08)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Autonomous camera calibration (robot moves, board fixed).")
    p.add_argument("--target", default="board",
                   help="'board' (checkerboard) or 'tag' / 'tag:ID' (a wall ArUco tag)")
    p.add_argument("--distances", default=None, help="station distances (m), comma-separated")
    p.add_argument("--no-orbit", action="store_true")
    p.add_argument("--turn-first", type=float, default=0.0,
                   help="turn the body this many degrees (CCW +) before searching")
    args = p.parse_args(argv)
    save_dir = os.path.join("datasets", "camera_cal", time.strftime("%Y%m%d_%H%M%S") + "_auto")
    os.makedirs(save_dir, exist_ok=True)
    if args.target.startswith("tag"):
        ids = [int(args.target.split(":")[1])] if ":" in args.target else None
        target = TagTarget(ids)
    else:
        target = CheckerTarget()
    rig = Rig(save_dir, target)
    t0 = time.monotonic()
    try:
        if args.turn_first:
            print(f"turning {rig.rotate(args.turn_first):+.0f} deg first", flush=True)
        if not search(rig):
            print("board not found anywhere around the robot")
            return 1
        face_board(rig)
        dists = ([float(v) for v in args.distances.split(",")] if args.distances
                 else list(target.near_m))
        for d in dists:
            got = goto_distance(rig, d)
            if got is None:
                face_board(rig)
                got = goto_distance(rig, d)
            if got is None:
                print(f"  lost the board going to {d} m", flush=True)
                continue
            print(f"station {got:.2f} m", flush=True)
            face_board(rig)
            sweep_station(rig, f"{got:.2f} m")
        if not args.no_orbit:
            mid = dists[len(dists) // 2]
            goto_distance(rig, mid)
            face_board(rig)
            oblique_views(rig, mid)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        rig.act.stop()
        rig.hd.stop()
        rig.servo(config.SERVO_PAN_CENTER, config.SERVO_TILT_REST, settle=0.2)
        rig.cap.release()
    print(f"\n{len(rig.views)} views in {time.monotonic() - t0:.0f} s", flush=True)
    objp = [rig.objp] * len(rig.views)
    cc.finish(objp, rig.views, rig.descs, (W, H), save_dir,
              flags=target.calib_flags, target=target.name, K0=target.K0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
