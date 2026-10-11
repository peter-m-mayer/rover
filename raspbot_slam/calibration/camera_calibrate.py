"""
Camera intrinsic calibration using a checkerboard pattern.

Usage (on the Pi, with the printed checkerboard from make_targets.py):
    python3 -m raspbot_slam.calibration.camera_calibrate --auto           # headless, over SSH
    python3 -m raspbot_slam.calibration.camera_calibrate --from-dir DIR   # re-solve offline
    python3 -m raspbot_slam.calibration.camera_calibrate                  # interactive (needs a display)

--auto captures by itself whenever the board is held still in a pose that
differs from the ones already captured, and prints which parts of the image
still need coverage. Frames are kept, so the solve can be redone offline.
Board size comes from config.CHECKER_INNER_CORNERS / CHECKER_SQUARE_MM (the
size as printed). Saves raspbot_slam/calibration/camera_calibration.json.
"""

import os
import sys
import json
import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV required. Install with: pip install opencv-python")
    sys.exit(1)

from .. import config


CAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_calibration.json")
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def board_points(board_size, square_size_mm):
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    return objp * square_size_mm


def pose_descriptor(corners, board_size, image_size):
    """Where / how big / how tilted the board looks: (cx, cy, size, tilt_lr, tilt_ud).

    Centre and size are fractions of the image; the tilts are log ratios of
    opposite edge lengths (0 = facing the camera squarely)."""
    c = corners.reshape(board_size[1], board_size[0], 2)
    w, h = image_size
    cx, cy = c[..., 0].mean() / w, c[..., 1].mean() / h
    top, bottom = np.linalg.norm(c[0, -1] - c[0, 0]), np.linalg.norm(c[-1, -1] - c[-1, 0])
    left, right = np.linalg.norm(c[-1, 0] - c[0, 0]), np.linalg.norm(c[-1, -1] - c[0, -1])
    size = np.sqrt(0.5 * (top + bottom) * 0.5 * (left + right) / (w * h))
    return np.array([cx, cy, size, np.log(right / left), np.log(bottom / top)])


def _region(d):
    col = ("left", "centre", "right")[min(2, max(0, int(d[0] * 3)))]
    row = ("top", "middle", "bottom")[min(2, max(0, int(d[1] * 3)))]
    return row, col


def solve(obj_points, img_points, image_size, flags=0, K0=None):
    K_init = None if K0 is None else np.array(K0, dtype=np.float64)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, K_init, None,
                                                     flags=flags)
    per_view = []
    for op, ip, rv, tv in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        err = proj.reshape(-1, 2) - np.asarray(ip).reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))))
    return rms, K, dist, per_view


def save(K, dist, rms, n, image_size, path=CAL_FILE, **extra):
    data = {"camera_matrix": K.tolist(), "dist_coeffs": dist.ravel().tolist(),
            "image_size": list(image_size), "reprojection_error": float(rms),
            "n_calibration_images": n, "board_square_mm": config.CHECKER_SQUARE_MM, **extra}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def calibrate_auto(seconds=240, n_images=20, still_px=1.0, min_diff=0.12,
                   save_dir=None, device=None, on_capture=None):
    """Headless capture: keep frames where the board is still and in a new pose."""
    import time
    board = tuple(config.CHECKER_INNER_CORNERS)
    objp = board_points(board, config.CHECKER_SQUARE_MM)
    size = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    save_dir = save_dir or os.path.join("datasets", "camera_cal", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(device if device is not None else config.CAMERA_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return None
    print(f"auto-capture: {board[0]}x{board[1]} corners, {config.CHECKER_SQUARE_MM} mm squares; "
          f"up to {seconds:.0f} s or {n_images} views -> {save_dir}", flush=True)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    obj_points, img_points, descs = [], [], []
    prev, last_note = None, 0.0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end and len(img_points) < n_images:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board, flags=flags)
        now = time.monotonic()
        if not found:
            prev = None
            if now - last_note > 10:
                print(f"  ... no board in view ({len(img_points)} captured)", flush=True)
                last_note = now
            continue
        corners = corners.reshape(-1, 1, 2)   # OpenCV 5 returns (N, 2), 4.x (N, 1, 2)
        still = prev is not None and np.mean(np.linalg.norm((corners - prev).reshape(-1, 2), axis=1)) < still_px
        prev = corners
        if not still:
            continue
        d = pose_descriptor(corners, board, size)
        if descs and min(np.linalg.norm(d - e) for e in descs) < min_diff:
            if now - last_note > 10:
                print(f"  ... same pose as an earlier one: move or tilt the board "
                      f"({len(img_points)} captured)", flush=True)
                last_note = now
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        cv2.imwrite(os.path.join(save_dir, f"view_{len(img_points):02d}.png"), gray)
        obj_points.append(objp)
        img_points.append(corners)
        descs.append(d)
        row, col = _region(d)
        print(f"  captured {len(img_points):2d}: {row}-{col}, size {d[2]:.0%}, "
              f"tilt L/R {d[3]:+.2f} U/D {d[4]:+.2f}", flush=True)
        last_note = now
        if on_capture is not None:
            on_capture(len(img_points))
    cap.release()
    return finish(obj_points, img_points, descs, size, save_dir)


def sanity(K, size):
    """Reasons a solution is physically implausible for this camera (empty = fine)."""
    w, h = size
    why = []
    if abs(K[0, 0] / K[1, 1] - 1) > 0.05:
        why.append(f"fx/fy = {K[0, 0] / K[1, 1]:.2f} (square pixels expected)")
    if abs(K[0, 2] - w / 2) > 0.15 * w or abs(K[1, 2] - h / 2) > 0.15 * h:
        why.append(f"principal point ({K[0, 2]:.0f}, {K[1, 2]:.0f}) far from the image centre")
    hfov = np.degrees(2 * np.arctan(w / 2 / K[0, 0]))
    if not 45 <= hfov <= 110:
        why.append(f"HFOV {hfov:.0f} deg outside 45-110")
    return why


def finish(obj_points, img_points, descs, size, save_dir, flags=0, target="checkerboard", K0=None):
    if len(img_points) < 6:
        print(f"ERROR: only {len(img_points)} views; need >= 6 (aim for 15-20)")
        return None
    covered = {_region(d) for d in descs}
    missing = [f"{r}-{c}" for r in ("top", "middle", "bottom")
               for c in ("left", "centre", "right") if (r, c) not in covered]
    rms, K, dist, per_view = solve(obj_points, img_points, size, flags, K0)
    worst = int(np.argmax(per_view))
    print(f"\nRMS reprojection {rms:.3f} px over {len(img_points)} views "
          f"(worst view {worst}: {per_view[worst]:.2f} px)")
    print(f"fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}  "
          f"-> HFOV {np.degrees(2 * np.arctan(size[0] / 2 / K[0, 0])):.1f} deg")
    print(f"distortion {np.round(dist.ravel(), 4).tolist()}")
    if missing:
        print(f"not covered: {', '.join(missing)} (image corners matter most for distortion)")
    tilts = np.abs(np.array(descs)[:, 3:]).max(axis=0)
    if tilts.max() < 0.15:
        print("WARN: little board tilt; tilted views pin down the focal length")
    bad = sanity(K, size)
    kw = dict(frames_dir=save_dir, target=target, per_view_px=[round(v, 3) for v in per_view],
              missing_regions=missing)
    if bad:
        path = save(K, dist, rms, len(img_points), size,
                    path=os.path.join(save_dir, "camera_calibration.rejected.json"), rejected=bad, **kw)
        print("REJECTED (not installed): " + "; ".join(bad))
        print(f"kept for inspection: {path}")
        return None
    path = save(K, dist, rms, len(img_points), size, **kw)
    print(f"saved {path}  ({'PASS' if rms < 0.5 else 'CHECK'}: target < 0.5 px)")
    return K, dist, rms


def calibrate_from_dir(path):
    """Re-solve from frames saved by --auto (e.g. after deleting a bad view)."""
    import glob
    board = tuple(config.CHECKER_INNER_CORNERS)
    objp = board_points(board, config.CHECKER_SQUARE_MM)
    obj_points, img_points, descs, size = [], [], [], None
    for f in sorted(glob.glob(os.path.join(path, "view_*.png"))):
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCornersSB(
            gray, board, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not found:
            print(f"  skip {os.path.basename(f)}: board not found")
            continue
        img_points.append(corners.reshape(-1, 1, 2).astype(np.float32))
        obj_points.append(objp)
        descs.append(pose_descriptor(corners, board, size))
    return finish(obj_points, img_points, descs, size, path)


def calibrate_camera(board_size=(9, 6), square_size_mm=25.0,
                     n_images=15, device=None):
    """Interactive camera calibration.

    Args:
        board_size: Inner corners of the checkerboard (columns, rows).
        square_size_mm: Size of one square in mm.
        n_images: Number of images to capture.
        device: Camera device index.
    """
    device = device if device is not None else config.CAMERA_DEVICE

    # Prepare object points (3D checkerboard corners in a flat plane)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera device {device}")
        return

    print(f"Camera calibration: capture {n_images} images of a "
          f"{board_size[0]}x{board_size[1]} checkerboard.")
    print("Press SPACE to capture, 'q' to quit early.\n")

    captured = 0
    while captured < n_images:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, board_size, corners, found)
            cv2.putText(display, f"Board detected! ({captured}/{n_images})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"No board found ({captured}/{n_images})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' ') and found:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            obj_points.append(objp)
            img_points.append(corners_refined)
            captured += 1
            print(f"  Captured image {captured}/{n_images}")

    cap.release()
    cv2.destroyAllWindows()

    if captured < 3:
        print("ERROR: Need at least 3 images for calibration.")
        return

    # Calibrate
    print(f"\nCalibrating with {captured} images...")
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        None, None)

    print(f"  Reprojection error: {ret:.4f} pixels")
    print(f"  Camera matrix:\n{K}")
    print(f"  Distortion: {dist_coeffs.ravel()}")

    # Save
    cal_file = CAL_FILE

    data = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
        "image_size": [config.CAMERA_WIDTH, config.CAMERA_HEIGHT],
        "reprojection_error": float(ret),
        "n_calibration_images": captured,
    }
    with open(cal_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {cal_file}")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Camera intrinsics from the printed checkerboard.")
    p.add_argument("--auto", action="store_true", help="headless auto-capture (no display)")
    p.add_argument("--from-dir", help="re-solve from frames saved by --auto")
    p.add_argument("--seconds", type=float, default=240)
    p.add_argument("--n", type=int, default=20, help="views to capture")
    args = p.parse_args(argv)
    if args.from_dir:
        calibrate_from_dir(args.from_dir)
    elif args.auto:
        beep = None
        try:   # robot buzzer = "captured, move to the next pose" for whoever holds the board
            from catchaser.hw import connect_bot
            from raspbot_slam.actuators import Actuators
            bot = connect_bot()
            if bot is not None:
                act = Actuators(bot=bot)
                beep = lambda n: act.beep(0.08)
        except Exception:
            pass
        calibrate_auto(seconds=args.seconds, n_images=args.n, on_capture=beep)
    else:
        calibrate_camera(board_size=tuple(config.CHECKER_INNER_CORNERS),
                         square_size_mm=config.CHECKER_SQUARE_MM)


if __name__ == "__main__":
    main()
