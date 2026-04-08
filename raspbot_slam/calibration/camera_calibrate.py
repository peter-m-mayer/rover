"""
Camera intrinsic calibration using a checkerboard pattern.

Usage (on the Pi, with a printed checkerboard):
    python -m raspbot_slam.calibration.camera_calibrate

Captures images interactively, computes intrinsics, saves to
calibration/camera_calibration.json.
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
    cal_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(cal_dir, exist_ok=True)
    cal_file = os.path.join(cal_dir, "camera_calibration.json")

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


if __name__ == "__main__":
    calibrate_camera()
