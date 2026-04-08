"""
Offline bundle adjustment: jointly optimizes all keyframe poses and
landmark positions to minimize total reprojection error.

This is too expensive for real-time on Pi 5 but dramatically improves
map quality when run after a mapping session. Uses scipy.optimize
for the nonlinear least-squares problem.

Usage:
    python -m raspbot_slam.offline.bundle_adjustment maps/ground_floor
"""

import os
import sys
import time
from typing import List, Tuple, Dict
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
except ImportError:
    least_squares = None
    lil_matrix = None

from .. import config
from ..map_manager import MapManager
from ..feature_extractor import FeatureExtractor


def load_keyframes(map_dir: str) -> List[dict]:
    """Load all keyframes from disk.

    Returns:
        List of dicts with keys: 'id', 'descriptors', 'keypoints', 'image_path'.
    """
    kf_dir = os.path.join(map_dir, "keyframes")
    if not os.path.isdir(kf_dir):
        return []

    keyframes = []
    idx = 0
    while True:
        prefix = os.path.join(kf_dir, f"kf_{idx:04d}")
        desc_path = f"{prefix}_desc.npy"
        kp_path = f"{prefix}_kp.npy"
        img_path = f"{prefix}.jpg"

        if not os.path.exists(desc_path):
            break

        kf = {
            "id": idx,
            "descriptors": np.load(desc_path),
            "keypoints": np.load(kp_path),
            "image_path": img_path if os.path.exists(img_path) else None,
        }
        keyframes.append(kf)
        idx += 1

    return keyframes


def build_observation_graph(keyframes: List[dict], map_mgr: MapManager,
                            fe: FeatureExtractor
                            ) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
    """Build the observation graph for bundle adjustment.

    Matches keyframe features against landmarks to establish
    2D-3D correspondences across the full trajectory.

    Returns:
        observations: List of (keyframe_idx, landmark_id, u, v) tuples
        initial_poses: Nx6 array (angle-axis rotation + translation per keyframe)
        initial_landmarks: Mx3 array (3D positions)
    """
    observations = []

    # Use trajectory poses as initial keyframe poses
    traj = np.array(map_mgr.trajectory) if map_mgr.trajectory else np.zeros((0, 3))
    n_kf = len(keyframes)

    # Sample trajectory at keyframe intervals
    initial_poses = np.zeros((n_kf, 6))  # [rx, ry, rz, tx, ty, tz]
    if len(traj) > 0:
        step = max(1, len(traj) // n_kf)
        for i in range(n_kf):
            t_idx = min(i * step, len(traj) - 1)
            x, y, theta = traj[t_idx]
            initial_poses[i] = [0, theta, 0, x, 0, y]  # rotation about Y axis

    # Build landmark array
    lm_ids = sorted(map_mgr.landmarks.keys())
    lm_id_to_idx = {lm_id: idx for idx, lm_id in enumerate(lm_ids)}
    initial_landmarks = np.array([
        map_mgr.landmarks[lm_id].position_3d for lm_id in lm_ids
    ])

    # Match each keyframe against landmarks
    for kf_idx, kf in enumerate(keyframes):
        desc = kf["descriptors"]
        kp = kf["keypoints"]

        if desc is None or len(desc) == 0:
            continue

        for i in range(len(desc)):
            match = map_mgr.match_observation(desc[i], max_descriptor_distance=50)
            if match is not None and match.id in lm_id_to_idx:
                lm_idx = lm_id_to_idx[match.id]
                u, v = kp[i]
                observations.append((kf_idx, lm_idx, float(u), float(v)))

    return observations, initial_poses, initial_landmarks


def reprojection_residuals(params, n_poses, n_landmarks, observations, K):
    """Compute reprojection errors for all observations.

    Args:
        params: Flattened array of [poses (n*6), landmarks (m*3)].
        n_poses: Number of camera poses.
        n_landmarks: Number of landmarks.
        observations: List of (pose_idx, lm_idx, u, v).
        K: 3x3 camera intrinsic matrix.

    Returns:
        Residual vector (2 * n_observations).
    """
    poses = params[:n_poses * 6].reshape(n_poses, 6)
    landmarks = params[n_poses * 6:].reshape(n_landmarks, 3)

    residuals = np.zeros(len(observations) * 2)

    for i, (pose_idx, lm_idx, u_obs, v_obs) in enumerate(observations):
        # Camera pose: angle-axis rotation + translation
        rvec = poses[pose_idx, :3]
        tvec = poses[pose_idx, 3:]

        # Rotation matrix from angle-axis
        R, _ = cv2.Rodrigues(rvec)

        # Project landmark into camera
        pt_world = landmarks[lm_idx]
        pt_cam = R @ pt_world + tvec

        if pt_cam[2] <= 0:
            # Behind camera -- large residual to push optimizer away
            residuals[2 * i] = 100.0
            residuals[2 * i + 1] = 100.0
            continue

        # Project to pixel coordinates
        u_proj = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
        v_proj = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]

        residuals[2 * i] = u_proj - u_obs
        residuals[2 * i + 1] = v_proj - v_obs

    return residuals


def build_sparsity_matrix(n_poses, n_landmarks, observations):
    """Build the Jacobian sparsity pattern for efficient optimization."""
    if lil_matrix is None:
        return None

    n_obs = len(observations)
    n_params = n_poses * 6 + n_landmarks * 3
    A = lil_matrix((n_obs * 2, n_params), dtype=int)

    for i, (pose_idx, lm_idx, _, _) in enumerate(observations):
        # Each observation depends on 6 pose params and 3 landmark params
        pose_start = pose_idx * 6
        lm_start = n_poses * 6 + lm_idx * 3

        for j in range(6):
            A[2 * i, pose_start + j] = 1
            A[2 * i + 1, pose_start + j] = 1
        for j in range(3):
            A[2 * i, lm_start + j] = 1
            A[2 * i + 1, lm_start + j] = 1

    return A


def run_bundle_adjustment(map_dir: str, camera_calibration_file: str = None):
    """Run full bundle adjustment on a stored map.

    Args:
        map_dir: Path to the map directory.
        camera_calibration_file: Path to camera calibration JSON.
    """
    if cv2 is None:
        print("ERROR: OpenCV required for bundle adjustment")
        return
    if least_squares is None:
        print("ERROR: scipy required for bundle adjustment")
        print("Install with: pip install scipy")
        return

    print(f"Bundle Adjustment: {map_dir}")
    print("=" * 50)

    # Load map and camera calibration
    from ..camera import Camera
    cam = Camera(calibration_file=camera_calibration_file)
    K = cam.K

    mm = MapManager()
    mm.load(map_dir)
    print(f"  Loaded {mm.landmark_count} landmarks, "
          f"{len(mm.trajectory)} trajectory poses")

    # Load keyframes
    fe = FeatureExtractor()
    keyframes = load_keyframes(map_dir)
    print(f"  Loaded {len(keyframes)} keyframes")

    if len(keyframes) < 2 or mm.landmark_count < 3:
        print("  Not enough data for bundle adjustment.")
        return

    # Build observation graph
    print("  Building observation graph...")
    observations, initial_poses, initial_landmarks = build_observation_graph(
        keyframes, mm, fe)
    print(f"  {len(observations)} observations")

    if len(observations) < 10:
        print("  Not enough observations for bundle adjustment.")
        return

    n_poses = len(keyframes)
    n_landmarks = len(initial_landmarks)

    # Flatten initial parameters
    x0 = np.hstack([initial_poses.ravel(), initial_landmarks.ravel()])

    # Build sparsity pattern
    print("  Building sparsity pattern...")
    A = build_sparsity_matrix(n_poses, n_landmarks, observations)

    # Run optimization
    print(f"  Optimizing {n_poses} poses + {n_landmarks} landmarks "
          f"({len(x0)} parameters)...")
    t0 = time.time()

    result = least_squares(
        reprojection_residuals,
        x0,
        jac_sparsity=A,
        verbose=2,
        x_scale='jac',
        ftol=1e-4,
        method='trf',
        args=(n_poses, n_landmarks, observations, K),
        max_nfev=100,
    )

    elapsed = time.time() - t0
    print(f"\n  Optimization completed in {elapsed:.1f}s")
    print(f"  Cost: {result.cost:.4f} -> final")
    print(f"  Iterations: {result.nfev}")

    # Extract optimized parameters
    opt_poses = result.x[:n_poses * 6].reshape(n_poses, 6)
    opt_landmarks = result.x[n_poses * 6:].reshape(n_landmarks, 3)

    # Update map landmarks with optimized positions
    lm_ids = sorted(mm.landmarks.keys())
    for idx, lm_id in enumerate(lm_ids):
        if idx < len(opt_landmarks):
            mm.landmarks[lm_id].position_3d = opt_landmarks[idx]

    # Update trajectory with optimized poses
    mm.trajectory = []
    for i in range(n_poses):
        x = opt_poses[i, 3]
        y = opt_poses[i, 5]
        theta = opt_poses[i, 1]  # rotation about Y
        mm.trajectory.append(np.array([x, y, theta]))

    # Save optimized map
    print(f"\n  Saving optimized map to {map_dir}...")
    mm.save(map_dir)
    print("  Done!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m raspbot_slam.offline.bundle_adjustment <map_dir>")
        sys.exit(1)
    run_bundle_adjustment(sys.argv[1])


if __name__ == "__main__":
    main()
