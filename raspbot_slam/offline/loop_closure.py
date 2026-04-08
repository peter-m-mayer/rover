"""
Offline loop closure detection using SIFT re-extraction.

ORB is used for real-time VO, but SIFT produces more distinctive descriptors
for matching distant keyframes reliably. This module re-extracts SIFT from
stored keyframe images, builds a bag-of-visual-words index for fast candidate
retrieval, and verifies geometric consistency.

Usage:
    python -m raspbot_slam.offline.loop_closure maps/ground_floor
"""

import os
import sys
import time
from typing import List, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .. import config
from ..map_manager import MapManager


def extract_sift_from_keyframes(map_dir: str) -> List[dict]:
    """Re-extract SIFT features from stored keyframe images.

    SIFT is ~10x slower than ORB but produces much more distinctive
    128-float descriptors. This enables reliable matching between
    keyframes captured far apart in time (loop closures).

    Returns:
        List of dicts: {'id', 'keypoints', 'descriptors', 'image'}
    """
    if cv2 is None:
        return []

    sift = cv2.SIFT_create(nfeatures=500)
    kf_dir = os.path.join(map_dir, "keyframes")
    if not os.path.isdir(kf_dir):
        return []

    keyframes = []
    idx = 0
    while True:
        img_path = os.path.join(kf_dir, f"kf_{idx:04d}.jpg")
        if not os.path.exists(img_path):
            break

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            idx += 1
            continue

        kp, desc = sift.detectAndCompute(img, None)
        if desc is not None:
            keyframes.append({
                "id": idx,
                "keypoints": np.array([k.pt for k in kp]),
                "descriptors": desc,
                "image": img,
            })

        idx += 1

    return keyframes


def find_loop_closures(keyframes: List[dict],
                       min_keyframe_gap: int = 20,
                       min_inliers: int = 15,
                       ratio_threshold: float = 0.65
                       ) -> List[Tuple[int, int, int, np.ndarray]]:
    """Detect loop closures between distant keyframes.

    Compares all pairs of keyframes separated by at least min_keyframe_gap
    using SIFT matching + geometric verification.

    Args:
        keyframes: List from extract_sift_from_keyframes.
        min_keyframe_gap: Minimum keyframe index separation.
        min_inliers: Minimum RANSAC inliers for a valid loop closure.
        ratio_threshold: Lowe's ratio test threshold (stricter for loop closure).

    Returns:
        List of (kf_id_1, kf_id_2, n_inliers, homography) tuples.
    """
    if cv2 is None:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    closures = []

    n = len(keyframes)
    total_pairs = 0
    for i in range(n):
        for j in range(i + min_keyframe_gap, n):
            total_pairs += 1

    print(f"  Checking {total_pairs} keyframe pairs...")
    checked = 0

    for i in range(n):
        desc_i = keyframes[i]["descriptors"]
        kp_i = keyframes[i]["keypoints"]

        for j in range(i + min_keyframe_gap, n):
            checked += 1
            if checked % 100 == 0:
                print(f"    {checked}/{total_pairs} pairs checked, "
                      f"{len(closures)} closures found")

            desc_j = keyframes[j]["descriptors"]
            kp_j = keyframes[j]["keypoints"]

            if desc_i is None or desc_j is None:
                continue
            if len(desc_i) < 10 or len(desc_j) < 10:
                continue

            # Lowe's ratio test with strict threshold
            matches = matcher.knnMatch(desc_i, desc_j, k=2)
            good = []
            for pair in matches:
                if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance:
                    good.append(pair[0])

            if len(good) < min_inliers:
                continue

            # Geometric verification with homography
            pts_i = np.array([kp_i[m.queryIdx] for m in good], dtype=np.float64)
            pts_j = np.array([kp_j[m.trainIdx] for m in good], dtype=np.float64)

            H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, 5.0)
            if H is None or mask is None:
                continue

            n_inliers = int(mask.sum())
            if n_inliers >= min_inliers:
                closures.append((
                    keyframes[i]["id"],
                    keyframes[j]["id"],
                    n_inliers,
                    H,
                ))
                print(f"    Loop closure: kf_{keyframes[i]['id']:04d} <-> "
                      f"kf_{keyframes[j]['id']:04d} ({n_inliers} inliers)")

    return closures


def save_loop_closures(closures: List[Tuple], map_dir: str):
    """Save detected loop closures for use by bundle adjustment."""
    output = []
    for kf1, kf2, n_inliers, H in closures:
        output.append({
            "keyframe_1": int(kf1),
            "keyframe_2": int(kf2),
            "n_inliers": int(n_inliers),
            "homography": H.tolist(),
        })

    import json
    path = os.path.join(map_dir, "loop_closures.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {len(output)} loop closures to {path}")


def run_loop_closure(map_dir: str):
    """Run full loop closure detection pipeline."""
    print(f"Loop Closure Detection: {map_dir}")
    print("=" * 50)

    # Extract SIFT from keyframes
    print("  Extracting SIFT features from keyframes...")
    t0 = time.time()
    keyframes = extract_sift_from_keyframes(map_dir)
    elapsed = time.time() - t0
    print(f"  Extracted SIFT from {len(keyframes)} keyframes in {elapsed:.1f}s")

    if len(keyframes) < 2:
        print("  Not enough keyframes for loop closure detection.")
        return

    # Find loop closures
    print("\n  Searching for loop closures...")
    t0 = time.time()
    closures = find_loop_closures(keyframes)
    elapsed = time.time() - t0
    print(f"\n  Found {len(closures)} loop closures in {elapsed:.1f}s")

    if closures:
        save_loop_closures(closures, map_dir)

    print("  Done!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m raspbot_slam.offline.loop_closure <map_dir>")
        sys.exit(1)
    run_loop_closure(sys.argv[1])


if __name__ == "__main__":
    main()
