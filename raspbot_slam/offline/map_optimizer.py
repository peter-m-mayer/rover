"""
Post-processing map optimization.

After bundle adjustment and loop closure, this module:
1. Reprojects depth observations using corrected poses to refine the occupancy grid
2. Smooths the trajectory using a moving average
3. Removes duplicate/outlier landmarks
4. Computes map quality statistics

Usage:
    python -m raspbot_slam.offline.map_optimizer maps/ground_floor
"""

import os
import sys
import json
from typing import Dict
import numpy as np

from .. import config
from ..map_manager import MapManager, OccupancyGrid, Landmark


def remove_outlier_landmarks(mm: MapManager, min_observations: int = 2,
                             max_covariance_trace: float = 10.0) -> int:
    """Remove landmarks with too few observations or excessive uncertainty.

    Returns:
        Number of landmarks removed.
    """
    to_remove = []
    for lm_id, lm in mm.landmarks.items():
        if lm.observation_count < min_observations:
            to_remove.append(lm_id)
            continue
        if np.trace(lm.covariance) > max_covariance_trace:
            to_remove.append(lm_id)

    for lm_id in to_remove:
        del mm.landmarks[lm_id]

    return len(to_remove)


def merge_duplicate_landmarks(mm: MapManager,
                              spatial_threshold: float = 0.10,
                              descriptor_threshold: int = 40) -> int:
    """Merge landmarks that are very close spatially with similar descriptors.

    Keeps the landmark with more observations, absorbs the other's count.

    Returns:
        Number of landmarks merged.
    """
    lm_list = list(mm.landmarks.values())
    to_remove = set()
    merged = 0

    for i in range(len(lm_list)):
        if lm_list[i].id in to_remove:
            continue
        for j in range(i + 1, len(lm_list)):
            if lm_list[j].id in to_remove:
                continue

            # Spatial distance
            dist = np.linalg.norm(
                lm_list[i].position_3d - lm_list[j].position_3d)
            if dist > spatial_threshold:
                continue

            # Descriptor distance
            desc_dist = int(np.sum(np.unpackbits(
                np.bitwise_xor(lm_list[i].descriptor, lm_list[j].descriptor))))
            if desc_dist > descriptor_threshold:
                continue

            # Merge: keep the one with more observations
            if lm_list[i].observation_count >= lm_list[j].observation_count:
                keeper, victim = lm_list[i], lm_list[j]
            else:
                keeper, victim = lm_list[j], lm_list[i]

            # Average positions weighted by observation count
            w_k = keeper.observation_count
            w_v = victim.observation_count
            keeper.position_3d = (
                keeper.position_3d * w_k + victim.position_3d * w_v
            ) / (w_k + w_v)
            keeper.observation_count += victim.observation_count

            to_remove.add(victim.id)
            merged += 1

    for lm_id in to_remove:
        del mm.landmarks[lm_id]

    return merged


def smooth_trajectory(mm: MapManager, window: int = 5) -> np.ndarray:
    """Apply moving average smoothing to the trajectory.

    Preserves start and end points. Smooths x, y independently.
    Angle is smoothed via sin/cos decomposition to handle wraparound.

    Returns:
        Smoothed trajectory as Nx3 array.
    """
    if len(mm.trajectory) < window:
        return np.array(mm.trajectory)

    traj = np.array(mm.trajectory)
    n = len(traj)
    smoothed = traj.copy()

    half = window // 2
    for i in range(half, n - half):
        segment = traj[i - half:i + half + 1]
        smoothed[i, 0] = np.mean(segment[:, 0])  # x
        smoothed[i, 1] = np.mean(segment[:, 1])  # y
        # Smooth angle via sin/cos
        sin_avg = np.mean(np.sin(segment[:, 2]))
        cos_avg = np.mean(np.cos(segment[:, 2]))
        smoothed[i, 2] = np.arctan2(sin_avg, cos_avg)

    mm.trajectory = [smoothed[i] for i in range(n)]
    return smoothed


def rebuild_occupancy_grid(mm: MapManager):
    """Rebuild the occupancy grid from the optimized trajectory.

    Marks all trajectory cells as free. This doesn't reproject depth
    (we'd need the raw observations for that), but it ensures the
    traversed path is marked correctly after pose optimization.
    """
    mm.occupancy = OccupancyGrid()
    for pose in mm.trajectory:
        mm.occupancy.mark_traversed(pose[0], pose[1])


def compute_statistics(mm: MapManager) -> Dict:
    """Compute map quality statistics."""
    stats = {
        "n_landmarks": len(mm.landmarks),
        "n_trajectory_poses": len(mm.trajectory),
    }

    if mm.landmarks:
        obs_counts = [lm.observation_count for lm in mm.landmarks.values()]
        stats["landmark_obs_mean"] = float(np.mean(obs_counts))
        stats["landmark_obs_median"] = float(np.median(obs_counts))
        stats["landmark_obs_max"] = int(np.max(obs_counts))

        cov_traces = [float(np.trace(lm.covariance)) for lm in mm.landmarks.values()]
        stats["landmark_cov_trace_mean"] = float(np.mean(cov_traces))
        stats["landmark_cov_trace_median"] = float(np.median(cov_traces))

        positions = np.array([lm.position_3d for lm in mm.landmarks.values()])
        stats["map_extent_x"] = float(positions[:, 0].max() - positions[:, 0].min())
        stats["map_extent_y"] = float(positions[:, 1].max() - positions[:, 1].min())

    grid = mm.occupancy
    prob = grid.get_probability_grid()
    stats["grid_free_cells"] = int(np.sum(prob < 0.4))
    stats["grid_occupied_cells"] = int(np.sum(prob > 0.6))
    stats["grid_unknown_cells"] = int(np.sum((prob >= 0.4) & (prob <= 0.6)))
    stats["grid_free_area_m2"] = stats["grid_free_cells"] * grid.resolution**2
    stats["grid_occupied_area_m2"] = stats["grid_occupied_cells"] * grid.resolution**2

    if mm.trajectory and len(mm.trajectory) > 1:
        traj = np.array(mm.trajectory)
        diffs = np.diff(traj[:, :2], axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        stats["trajectory_length_m"] = float(np.sum(dists))

    return stats


def run_map_optimizer(map_dir: str):
    """Run full map optimization pipeline."""
    print(f"Map Optimization: {map_dir}")
    print("=" * 50)

    mm = MapManager()
    mm.load(map_dir)
    print(f"  Loaded {mm.landmark_count} landmarks, "
          f"{len(mm.trajectory)} trajectory poses")

    # Remove outliers
    n_removed = remove_outlier_landmarks(mm)
    print(f"  Removed {n_removed} outlier landmarks")

    # Merge duplicates
    n_merged = merge_duplicate_landmarks(mm)
    print(f"  Merged {n_merged} duplicate landmarks")

    # Smooth trajectory
    smooth_trajectory(mm, window=5)
    print(f"  Trajectory smoothed (window=5)")

    # Rebuild occupancy from optimized trajectory
    rebuild_occupancy_grid(mm)
    print(f"  Occupancy grid rebuilt from trajectory")

    # Statistics
    stats = compute_statistics(mm)
    print(f"\n  Map Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")

    # Save
    mm.save(map_dir)
    stats_path = os.path.join(map_dir, "statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved optimized map and statistics to {map_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m raspbot_slam.offline.map_optimizer <map_dir>")
        sys.exit(1)
    run_map_optimizer(sys.argv[1])


if __name__ == "__main__":
    main()
