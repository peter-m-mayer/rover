"""
Visualize a stored SLAM map: occupancy grid, landmarks, and trajectory.

Usage:
    python -m raspbot_slam.visualize_map maps/ground_floor

Works on any machine (no hardware needed) -- just needs matplotlib and numpy.
"""

import sys
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)

from .map_manager import MapManager


def visualize(map_dir: str, save_path: str = None):
    """Render the map as a figure.

    Args:
        map_dir: Path to the map directory.
        save_path: If provided, save figure to this path instead of showing.
    """
    mm = MapManager()
    mm.load(map_dir)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"SLAM Map: {os.path.basename(map_dir)}", fontsize=14)

    # --- Left panel: Occupancy Grid ---
    ax = axes[0]
    prob_grid = mm.occupancy.get_probability_grid()
    # Color: white=free, black=occupied, gray=unknown
    display = np.ones((*prob_grid.shape, 3))
    # Occupied cells (high probability) -> black
    occupied_mask = prob_grid > 0.6
    display[occupied_mask] = [0, 0, 0]
    # Free cells (low probability) -> white (already default)
    # Unknown cells (near 0.5) -> light gray
    unknown_mask = (prob_grid > 0.35) & (prob_grid < 0.65)
    display[unknown_mask] = [0.8, 0.8, 0.8]

    res = mm.occupancy.resolution
    extent_m = mm.occupancy.size * res
    offset = mm.occupancy.origin_offset * res
    extent = [-offset, extent_m - offset, -offset, extent_m - offset]

    ax.imshow(display, origin='lower', extent=extent)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Occupancy Grid")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Overlay trajectory
    if mm.trajectory:
        traj = np.array(mm.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=0.8, alpha=0.6, label="Trajectory")
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label="Start")
        ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=8, label="End")
        ax.legend(loc='upper right', fontsize=8)

    # --- Right panel: Landmarks ---
    ax = axes[1]
    ax.imshow(display, origin='lower', extent=extent, alpha=0.3)

    if mm.landmarks:
        positions = np.array([lm.position_3d for lm in mm.landmarks.values()])
        obs_counts = np.array([lm.observation_count for lm in mm.landmarks.values()])

        # Color by observation count (more observations = more confident)
        scatter = ax.scatter(positions[:, 0], positions[:, 1],
                             c=obs_counts, cmap='viridis', s=3, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Observation count", shrink=0.6)

    if mm.trajectory:
        traj = np.array(mm.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=0.5, alpha=0.4)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"Landmarks ({mm.landmark_count} total)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m raspbot_slam.visualize_map <map_dir> [output.png]")
        sys.exit(1)

    map_dir = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.isdir(map_dir):
        print(f"ERROR: {map_dir} is not a directory")
        sys.exit(1)

    visualize(map_dir, save_path)


if __name__ == "__main__":
    main()
