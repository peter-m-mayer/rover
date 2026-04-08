"""
Dual map manager: sparse 3D landmark database + 2D occupancy grid.

The landmark database supports localization (feature matching against known
3D positions). The occupancy grid supports navigation (path planning through
free space).

Occupancy is updated from three sources:
1. Ultrasonic ray-cast (forward-facing range)
2. Synthetic stereo depth points (projected to ground plane)
3. Robot traversal path (known free space)
"""

import json
import math
import os
import pickle
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import numpy as np

from . import config


@dataclass
class Landmark:
    """A mapped 3D landmark with visual descriptor."""
    id: int
    position_3d: np.ndarray           # [x, y, z] world frame
    covariance: np.ndarray            # 3x3
    descriptor: np.ndarray            # ORB descriptor (32 bytes)
    observation_count: int = 0
    last_seen_keyframe: int = -1
    status: str = "active"            # 'active' or 'frozen'


class OccupancyGrid:
    """2D log-odds occupancy grid for navigation."""

    def __init__(self, size_cells: int = None, resolution_m: float = None):
        self.size = size_cells or config.GRID_SIZE_CELLS
        self.resolution = resolution_m or config.GRID_RESOLUTION_M
        self.origin_offset = config.GRID_ORIGIN_OFFSET

        # Log-odds grid, initialized to 0 (unknown prior)
        self.grid = np.full((self.size, self.size), config.LOG_ODDS_PRIOR,
                            dtype=np.float32)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid cell indices."""
        cx = int(x / self.resolution) + self.origin_offset
        cy = int(y / self.resolution) + self.origin_offset
        return (cx, cy)

    def cell_to_world(self, cx: int, cy: int) -> Tuple[float, float]:
        """Convert grid cell indices to world coordinates (cell center)."""
        x = (cx - self.origin_offset + 0.5) * self.resolution
        y = (cy - self.origin_offset + 0.5) * self.resolution
        return (x, y)

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.size and 0 <= cy < self.size

    def is_free(self, cx: int, cy: int) -> bool:
        """True if cell is classified as free (log-odds < 0)."""
        if not self.in_bounds(cx, cy):
            return False
        return self.grid[cy, cx] < 0

    def is_occupied(self, cx: int, cy: int) -> bool:
        """True if cell is classified as occupied (log-odds > 0.5)."""
        if not self.in_bounds(cx, cy):
            return True  # out of bounds treated as occupied
        return self.grid[cy, cx] > 0.5

    def is_unknown(self, cx: int, cy: int) -> bool:
        """True if cell is still near the prior (not significantly observed)."""
        if not self.in_bounds(cx, cy):
            return False
        return abs(self.grid[cy, cx]) < 0.3

    def update_cell(self, cx: int, cy: int, log_odds_delta: float):
        """Update a single cell with a log-odds observation."""
        if not self.in_bounds(cx, cy):
            return
        self.grid[cy, cx] = np.clip(
            self.grid[cy, cx] + log_odds_delta,
            config.LOG_ODDS_MIN,
            config.LOG_ODDS_MAX)

    def update_ultrasonic(self, robot_x: float, robot_y: float,
                          robot_theta: float, range_m: float):
        """Update grid from an ultrasonic reading.

        Marks cells along the beam as free, and the endpoint as occupied.
        """
        if range_m <= 0 or range_m > 10.0:
            return

        # Ray-cast from robot position along heading
        step = self.resolution * 0.5
        n_steps = int(range_m / step)
        cos_t = math.cos(robot_theta)
        sin_t = math.sin(robot_theta)

        for i in range(n_steps):
            d = i * step
            wx = robot_x + d * cos_t
            wy = robot_y + d * sin_t
            cx, cy = self.world_to_cell(wx, wy)
            self.update_cell(cx, cy, config.LOG_ODDS_FREE)

        # Endpoint: occupied
        wx = robot_x + range_m * cos_t
        wy = robot_y + range_m * sin_t
        cx, cy = self.world_to_cell(wx, wy)
        self.update_cell(cx, cy, config.LOG_ODDS_OCCUPIED)

    def update_depth_points(self, robot_x: float, robot_y: float,
                            robot_theta: float, points_3d: np.ndarray):
        """Update grid from synthetic stereo depth observations.

        Args:
            robot_x, robot_y, robot_theta: Robot pose in world frame.
            points_3d: Nx3 array of (X, Y, Z) in camera frame.
                Camera: X=right, Y=down, Z=forward.
        """
        if points_3d is None or len(points_3d) == 0:
            return

        cos_t = math.cos(robot_theta)
        sin_t = math.sin(robot_theta)

        for pt in points_3d:
            cam_x, cam_y, cam_z = pt

            # Height filter: ignore floor noise and ceiling
            height = -cam_y  # camera Y is down, height is up
            if height < config.OBSTACLE_HEIGHT_MIN_M:
                continue
            if height > config.OBSTACLE_HEIGHT_MAX_M:
                continue

            # Transform camera frame to world frame
            # Camera Z=forward, X=right → World frame rotated by robot_theta
            wx = robot_x + cam_z * cos_t - cam_x * sin_t
            wy = robot_y + cam_z * sin_t + cam_x * cos_t

            cx, cy = self.world_to_cell(wx, wy)

            if height < 0.30:
                # Low obstacle (furniture leg, step)
                self.update_cell(cx, cy, config.LOG_ODDS_OCCUPIED)
            else:
                # Wall or tall furniture
                self.update_cell(cx, cy, config.LOG_ODDS_OCCUPIED)

            # Mark cells between robot and this point as free
            n_steps = max(1, int(cam_z / (self.resolution * 2)))
            for i in range(n_steps):
                frac = i / n_steps
                fx = robot_x + frac * (wx - robot_x)
                fy = robot_y + frac * (wy - robot_y)
                fcx, fcy = self.world_to_cell(fx, fy)
                self.update_cell(fcx, fcy, config.LOG_ODDS_FREE * 0.5)

    def mark_traversed(self, robot_x: float, robot_y: float):
        """Mark the robot's current cell (and neighbors) as free."""
        cx, cy = self.world_to_cell(robot_x, robot_y)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                self.update_cell(cx + dx, cy + dy, config.LOG_ODDS_FREE)

    def get_frontiers(self) -> List[np.ndarray]:
        """Find frontier cells: free cells adjacent to unknown cells.

        Returns:
            List of (cx, cy) arrays representing frontier cell positions.
        """
        frontiers = []
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                if not self.is_free(x, y):
                    continue
                # Check 4-connected neighbors for unknown
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if self.is_unknown(x + dx, y + dy):
                        frontiers.append(np.array([x, y]))
                        break
        return frontiers

    def get_probability_grid(self) -> np.ndarray:
        """Convert log-odds to probability [0, 1] for visualization."""
        return 1.0 / (1.0 + np.exp(-self.grid))

    def save(self, filepath: str):
        """Save occupancy grid as numpy file."""
        np.save(filepath, self.grid)

    def load(self, filepath: str):
        """Load occupancy grid from numpy file."""
        self.grid = np.load(filepath).astype(np.float32)
        self.size = self.grid.shape[0]


class MapManager:
    """Manages the dual map: sparse landmarks + occupancy grid.

    Usage:
        mm = MapManager()
        # During mapping:
        mm.add_landmark(position_3d, descriptor)
        mm.occupancy.update_ultrasonic(...)
        mm.save("maps/ground_floor")
        # During localization:
        mm.load("maps/ground_floor")
        matches = mm.get_visible_landmarks(pose, fov_deg=90)
    """

    def __init__(self):
        self.landmarks: Dict[int, Landmark] = {}
        self.occupancy = OccupancyGrid()
        self.trajectory: List[np.ndarray] = []  # list of (x, y, theta)
        self._next_id = 0

    # =========================================================================
    # Landmark Operations
    # =========================================================================

    def add_landmark(self, position_3d: np.ndarray, descriptor: np.ndarray,
                     covariance: np.ndarray = None) -> int:
        """Add a new landmark to the map.

        Returns:
            Landmark ID.
        """
        lm_id = self._next_id
        self._next_id += 1

        if covariance is None:
            covariance = np.eye(3, dtype=np.float64)

        self.landmarks[lm_id] = Landmark(
            id=lm_id,
            position_3d=np.array(position_3d, dtype=np.float64),
            covariance=np.array(covariance, dtype=np.float64),
            descriptor=np.array(descriptor, dtype=np.uint8),
            observation_count=1,
        )
        return lm_id

    def update_landmark(self, lm_id: int, position_3d: np.ndarray = None,
                        covariance: np.ndarray = None, keyframe_id: int = -1):
        """Update a landmark's position and/or metadata."""
        if lm_id not in self.landmarks:
            return
        lm = self.landmarks[lm_id]
        if position_3d is not None:
            lm.position_3d = np.array(position_3d, dtype=np.float64)
        if covariance is not None:
            lm.covariance = np.array(covariance, dtype=np.float64)
        lm.observation_count += 1
        lm.last_seen_keyframe = max(lm.last_seen_keyframe, keyframe_id)

    def get_visible_landmarks(self, pose: Tuple[float, float, float],
                              fov_deg: float = 90.0,
                              max_range: float = 5.0) -> List[Landmark]:
        """Find landmarks expected to be visible from the given pose.

        Args:
            pose: (x, y, theta) robot pose.
            fov_deg: Field of view in degrees (total, centered on heading).
            max_range: Maximum range in meters.

        Returns:
            List of Landmark objects within FOV and range.
        """
        x, y, theta = pose
        half_fov = math.radians(fov_deg / 2.0)
        visible = []

        for lm in self.landmarks.values():
            dx = lm.position_3d[0] - x
            dy = lm.position_3d[1] - y
            dist = math.sqrt(dx**2 + dy**2)

            if dist > max_range:
                continue

            bearing = math.atan2(dy, dx) - theta
            # Normalize to [-pi, pi]
            while bearing > math.pi:
                bearing -= 2 * math.pi
            while bearing < -math.pi:
                bearing += 2 * math.pi

            if abs(bearing) <= half_fov:
                visible.append(lm)

        return visible

    def match_observation(self, descriptor: np.ndarray,
                          position_hint: np.ndarray = None,
                          max_descriptor_distance: int = 60,
                          max_spatial_distance: float = 2.0
                          ) -> Optional[Landmark]:
        """Find the best matching landmark by descriptor and optionally position.

        Args:
            descriptor: ORB descriptor to match.
            position_hint: Optional 3D position to limit spatial search.
            max_descriptor_distance: Maximum Hamming distance.
            max_spatial_distance: Maximum spatial distance (if position_hint given).

        Returns:
            Best matching Landmark, or None.
        """
        best_lm = None
        best_dist = max_descriptor_distance + 1

        for lm in self.landmarks.values():
            # Spatial filter
            if position_hint is not None:
                spatial_dist = np.linalg.norm(lm.position_3d[:2] - position_hint[:2])
                if spatial_dist > max_spatial_distance:
                    continue

            # Descriptor distance
            d = _hamming_distance(descriptor, lm.descriptor)
            if d < best_dist:
                best_dist = d
                best_lm = lm

        return best_lm if best_dist <= max_descriptor_distance else None

    # =========================================================================
    # Trajectory
    # =========================================================================

    def record_pose(self, x: float, y: float, theta: float):
        """Record a robot pose in the trajectory."""
        self.trajectory.append(np.array([x, y, theta]))
        self.occupancy.mark_traversed(x, y)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, map_dir: str):
        """Save the complete map to a directory.

        Creates:
            map_dir/metadata.json
            map_dir/landmarks.pkl
            map_dir/occupancy_grid.npy
            map_dir/trajectory.npy
        """
        os.makedirs(map_dir, exist_ok=True)

        # Metadata
        metadata = {
            "version": "1.0",
            "n_landmarks": len(self.landmarks),
            "n_trajectory_poses": len(self.trajectory),
            "grid_resolution_m": self.occupancy.resolution,
            "grid_size_cells": self.occupancy.size,
        }
        with open(os.path.join(map_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Landmarks
        with open(os.path.join(map_dir, "landmarks.pkl"), 'wb') as f:
            pickle.dump(self.landmarks, f)

        # Occupancy grid
        self.occupancy.save(os.path.join(map_dir, "occupancy_grid.npy"))

        # Trajectory
        if self.trajectory:
            traj_array = np.array(self.trajectory)
            np.save(os.path.join(map_dir, "trajectory.npy"), traj_array)

    def load(self, map_dir: str):
        """Load a complete map from a directory."""
        # Landmarks
        lm_path = os.path.join(map_dir, "landmarks.pkl")
        if os.path.exists(lm_path):
            with open(lm_path, 'rb') as f:
                self.landmarks = pickle.load(f)
            if self.landmarks:
                self._next_id = max(self.landmarks.keys()) + 1

        # Occupancy grid
        grid_path = os.path.join(map_dir, "occupancy_grid.npy")
        if os.path.exists(grid_path):
            self.occupancy.load(grid_path)

        # Trajectory
        traj_path = os.path.join(map_dir, "trajectory.npy")
        if os.path.exists(traj_path):
            traj_array = np.load(traj_path)
            self.trajectory = [traj_array[i] for i in range(len(traj_array))]

    @property
    def landmark_count(self) -> int:
        return len(self.landmarks)


def _hamming_distance(d1: np.ndarray, d2: np.ndarray) -> int:
    """Hamming distance between two binary descriptors."""
    return int(np.sum(np.unpackbits(np.bitwise_xor(d1, d2))))
