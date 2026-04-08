"""
Frontier-based exploration for autonomous mapping.

Identifies boundaries between known-free and unknown space on the
occupancy grid, selects the most informative frontier as the next
exploration target, and plans a path via A*.
"""

import math
import heapq
from typing import List, Tuple, Optional
import numpy as np

from . import config
from .map_manager import MapManager, OccupancyGrid


class Explorer:
    """Frontier-based exploration strategy for mapping mode.

    Usage:
        explorer = Explorer(map_manager)
        target = explorer.select_target(robot_pose)
        if target is None:
            print("Mapping complete!")
        else:
            path = explorer.plan_path(robot_pose[:2], target)
            # follow path with MotionController
    """

    def __init__(self, map_manager: MapManager,
                 size_weight: float = 1.0,
                 distance_weight: float = 2.0,
                 info_weight: float = 0.5):
        self._map = map_manager
        self._w_size = size_weight
        self._w_dist = distance_weight
        self._w_info = info_weight
        self._prev_mapped_area = 0
        self._stagnation_count = 0

    def select_target(self, robot_pose: Tuple[float, float, float]
                      ) -> Optional[Tuple[float, float]]:
        """Find the best frontier to explore next.

        Args:
            robot_pose: (x, y, theta) in world frame.

        Returns:
            (x, y) world coordinates of the target, or None if mapping is complete.
        """
        grid = self._map.occupancy
        frontiers = grid.get_frontiers()

        if not frontiers:
            return None

        # Cluster frontiers into connected segments
        segments = self._cluster_frontiers(frontiers, grid)

        if not segments:
            return None

        # Filter out tiny segments
        segments = [s for s in segments if len(s) >= 3]
        if not segments:
            return None

        # Score each segment
        robot_cell = grid.world_to_cell(robot_pose[0], robot_pose[1])
        best_score = -float('inf')
        best_target = None

        for segment in segments:
            centroid = np.mean(segment, axis=0).astype(int)
            cx, cy = int(centroid[0]), int(centroid[1])

            # Distance from robot (in cells)
            dist = math.sqrt((cx - robot_cell[0])**2 + (cy - robot_cell[1])**2)
            if dist < 1:
                dist = 1

            # Count unknown cells behind this frontier (information gain)
            unknown_count = self._count_unknown_behind(segment, grid)

            score = (self._w_size * len(segment)
                     + self._w_dist * (1.0 / dist)
                     + self._w_info * unknown_count)

            if score > best_score:
                best_score = score
                best_target = grid.cell_to_world(cx, cy)

        return best_target

    def plan_path(self, start_xy: Tuple[float, float],
                  goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Plan a path from start to goal using A* on the occupancy grid.

        Args:
            start_xy: (x, y) start position in world frame.
            goal_xy: (x, y) goal position in world frame.

        Returns:
            List of (x, y) waypoints in world frame. Empty if no path found.
        """
        grid = self._map.occupancy
        start_cell = grid.world_to_cell(*start_xy)
        goal_cell = grid.world_to_cell(*goal_xy)

        cell_path = self._astar(grid, start_cell, goal_cell)
        if not cell_path:
            return []

        # Convert cell path to world coordinates
        world_path = [grid.cell_to_world(cx, cy) for cx, cy in cell_path]

        # Simplify path: remove redundant waypoints on straight segments
        return self._simplify_path(world_path)

    def is_mapping_complete(self) -> bool:
        """Check if mapping is complete based on frontier count and area growth."""
        grid = self._map.occupancy
        frontiers = grid.get_frontiers()
        large_frontiers = self._cluster_frontiers(frontiers, grid)
        large_frontiers = [s for s in large_frontiers if len(s) >= 3]

        if not large_frontiers:
            return True

        # Check for stagnation (mapped area not growing)
        free_count = int(np.sum(grid.grid < -0.3))
        if free_count <= self._prev_mapped_area * 1.01:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0
        self._prev_mapped_area = free_count

        return self._stagnation_count >= 3

    # =========================================================================
    # A* Path Planning
    # =========================================================================

    @staticmethod
    def _astar(grid: OccupancyGrid,
               start: Tuple[int, int],
               goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* on occupancy grid. Returns list of (cx, cy) cells, or empty list."""
        if not grid.in_bounds(*start) or not grid.in_bounds(*goal):
            return []

        # 8-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        costs = [1.0, 1.0, 1.0, 1.0,
                 1.414, 1.414, 1.414, 1.414]

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cx, cy = current
            for (dx, dy), cost in zip(neighbors, costs):
                nx, ny = cx + dx, cy + dy
                if not grid.in_bounds(nx, ny):
                    continue
                if grid.is_occupied(nx, ny):
                    continue

                # Penalize cells near obstacles (inflation)
                inflation_cost = 0.0
                for idx, idy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if grid.is_occupied(nx + idx, ny + idy):
                        inflation_cost += 2.0

                tentative_g = g_score[current] + cost + inflation_cost
                neighbor = (nx, ny)

                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue

                g_score[neighbor] = tentative_g
                # Heuristic: Euclidean distance
                h = math.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
                f = tentative_g + h
                heapq.heappush(open_set, (f, neighbor))
                came_from[neighbor] = current

        return []  # no path found

    # =========================================================================
    # Frontier Clustering
    # =========================================================================

    @staticmethod
    def _cluster_frontiers(frontiers: List[np.ndarray],
                           grid: OccupancyGrid) -> List[List[np.ndarray]]:
        """Cluster frontier cells into connected segments using flood fill."""
        if not frontiers:
            return []

        frontier_set = set()
        for f in frontiers:
            frontier_set.add((int(f[0]), int(f[1])))

        visited = set()
        segments = []

        for cell in frontier_set:
            if cell in visited:
                continue
            # BFS flood fill
            segment = []
            queue = [cell]
            while queue:
                c = queue.pop(0)
                if c in visited:
                    continue
                if c not in frontier_set:
                    continue
                visited.add(c)
                segment.append(np.array(c))
                # 8-connected
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        n = (c[0] + dx, c[1] + dy)
                        if n not in visited and n in frontier_set:
                            queue.append(n)
            if segment:
                segments.append(segment)

        return segments

    @staticmethod
    def _count_unknown_behind(segment: List[np.ndarray],
                              grid: OccupancyGrid) -> int:
        """Count unknown cells in a region beyond the frontier segment."""
        count = 0
        for cell in segment:
            cx, cy = int(cell[0]), int(cell[1])
            # Check cells 1-3 steps beyond in all directions
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for step in range(1, 4):
                        nx = cx + dx * step
                        ny = cy + dy * step
                        if grid.is_unknown(nx, ny):
                            count += 1
        return count

    @staticmethod
    def _simplify_path(path: List[Tuple[float, float]],
                       tolerance: float = 0.05) -> List[Tuple[float, float]]:
        """Remove redundant waypoints on straight-line segments."""
        if len(path) <= 2:
            return path

        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            # Check if point i is collinear with i-1 and i+1
            x0, y0 = simplified[-1]
            x1, y1 = path[i]
            x2, y2 = path[i + 1]

            # Cross product magnitude (distance from point to line)
            cross = abs((x2 - x0) * (y1 - y0) - (x1 - x0) * (y2 - y0))
            line_len = math.sqrt((x2 - x0)**2 + (y2 - y0)**2)
            if line_len < 1e-6:
                continue
            dist = cross / line_len

            if dist > tolerance:
                simplified.append(path[i])

        simplified.append(path[-1])
        return simplified
