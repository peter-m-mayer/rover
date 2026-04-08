"""Tests for explorer.py -- frontier detection, A*, path planning."""

import math
import numpy as np
import pytest

from raspbot_slam.explorer import Explorer
from raspbot_slam.map_manager import MapManager, OccupancyGrid


class TestAstar:

    def test_straight_line_path(self):
        """Path through open space should be roughly straight."""
        grid = OccupancyGrid(size_cells=100, resolution_m=0.05)
        grid.origin_offset = 50
        # Mark a corridor of free space
        for x in range(20, 80):
            for y in range(45, 55):
                grid.update_cell(x, y, -2.0)

        mm = MapManager()
        mm.occupancy = grid
        explorer = Explorer(mm)

        start = grid.cell_to_world(25, 50)
        goal = grid.cell_to_world(75, 50)
        path = explorer.plan_path(start, goal)

        assert len(path) >= 2
        # Start and end should be close to requested
        assert abs(path[0][0] - start[0]) < 0.1
        assert abs(path[-1][0] - goal[0]) < 0.1

    def test_path_avoids_obstacle(self):
        """Path should route around an obstacle."""
        grid = OccupancyGrid(size_cells=100, resolution_m=0.05)
        grid.origin_offset = 50
        # Free space
        for x in range(10, 90):
            for y in range(10, 90):
                grid.update_cell(x, y, -2.0)

        # Wall blocking direct path at x=50
        for y in range(20, 70):
            grid.update_cell(50, y, 3.0)

        mm = MapManager()
        mm.occupancy = grid
        explorer = Explorer(mm)

        start = grid.cell_to_world(30, 50)
        goal = grid.cell_to_world(70, 50)
        path = explorer.plan_path(start, goal)

        assert len(path) >= 2
        # Path should go around the wall (not through it)
        for wx, wy in path:
            cx, cy = grid.world_to_cell(wx, wy)
            assert not grid.is_occupied(cx, cy)

    def test_no_path_blocked(self):
        """Fully blocked goal should return empty path."""
        grid = OccupancyGrid(size_cells=50, resolution_m=0.1)
        grid.origin_offset = 25
        # Make everything free
        for x in range(50):
            for y in range(50):
                grid.update_cell(x, y, -2.0)
        # Wall surrounding the goal
        for i in range(15, 25):
            grid.update_cell(i, 15, 3.0)
            grid.update_cell(i, 24, 3.0)
            grid.update_cell(15, i, 3.0)
            grid.update_cell(24, i, 3.0)

        mm = MapManager()
        mm.occupancy = grid
        explorer = Explorer(mm)

        start = grid.cell_to_world(5, 5)
        goal = grid.cell_to_world(20, 20)  # inside walled box
        path = explorer.plan_path(start, goal)
        assert len(path) == 0


class TestFrontierDetection:

    def test_partially_explored_has_frontiers(self):
        """A partially explored area should have frontiers at the boundary."""
        grid = OccupancyGrid(size_cells=50, resolution_m=0.1)
        grid.origin_offset = 25
        # Explore only the left half
        for y in range(10, 40):
            for x in range(10, 25):
                grid.update_cell(x, y, -2.0)
        # Right half remains unknown → frontiers at x=25
        frontiers = grid.get_frontiers()
        assert len(frontiers) > 0

    def test_select_target_returns_point(self, simple_map):
        explorer = Explorer(simple_map)
        target = explorer.select_target((0, 0, 0))
        # simple_map has frontiers at the room boundary
        # target should be a valid (x, y) tuple or None
        if target is not None:
            assert len(target) == 2


class TestFrontierClustering:

    def test_clustering_separates_groups(self):
        grid = OccupancyGrid(size_cells=50, resolution_m=0.1)
        grid.origin_offset = 25

        # Two separate free regions with unknown between them
        for x in range(5, 15):
            for y in range(5, 15):
                grid.update_cell(x, y, -2.0)
        for x in range(30, 40):
            for y in range(30, 40):
                grid.update_cell(x, y, -2.0)

        frontiers = grid.get_frontiers()
        segments = Explorer._cluster_frontiers(frontiers, grid)

        # Should produce at least 2 separate clusters
        assert len(segments) >= 2


class TestPathSimplification:

    def test_straight_path_simplified(self):
        # Points on a straight line should simplify to just endpoints
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        simplified = Explorer._simplify_path(path, tolerance=0.01)
        assert len(simplified) == 2
        assert simplified[0] == (0, 0)
        assert simplified[-1] == (4, 0)

    def test_corner_preserved(self):
        # L-shaped path: straight, then 90 degree turn
        path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        simplified = Explorer._simplify_path(path, tolerance=0.01)
        assert len(simplified) >= 3  # at least start, corner, end

    def test_short_path_unchanged(self):
        path = [(0, 0), (1, 1)]
        simplified = Explorer._simplify_path(path)
        assert simplified == path


class TestMappingComplete:

    def test_fully_explored_is_complete(self):
        grid = OccupancyGrid(size_cells=30)
        # Mark everything as free (no unknowns)
        for x in range(30):
            for y in range(30):
                grid.update_cell(x, y, -2.0)

        mm = MapManager()
        mm.occupancy = grid
        explorer = Explorer(mm)

        assert explorer.is_mapping_complete()

    def test_unexplored_is_not_complete(self):
        """A partially explored map should not be considered complete."""
        grid = OccupancyGrid(size_cells=50, resolution_m=0.1)
        grid.origin_offset = 25
        # Only explore a small patch -- lots of unknown remains
        for y in range(10, 20):
            for x in range(10, 20):
                grid.update_cell(x, y, -2.0)
        mm = MapManager()
        mm.occupancy = grid
        explorer = Explorer(mm)
        assert not explorer.is_mapping_complete()
