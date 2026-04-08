"""Tests for map_manager.py -- OccupancyGrid and MapManager."""

import math
import os
import tempfile
import numpy as np
import pytest

from raspbot_slam.map_manager import MapManager, OccupancyGrid, Landmark


# =============================================================================
# OccupancyGrid
# =============================================================================

class TestOccupancyGridCoordinates:

    def test_world_to_cell_origin(self):
        grid = OccupancyGrid(size_cells=100, resolution_m=0.05)
        grid.origin_offset = 50
        cx, cy = grid.world_to_cell(0.0, 0.0)
        assert cx == 50
        assert cy == 50

    def test_world_to_cell_positive(self):
        grid = OccupancyGrid(size_cells=100, resolution_m=0.05)
        grid.origin_offset = 50
        cx, cy = grid.world_to_cell(1.0, 0.5)
        assert cx == 70  # 1.0/0.05 + 50
        assert cy == 60  # 0.5/0.05 + 50

    def test_cell_to_world_roundtrip(self):
        grid = OccupancyGrid(size_cells=400, resolution_m=0.05)
        x, y = 2.5, -1.3
        cx, cy = grid.world_to_cell(x, y)
        x2, y2 = grid.cell_to_world(cx, cy)
        assert abs(x2 - x) < grid.resolution
        assert abs(y2 - y) < grid.resolution

    def test_in_bounds(self):
        grid = OccupancyGrid(size_cells=100)
        assert grid.in_bounds(0, 0)
        assert grid.in_bounds(99, 99)
        assert not grid.in_bounds(-1, 0)
        assert not grid.in_bounds(100, 0)


class TestOccupancyGridUpdate:

    def test_initial_state_unknown(self):
        grid = OccupancyGrid(size_cells=50)
        assert grid.is_unknown(25, 25)
        assert not grid.is_free(25, 25)
        assert not grid.is_occupied(25, 25)

    def test_update_free(self):
        grid = OccupancyGrid(size_cells=50)
        grid.update_cell(25, 25, -2.0)  # strong free observation
        assert grid.is_free(25, 25)

    def test_update_occupied(self):
        grid = OccupancyGrid(size_cells=50)
        grid.update_cell(25, 25, 2.0)  # strong occupied observation
        assert grid.is_occupied(25, 25)

    def test_log_odds_clamp(self):
        from raspbot_slam import config
        grid = OccupancyGrid(size_cells=50)
        grid.update_cell(25, 25, 100.0)  # very strong
        assert grid.grid[25, 25] <= config.LOG_ODDS_MAX
        grid.update_cell(25, 25, -200.0)  # very strong free
        assert grid.grid[25, 25] >= config.LOG_ODDS_MIN

    def test_ultrasonic_ray_cast(self):
        grid = OccupancyGrid(size_cells=200, resolution_m=0.05)
        grid.origin_offset = 100
        # Robot at origin, facing right (theta=0), obstacle at 1m
        grid.update_ultrasonic(0.0, 0.0, 0.0, 1.0)

        # Cells along the ray should be free
        cx_mid, cy_mid = grid.world_to_cell(0.5, 0.0)
        assert grid.is_free(cx_mid, cy_mid)

        # Cell at the endpoint should be occupied
        cx_end, cy_end = grid.world_to_cell(1.0, 0.0)
        assert grid.is_occupied(cx_end, cy_end)

    def test_mark_traversed(self):
        grid = OccupancyGrid(size_cells=100, resolution_m=0.05)
        grid.origin_offset = 50
        grid.mark_traversed(0.0, 0.0)
        cx, cy = grid.world_to_cell(0.0, 0.0)
        assert grid.is_free(cx, cy)

    def test_probability_grid_range(self):
        grid = OccupancyGrid(size_cells=20)
        grid.update_cell(5, 5, 3.0)   # occupied
        grid.update_cell(10, 10, -3.0)  # free
        prob = grid.get_probability_grid()
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0
        assert prob[5, 5] > 0.9
        assert prob[10, 10] < 0.1


class TestOccupancyGridFrontiers:

    def test_frontiers_at_explored_boundary(self):
        """Free cells adjacent to unknown cells should be detected as frontiers."""
        grid = OccupancyGrid(size_cells=50, resolution_m=0.1)
        grid.origin_offset = 25
        # Explore only left half -- frontier at the boundary
        for y in range(10, 40):
            for x in range(10, 25):
                grid.update_cell(x, y, -2.0)
        frontiers = grid.get_frontiers()
        assert len(frontiers) > 0
        # At least some frontiers should be at x=24 (right boundary)
        right_boundary = [f for f in frontiers if f[0] == 24]
        assert len(right_boundary) > 0

    def test_fully_explored_no_frontiers(self):
        """A fully observed grid should have no frontiers."""
        grid = OccupancyGrid(size_cells=20)
        # Mark everything as free
        for y in range(20):
            for x in range(20):
                grid.update_cell(x, y, -2.0)
        frontiers = grid.get_frontiers()
        assert len(frontiers) == 0


class TestOccupancyGridPersistence:

    def test_save_load_roundtrip(self):
        grid = OccupancyGrid(size_cells=50)
        grid.update_cell(10, 20, 2.5)
        grid.update_cell(30, 40, -1.5)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        try:
            grid.save(path)
            grid2 = OccupancyGrid(size_cells=50)
            grid2.load(path)
            assert np.allclose(grid.grid, grid2.grid)
        finally:
            os.unlink(path)


# =============================================================================
# MapManager
# =============================================================================

class TestMapManagerLandmarks:

    def test_add_landmark(self):
        mm = MapManager()
        desc = np.random.randint(0, 256, 32, dtype=np.uint8)
        lm_id = mm.add_landmark(np.array([1.0, 2.0, 0.5]), desc)
        assert lm_id == 0
        assert mm.landmark_count == 1

    def test_add_multiple_landmarks(self):
        mm = MapManager()
        for i in range(10):
            desc = np.random.randint(0, 256, 32, dtype=np.uint8)
            mm.add_landmark(np.array([float(i), 0, 0]), desc)
        assert mm.landmark_count == 10

    def test_get_visible_landmarks(self, simple_map):
        # Robot at origin, facing right (theta=0), 90 degree FOV
        visible = simple_map.get_visible_landmarks((0, 0, 0), fov_deg=90, max_range=3.0)
        # Should see landmarks on the right wall (x=1.8)
        assert len(visible) > 0
        for lm in visible:
            assert lm.position_3d[0] > 0  # all visible landmarks in front

    def test_get_visible_respects_range(self, simple_map):
        visible_near = simple_map.get_visible_landmarks((0, 0, 0), max_range=0.5)
        visible_far = simple_map.get_visible_landmarks((0, 0, 0), max_range=5.0)
        assert len(visible_near) <= len(visible_far)

    def test_match_observation_exact(self):
        mm = MapManager()
        desc = np.array([42] * 32, dtype=np.uint8)
        mm.add_landmark(np.array([1, 0, 0]), desc)

        match = mm.match_observation(desc)
        assert match is not None
        assert match.id == 0

    def test_match_observation_no_match(self):
        mm = MapManager()
        desc1 = np.zeros(32, dtype=np.uint8)
        mm.add_landmark(np.array([1, 0, 0]), desc1)

        desc2 = np.full(32, 255, dtype=np.uint8)
        match = mm.match_observation(desc2, max_descriptor_distance=30)
        assert match is None


class TestMapManagerTrajectory:

    def test_record_pose(self):
        mm = MapManager()
        mm.record_pose(1.0, 2.0, 0.5)
        mm.record_pose(2.0, 3.0, 0.6)
        assert len(mm.trajectory) == 2
        assert np.allclose(mm.trajectory[0], [1.0, 2.0, 0.5])


class TestMapManagerPersistence:

    def test_save_load_roundtrip(self, simple_map):
        with tempfile.TemporaryDirectory() as tmpdir:
            simple_map.record_pose(0, 0, 0)
            simple_map.record_pose(1, 0, 0)
            simple_map.save(tmpdir)

            mm2 = MapManager()
            mm2.load(tmpdir)

            assert mm2.landmark_count == simple_map.landmark_count
            assert len(mm2.trajectory) == 2

            # Check landmark positions match
            for lm_id in simple_map.landmarks:
                pos1 = simple_map.landmarks[lm_id].position_3d
                pos2 = mm2.landmarks[lm_id].position_3d
                assert np.allclose(pos1, pos2)
