"""Room map, planner, localizer and heading history (NAV plan v2)."""

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from raspbot_slam.localizer import Localizer
from raspbot_slam.online_odometry import HeadingHistory
from raspbot_slam.planner import astar, path_length, plan, tag_viewpoint
from raspbot_slam.room_map import RoomMap
from raspbot_slam.tag_localizer import IN, load_survey


@pytest.fixture
def office():
    return load_survey("office")


def test_survey_seeds_walls_and_outside(office):
    m = RoomMap.from_survey(office)
    X, Y = office["xmax_in"] * IN, office["ymax_in"] * IN
    occ = m.occupied()
    assert occ[m.cell(X / 2, 0.0)[::-1]]            # south wall
    assert occ[m.cell(X / 2, Y + 0.2)[::-1]]        # outside
    assert not occ[m.cell(X / 2, Y / 2)[::-1]]      # middle of the room
    assert not occ[m.cell(-0.5, 0.5)[::-1]]         # inside the alcove
    assert occ[m.cell(0.0, 0.5)[::-1]]              # mirror door
    assert not occ[m.cell(0.0, 1.3)[::-1]]          # alcove opening above the mirror


def test_camera_scan_marks_free_and_hit(office):
    m = RoomMap.from_survey(office)
    pose = (2.0, 2.0, 0.0)
    b = np.radians(np.linspace(-20, 20, 9))
    for _ in range(3):
        m.add_camera_scan(pose, b, np.full(9, 1.0), np.ones(9, bool), cam_offset_m=0.0)
    assert m.free()[m.cell(2.5, 2.0)[::-1]]
    assert m.occupied()[m.cell(3.0, 2.0)[::-1]]


def test_contact_marks_ahead_and_footprint_clears(office):
    m = RoomMap.from_survey(office)
    m.add_contact((2.0, 2.0, math.pi / 2))
    assert m.occupied()[m.cell(2.0, 2.18)[::-1]]
    m.add_footprint(2.0, 2.18)
    assert not m.occupied()[m.cell(2.0, 2.18)[::-1]]


def test_plan_goes_around_a_block(office):
    m = RoomMap.from_survey(office)
    # a "dresser" between start and goal
    for x in np.arange(2.4, 2.8, 0.02):
        for y in np.arange(1.0, 3.5, 0.02):
            i, j = m.cell(x, y)
            m.L[j, i] = 3.0
    path = plan(m, (1.5, 2.0), (3.8, 2.0))
    assert path is not None
    assert path_length(path) > 2.3 + 0.5                # had to detour
    clear = m.clearance()
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        for s in np.linspace(0, 1, 20):
            i, j = m.cell(x0 + s * (x1 - x0), y0 + s * (y1 - y0))
            assert clear[j, i] > 0.12


def test_plan_none_when_goal_blocked(office):
    m = RoomMap.from_survey(office)
    assert plan(m, (1.5, 2.0), (-2.0, 3.0)) is None     # outside the room


def test_astar_start_inside_margin_can_leave():
    cost = np.ones((20, 20), np.float32)
    cost[:, 8] = np.inf
    cost[9:12, 6:8] = np.inf                            # inflated margin around the start
    assert astar(cost, (7, 10), (2, 10)) is not None


def test_tag_viewpoint_avoids_furniture(office):
    m = RoomMap.from_survey(office)
    tag3 = office["markers"]["3"]
    tx, ty = tag3["x"] * IN, tag3["y"] * IN
    # furniture right in front of tag 3 (the old fixed standpoint)
    for x in np.arange(tx - 0.4, tx + 0.4, 0.02):
        for y in np.arange(ty - 1.0, ty - 0.55, 0.02):
            i, j = m.cell(x, y)
            m.L[j, i] = 3.0
    vp = tag_viewpoint(m, office, 3, (2.0, 2.0))
    assert vp is not None
    (x, y), face, path = vp
    i, j = m.cell(x, y)
    assert np.isfinite(m.costmap()[j, i])
    assert m.line_of_sight((x, y), (tx, ty - 0.05))
    assert abs(math.degrees(math.atan2(ty - y, tx - x)) - face) < 1e-6


class FakeOdo:
    def __init__(self, x=0.0, y=0.0, h=0.0):
        self.x, self.y, self.h = x, y, h
        self.travelled_m = 0.0

    @property
    def pose(self):
        return self.x, self.y, self.h

    def shift(self, dx, dy, dh=0.0):
        self.x += dx
        self.y += dy
        self.h += dh


def test_localizer_applies_and_gates():
    odo = FakeOdo(1.0, 1.0, 90.0)
    loc = Localizer(odo)
    odo.travelled_m = 2.0                                # 2 m since start: ~10 cm sigma
    r = loc.add_fix({"x": 1.08, "y": 1.0, "heading": 91.0, "range_m": 0.8})
    assert r.startswith("used")
    assert 1.0 < odo.x < 1.08
    r = loc.add_fix({"x": 3.0, "y": 1.0, "heading": 91.0, "range_m": 0.8})
    assert r.startswith("gated")


def test_localizer_scoring_mode_never_moves_pose():
    odo = FakeOdo(1.0, 1.0, 90.0)
    loc = Localizer(odo, correct=False)
    odo.travelled_m = 2.0
    loc.add_fix({"x": 1.08, "y": 1.0, "heading": 91.0, "range_m": 0.8})
    assert (odo.x, odo.y) == (1.0, 1.0)


def test_heading_watchdog_reanchors_after_three_fixes():
    odo = FakeOdo(5.0, 1.0, -166.0)                     # the 2026-10-11 tag-1 case
    msgs = []
    loc = Localizer(odo, correct=False, log=msgs.append)
    for _ in range(3):
        loc.add_fix({"x": 5.0, "y": 1.0, "heading": 5.0, "range_m": 1.0})
    assert loc.stats["heading_resets"] == 1
    assert abs(((odo.h - 5.0) + 180) % 360 - 180) < 1.0
    assert any("heading_reset" in m for m in msgs)


def test_position_reset_after_consistent_gated_fixes():
    odo = FakeOdo(1.0, 1.0, 0.0)
    loc = Localizer(odo)
    for _ in range(3):
        loc.add_fix({"x": 2.5, "y": 1.0, "heading": 0.0, "range_m": 0.8})
    assert loc.stats["position_resets"] == 1
    assert abs(odo.x - 2.5) < 1e-6


def test_heading_history_interpolates():
    h = HeadingHistory()
    h.add(0.0, 0.0)
    h.add(1.0, 10.0)
    assert h.at(0.25) == pytest.approx(2.5)
    assert h.at(-1.0) == 0.0
    assert h.at(5.0) == 10.0
