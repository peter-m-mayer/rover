"""Learning across runs: persistent obstacles, prior maps, run scoring, wedged detection."""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from raspbot_slam.offline import score_run
from raspbot_slam.room_map import RoomMap
from raspbot_slam.tag_localizer import load_survey


@pytest.fixture
def office():
    return load_survey("office")


def scan(m, pose, r=1.0, n=9):
    b = np.radians(np.linspace(-20, 20, n))
    m.add_camera_scan(pose, b, np.full(n, r), np.ones(n, bool), cam_offset_m=0.0)


def test_obstacle_seen_from_two_places_survives_free_rays(office):
    m = RoomMap.from_survey(office)
    target = (3.0, 2.0)
    scan(m, (2.0, 2.0, 0.0))
    scan(m, (2.0, 2.0, 0.0))
    scan(m, (2.0, 2.3, -math.atan2(0.3, 1.0)), r=math.hypot(1.0, 0.3))
    c = m.cell(*target)[::-1]
    assert m.solid()[c]
    # later views see "floor" through it (a boundary one cell further away)
    for _ in range(8):
        m.add_camera_scan((2.0, 2.0, 0.0), np.zeros(1), np.full(1, 1.2), np.zeros(1, bool), cam_offset_m=0.0)
    assert m.L[c] < 0.8                     # log-odds alone would have erased it
    assert m.occupied()[c]


def test_hits_from_one_place_are_not_solid(office):
    m = RoomMap.from_survey(office)
    for _ in range(5):
        scan(m, (2.0, 2.0, 0.0))
    c = m.cell(3.0, 2.0)[::-1]
    assert not m.solid()[c]


def test_driving_over_a_solid_cell_clears_it(office):
    m = RoomMap.from_survey(office)
    for p in ((2.0, 2.0, 0.0), (2.0, 2.0, 0.0), (2.0, 2.3, -math.atan2(0.3, 1.0))):
        scan(m, p, r=math.hypot(1.0, p[1] - 2.0))
    m.add_footprint(3.0, 2.0)
    assert not m.occupied()[m.cell(3.0, 2.0)[::-1]]


def test_prior_round_trip_keeps_solid_and_contacts(office, tmp_path):
    m = RoomMap.from_survey(office)
    for p in ((2.0, 2.0, 0.0), (2.0, 2.0, 0.0), (2.0, 2.3, -math.atan2(0.3, 1.0))):
        scan(m, p, r=math.hypot(1.0, p[1] - 2.0))
    m.add_contact((1.0, 1.0, 0.0))
    f = str(tmp_path / "prior.npz")
    m.save(f)
    n = RoomMap.from_survey(office)
    assert n.load_prior(f)
    assert n.occupied()[n.cell(3.0, 2.0)[::-1]]
    assert n.solid()[n.cell(1.18, 1.0)[::-1]]           # the contact was saved as solid
    assert abs(float(n.L.max()) - 0.5 * float(m.L.max())) < 1e-5
    other = RoomMap(0, 0, 1, 1)
    assert not other.load_prior(f)                      # different grid: ignored


def test_score_counts_events_and_stops(tmp_path, office):
    rows = [(0.0, "start"),
            (0.5, "battery " + json.dumps({"v_mps": 0.3, "ratio": 0.86, "verdict": "ok"})),
            (1.0, "blocked ahead (0.20 m): replanning"),
            (2.0, "contact: hop moved 1 cm of 6"),
            (3.0, "turn stalled -40 deg short: backing off and retrying"),
            (4.0, "blocked too often: giving up on this goal"),
            (5.0, "tourstop " + json.dumps({"leg": 0, "tag": 1, "reached": True, "err_m": 0.1, "err_deg": -2.0})),
            (6.0, "tourstop " + json.dumps({"leg": 1, "tag": 0, "reached": False, "fix": None, "why": "blocked"})),
            (9.0, "stop")]
    import csv
    with open(tmp_path / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "label"])
        w.writerows(rows)
    with open(tmp_path / "nav.json", "w") as f:
        json.dump({"path": [[0, 0], [3, 0], [3, 4]]}, f)
    m = RoomMap.from_survey(office)
    scan(m, (2.0, 2.0, 0.0))
    m.save(str(tmp_path / "room_map.npz"))
    s = score_run.score(str(tmp_path))
    assert (s["goals"], s["reached"]) == (2, 1)
    assert (s["contacts"], s["blocked"], s["turn_stalls"], s["gave_up"]) == (1, 1, 1, 1)
    assert s["why"] == ["blocked"]
    assert (s["battery_mps"], s["battery_ratio"]) == (0.3, 0.86)
    assert s["distance_m"] == pytest.approx(7.0)
    assert s["seconds"] == pytest.approx(9.0)
    assert 0 < s["coverage"] < 0.2
    out = tmp_path / "scores.csv"
    score_run.append_csv(s, str(out))
    score_run.append_csv(s, str(out))                    # re-scoring replaces the row
    assert len(out.read_text().strip().splitlines()) == 2


def test_wedged_after_troubles_in_one_spot():
    from raspbot_slam.nav_executor import Navigator
    nav = Navigator.__new__(Navigator)
    nav.troubles = []
    nav.odo = SimpleNamespace(pose=(1.0, 1.0, 0.0))
    nav.last_us = 0.272
    assert not any(nav.trouble() for _ in range(3))
    assert nav.trouble()                                # 4th trouble, same spot, same sonar
    nav.troubles = []
    for k in range(6):
        nav.last_us = 0.27 + 0.05 * k                    # the sonar changes: still making progress
        assert not nav.trouble()


def test_explore_picks_reachable_goal_toward_unseen_floor(office):
    from raspbot_slam.explore import pick_goal
    m = RoomMap.from_survey(office)
    # the south half (y < 2.5) has been seen and is free; the north half is unseen
    xs = m.x0 + (np.arange(m.nx) + 0.5) * m.res
    ys = m.y0 + (np.arange(m.ny) + 0.5) * m.res
    gx, gy = np.meshgrid(xs, ys)
    south = (gy < 2.5) & ~m.wall
    m.seen[south] = True
    m.L[south] = -2.0
    g = pick_goal(m, (2.0, 1.0))
    assert g is not None
    (x, y), gain, d = g
    assert y > 1.6 and gain > 0.5                 # heads toward the unseen half
    assert m.free()[m.cell(x, y)[::-1]]
    # a failed goal isn't picked again
    g2 = pick_goal(m, (2.0, 1.0), failed=[(x, y)])
    assert g2 is not None and math.hypot(g2[0][0] - x, g2[0][1] - y) >= 0.4
    # everything seen: nothing left
    m.seen[~m.wall] = True
    assert pick_goal(m, (2.0, 1.0)) is None


def test_speed_check_verdicts():
    from raspbot_slam import nav_executor as ne
    nav = ne.Navigator.__new__(ne.Navigator)
    marks = []
    nav.rec = type("R", (), {"mark": lambda self, m: marks.append(m)})()
    nav.sense = lambda sweep=True: 1.0
    for per_hop, verdict in ((0.14, "ok"), (0.115, "weak"), (0.08, "low")):
        nav.hop = lambda seconds, reverse=False, d=per_hop: d
        b = nav.speed_check(seconds=0.5)
        assert b["verdict"] == verdict, (per_hop, b)
    assert marks[-1].startswith("battery ")
    nav.sense = lambda sweep=True: 0.3                    # no room ahead: don't drive
    nav.hop = None
    assert nav.speed_check()["verdict"] in ("skipped", "low")
