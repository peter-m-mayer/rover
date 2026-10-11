"""Tag localiser: recover a known camera pose from projected survey tags."""

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from raspbot_slam.tag_localizer import TagLocalizer, robot_heading, tag_corners_world

K = np.array([[480.0, 0, 320], [0, 480.0, 240], [0, 0, 1]])
SURVEY = {
    "marker_center_height_in": 10.25,
    "markers": {
        "0": {"x": 4.25, "y": 0.0, "facing": "+y"},
        "1": {"x": 237.0, "y": 34.0, "facing": "-x"},
        "4": {"x": 0.0, "y": 118.0, "facing": "+x"},
    },
}


def camera_pose(x, y, z, yaw_deg, pitch_deg=0.0):
    """World->camera (R, t) for a camera at (x, y, z) looking along yaw/pitch."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])        # rows: camera x (right), y (down), z (forward)
    t = -R @ np.array([x, y, z])
    return R, t


def project(loc, ids, R, t):
    det = {}
    for i in ids:
        p = (R @ loc.world[i].T).T + t
        det[i] = (K @ (p / p[:, 2:3]).T).T[:, :2]
    return det


def test_corners_face_the_room():
    c = tag_corners_world({"x": 0.0, "y": 100.0, "facing": "+x"}, 0.15, 0.26)
    assert np.allclose(c[:, 0], 0.0)                     # on the x = 0 wall
    assert c[0, 2] > c[3, 2]                             # TL above BL
    # facing +x: the viewer stands at +x looking toward -x, so their right is +y
    assert c[1, 1] > c[0, 1]


@pytest.mark.parametrize("ids", [[4], [0, 4], [0, 1, 4]])
def test_recovers_pose(ids):
    loc = TagLocalizer(K, survey=SURVEY, size_m=0.1429)
    # camera 1.2 m into the room, 0.11 m up, looking roughly at tag 4 / tag 0
    R, t = camera_pose(1.2, 1.5, 0.107, yaw_deg=200, pitch_deg=4)
    vis = [i for i in ids if np.all((R @ loc.world[i].T).T[:, 2] + t[2] > 0)]
    if not vis:
        pytest.skip("tag behind the camera in this pose")
    est = loc.solve(project(loc, vis, R, t))
    assert est is not None and est["reproj_px"] < 0.5
    assert (est["x"], est["y"], est["z"]) == pytest.approx((1.2, 1.5, 0.107), abs=0.01)
    assert est["yaw"] == pytest.approx(-160.0, abs=0.5)
    assert est["pitch"] == pytest.approx(4.0, abs=0.5)


def test_robot_heading_removes_pan():
    from raspbot_slam import config
    assert robot_heading(30.0, config.SERVO_PAN_CENTER + 20) == pytest.approx(10.0)
    assert robot_heading(170.0, config.SERVO_PAN_CENTER - 20) == pytest.approx(-170.0)
