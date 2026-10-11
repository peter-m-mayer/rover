"""
Absolute pose from the surveyed wall tags (AUTONOMY_PLAN.md M0/M2 ground truth).

The room survey (surveys/<name>.json) gives each ArUco tag's centre (x, y) in
inches, its height, and the direction its face points (+x, -x, +y, -y). From
one camera frame with one or more tags in view, solvePnP on all their corners
gives the camera's position and heading in room coordinates:

    x, y   metres, room frame (origin and axes as in the survey)
    z      camera height above the floor (m)
    yaw    direction of the optical axis on the floor, degrees CCW from +x
    pitch  optical axis above (+) / below (-) horizontal, degrees

Robot heading = camera yaw - (pan - SERVO_PAN_CENTER) (pan > centre looks
left), give or take the pan servo's few degrees of backlash.
"""

import json
import math
import os

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from . import config

IN = 0.0254
FACING = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}
SURVEY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surveys")


def load_survey(name="office"):
    path = name if name.endswith(".json") else os.path.join(SURVEY_DIR, name + ".json")
    with open(path) as f:
        return json.load(f)


def tag_corners_world(tag, size_m, height_m):
    """World corners (m) in ArUco order TL, TR, BR, BL as seen facing the tag."""
    n = np.array(FACING[tag["facing"]], float)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(-n, up)                  # viewer looks along -n
    c = np.array([tag["x"] * IN, tag["y"] * IN, height_m])
    h = size_m / 2
    return np.array([c + h * (-right + up), c + h * (right + up),
                     c + h * (right - up), c + h * (-right - up)])


class TagLocalizer:
    def __init__(self, K, dist=None, survey="office", size_m=None):
        self.K = np.asarray(K, float)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, float).ravel()
        self.survey = load_survey(survey) if isinstance(survey, str) else survey
        size = size_m or config.ARUCO_MARKER_SIZE_M
        height = self.survey.get("marker_center_height_in", 10.25) * IN
        self.world = {int(k): tag_corners_world(t, size, height)
                      for k, t in self.survey["markers"].items()}
        if cv2 is not None:
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.det = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT)), params)

    def detect(self, gray):
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if int(i) in self.world}

    def solve(self, detections, max_dz=0.05, max_pitch=20.0, max_single_range=2.5):
        """Camera pose from {tag_id: (4, 2) image corners}; None if nothing usable.

        A small flat target seen from afar has two nearly equally good poses
        (the planar-PnP flip). Both candidates are scored against what we
        know -- camera height ~CAMERA_HEIGHT_M, optical axis near level -- and
        the pose is rejected if neither fits.
        """
        if not detections:
            return None
        ids = sorted(detections)
        obj = np.concatenate([self.world[i] for i in ids]).astype(np.float64)
        img = np.concatenate([detections[i] for i in ids]).astype(np.float64)
        method = cv2.SOLVEPNP_IPPE if len(ids) == 1 else cv2.SOLVEPNP_SQPNP
        n, rvs, tvs, _ = cv2.solvePnPGeneric(obj, img, self.K, self.dist, flags=method)
        best = None
        for rv, tv in zip(rvs[:n], tvs[:n]):
            if len(ids) > 1:
                _, rv, tv = cv2.solvePnP(obj, img, self.K, self.dist, rv.copy(), tv.copy(),
                                         useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            est = self._pose(rv, tv, obj, img, ids)
            est["dz"] = abs(est["z"] - config.CAMERA_HEIGHT_M)
            if est["dz"] > max_dz or abs(est["pitch"]) > max_pitch:
                continue
            if len(ids) == 1 and est["range_m"] > max_single_range:
                continue            # one small tag far away: position unreliable
            if best is None or est["reproj_px"] + 20 * est["dz"] < best["reproj_px"] + 20 * best["dz"]:
                best = est
        return best

    def _pose(self, rv, tv, obj, img, ids):
        R, _ = cv2.Rodrigues(rv)
        cam = (-R.T @ tv).ravel()                 # camera centre in the world
        axis = R.T @ np.array([0.0, 0.0, 1.0])    # optical axis in the world
        proj, _ = cv2.projectPoints(obj, rv, tv, self.K, self.dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        rng = float(np.linalg.norm(obj.mean(0) - cam))
        return {"x": float(cam[0]), "y": float(cam[1]), "z": float(cam[2]),
                "yaw": math.degrees(math.atan2(axis[1], axis[0])),
                "pitch": math.degrees(math.asin(np.clip(axis[2], -1, 1))),
                "tags": ids, "reproj_px": err, "range_m": rng}

    def localize(self, gray):
        return self.solve(self.detect(gray))


def robot_heading(cam_yaw_deg, pan):
    """Body heading from camera yaw and the pan command (pan > centre looks left)."""
    return (cam_yaw_deg - (pan - config.SERVO_PAN_CENTER) + 180) % 360 - 180
