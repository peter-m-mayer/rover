"""
Map-based navigation on the robot (NAV plan step 6): one loop per stop,

    sense (pan sweep: camera free space, sonar, tag fix)
      -> update RoomMap and Localizer
      -> plan (A* on the costmap)
      -> turn to the first waypoint, one heading-held hop into seen-free floor
      -> check progress (a hop that doesn't move = contact: mark it, back off, replan)

Replaces tag_tour.go_to's reactive look-ahead-and-detour, which bounced
between furniture on 2026-10-11 (1 of 5 stops reached).

    nav = Navigator(rig, rec, model, survey, loc_tags)
    nav.start(fix)                    # anchor from a tag fix
    nav.go_to((x, y))                 # True when within tol
    nav.fix()                         # tag fix with the pan centred (or None)
"""

import json
import math
import os
import signal
import time

import numpy as np

from . import config
from .free_space import FreeSpaceScanner
from .heading import drive_straight, turn_to, wrap180
from .localizer import Localizer
from .m1_run import ultrasonic_mm
from .online_odometry import OnlineOdometry
from .planner import plan
from .room_map import RoomMap

SPEED_MPS = config.NAV_SPEED_MPS  # at config.NAV_SPEED on carpet (calibrate.py refits it)
SONAR_AHEAD_M = 0.10          # sonar sits about this far ahead of the robot centre
TURN_STALL_DEG = 20.0         # a turn that ends further off than this stalled
FIX_EVERY_HOPS = 3            # try a tag fix (pan centred) every few hops
STOP_FILE = "/tmp/rover_stop"  # touch it to stop a mission between hops
WEDGED_EVENTS = 4             # this many troubles (stall, contact, block) in one spot = wedged
WEDGED_RADIUS_M = 0.15
WEDGED_SONAR_M = 0.03         # ... with the sonar range barely changing


def install_stop_handlers():
    """Make SIGINT and SIGTERM raise KeyboardInterrupt, so the mission's finally
    block stops the motors and saves the map. A job started with `nohup ... &`
    from a non-interactive shell inherits SIGINT ignored (that's why
    `pkill -INT` didn't stop the 2026-10-12 tour)."""
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        os.remove(STOP_FILE)
    except OSError:
        pass


def check_stop():
    if os.path.exists(STOP_FILE):
        raise KeyboardInterrupt(f"{STOP_FILE} exists")


class Navigator:
    def __init__(self, rig, rec, model, survey, tagloc, correct=True, log=print, prior=None):
        self.rig, self.rec, self.model = rig, rec, model
        self.survey, self.tagloc = survey, tagloc
        self.correct = correct
        self.log = log
        self.map = RoomMap.from_survey(survey)
        if prior:
            ok = self.map.load_prior(prior)
            self.log(f"prior map {prior}: {'loaded' if ok else 'grid mismatch, ignored'}")
        self.fs = FreeSpaceScanner(model, max_range_m=1.6)
        self.odo = None
        self.loc = None
        self.contacts = 0
        self.path = []                       # driven (x, y) for plots
        self.why = None                      # why the last go_to gave up
        self.last_us = float("nan")          # last sonar range (m)
        self.troubles = []                   # (x, y, sonar) of recent stalls, contacts and blocks
        self.events = {"contacts": 0, "blocked": 0, "turn_stalls": 0, "relaxed": 0}

    # ---- setup ---------------------------------------------------------------------
    def start(self, fix):
        self.odo = OnlineOdometry(self.rec, self.rig.hd, self.model).start(fix["x"], fix["y"], fix["heading"])
        self.loc = Localizer(self.odo, correct=self.correct, log=self.mark)
        return self

    def stop(self):
        if self.odo:
            self.odo.stop()

    def mark(self, msg):
        self.log("    " + msg)
        self.rec.mark(msg)

    @property
    def pose(self):
        return self.odo.pose

    def pose_rad(self):
        x, y, h = self.pose
        return x, y, math.radians(h)

    def save(self, folder):
        """Map (npz + png) and nav.json (driven path, trouble counts, localizer stats) for score_run."""
        import cv2
        import os
        self.map.save(os.path.join(folder, "room_map.npz"))
        cv2.imwrite(os.path.join(folder, "room_map.png"), self.map.render(path=self.path, robot=self.pose_rad()))
        with open(os.path.join(folder, "nav.json"), "w") as f:
            json.dump({"path": [[round(x, 3), round(y, 3)] for x, y in self.path], "events": self.events,
                       "localizer": self.loc.stats if self.loc else None}, f)

    # ---- sensing -------------------------------------------------------------------
    def frame_after(self, t, timeout=1.0):
        """First recorded frame captured after monotonic time t (+ camera latency)."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            item = self.rec._latest
            if item is not None and item[0] - config.CAMERA_LATENCY_S > t:
                return item[1]
            time.sleep(0.02)
        return self.rig.capture_color()

    def look(self, pan_offset=0, settle=0.35):
        c0 = config.SERVO_PAN_CENTER
        if pan_offset == 0:                                   # centre: approach from below
            self.rig.servo(c0 - 10, config.SERVO_TILT_LEVEL, settle=0.2)
        self.rig.servo(c0 + pan_offset, config.SERVO_TILT_LEVEL, settle=settle)
        return self.frame_after(time.monotonic())

    def sense(self, sweep=True):
        """Update the map from a sweep (left, centre, right) and the sonar; returns
        the free distance straight ahead (min of sonar and camera, m)."""
        offsets = (-40, 0, 40) if sweep else (0,)
        ahead = 9.9
        us = ultrasonic_mm(self.rig, 3) / 1000
        self.last_us = us
        for off in offsets:
            frame = self.look(off)
            x, y, th = self.pose_rad()
            if off == 0:
                if us > 0.5:                     # the reference patch ahead is clear floor
                    self.fs.learn(frame)
                self.map.add_sonar((x, y, th), us)
            res = self.fs.scan(frame)
            if res is None:
                continue
            b, r, h = res
            self.map.add_camera_scan((x, y, th + math.radians(off)), b, r, h)
            if off == 0:
                c = np.abs(np.degrees(b)) < 10
                if c.any():
                    ahead = min(ahead, float(np.min(r[c])))
        if sweep:
            self.look(0, settle=0.2)
        self.map.add_footprint(*self.pose[:2])
        self.map.decay_contacts()
        return min(v for v in (us, ahead) if v == v)

    def fix(self, frames=5):
        """Tag fix with the pan centred; feeds the Localizer. Returns the fix or None."""
        self.look(0)
        ests = [e for e in (self.tagloc.localize(self.rig.gray()) for _ in range(frames)) if e]
        if len(ests) < config.TAG_MIN_FRAMES:
            return None
        med = {k: float(np.median([e[k] for e in ests])) for k in ("x", "y", "z", "yaw", "range_m")}
        med.update(heading=med["yaw"], tags=ests[0]["tags"], n=len(ests))
        if self.loc is not None:
            med["result"] = self.loc.add_fix(med)
            self.rec.mark("fix " + json.dumps(dict(med, ok=True, dr=list(self.pose))))
        return med

    # ---- moving --------------------------------------------------------------------
    def turn_to_heading(self, room_deg, retry=True):
        """Turn to a room heading; on a stalled turn back off once and retry.
        Returns the final error (deg)."""
        _, _, h = self.pose
        err = turn_to(self.rig.act, self.rig.hd, self.rig.hd.heading_deg + wrap180(room_deg - h))
        if abs(err) > TURN_STALL_DEG and retry:
            self.events["turn_stalls"] += 1
            self.trouble()
            self.mark(f"turn stalled {err:+.0f} deg short: backing off and retrying")
            self.back_off()
            err = self.turn_to_heading(room_deg, retry=False)
        return err

    def hop(self, seconds):
        x0, y0, _ = self.pose
        drive_straight(self.rig.act, self.rig.hd, config.NAV_SPEED, seconds)
        time.sleep(0.35)                          # let floor odometry catch up
        x1, y1, _ = self.pose
        moved = math.hypot(x1 - x0, y1 - y0)
        self.path.append((x1, y1))
        return moved

    def back_off(self, metres=0.15):
        drive_straight(self.rig.act, self.rig.hd, -config.NAV_SPEED, max(config.NAV_MIN_HOP_S, metres / SPEED_MPS))
        time.sleep(0.3)

    def trouble(self):
        """Note a stall, contact or block here; True when the robot is wedged:
        several troubles in one spot with the sonar range barely changing
        (2026-10-12: against the south wall, sonar 272 mm, every turn stalled)."""
        x, y, _ = self.pose
        self.troubles = [t for t in self.troubles
                         if math.hypot(t[0] - x, t[1] - y) < WEDGED_RADIUS_M] + [(x, y, self.last_us)]
        recent = self.troubles[-WEDGED_EVENTS:]
        sonar = [t[2] for t in recent if t[2] == t[2]]
        return (len(recent) >= WEDGED_EVENTS and len(sonar) == len(recent)
                and max(sonar) - min(sonar) < WEDGED_SONAR_M)

    def wedged(self):
        if self.trouble():
            self.mark("wedged: troubles in one spot with an unchanging sonar range; stopping")
            self.why = "wedged"
            return True
        return False

    def go_to(self, goal, tol=0.15, max_hops=60, max_contacts=3):
        """Drive to room point `goal`; True when within tol. On False, self.why
        says why: no path, blocked, contacts, stuck, wedged or hops."""
        contacts = blocked = 0
        self.why = None
        for k in range(max_hops):
            check_stop()
            x, y, h = self.pose
            dist = math.hypot(goal[0] - x, goal[1] - y)
            if dist < tol:
                return True
            self.sense(sweep=(k % 2 == 0))
            if self.loc is not None and k % FIX_EVERY_HOPS == FIX_EVERY_HOPS - 1:
                self.fix()                       # corrections en route, not only at stops
                x, y, h = self.pose
            path = plan(self.map, (x, y), goal)
            if path is None:
                path = plan(self.map, (x, y), goal, relaxed=True)
                if path is not None:
                    self.events["relaxed"] += 1
                    self.mark("no normal path: using the relaxed plan (contacts ignored, tight margin)")
            if path is None or len(path) < 2:
                self.mark(f"no path to ({goal[0]:.2f}, {goal[1]:.2f})")
                self.why = "no path"
                return False
            wx, wy = path[1]
            leg = math.hypot(wx - x, wy - y)
            bearing = math.degrees(math.atan2(wy - y, wx - x))
            if abs(wrap180(bearing - h)) > 6:
                if abs(self.turn_to_heading(bearing)) > TURN_STALL_DEG:
                    if self.wedged():
                        return False
                    self.mark("can't turn here: giving up on this goal")
                    self.why = "stuck"
                    return False
            free = self.sense(sweep=False)
            room = free - config.NAV_FRONT_CLEAR_M
            if room < SPEED_MPS * (config.NAV_MIN_HOP_S - config.NAV_START_LAG_S):
                # something the map didn't know is right ahead: mark it lethal (one
                # weak sonar reading doesn't flip the cells, and the planner would
                # choose the same path again), then replan
                self.mark(f"blocked ahead ({free:.2f} m): replanning")
                self.map.add_contact(self.pose_rad(), ahead_m=free + SONAR_AHEAD_M)
                self.events["blocked"] += 1
                if self.wedged():
                    return False
                if free < 0.12:
                    self.back_off()              # too close to turn freely
                blocked += 1
                if blocked >= max_contacts * 2:
                    self.mark("blocked too often: giving up on this goal")
                    self.why = "blocked"
                    return False
                continue
            seconds = float(np.clip(min(leg, room) / SPEED_MPS + config.NAV_START_LAG_S,
                                    config.NAV_MIN_HOP_S, config.NAV_HOP_S))
            moved = self.hop(seconds)
            expected = SPEED_MPS * (seconds - config.NAV_START_LAG_S)
            if moved < config.NAV_STALL_FRAC * expected:
                contacts += 1
                self.contacts += 1
                self.mark(f"contact: hop moved {moved * 100:.0f} cm of {expected * 100:.0f}")
                self.events["contacts"] += 1
                self.map.add_contact(self.pose_rad())
                if self.wedged():
                    return False
                self.back_off()
                if contacts >= max_contacts:
                    self.mark("too many contacts: giving up on this goal")
                    self.why = "contacts"
                    return False
        self.why = "hops"
        return False
