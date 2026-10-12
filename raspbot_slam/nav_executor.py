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

SPEED_MPS = 0.28              # speed 50 on carpet
SONAR_AHEAD_M = 0.10          # sonar sits about this far ahead of the robot centre
FIX_EVERY_HOPS = 3            # try a tag fix (pan centred) every few hops


class Navigator:
    def __init__(self, rig, rec, model, survey, tagloc, correct=True, log=print):
        self.rig, self.rec, self.model = rig, rec, model
        self.survey, self.tagloc = survey, tagloc
        self.correct = correct
        self.log = log
        self.map = RoomMap.from_survey(survey)
        self.fs = FreeSpaceScanner(model, max_range_m=1.6)
        self.odo = None
        self.loc = None
        self.contacts = 0
        self.path = []                       # driven (x, y) for plots
        self.why = None                      # why the last go_to gave up

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
    def turn_to_heading(self, room_deg):
        _, _, h = self.pose
        turn_to(self.rig.act, self.rig.hd, self.rig.hd.heading_deg + wrap180(room_deg - h))

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

    def go_to(self, goal, tol=0.15, max_hops=60, max_contacts=3):
        """Drive to room point `goal`; True when within tol."""
        contacts = blocked = 0
        self.why = None
        for k in range(max_hops):
            x, y, h = self.pose
            dist = math.hypot(goal[0] - x, goal[1] - y)
            if dist < tol:
                return True
            self.sense(sweep=(k % 2 == 0))
            if self.loc is not None and k % FIX_EVERY_HOPS == FIX_EVERY_HOPS - 1:
                self.fix()                       # corrections en route, not only at stops
                x, y, h = self.pose
            path = plan(self.map, (x, y), goal)
            if path is None or len(path) < 2:
                self.mark(f"no path to ({goal[0]:.2f}, {goal[1]:.2f})")
                self.why = "no path"
                return False
            wx, wy = path[1]
            leg = math.hypot(wx - x, wy - y)
            bearing = math.degrees(math.atan2(wy - y, wx - x))
            if abs(wrap180(bearing - h)) > 6:
                self.turn_to_heading(bearing)
            free = self.sense(sweep=False)
            room = free - config.NAV_FRONT_CLEAR_M
            if room < 0.05:
                # something the map didn't know is right ahead: mark it lethal (one
                # weak sonar reading doesn't flip the cells, and the planner would
                # choose the same path again), then replan
                self.mark(f"blocked ahead ({free:.2f} m): replanning")
                self.map.add_contact(self.pose_rad(), ahead_m=free + SONAR_AHEAD_M)
                blocked += 1
                if blocked >= max_contacts * 2:
                    self.mark("blocked too often: giving up on this goal")
                    self.why = "blocked"
                    return False
                continue
            seconds = float(np.clip(min(leg, room) / SPEED_MPS, config.NAV_MIN_HOP_S, config.NAV_HOP_S))
            moved = self.hop(seconds)
            expected = SPEED_MPS * seconds
            if moved < config.NAV_STALL_M and expected > 0.05:
                contacts += 1
                self.contacts += 1
                self.mark(f"contact: hop moved {moved * 100:.0f} cm of {expected * 100:.0f}")
                self.map.add_contact(self.pose_rad())
                self.back_off()
                if contacts >= max_contacts:
                    self.mark("too many contacts: giving up on this goal")
                    self.why = "contacts"
                    return False
        self.why = "hops"
        return False
