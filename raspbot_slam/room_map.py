"""
Occupancy map in room coordinates (NAV plan step 3), used live and offline.

    m = RoomMap.from_survey(load_survey("office"))
    m.add_camera_scan(pose, bearings, ranges, hits)    # free_space.FreeSpaceScanner.scan
    m.add_sonar(pose, range_m)                         # forward sonar, as a cone
    m.add_contact(pose)                                # stall / bump: lethal ahead
    m.add_footprint(x, y)                              # where the robot has been: free
    cost = m.costmap()                                 # for planner.py

Evidence is log-odds per 4 cm cell. Surveyed walls (and the alcove mirror) are
seeded lethal and never cleared: the camera sees "floor" in the mirror, and the
sonar slides off walls at shallow angles. Contacts are lethal too but fade
slowly (furniture gets moved; a bad pose can put one in the wrong cell).

Why both sensors: the sonar can't see low things (the round desk's ring base,
2026-10-11) and the camera floor model can't see overhangs (a desk top); a cell
either one calls occupied stays occupied until seen free many times.
"""

import math

import numpy as np

from .tag_localizer import IN

L_FREE, L_OCC, L_MIN, L_MAX = -0.4, 1.2, -4.0, 4.0
OCC_T, FREE_T = 0.8, -0.5           # log-odds thresholds for occupied / free
ROBOT_RADIUS_M = 0.13               # chassis half-diagonal (~0.24 x 0.16 m) plus camera head
SOFT_MARGIN_M = 0.10                # extra cost band outside the lethal inflation
SONAR_HALF_ANGLE = math.radians(7.5)


class RoomMap:
    def __init__(self, x0, y0, x1, y1, res=0.04):
        """Covers [x0, x1] x [y0, y1] metres (room frame)."""
        self.res = res
        self.x0, self.y0 = x0, y0
        self.nx = int(math.ceil((x1 - x0) / res))
        self.ny = int(math.ceil((y1 - y0) / res))
        self.L = np.zeros((self.ny, self.nx), np.float32)       # [row = y, col = x]
        self.wall = np.zeros((self.ny, self.nx), bool)          # surveyed, permanent
        self.contact = np.zeros((self.ny, self.nx), np.float32)  # 1 = fresh contact, fades
        self.seen = np.zeros((self.ny, self.nx), bool)

    # ---- construction -----------------------------------------------------------
    @classmethod
    def from_survey(cls, survey, res=0.04, margin=0.3):
        X, Y = survey["xmax_in"] * IN, survey["ymax_in"] * IN
        a = survey.get("alcove")
        xlo = (a["x0"] * IN if a else 0.0) - margin
        m = cls(xlo, -margin, X + margin, Y + margin, res)
        walls = [((0, 0), (X, 0)), ((X, 0), (X, Y)), ((X, Y), (0, Y))]
        if a:
            # the alcove opens in the x=0 wall between y0 and y1
            ax0, ay0, ay1 = a["x0"] * IN, a["y0"] * IN, a["y1"] * IN
            walls += [((0, Y), (0, ay1)), ((0, ay1), (ax0, ay1)), ((ax0, ay1), (ax0, ay0)),
                      ((ax0, ay0), (0, ay0)), ((0, ay0), (0, 0))]
            mir = a.get("sliding_door_mirror")
            if mir:
                walls.append(((mir["x"] * IN, mir["y0"] * IN), (mir["x"] * IN, mir["y1"] * IN)))
        else:
            walls.append(((0, Y), (0, 0)))
        for p, q in walls:
            m.add_wall(p, q)
        # outside the surveyed outline is not floor we can drive on
        inside = np.zeros_like(m.wall)
        xs = m.x0 + (np.arange(m.nx) + 0.5) * res
        ys = m.y0 + (np.arange(m.ny) + 0.5) * res
        gx, gy = np.meshgrid(xs, ys)
        inside |= (gx >= 0) & (gx <= X) & (gy >= 0) & (gy <= Y)
        if a:
            inside |= (gx >= a["x0"] * IN) & (gx <= 0) & (gy >= a["y0"] * IN) & (gy <= a["y1"] * IN)
        m.wall |= ~inside
        return m

    def add_wall(self, p, q, thick_m=0.04):
        n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / (self.res * 0.5)))
        r = max(0, int(round(thick_m / 2 / self.res)))
        for x, y in zip(np.linspace(p[0], q[0], n), np.linspace(p[1], q[1], n)):
            i, j = self.cell(x, y)
            self.wall[max(0, j - r):j + r + 1, max(0, i - r):i + r + 1] = True

    # ---- geometry ----------------------------------------------------------------
    def cell(self, x, y):
        """(col, row) of a room point (may lie outside the grid)."""
        return int(math.floor((x - self.x0) / self.res)), int(math.floor((y - self.y0) / self.res))

    def centre(self, i, j):
        return self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res

    def inside(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny

    def _ray_cells(self, x0, y0, x1, y1):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / (self.res * 0.5)))
        cells = []
        for x, y in zip(np.linspace(x0, x1, n + 1), np.linspace(y0, y1, n + 1)):
            c = self.cell(x, y)
            if not cells or cells[-1] != c:
                cells.append(c)
        return cells

    # ---- evidence ----------------------------------------------------------------
    def ray(self, x0, y0, x1, y1, hit, w_free=1.0, w_occ=1.0):
        cells = self._ray_cells(x0, y0, x1, y1)
        end = cells[-1]
        for i, j in cells[:-1]:
            if self.inside(i, j):
                self.L[j, i] = max(L_MIN, self.L[j, i] + w_free * L_FREE)
                self.seen[j, i] = True
        i, j = end
        if self.inside(i, j):
            if hit:
                self.L[j, i] = min(L_MAX, self.L[j, i] + w_occ * L_OCC)
            else:
                self.L[j, i] = max(L_MIN, self.L[j, i] + w_free * L_FREE)
            self.seen[j, i] = True

    def add_camera_scan(self, pose, bearings, ranges, hits, cam_offset_m=0.06):
        """pose = (x, y, heading_rad) of the robot; bearings relative to the camera axis.
        The camera sits ~6 cm ahead of the turning centre."""
        x, y, th = pose
        cx, cy = x + cam_offset_m * math.cos(th), y + cam_offset_m * math.sin(th)
        for b, r, h in zip(bearings, ranges, hits):
            if b != b:
                continue
            a = th + b
            self.ray(cx, cy, cx + r * math.cos(a), cy + r * math.sin(a), bool(h))

    def add_sonar(self, pose, range_m, max_m=2.5, min_m=0.03):
        """Forward sonar as a cone: free inside it, occupied on its arc (weak per ray)."""
        x, y, th = pose
        if not (range_m == range_m) or range_m < min_m:
            return
        hit = range_m <= max_m
        r = min(range_m, max_m)
        k = 5
        for a in np.linspace(-SONAR_HALF_ANGLE, SONAR_HALF_ANGLE, k):
            self.ray(x, y, x + r * math.cos(th + a), y + r * math.sin(th + a), hit,
                     w_free=0.6, w_occ=2.0 / k)

    def add_contact(self, pose, ahead_m=0.18, half_width_m=0.12):
        """The robot pushed against something it didn't see: the strip just ahead
        of its front is lethal."""
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        for d in np.arange(ahead_m - 0.04, ahead_m + 0.05, self.res / 2):
            for w in np.arange(-half_width_m, half_width_m + 1e-9, self.res / 2):
                i, j = self.cell(x + d * c - w * s, y + d * s + w * c)
                if self.inside(i, j):
                    self.contact[j, i] = 1.0
                    self.seen[j, i] = True

    def add_footprint(self, x, y, radius_m=0.10):
        r = int(radius_m / self.res)
        i, j = self.cell(x, y)
        j0, j1, i0, i1 = max(0, j - r), min(self.ny, j + r + 1), max(0, i - r), min(self.nx, i + r + 1)
        if j0 >= j1 or i0 >= i1:
            return
        self.L[j0:j1, i0:i1] = np.minimum(self.L[j0:j1, i0:i1], L_MIN / 2)
        self.contact[j0:j1, i0:i1] = 0.0
        self.seen[j0:j1, i0:i1] = True

    def decay_contacts(self, factor=0.98):
        self.contact *= factor

    # ---- products ----------------------------------------------------------------
    def occupied(self):
        return self.wall | (self.L > OCC_T) | (self.contact > 0.3)

    def free(self):
        return ~self.occupied() & (self.L < FREE_T)

    def unknown(self):
        return ~self.occupied() & ~self.free()

    def costmap(self, radius_m=ROBOT_RADIUS_M, soft_m=SOFT_MARGIN_M, unknown_cost=3.0):
        """Per-cell traversal cost: inf within radius_m of an obstacle, a falling
        penalty in the soft band beyond it, unknown_cost in unseen cells, 1 elsewhere."""
        import cv2
        occ = self.occupied().astype(np.uint8)
        dist = cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 5) * self.res
        cost = np.ones(occ.shape, np.float32)
        cost[self.unknown()] = unknown_cost
        band = (dist > radius_m) & (dist < radius_m + soft_m)
        cost[band] += 4.0 * (1 - (dist[band] - radius_m) / soft_m)
        cost[dist <= radius_m] = np.inf
        return cost

    def clearance(self):
        """Distance (m) from each cell to the nearest obstacle."""
        import cv2
        occ = self.occupied().astype(np.uint8)
        return cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 5) * self.res

    def line_of_sight(self, p, q):
        """True if no occupied cell (walls excluded at the end point) lies between p and q."""
        occ = self.occupied()
        cells = self._ray_cells(p[0], p[1], q[0], q[1])[:-2]
        return all(not (self.inside(i, j) and occ[j, i]) for i, j in cells)

    def render(self, scale=4, path=None, robot=None, marks=()):
        """BGR image, +y up: walls blue, obstacles black, free white, unknown grey."""
        import cv2
        img = np.full((self.ny, self.nx, 3), 200, np.uint8)
        img[self.free()] = (250, 250, 250)
        img[(self.L > OCC_T) & ~self.wall] = (30, 30, 30)
        img[(self.contact > 0.3) & ~self.wall] = (0, 0, 200)
        img[self.wall] = (160, 110, 40)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        def px(x, y):
            return (int((x - self.x0) / self.res * scale), int((y - self.y0) / self.res * scale))
        if path is not None and len(path) > 1:
            cv2.polylines(img, [np.array([px(x, y) for x, y in path], np.int32)], False, (0, 0, 255), 1)
        for (x, y), colour in marks:
            cv2.circle(img, px(x, y), 4, colour, -1)
        if robot is not None:
            x, y, th = robot
            cv2.circle(img, px(x, y), 5, (0, 160, 0), -1)
            cv2.line(img, px(x, y), px(x + 0.2 * math.cos(th), y + 0.2 * math.sin(th)), (0, 160, 0), 2)
        return cv2.flip(img, 0)

    def save(self, path):
        np.savez_compressed(path, L=self.L, wall=self.wall, contact=self.contact, seen=self.seen,
                            geom=np.array([self.x0, self.y0, self.res, self.nx, self.ny]))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        x0, y0, res, nx, ny = d["geom"]
        m = cls(x0, y0, x0 + nx * res, y0 + ny * res, res)
        m.L, m.wall, m.contact, m.seen = d["L"], d["wall"], d["contact"], d["seen"]
        return m
