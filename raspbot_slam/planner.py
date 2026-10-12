"""
Path planning on a RoomMap (NAV plan step 5).

    path = plan(m, start_xy, goal_xy)          # [(x, y), ...] or None
    vp = tag_viewpoint(m, survey, tag, start)  # where to stand to read a tag

One Dijkstra from the robot over the costmap (8-connected, edge cost = step
length x mean cell cost, scipy.sparse.csgraph: ~0.1 s for the office on a PC
where a pure-Python A* took 0.8 s per goal) gives the cost to every cell, so
choosing among many goals (tag viewpoints, frontiers) costs one search. Paths
keep away from furniture because cells near obstacles cost more. The start's
neighbourhood is exempt from the lethal inflation (after a contact the robot
may sit inside its own safety margin), and the cell path is shortened to
straight runs that stay clear (shortcut).

Tag viewpoints (why: on 2026-10-11 the fixed "0.8 m in front of the tag"
standpoints were inside furniture for 3 of 5 tags): candidate cells 0.5-1.4 m
out from the tag, within +-35 deg of its normal, drivable, with a clear line of
sight; the best mix of reach cost and viewing geometry wins.
"""

import math

import numpy as np

from .room_map import ROBOT_RADIUS_M
from .tag_localizer import FACING, IN

SQ2 = math.sqrt(2.0)
START_FREE_CELLS = 4
START_ESCAPE_COST = 20.0
RELAXED_RADIUS_M = 0.09       # escape plans: contacts ignored, smaller margin


class CostField:
    """Single-source shortest paths from `start` (col, row) over `cost` (ny, nx).
    `hard` (bool, ny x nx): obstacle cells that stay blocked even next to the
    start (only the inflation margin around the start is waived)."""

    def __init__(self, cost, start, hard=None):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        ny, nx = cost.shape
        c = cost.astype(np.float64).copy()
        si, sj = start
        self.ok = 0 <= si < nx and 0 <= sj < ny
        if not self.ok:
            return
        r = START_FREE_CELLS
        win = (slice(max(0, sj - r), sj + r + 1), slice(max(0, si - r), si + r + 1))
        blk = c[win]
        waive = np.isinf(blk)
        if hard is not None:
            waive &= ~hard[win]
        blk[waive] = START_ESCAPE_COST
        idx = np.arange(nx * ny).reshape(ny, nx)
        rows, cols, w = [], [], []
        for dj, di, step in [(0, 1, 1.0), (1, 0, 1.0), (1, 1, SQ2), (1, -1, SQ2)]:
            r_src, r_dst = slice(0, ny - dj), slice(dj, ny)
            c_src, c_dst = (slice(0, nx - di), slice(di, nx)) if di >= 0 else (slice(-di, nx), slice(0, nx + di))
            a, b = c[r_src, c_src], c[r_dst, c_dst]
            ok = np.isfinite(a) & np.isfinite(b)
            if di and dj:                          # no corner cutting past a blocked cell
                ok &= np.isfinite(c[r_src, c_dst]) & np.isfinite(c[r_dst, c_src])
            wt = step * 0.5 * (a + b)
            rows.append(idx[r_src, c_src][ok]); cols.append(idx[r_dst, c_dst][ok]); w.append(wt[ok])
        rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
        g = coo_matrix((w, (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()
        self.dist, self.pred = dijkstra(g, directed=False, indices=int(idx[sj, si]),
                                        return_predecessors=True)
        self.dist = self.dist.reshape(ny, nx)
        self.shape = (ny, nx)
        self.start = start

    def reachable(self, cell):
        i, j = cell
        ny, nx = self.shape
        return self.ok and 0 <= i < nx and 0 <= j < ny and np.isfinite(self.dist[j, i])

    def cells_to(self, cell):
        if not self.reachable(cell):
            return None
        ny, nx = self.shape
        k = cell[1] * nx + cell[0]
        out = []
        while k >= 0:
            out.append((k % nx, k // nx))
            k = self.pred[k]
        return out[::-1]


def astar(cost, start, goal):
    """Cell path [(col, row), ...] from start to goal, or None (kept for callers
    that want a single goal; it's a Dijkstra underneath)."""
    return CostField(cost, start).cells_to(goal)


def path_length(path):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def segment_ok(m, clear, p, q, need_m):
    """Straight drive p -> q keeps need_m from every obstacle (the start cell
    excepted: the robot may be inside a margin after a contact)."""
    n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / (m.res * 0.5)))
    first = m.cell(*p)
    for x, y in zip(np.linspace(p[0], q[0], n), np.linspace(p[1], q[1], n)):
        i, j = m.cell(x, y)
        if (i, j) == first:
            continue
        if not m.inside(i, j) or clear[j, i] < need_m:
            return False
    return True


def shortcut(m, pts, clear=None, need_m=ROBOT_RADIUS_M + 0.04):
    """Greedy: from each kept point, jump to the farthest later point reachable
    in a straight line that stays clear. Fewer turns than the 8-connected path,
    and unlike plain collinearity simplification it can't cut a corner."""
    clear = m.clearance() if clear is None else clear
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not segment_ok(m, clear, pts[i], pts[j], need_m):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def costs(m, relaxed=False):
    """(cost, hard) for planning. relaxed: contact marks left out and a smaller
    margin -- for when the normal plan finds no way out (2026-10-12 the robot
    sat in a corner boxed in by its own contact marks and every leg failed)."""
    if relaxed:
        return m.costmap(radius_m=RELAXED_RADIUS_M, contacts=False), m.occupied(contacts=False)
    return m.costmap(), m.occupied()


def plan(m, start_xy, goal_xy, cost=None, field=None, relaxed=False):
    """Room-frame path from start to goal on RoomMap m, or None."""
    if field is None:
        c, hard = costs(m, relaxed)
        cost = c if cost is None else cost
        field = CostField(cost, m.cell(*start_xy), hard=hard)
    cells = field.cells_to(m.cell(*goal_xy))
    if cells is None:
        return None
    pts = [m.centre(i, j) for i, j in cells]
    pts[0] = tuple(start_xy)
    pts[-1] = tuple(goal_xy)
    return shortcut(m, pts)


def tag_viewpoint(m, survey, tag_id, start_xy, rmin=0.5, rmax=1.4, max_angle_deg=35, cost=None, relaxed=False):
    """Best place to stand to read tag `tag_id`: returns (point, facing_deg, path) or None."""
    c, hard = costs(m, relaxed)
    cost = c if cost is None else cost
    field = CostField(cost, m.cell(*start_xy), hard=hard)
    t = survey["markers"][str(tag_id)]
    n = np.array(FACING[t["facing"]][:2], float)
    tag = np.array([t["x"] * IN, t["y"] * IN])
    best = None
    for r in np.arange(rmin, rmax + 1e-9, 0.1):
        for a in np.radians(np.arange(-max_angle_deg, max_angle_deg + 1e-9, 7)):
            d = np.array([n[0] * math.cos(a) - n[1] * math.sin(a), n[0] * math.sin(a) + n[1] * math.cos(a)])
            p = tag + r * d
            cell = m.cell(*p)
            if not field.reachable(cell) or not np.isfinite(cost[cell[1], cell[0]]):
                continue
            if not m.line_of_sight(p, tag + 0.05 * n):
                continue
            reach = field.dist[cell[1], cell[0]] * m.res           # ~metres, cost-weighted
            score = reach + 2.0 * abs(r - 0.8) + 0.6 * abs(a) + 0.4 * (cost[cell[1], cell[0]] - 1)
            if best is None or score < best[0]:
                best = (score, p)
    if best is None:
        return None
    p = best[1]
    path = plan(m, start_xy, p, field=field)
    face = math.degrees(math.atan2(tag[1] - p[1], tag[0] - p[0]))
    return (float(p[0]), float(p[1])), face, path
