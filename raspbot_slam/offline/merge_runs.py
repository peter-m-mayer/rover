"""
Merge several recorded runs into one room map: the prior a new run starts from.

    python -m raspbot_slam.offline.merge_runs datasets/slam_runs/<run> [...] [--out maps/office_prior]

Each run is replayed (eval_tags.replay) with the Localizer correcting the pose
from tag fixes, into one shared RoomMap, oldest first; its persistent obstacle
counts (room_map.py: hits from two places) accumulate across runs, so furniture
seen from several runs stays solid. Writes <out>.npz (RoomMap.save; load with
tag_tour --prior <out>.npz) and <out>.png. Runs from before the survey frame fix
(2026-10-12 08:00) are in the old mirrored frame: leave them out.
"""

import argparse
import os

import numpy as np

from ..room_map import RoomMap
from ..tag_localizer import IN, load_survey
from .eval_tags import replay


def merge(runs, survey="office", log=print):
    sv = load_survey(survey)
    m = RoomMap.from_survey(sv)
    for r in sorted(runs, key=lambda p: os.path.basename(os.path.normpath(p))):
        out = replay(r, m, localizer=True, correct=True, log=lambda s: None)
        if out is None:
            log(f"{r}: no tag fix, skipped")
            continue
        log(f"{r}: {len(out['path'])} poses, localizer {out['stats']}")
    return m, sv


def plot(m, sv, path, title="Merged room map"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, Y = sv["xmax_in"] * IN, sv["ymax_in"] * IN
    img = np.full(m.L.shape + (3,), 0.85)
    img[m.free()] = (1, 1, 1)
    img[m.occupied(contacts=False) & ~m.wall] = (0.15, 0.15, 0.15)
    img[(m.contact > 0.3) & ~m.wall] = (0.8, 0.1, 0.1)
    img[m.wall] = (0.55, 0.6, 0.7)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(img, origin="lower", interpolation="nearest",
              extent=(m.x0, m.x0 + m.nx * m.res, m.y0, m.y0 + m.ny * m.res))
    for k, mk in sv["markers"].items():
        ax.plot(mk["x"] * IN, mk["y"] * IN, "s", color="crimson", ms=8)
        ax.annotate(f"tag {k}", (mk["x"] * IN, mk["y"] * IN), xytext=(6, 6), textcoords="offset points",
                    color="crimson")
    a = sv.get("alcove")
    ax.set_xlim((a["x0"] * IN if a else 0) - 0.3, X + 0.3)
    ax.set_ylim(-0.3, Y + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title}: white free, black obstacles, red contacts, grey unknown")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Merge runs into one prior room map.")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=os.path.join("maps", "office_prior"))
    p.add_argument("--survey", default="office")
    args = p.parse_args(argv)
    m, sv = merge(args.runs, args.survey)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    m.save(args.out + ".npz")
    plot(m, sv, args.out + ".png", title=f"{len(args.runs)} runs merged")
    from .score_run import map_scores
    sc = map_scores(m)
    print(f"saved {args.out}.npz / .png: coverage {sc['coverage']:.0%}, "
          f"solid cells {int((m.solid() & ~m.wall).sum())}, wall offset {sc['wall_offset_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
