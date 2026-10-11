"""
Print-ready calibration targets: checkerboard, ArUco landmarks, and a
measurement sheet, as one US-Letter PDF at exact physical scale.

Usage:
    python -m raspbot_slam.calibration.make_targets [--out docs/print/calibration_targets.pdf]

Print at 100% / "Actual size" (not "Fit to page"), then check the 100 mm
scale bar on every page with a ruler. The PDF is drawn at the design sizes
below; config.py holds the sizes *as printed* (measure them and update
CHECKER_SQUARE_MM / ARUCO_MARKER_SIZE_M if your printer scaled the page).
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from .. import config

LETTER_MM = (215.9, 279.4)   # portrait (w, h)
MM_PER_IN = 25.4
DESIGN_SQUARE_MM = 25.0
DESIGN_MARKER_MM = 150.0


def _page(pdf, landscape=False):
    w, h = LETTER_MM[::-1] if landscape else LETTER_MM
    fig = plt.figure(figsize=(w / MM_PER_IN, h / MM_PER_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax, w, h


def _scale_bar(ax, x, y):
    """A 100 mm bar with end ticks: measure it to confirm the print scale."""
    ax.plot([x, x + 100], [y, y], color="k", lw=0.8)
    for xx in (x, x + 100):
        ax.plot([xx, xx], [y - 2, y + 2], color="k", lw=0.8)
    ax.text(x + 104, y, "must measure exactly 100 mm", va="center", fontsize=7)


def checkerboard_page(pdf):
    cols, rows = config.CHECKER_INNER_CORNERS
    sq = DESIGN_SQUARE_MM
    fig, ax, w, h = _page(pdf, landscape=True)
    bw, bh = (cols + 1) * sq, (rows + 1) * sq
    x0, y0 = (w - bw) / 2, (h - bh) / 2 + 6
    for i in range(cols + 1):
        for j in range(rows + 1):
            if (i + j) % 2 == 0:
                ax.add_patch(Rectangle((x0 + i * sq, y0 + j * sq), sq, sq,
                                       facecolor="k", edgecolor="none"))
    ax.text(w / 2, h - 8,
            f"Camera calibration checkerboard: {cols}x{rows} inner corners, "
            f"{sq:g} mm squares. Tape flat to stiff cardboard.",
            ha="center", va="center", fontsize=8)
    _scale_bar(ax, x0, 9)
    pdf.savefig(fig)
    plt.close(fig)


def marker_page(pdf, dictionary, marker_id):
    size_mm = DESIGN_MARKER_MM
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, 600)
    fig, ax, w, h = _page(pdf)
    x0, y0 = (w - size_mm) / 2, (h - size_mm) / 2 + 10
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=(x0, x0 + size_mm, y0, y0 + size_mm))
    # cross-hair ticks at the marker centre line, outside the white margin
    cx, cy = x0 + size_mm / 2, y0 + size_mm / 2
    for d in (-1, 1):
        ax.plot([cx + d * (size_mm / 2 + 6), cx + d * (size_mm / 2 + 14)], [cy, cy], "k", lw=0.8)
        ax.plot([cx, cx], [cy + d * (size_mm / 2 + 6), cy + d * (size_mm / 2 + 14)], "k", lw=0.8)
    ax.text(w / 2, y0 + size_mm + 34, f"ID {marker_id}", ha="center", fontsize=28, weight="bold")
    ax.text(w / 2, y0 + size_mm + 24, "this way up ↑", ha="center", fontsize=9)
    ax.text(w / 2, y0 - 24,
            f"{config.ARUCO_DICT}  •  black square {size_mm:g} mm  •  "
            "keep the white border; mount vertical, flat, centre 15–25 cm above the floor",
            ha="center", fontsize=7.5)
    ax.text(w / 2, y0 - 31, "Measure to the centre (where the tick marks point).",
            ha="center", fontsize=7.5)
    _scale_bar(ax, (w - 100) / 2 - 30, 14)
    pdf.savefig(fig)
    plt.close(fig)


def sheet_page(pdf):
    fig, ax, w, h = _page(pdf)
    y = h - 22
    ax.text(18, y, "Room survey sheet", fontsize=16, weight="bold")
    notes = [
        "Origin: pick one room corner. +X runs along one wall, +Y along the other, as seen from above.",
        "Write which corner and which walls in the box below (e.g. 'NW corner, X along window wall').",
        "Markers: centre position X, Y (cm from the origin) and height of the centre above the floor.",
        "Facing: the direction the printed face looks into the room (+X, -X, +Y, -Y, or an angle).",
        "Floor marks: start pose (where the robot's centre sits, which way it points) and the 1 m square corners.",
    ]
    for i, t in enumerate(notes):
        ax.text(18, y - 12 - 6 * i, t, fontsize=8)
    y = y - 12 - 6 * len(notes) - 10
    ax.add_patch(Rectangle((18, y - 14), w - 36, 16, fill=False, lw=0.6))
    ax.text(20, y - 3, "Origin corner / axes:", fontsize=8)
    y -= 26
    headers = ["Marker ID", "X (cm)", "Y (cm)", "Height (cm)", "Facing", "Mounted on"]
    xs = [18, 46, 74, 102, 132, 160, w - 18]
    rows = len(config.ARUCO_MARKER_IDS) + 4
    rh = 10
    for r in range(rows + 1):
        ax.plot([xs[0], xs[-1]], [y - r * rh, y - r * rh], "k", lw=0.5)
    for x in xs:
        ax.plot([x, x], [y, y - rows * rh], "k", lw=0.5)
    for i, t in enumerate(headers):
        ax.text(xs[i] + 2, y - 6.5, t, fontsize=8, weight="bold")
    labels = [str(m) for m in config.ARUCO_MARKER_IDS] + ["start", "square A", "square B..D"]
    for r, t in enumerate(labels):
        ax.text(xs[0] + 2, y - (r + 1) * rh - 6.5, t, fontsize=8)
    y -= rows * rh + 14
    ax.text(18, y, "Room sketch (outer walls, doors, furniture footprints, with dimensions):", fontsize=8)
    ax.add_patch(Rectangle((18, 20), w - 36, y - 24, fill=False, lw=0.6))
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", default=os.path.join("docs", "print", "calibration_targets.pdf"))
    args = p.parse_args(argv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
    with PdfPages(args.out) as pdf:
        checkerboard_page(pdf)
        for mid in config.ARUCO_MARKER_IDS:
            marker_page(pdf, dictionary, mid)
        sheet_page(pdf)
    print(f"wrote {args.out}: 1 checkerboard, {len(config.ARUCO_MARKER_IDS)} markers, 1 survey sheet")


if __name__ == "__main__":
    main()
