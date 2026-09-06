"""Dataset harvesting: turn chase-loop frames into review-ready training data.

Writes the layout that catchaser.review triages and YOLO training consumes:

    <root>/images/<stem>.jpg     clean frame (no annotations burned in)
    <root>/labels/<stem>.txt     one "0 cx cy w h" line per detection,
                                 normalized to [0,1]; empty file = negative
    <root>/preview/<stem>.jpg    annotated copy (boxes + confidence) for
                                 quick human/remote inspection only

Class id is always 0: the dataset is single-class ("cat"), regardless of the
COCO id the detector used.
"""

import os
from typing import List

from .detector import Detection


def yolo_lines(detections: List[Detection], width: int, height: int) -> str:
    """Render detections as YOLO label-file content (normalized xywh)."""
    lines = []
    for d in detections:
        cx = min(max(d.cx / width, 0.0), 1.0)
        cy = min(max(d.cy / height, 0.0), 1.0)
        w = min(max(d.width / width, 0.0), 1.0)
        h = min(max(d.height / height, 0.0), 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def save_sample(root: str, stem: str, frame_bgr, detections: List[Detection],
                preview: bool = True) -> None:
    """Save one frame + weak labels (and an annotated preview) under root."""
    import cv2

    h, w = frame_bgr.shape[:2]
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels"), exist_ok=True)
    cv2.imwrite(os.path.join(root, "images", stem + ".jpg"), frame_bgr)
    with open(os.path.join(root, "labels", stem + ".txt"), "w") as f:
        f.write(yolo_lines(detections, w, h))

    if preview:
        os.makedirs(os.path.join(root, "preview"), exist_ok=True)
        annotated = frame_bgr.copy()
        for d in detections:
            cv2.rectangle(annotated, (int(d.x1), int(d.y1)),
                          (int(d.x2), int(d.y2)), (0, 255, 0), 2)
            cv2.putText(annotated, f"{d.confidence:.2f}",
                        (int(d.x1), max(15, int(d.y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(root, "preview", stem + ".jpg"), annotated)
