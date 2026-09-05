"""Cat detector: YOLOv8n (COCO) via ONNX Runtime.

Pure numpy + cv2 + onnxruntime inference — no ultralytics dependency on the
robot. The model file (yolov8n.onnx, 320px, opset 12) is exported offline and
committed under catchaser/models/.

COCO class indices (YOLO 80-class ordering): 15 = cat, 16 = dog.

Typical use:
    det = CatDetector("catchaser/models/yolov8n_320.onnx")
    cats = det.detect(frame_bgr)           # list of Detection
    target = best_target(cats)             # highest-confidence cat or None
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised only where ORT is absent
    ort = None

COCO_CAT = 15
COCO_DOG = 16

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "models", "yolov8n_320.onnx")


@dataclass
class Detection:
    """One detected object in original-image pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


def letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize with unchanged aspect ratio, pad to (size, size) with gray.

    Returns:
        (padded_image, scale, (pad_x, pad_y)) where original coords map as
        orig = (padded - pad) / scale.
    """
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    out = np.full((size, size, 3), 114, dtype=img.dtype)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    out[top:top + new_h, left:left + new_w] = resized
    return out, scale, (left, top)


class CatDetector:
    """YOLOv8n ONNX detector filtered to target classes (default: cat)."""

    def __init__(self, model_path: str = DEFAULT_MODEL,
                 conf_threshold: float = 0.35,
                 iou_threshold: float = 0.45,
                 target_classes: Tuple[int, ...] = (COCO_CAT,),
                 num_threads: Optional[int] = None):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        so = ort.SessionOptions()
        if num_threads:
            so.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(model_path, sess_options=so,
                                            providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = int(inp.shape[-1])  # e.g. 320
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = set(target_classes)
        self.last_inference_ms: float = 0.0

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run detection on a BGR frame (any size).

        Returns:
            Detections of target classes, highest confidence first,
            in original frame pixel coordinates.
        """
        img, scale, (pad_x, pad_y) = letterbox(frame_bgr, self.input_size)
        blob = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR->RGB, 0-1
        blob = np.transpose(blob, (2, 0, 1))[None]           # NCHW

        t0 = time.perf_counter()
        (out,) = self.session.run(None, {self.input_name: blob})
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

        # out: (1, 84, N) -> (N, 84); rows = [cx, cy, w, h, 80 class scores]
        pred = out[0].T
        scores_all = pred[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        confidences = scores_all[np.arange(len(class_ids)), class_ids]

        keep = confidences >= self.conf_threshold
        if self.target_classes:
            keep &= np.isin(class_ids, list(self.target_classes))
        pred, class_ids, confidences = pred[keep], class_ids[keep], confidences[keep]
        if len(pred) == 0:
            return []

        # xywh (letterboxed px) -> xyxy (original px)
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        boxes_xywh = [[float(a), float(b), float(c - a), float(d - b)]
                      for a, b, c, d in zip(x1, y1, x2, y2)]
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, confidences.astype(float).tolist(),
                                self.conf_threshold, self.iou_threshold)
        idxs = np.array(idxs).flatten() if len(idxs) else np.array([], dtype=int)

        dets = [Detection(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                          float(confidences[i]), int(class_ids[i])) for i in idxs]
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets


def best_target(detections: List[Detection]) -> Optional[Detection]:
    """Pick the detection to chase: highest confidence (list is pre-sorted)."""
    return detections[0] if detections else None
