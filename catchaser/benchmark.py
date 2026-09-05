"""On-device detector benchmark for the Pi.

Measures real inference rate — the number that decides whether the chase
loop runs at the target >=1 Hz (expected: comfortably above on a Pi 5).

    python3 -m catchaser.benchmark              # synthetic frames
    python3 -m catchaser.benchmark --camera     # live camera frames
    python3 -m catchaser.benchmark --threads 4  # thread sweep candidate
"""

import argparse
import sys
import time

import numpy as np

from .detector import CatDetector


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cat detector benchmark")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--camera", action="store_true",
                        help="benchmark on live camera frames (includes capture cost)")
    parser.add_argument("--threads", type=int, default=None,
                        help="onnxruntime intra-op threads (default: ORT's choice)")
    args = parser.parse_args(argv)

    det = CatDetector(num_threads=args.threads)
    print(f"model input: {det.input_size}px, threads: {args.threads or 'auto'}")

    camera = None
    if args.camera:
        from raspbot_slam.camera import Camera
        camera = Camera()
        camera.open()

    def get_frame():
        if camera is not None:
            return camera.capture_color()
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    try:
        for _ in range(args.warmup):
            det.detect(get_frame())

        total_ms, infer_ms = [], []
        for i in range(args.runs):
            t0 = time.perf_counter()
            frame = get_frame()
            dets = det.detect(frame)
            total_ms.append((time.perf_counter() - t0) * 1000.0)
            infer_ms.append(det.last_inference_ms)
            if dets:
                d = dets[0]
                print(f"  run {i}: class {d.class_id} conf {d.confidence:.2f}")
    finally:
        if camera is not None:
            camera.close()

    total_ms, infer_ms = np.array(total_ms), np.array(infer_ms)
    label = "capture+detect" if args.camera else "detect (synthetic frame)"
    print(f"\n{label}, {args.runs} runs:")
    print(f"  inference : median {np.median(infer_ms):6.1f} ms   "
          f"p90 {np.percentile(infer_ms, 90):6.1f} ms")
    print(f"  end-to-end: median {np.median(total_ms):6.1f} ms   "
          f"p90 {np.percentile(total_ms, 90):6.1f} ms")
    print(f"  rate      : {1000.0 / np.median(total_ms):.1f} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
