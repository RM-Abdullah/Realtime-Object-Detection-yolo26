"""
Real-time object detection with YOLO26n, running on the CPU.

Usage:
    python realtime_detect.py                      # demo: every image and video in this folder
    python realtime_detect.py --source bmw2.jpg    # a single image
    python realtime_detect.py --source BG2.mp4     # a video file
    python realtime_detect.py --source 0           # live webcam (use 1, 2... for other cameras)

Press 'q' in the video window to stop. Images wait for any key.
Annotated results are saved to the outputs/ folder.
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "yolo26n.pt"  # downloaded automatically on first run
OUTPUT_DIR = ROOT / "outputs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def count_classes(result, names):
    return Counter(names[int(c)] for c in result.boxes.cls)


def format_counts(counts):
    if not counts:
        return "nothing detected"
    return ", ".join(f"{name} x{n}" for name, n in counts.most_common())


def detect_image(model, path, args):
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Cannot read image {path}")

    result = model(img, conf=args.conf, imgsz=args.imgsz, device="cpu", verbose=False)[0]
    annotated = result.plot()

    out_path = OUTPUT_DIR / f"{path.stem}_detected.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"[image] {path.name}: {format_counts(count_classes(result, model.names))}"
          f"  ({result.speed['inference']:.0f} ms) -> {out_path.relative_to(ROOT)}")

    if args.show:
        cv2.imshow(f"YOLO26n - {path.name} (press any key)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_stream(model, source, args):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source {source}")

    name = f"webcam{source}" if isinstance(source, int) else Path(source).stem
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30  # webcams often report 0
    out_path = OUTPUT_DIR / f"{name}_detected.mp4"
    writer = None

    frames, fps, start = 0, 0.0, time.perf_counter()
    peak = Counter()  # most objects of each class seen in a single frame
    total = Counter()  # detections of each class summed over all frames
    conf_sum = Counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        result = model(frame, conf=args.conf, imgsz=args.imgsz, device="cpu", verbose=False)[0]
        annotated = result.plot()
        dt = time.perf_counter() - t0
        fps = 1 / dt if frames == 0 else 0.9 * fps + 0.1 / dt  # smoothed

        counts = count_classes(result, model.names)
        peak |= counts
        total += counts
        for c, p in zip(result.boxes.cls, result.boxes.conf):
            conf_sum[model.names[int(c)]] += float(p)
        cv2.putText(annotated, f"FPS: {fps:.1f}  objects: {len(result.boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if args.save:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(annotated)

        frames += 1
        if args.show:
            cv2.imshow(f"YOLO26n - {name} (press q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.perf_counter() - start
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    saved = f" -> {out_path.relative_to(ROOT)}" if writer is not None else ""
    print(f"[video] {name}: {frames} frames in {elapsed:.1f}s, average {frames / max(elapsed, 1e-9):.1f} FPS{saved}")
    print(f"        peak per frame: {format_counts(peak)}")
    if total:
        print("        avg per frame @ avg confidence: " + ", ".join(
            f"{name} {total[name] / frames:.1f} @ {conf_sum[name] / total[name]:.2f}"
            for name, _ in total.most_common()))


def collect_sources(source):
    if source is None:
        source = ROOT
    if str(source).isdigit():
        return [int(source)]

    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
        # images first, then videos
        return sorted(files, key=lambda p: p.suffix.lower() in VIDEO_EXTS)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    return [path]


def main():
    parser = argparse.ArgumentParser(description="YOLO26n object detection on images, videos or a webcam (CPU).")
    parser.add_argument("--source", help="image, video, folder, or webcam index (0, 1...). Default: this folder")
    parser.add_argument("--conf", type=float, default=0.4, help="minimum confidence (default 0.4)")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size; 320 is faster, 640 is more accurate")
    parser.add_argument("--no-show", dest="show", action="store_false", help="don't open preview windows")
    parser.add_argument("--no-save", dest="save", action="store_false", help="don't save annotated videos")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    for src in collect_sources(args.source):
        if isinstance(src, int) or src.suffix.lower() in VIDEO_EXTS:
            detect_stream(model, src if isinstance(src, int) else str(src), args)
        else:
            detect_image(model, src, args)


if __name__ == "__main__":
    main()
