"""
Real-Time Object Detection and Tracking
Local educational prototype using Ultralytics YOLO tracking and OpenCV.
Run: python app.py --source 0
     python app.py --source sample_video.mp4
"""
import argparse
import os
import time
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking")
    parser.add_argument("--source", default="0",
                        help="Webcam index (0) or path to a video file")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Ultralytics model weights (downloaded on first use if needed)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Minimum detection confidence, from 0.0 to 1.0")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output to outputs/annotated_output.mp4")
    return parser.parse_args()


def open_source(source_text):
    source = int(source_text) if source_text.isdigit() else source_text
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open video source '{source_text}'. Check the camera, path, and permissions."
        )
    return cap


def main():
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0.0 and 1.0")

    model = YOLO(args.model)
    cap = open_source(args.source)
    writer = None
    previous_time = time.perf_counter()
    fps_smooth = 0.0
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # persist=True asks the tracking pipeline to keep track state between frames.
            results = model.track(
                source=frame, persist=True, tracker="bytetrack.yaml",
                conf=args.conf, verbose=False
            )
            annotated = results[0].plot()
            frame_count += 1

            now = time.perf_counter()
            elapsed = max(now - previous_time, 1e-9)
            current_fps = 1.0 / elapsed
            fps_smooth = current_fps if fps_smooth == 0 else 0.85 * fps_smooth + 0.15 * current_fps
            previous_time = now

            boxes = results[0].boxes
            object_count = 0 if boxes is None else len(boxes)
            cv2.putText(annotated, f"Objects: {object_count}  FPS: {fps_smooth:.1f}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if args.save:
                if writer is None:
                    os.makedirs("outputs", exist_ok=True)
                    height, width = annotated.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 0:
                        fps = 25.0
                    writer = cv2.VideoWriter(
                        "outputs/annotated_output.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                writer.write(annotated)

            cv2.imshow("Real-Time Object Detection and Tracking | Q: quit",
                       annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Session finished. Frames processed: {frame_count}")
    if args.save:
        print("Annotated video (if frames were processed): outputs/annotated_output.mp4")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")
