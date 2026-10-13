# Real-Time Object Detection and Tracking

## Project overview
A local computer-vision prototype that detects supported objects in webcam/video frames and tracks detections between frames. Annotated output includes bounding boxes, class labels, confidence values (as supplied by the model), tracking IDs where available, approximate FPS, and the number of detections in the current frame.

## Requirements
- Python 3.10–3.12 (verify compatibility with your installed package versions)
- Webcam or a supported video file
- CPU-capable computer; compatible GPU is optional
- Internet may be required the first time to install packages and download model weights

## Installation
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
# Default webcam (camera index 0)
python app.py

# A video file
python app.py --source "sample_video.mp4"

# Adjust confidence threshold
python app.py --source 0 --conf 0.50

# Save annotated output video
python app.py --source "sample_video.mp4" --save
```
Press **Q** or **Esc** in the display window to stop. The camera/video handle and display resources are released when the program exits.

## How it works
1. OpenCV reads frames from the selected source.
2. Ultralytics YOLO predicts supported object classes, bounding boxes and confidence scores.
3. The tracking pipeline associates detections between frames using ByteTrack.
4. The result is rendered with labels and available tracking IDs.
5. FPS and current-frame detection count are displayed.

## Limitations and privacy
- This is an educational prototype, not a certified safety, medical, surveillance, or access-control system.
- The model only detects classes represented by its training labels; results can be wrong.
- Tracking IDs are temporary track identifiers, not proof of a person's identity.
- Video is processed locally by default. Output is only saved when `--save` is supplied.
- Sample flight-like/demo data is not included; use a video you have permission to process.

## Troubleshooting
- If the camera fails, close other apps using it and check OS camera permissions.
- If a model file cannot be downloaded, check network access or provide a compatible local weights file with `--model`.
- If processing is slow, lower the input resolution or use a smaller model.
- If OpenCV display is unavailable in a notebook/headless environment, run the script on a desktop session with a display.

## References
- Python: https://docs.python.org/3/
- OpenCV: https://docs.opencv.org/
- Ultralytics: https://docs.ultralytics.com/
