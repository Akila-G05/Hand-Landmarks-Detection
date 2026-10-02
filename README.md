# Hand Landmarks Detection

Real-time hand tracking with [MediaPipe](https://ai.google.dev/edge/mediapipe).
Reads from your webcam, detects up to two hands, and overlays the 21 hand
landmarks and their connections, labelled with handedness (Left / Right).

## How it works

1. Loads `hand_landmarker.task`, MediaPipe's pre-trained model, through the
   Tasks API (`vision.HandLandmarker`).
2. Opens the webcam at 1920x1080 and reads frames in a loop.
3. Mirrors each frame horizontally (`cv2.flip`) so the preview behaves like a
   mirror.
4. Converts BGR to RGB and wraps it in an `mp.Image`, which is what the
   MediaPipe API expects.
5. Calls `detector.detect()` to get landmarks (normalised 0-1 x/y coordinates)
   and a handedness label per hand.
6. Draws the landmarks, connections, and handedness label, then displays the
   result fullscreen.

## Requirements

- Python 3.9+
- A webcam
- Windows, macOS, or Linux

```bash
pip install -r requirements.txt
```

> Install **one** OpenCV distribution only. If `opencv-python` and
> `opencv-contrib-python` are both present in the same environment, imports
> conflict and OpenCV silently misbehaves.

The `hand_landmarker.task` model file (~7.5 MB) is committed to the repo, so
there is nothing extra to download.

## Usage

```bash
python index.py
```

The window opens fullscreen. Press `ESC` to quit.

To run against a video file instead of the webcam, replace the `0` in
`cv2.VideoCapture(0)` with the filename:

```python
cap = cv2.VideoCapture('clip.mp4')
```

## Tuning

| What | Where | Default | Notes |
| --- | --- | --- | --- |
| Hands detected | `num_hands` in `HandLandmarkerOptions` | `2` | Raise for group shots, lower for speed |
| Frame width | `CAP_PROP_FRAME_WIDTH` | `1920` | Drop to 1280 for weaker machines |
| Frame height | `CAP_PROP_FRAME_HEIGHT` | `1080` | |
| Label colour | `HANDEDNESS_TEXT_COLOR` | `(88, 205, 54)` | BGR tuple |
| Label offset | `MARGIN` | `10` | Pixels above the hand's top-left corner |
| Mirror preview | `cv2.flip(frame, 1)` | on | Remove for a raw camera view |

To draw without the default blue landmark dots, swap
`mp_drawing_styles.get_default_hand_landmarks_style()` for
`mp_drawing_styles.get_default_hand_connections_style()`, or pass
`mp_drawing.DrawingSpec(color=..., thickness=..., radius=...)` directly.

## Files

```
index.py                   Webcam loop, detection, and overlay drawing
hand_landmarker.task       MediaPipe hand landmark model
hand.png                   Reference image
requirements.txt
```

## Notes

- Handedness is reported from the camera's point of view. With the mirror flip
  enabled it reads as a mirror, so the labels may feel reversed to you.
- The Tasks API expects an RGB image. The BGR-to-RGB conversion in the loop is
  required, not incidental.
- `detector` is created once outside the loop on purpose, since model
  construction is the expensive part.

## License

MIT — see [LICENSE](LICENSE).