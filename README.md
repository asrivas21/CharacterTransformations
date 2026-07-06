# Anime Transformation Filter

Real-time webcam filter. Form a rectangle with a specific finger pair on **both
hands** to instantly switch between three anime transformations. Each
transformation applies a full-frame art style **and** composites character
features onto your face and body using MediaPipe landmarks.

| Gesture (finger corners, both hands) | Show | Style treatment | Character features |
|---|---|---|---|
| Thumb + Index | Naruto | Cel-shaded, bold ink outlines, orange/blue palette | Hidden Leaf headband, Sage Mode eyes |
| Index + Middle | ATLA | Watercolor ink-wash (GLSL), warm desaturated | Avatar State glow, forehead & hand arrows |
| Middle + Pinky | JJK | Dark high-contrast, purple/blue accents | Gojo's Six Eyes |

## Stack

Python · MediaPipe Holistic · OpenCV · NumPy · ModernGL (GLSL fragment shaders).

## Pipeline

```
Webcam frame ─► MediaPipe Holistic (hands + face mesh + pose)
             ├─ Hands  ─► gesture detector ─► debouncer ─► active show
             └─ Face   ─► feature compositor
Frame ─► STYLE renderer (full frame) ─► FEATURE compositor (overlays on top) ─► display
```

Style is applied first, character features on top, so the headband and eyes stay
crisp instead of being cel-shaded into mush.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

- Mirror view is on by default (natural interaction).
- Form the finger-pair rectangle with **both** hands to switch shows.
- A 3-frame debounce (~100 ms) confirms a gesture before switching — instant feel,
  no single-frame flicker.
- Press **`q`** to quit. The HUD shows the active show and current FPS.

## Gestures

MediaPipe hand fingertip indices used as rectangle corners:

```
THUMB_TIP=4  INDEX_TIP=8  MIDDLE_TIP=12  RING_TIP=16  PINKY_TIP=20

NARUTO → (THUMB, INDEX)    ATLA → (INDEX, MIDDLE)    JJK → (MIDDLE, PINKY)
```

A show activates only when the required pair is extended on **both** hands. When
poses overlap, the detector prefers the match with the fewest extra extended
fingers. Expect to tune the extension thresholds in `src/gesture/detector.py`
against your own hands on camera.

## Project structure

```
src/
├── main.py                 # capture loop, orchestration
├── landmarks.py            # MediaPipe Holistic wrapper (refined iris landmarks)
├── segmentation.py         # selfie segmentation (person vs background)
├── gesture/
│   ├── detector.py         # rectangle-gesture recognition
│   └── smoothing.py        # temporal debounce
├── styles/
│   ├── base.py             # StyleRenderer interface
│   ├── naruto.py           # cel-shade + ink outline
│   ├── atla.py             # GLSL watercolor (CPU fallback)
│   └── jjk.py              # dark high-contrast
├── features/
│   ├── compositor.py       # landmark-driven asset placement
│   ├── naruto_features.py  # headband + sage eyes
│   ├── atla_features.py    # avatar glow + arrow tattoos
│   └── jjk_features.py     # six eyes
└── shaders/
    ├── watercolor.glsl     # ATLA ink-wash fragment shader
    └── outline.glsl        # edge-detect + ink outline
assets/                     # transparent PNG overlays (see below)
tools/                      # placeholder-asset generator
tests/                      # unit tests
```

## Assets

The PNGs under `assets/` are **stylized placeholders** so the pipeline runs
end-to-end. Replace them with real traced or AI-generated art (see the
`INSTRUCTIONS.md` asset guidance). Regenerate the placeholders anytime with:

```bash
python tools/gen_placeholder_assets.py
```

Asset requirements: RGBA (alpha channel), roughly front-facing/symmetric (the
compositor rotates them), iris assets square with a centered circle.

## Tests

Unit tests cover the gesture detector, the debouncer, and the compositor math
(no webcam or GPU required):

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

- **No iris/eye alignment:** `refine_face_landmarks=True` must be on (it is in
  `landmarks.py`); without it you only get 468 points instead of 478.
- **ATLA style looks like a CPU wash:** a standalone GL context could not be
  created; `atla.py` falls back to an OpenCV approximation automatically.
- **Below ~24 FPS:** lower `k` in `NarutoStyle` (K-means quantization is the
  heaviest step) or reduce the capture resolution in `main.py`.
- **Webcam won't open:** confirm no other app holds the camera and that the
  index in `cv2.VideoCapture(0)` is correct.
