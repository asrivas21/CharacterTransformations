# Anime Transformation Filter — Implementation Plan
**What it does:** Real-time webcam filter. Form a rectangle with specific finger pairs on both hands to instantly switch between three anime transformations. Each transformation applies a full-frame art style AND composites character features onto your face and body using facial landmarks.

**The three transformations:**
| Gesture (finger corners, both hands) | Show | Style treatment | Character features |
|---|---|---|---|
| Thumb + Index | Naruto | Cel-shaded, bold ink outlines, orange/blue palette | Hidden Leaf headband, Sage Mode eyes |
| Index + Middle | ATLA | Watercolor ink-wash, warm desaturated | Avatar State glowing eyes, arrow tattoos |
| Middle + Pinky | JJK | Dark high-contrast, purple/blue accents | Gojo's Six Eyes |

**Stack:** Python + MediaPipe + OpenCV + NumPy + ModernGL (GLSL shaders). Optional Phase 2: TypeScript + WebGL browser port.

---

## Core technical concept

Two independent systems running on the same MediaPipe output:

```
Webcam frame (1080p, target 30fps)
        │
        ▼
MediaPipe Holistic
├── Hands (21 landmarks × 2)  ──►  GESTURE DETECTOR  ──►  active_show (naruto|atla|jjk|none)
├── Face Mesh (468 landmarks) ──►  FEATURE COMPOSITOR (headband, eyes, tattoos, six-eyes)
└── Pose (33 landmarks)       ──►  BODY SEGMENTATION (for background/clothing style)
        │
        ▼
STYLE RENDERER (full frame, per active_show)
        │
        ▼
FEATURE COMPOSITOR (overlays on top of styled frame)
        │
        ▼
Display / record
```

The gesture detector picks the show. The style renderer restyles the whole frame. The feature compositor pastes character assets onto face/body landmarks. Order matters: style first, features on top, so the headband and eyes stay crisp instead of getting cel-shaded into mush.

---

## Repo structure

```
anime-transform/
├── src/
│   ├── main.py                    # capture loop, orchestration
│   ├── landmarks.py               # MediaPipe Holistic wrapper
│   ├── gesture/
│   │   ├── detector.py            # rectangle-gesture recognition
│   │   └── smoothing.py           # temporal debounce (no flicker)
│   ├── styles/
│   │   ├── base.py                # StyleRenderer interface
│   │   ├── naruto.py              # cel-shade + ink outline
│   │   ├── atla.py                # watercolor wash
│   │   └── jjk.py                 # dark high-contrast
│   ├── features/
│   │   ├── compositor.py          # landmark-driven asset placement
│   │   ├── naruto_features.py     # headband + sage eyes
│   │   ├── atla_features.py       # avatar eyes + arrow tattoos
│   │   └── jjk_features.py        # six eyes
│   ├── segmentation.py            # selfie segmentation (person vs bg)
│   └── shaders/
│       ├── watercolor.glsl        # ATLA ink-wash fragment shader
│       └── outline.glsl           # edge-detect + ink outline
├── assets/
│   ├── naruto/
│   │   ├── headband.png           # transparent PNG, front-facing
│   │   ├── headband_metal.png     # the metal plate w/ leaf symbol
│   │   └── sage_iris.png          # toad-eye iris texture
│   ├── atla/
│   │   ├── arrow_head.png         # forehead arrow
│   │   ├── arrow_hand.png         # hand arrows
│   │   └── avatar_glow.png        # eye glow sprite
│   └── jjk/
│       ├── six_eyes_overlay.png   # Gojo eye texture
│       └── blindfold.png          # optional blindfold variant
├── demo/
│   └── clips/                     # recorded demos for README
├── requirements.txt
└── README.md
```

---

## Phase 0 — Setup and MediaPipe pipeline

### 0.1 — Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install mediapipe opencv-python numpy moderngl moderngl-window pillow
```

`requirements.txt`:
```
mediapipe==0.10.14
opencv-python==4.10.0.84
numpy==1.26.4
moderngl==5.10.0
moderngl-window==2.4.6
pillow==10.4.0
```

### 0.2 — Landmark wrapper

`src/landmarks.py`:
```python
"""MediaPipe Holistic wrapper. Returns hands, face mesh, and pose landmarks
in a single normalized structure per frame."""
from __future__ import annotations
import mediapipe as mp
import numpy as np
from dataclasses import dataclass

mp_holistic = mp.solutions.holistic


@dataclass
class FrameLandmarks:
    left_hand: np.ndarray | None   # (21, 3) or None
    right_hand: np.ndarray | None  # (21, 3) or None
    face: np.ndarray | None        # (468, 3) or None
    pose: np.ndarray | None        # (33, 4) or None
    frame_w: int
    frame_h: int


class LandmarkExtractor:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,        # 0=fast, 1=balanced, 2=accurate
            smooth_landmarks=True,
            refine_face_landmarks=True,  # CRITICAL: enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray) -> FrameLandmarks:
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]  # BGR → RGB, no copy
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)

        def to_array(landmarks, n, dims=3):
            if landmarks is None:
                return None
            arr = np.zeros((n, dims), dtype=np.float32)
            for i, lm in enumerate(landmarks.landmark):
                arr[i, 0] = lm.x * w
                arr[i, 1] = lm.y * h
                arr[i, 2] = lm.z
                if dims == 4:
                    arr[i, 3] = lm.visibility
            return arr

        return FrameLandmarks(
            left_hand=to_array(results.left_hand_landmarks, 21),
            right_hand=to_array(results.right_hand_landmarks, 21),
            face=to_array(results.face_landmarks, 468),
            pose=to_array(results.pose_landmarks, 33, dims=4),
            frame_w=w,
            frame_h=h,
        )

    def close(self):
        self.holistic.close()
```

> **Note:** `refine_face_landmarks=True` is mandatory. Without it you don't get iris landmarks (468 points instead of 478), and the eye-replacement features won't align.

---

## Phase 1 — Gesture detection (the rectangle system)

This is the heart of the interaction. Each show maps to a specific pair of fingers used as rectangle corners on both hands.

### 1.1 — MediaPipe hand landmark indices

```
Fingertip indices (MediaPipe hand model):
  THUMB_TIP   = 4
  INDEX_TIP   = 8
  MIDDLE_TIP  = 12
  RING_TIP    = 16
  PINKY_TIP   = 20
```

### 1.2 — The three rectangle definitions

Each gesture forms a quadrilateral using two fingertips from each hand as the four corners:

```
NARUTO  →  Left(THUMB, INDEX)  ×  Right(THUMB, INDEX)
ATLA    →  Left(INDEX, MIDDLE) ×  Right(INDEX, MIDDLE)
JJK     →  Left(MIDDLE, PINKY) ×  Right(MIDDLE, PINKY)
```

### 1.3 — Detection logic

The challenge: distinguish which fingers are extended and forming the rectangle. The approach — check which fingertips are extended (far from palm) AND which two extended fingertips per hand are the "active" pair based on separation.

`src/gesture/detector.py`:
```python
"""Rectangle gesture detector. Determines which show is active based on
which finger pair forms the rectangle on both hands."""
from __future__ import annotations
import numpy as np
from enum import Enum

THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
WRIST = 0
# MCP joints (knuckles) for extension check
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17


class Show(Enum):
    NONE = "none"
    NARUTO = "naruto"
    ATLA = "atla"
    JJK = "jjk"


# Which finger pair defines each show
SHOW_FINGER_PAIRS = {
    Show.NARUTO: (THUMB_TIP, INDEX_TIP),
    Show.ATLA:   (INDEX_TIP, MIDDLE_TIP),
    Show.JJK:    (MIDDLE_TIP, PINKY_TIP),
}


def _finger_extended(hand: np.ndarray, tip_idx: int, mcp_idx: int) -> bool:
    """A finger is extended if its tip is farther from the wrist than its MCP."""
    wrist = hand[WRIST, :2]
    tip_dist = np.linalg.norm(hand[tip_idx, :2] - wrist)
    mcp_dist = np.linalg.norm(hand[mcp_idx, :2] - wrist)
    return tip_dist > mcp_dist * 1.15


def _extended_fingers(hand: np.ndarray) -> set[int]:
    """Return the set of extended fingertip indices."""
    extended = set()
    # Thumb: compare tip to IP joint horizontally (thumb geometry differs)
    if np.linalg.norm(hand[THUMB_TIP, :2] - hand[WRIST, :2]) > \
       np.linalg.norm(hand[2, :2] - hand[WRIST, :2]) * 1.1:
        extended.add(THUMB_TIP)
    for tip, mcp in [(INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
                     (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP)]:
        if _finger_extended(hand, tip, mcp):
            extended.add(tip)
    return extended


def _matches_pair(extended: set[int], pair: tuple[int, int]) -> bool:
    """
    A hand matches a show's finger pair if BOTH fingers in the pair are extended.
    We also require the OTHER show-defining fingers to help disambiguate:
    e.g. for NARUTO (thumb+index) we prefer middle/ring/pinky curled.
    """
    a, b = pair
    if not (a in extended and b in extended):
        return False
    return True


def detect_show(left_hand: np.ndarray | None,
                right_hand: np.ndarray | None) -> Show:
    """
    Determine active show. Both hands must present the same finger pair.
    Priority order disambiguates overlapping poses (thumb+index also has
    index extended, which could look like index+middle).
    """
    if left_hand is None or right_hand is None:
        return Show.NONE

    left_ext = _extended_fingers(left_hand)
    right_ext = _extended_fingers(right_hand)

    # Score each show by how cleanly both hands match its pair
    # Check in an order that resolves ambiguity — most specific first
    candidates = []
    for show, pair in SHOW_FINGER_PAIRS.items():
        a, b = pair
        left_match = a in left_ext and b in right_ext or (a in left_ext and b in left_ext)
        # Require the pair extended on BOTH hands
        if (a in left_ext and b in left_ext) and (a in right_ext and b in right_ext):
            # Count "clean" match — fewer extra extended fingers = cleaner
            extra = (len(left_ext) - 2) + (len(right_ext) - 2)
            candidates.append((show, extra))

    if not candidates:
        return Show.NONE

    # Prefer the cleanest match (fewest extra extended fingers)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def rectangle_corners(left_hand: np.ndarray, right_hand: np.ndarray,
                      show: Show) -> np.ndarray | None:
    """Return the 4 corner points of the rectangle for compositing/masking."""
    if show == Show.NONE:
        return None
    a, b = SHOW_FINGER_PAIRS[show]
    return np.array([
        left_hand[a, :2],
        left_hand[b, :2],
        right_hand[b, :2],
        right_hand[a, :2],
    ], dtype=np.float32)
```

### 1.4 — Temporal smoothing (kill the flicker)

Instant switching per your spec — but instant on *confirmed* gesture, not on a single noisy frame. Require N consecutive frames of the same detection before switching.

`src/gesture/smoothing.py`:
```python
"""Debounce gesture detection. Switch instantly once a gesture is confirmed
across a few frames, but never flicker on single-frame noise."""
from collections import deque
from gesture.detector import Show


class GestureDebouncer:
    def __init__(self, confirm_frames: int = 3):
        self.confirm_frames = confirm_frames
        self.history = deque(maxlen=confirm_frames)
        self.current = Show.NONE

    def update(self, detected: Show) -> Show:
        self.history.append(detected)
        # Switch only when the last N frames unanimously agree
        if len(self.history) == self.confirm_frames and \
           len(set(self.history)) == 1 and self.history[0] != self.current:
            self.current = self.history[0]
        return self.current
```

3 frames at 30fps = 100ms confirmation delay. Feels instant, eliminates flicker.

---

## Phase 2 — Style renderers (full-frame treatment)

Each show gets a distinct art style applied to the entire frame — you, your clothes, and the background.

### 2.1 — Style interface

`src/styles/base.py`:
```python
from abc import ABC, abstractmethod
import numpy as np

class StyleRenderer(ABC):
    @abstractmethod
    def render(self, frame: np.ndarray, person_mask: np.ndarray | None) -> np.ndarray:
        """Apply the art style to the full frame. person_mask (H,W) in [0,1]
        lets styles treat foreground and background differently."""
        ...
```

### 2.2 — Naruto: cel-shade + bold ink outlines

`src/styles/naruto.py`:
```python
"""Naruto style: flat cel-shaded colors, thick black ink outlines,
orange/blue palette push."""
import cv2
import numpy as np
from styles.base import StyleRenderer

NARUTO_PALETTE = np.array([
    [20, 20, 20],      # near-black (outlines/shadows)
    [235, 240, 245],   # off-white
    [30, 90, 220],     # naruto blue (BGR)
    [20, 120, 240],    # orange (BGR)
    [60, 60, 180],     # muted red
    [180, 140, 60],    # tan skin base (BGR)
], dtype=np.uint8)


def _quantize_colors(frame: np.ndarray, k: int = 8) -> np.ndarray:
    """K-means color quantization → flat cel-shaded regions."""
    data = frame.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(frame.shape).astype(np.uint8)
    return quantized


def _ink_outline(frame: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Thick black outlines via edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=9, C=2
    )
    # Thicken edges
    kernel = np.ones((thickness, thickness), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges  # white=keep, black=outline


class NarutoStyle(StyleRenderer):
    def render(self, frame, person_mask=None):
        # Bilateral filter smooths while preserving edges (cel look)
        smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        quantized = _quantize_colors(smooth, k=8)
        edges = _ink_outline(frame, thickness=2)
        # Composite outlines over quantized color
        result = cv2.bitwise_and(quantized, quantized, mask=edges)
        # Warm the whole frame slightly toward orange
        result = cv2.addWeighted(result, 0.85,
                                 np.full_like(result, (20, 60, 90)), 0.15, 0)
        return result
```

### 2.3 — ATLA: watercolor ink-wash (GLSL)

The watercolor effect is much cleaner as a GLSL shader. Pure NumPy watercolor looks muddy.

`src/shaders/watercolor.glsl`:
```glsl
#version 330
uniform sampler2D tex;
uniform float time;
in vec2 uv;
out vec4 fragColor;

// Simple value-noise for paper texture + edge bleeding
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(hash(i), hash(i+vec2(1,0)), u.x),
               mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), u.x), u.y);
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(tex, 0));
    // Slight UV distortion → color bleeding
    vec2 warp = vec2(noise(uv*40.0), noise(uv*40.0+7.3)) - 0.5;
    vec3 col = texture(tex, uv + warp * texel * 3.0).rgb;

    // Posterize gently for flat wash regions
    col = floor(col * 6.0) / 6.0;

    // Warm desaturation (ATLA palette)
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(lum), col, 0.65);
    col *= vec3(1.08, 1.0, 0.9);   // warm bias

    // Paper grain
    float grain = noise(uv * 800.0) * 0.06;
    col += grain - 0.03;

    fragColor = vec4(col, 1.0);
}
```

`src/styles/atla.py`:
```python
"""ATLA style: watercolor ink-wash via GLSL shader."""
import moderngl
import numpy as np
from styles.base import StyleRenderer

class ATLAStyle(StyleRenderer):
    def __init__(self, width, height):
        self.ctx = moderngl.create_standalone_context()
        with open("src/shaders/watercolor.glsl") as f:
            frag = f.read()
        vert = """
        #version 330
        in vec2 in_pos; out vec2 uv;
        void main() { uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos,0,1); }
        """
        self.prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        quad = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype='f4')
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_pos')
        self.fbo = self.ctx.simple_framebuffer((width, height))
        self.width, self.height = width, height

    def render(self, frame, person_mask=None):
        tex = self.ctx.texture((self.width, self.height), 3,
                               frame[:, :, ::-1].tobytes())  # BGR→RGB
        tex.use(0)
        self.fbo.use()
        self.prog['tex'] = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)
        data = self.fbo.read(components=3)
        out = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
        tex.release()
        return out[:, :, ::-1].copy()  # RGB→BGR
```

### 2.4 — JJK: dark high-contrast

`src/styles/jjk.py`:
```python
"""JJK style: crushed blacks, boosted highlights, purple/blue accent grade."""
import cv2
import numpy as np
from styles.base import StyleRenderer

class JJKStyle(StyleRenderer):
    def render(self, frame, person_mask=None):
        # High contrast S-curve
        lut = np.array([
            np.clip(255 * ((i/255.0)**1.4), 0, 255) for i in range(256)
        ], dtype=np.uint8)
        graded = cv2.LUT(frame, lut)
        # Desaturate then push toward purple/blue in shadows
        hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.55  # desaturate
        graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # Purple/blue tint in the darks
        dark_mask = (cv2.cvtColor(graded, cv2.COLOR_BGR2GRAY) < 100)
        graded[dark_mask] = cv2.addWeighted(
            graded[dark_mask], 0.7,
            np.full_like(graded[dark_mask], (120, 40, 80)), 0.3, 0)
        return graded
```

---

## Phase 3 — Character feature compositing

This is the standout technical work. Place character assets precisely on facial landmarks in real time.

### 3.1 — Key MediaPipe Face Mesh landmarks

```
With refine_face_landmarks=True, you get 478 points including iris:
  Forehead center:      10, 151, 9
  Left eye center:      468 (left iris center)
  Right eye center:     473 (right iris center)
  Left eye corners:     33 (outer), 133 (inner)
  Right eye corners:    263 (outer), 362 (inner)
  Left eyebrow:         70, 63, 105, 66, 107
  Right eyebrow:        336, 296, 334, 293, 300
  Nose bridge:          6, 197, 195, 5
  Face width ref:       234 (left cheek), 454 (right cheek)
```

### 3.2 — Compositor base

`src/features/compositor.py`:
```python
"""Landmark-driven asset compositing. Warps and blends transparent PNG assets
onto facial landmark positions with correct scale and rotation."""
import cv2
import numpy as np


def overlay_rgba(base: np.ndarray, asset_rgba: np.ndarray,
                 center: tuple[int, int], scale: float, angle: float) -> np.ndarray:
    """Rotate + scale an RGBA asset and alpha-composite onto base at center."""
    ah, aw = asset_rgba.shape[:2]
    new_w, new_h = int(aw * scale), int(ah * scale)
    if new_w < 1 or new_h < 1:
        return base
    resized = cv2.resize(asset_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Rotate around asset center
    M = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle, 1.0)
    rotated = cv2.warpAffine(resized, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

    # Compute placement bounds
    cx, cy = center
    x0, y0 = cx - new_w // 2, cy - new_h // 2
    x1, y1 = x0 + new_w, y0 + new_h

    # Clip to frame
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(base.shape[1], x1), min(base.shape[0], y1)
    if bx0 >= bx1 or by0 >= by1:
        return base

    ax0, ay0 = bx0 - x0, by0 - y0
    ax1, ay1 = ax0 + (bx1 - bx0), ay0 + (by1 - by0)

    asset_region = rotated[ay0:ay1, ax0:ax1]
    alpha = asset_region[:, :, 3:4].astype(np.float32) / 255.0
    rgb = asset_region[:, :, :3].astype(np.float32)

    base_region = base[by0:by1, bx0:bx1].astype(np.float32)
    blended = alpha * rgb + (1 - alpha) * base_region
    base[by0:by1, bx0:bx1] = blended.astype(np.uint8)
    return base


def face_angle(face: np.ndarray) -> float:
    """Roll angle of the face in degrees, from eye-to-eye vector."""
    left_eye = face[33, :2]
    right_eye = face[263, :2]
    dx, dy = right_eye - left_eye
    return -np.degrees(np.arctan2(dy, dx))


def face_width(face: np.ndarray) -> float:
    """Pixel width of the face for scale reference."""
    return np.linalg.norm(face[454, :2] - face[234, :2])
```

### 3.3 — Naruto features: headband + Sage Mode eyes

`src/features/naruto_features.py`:
```python
"""Naruto: Hidden Leaf headband on forehead, Sage Mode iris on eyes."""
import cv2
import numpy as np
from features.compositor import overlay_rgba, face_angle, face_width

_headband = cv2.imread("assets/naruto/headband.png", cv2.IMREAD_UNCHANGED)
_sage_iris = cv2.imread("assets/naruto/sage_iris.png", cv2.IMREAD_UNCHANGED)


def apply(frame: np.ndarray, face: np.ndarray) -> np.ndarray:
    if face is None:
        return frame
    fw = face_width(face)
    angle = face_angle(face)

    # Headband: anchor across forehead (landmark 10 = forehead top center,
    # 151 = slightly lower). Place between brows and hairline.
    forehead = face[10, :2].astype(int)
    hairline = face[151, :2].astype(int)
    hb_center = ((forehead + hairline) // 2)
    hb_scale = fw / _headband.shape[1] * 1.4
    frame = overlay_rgba(frame, _headband, tuple(hb_center), hb_scale, angle)

    # Sage Mode eyes: overlay orange toad-iris on each iris center.
    # Iris landmarks: 468 (left), 473 (right) with refine_face_landmarks
    for iris_idx, corner_a, corner_b in [(468, 33, 133), (473, 362, 263)]:
        iris_center = face[iris_idx, :2].astype(int)
        eye_w = np.linalg.norm(face[corner_a, :2] - face[corner_b, :2])
        iris_scale = eye_w / _sage_iris.shape[1] * 0.9
        frame = overlay_rgba(frame, _sage_iris, tuple(iris_center), iris_scale, angle)

    return frame
```

### 3.4 — ATLA features: Avatar State eyes + arrow tattoos

`src/features/atla_features.py`:
```python
"""ATLA: glowing white Avatar State eyes + blue arrow tattoos on forehead & hands."""
import cv2
import numpy as np
from features.compositor import overlay_rgba, face_angle, face_width

_arrow_head = cv2.imread("assets/atla/arrow_head.png", cv2.IMREAD_UNCHANGED)
_avatar_glow = cv2.imread("assets/atla/avatar_glow.png", cv2.IMREAD_UNCHANGED)
_arrow_hand = cv2.imread("assets/atla/arrow_hand.png", cv2.IMREAD_UNCHANGED)


def apply(frame, face, left_hand=None, right_hand=None):
    if face is not None:
        fw = face_width(face)
        angle = face_angle(face)

        # Forehead arrow: point starts at brow center (9), extends up over forehead (10)
        brow = face[9, :2]
        top = face[10, :2]
        arrow_center = ((brow + top) / 2).astype(int)
        arrow_scale = fw / _arrow_head.shape[1] * 0.5
        frame = overlay_rgba(frame, _arrow_head, tuple(arrow_center), arrow_scale, angle)

        # Avatar State glow over both eyes (additive white glow)
        for iris_idx, ca, cb in [(468, 33, 133), (473, 362, 263)]:
            eye_center = face[iris_idx, :2].astype(int)
            eye_w = np.linalg.norm(face[ca, :2] - face[cb, :2])
            glow_scale = eye_w / _avatar_glow.shape[1] * 1.6
            frame = overlay_rgba(frame, _avatar_glow, tuple(eye_center), glow_scale, angle)

    # Hand arrows: back of each hand (landmark 9 = middle finger MCP ≈ hand center)
    for hand in [left_hand, right_hand]:
        if hand is not None:
            hand_center = hand[9, :2].astype(int)
            hand_w = np.linalg.norm(hand[5, :2] - hand[17, :2])  # index to pinky MCP
            hand_scale = hand_w / _arrow_hand.shape[1] * 1.2
            # Angle from wrist to middle finger
            vec = hand[12, :2] - hand[0, :2]
            hand_angle = -np.degrees(np.arctan2(vec[1], vec[0])) - 90
            frame = overlay_rgba(frame, _arrow_hand, tuple(hand_center),
                                 hand_scale, hand_angle)
    return frame
```

### 3.5 — JJK features: Gojo's Six Eyes

`src/features/jjk_features.py`:
```python
"""JJK: Gojo's Six Eyes — bright blue glowing eyes with the distinctive look.
Simplest is a stylized blue iris + glow; the 'six' is aesthetic, not literal."""
import cv2
import numpy as np
from features.compositor import overlay_rgba, face_angle

_six_eyes = cv2.imread("assets/jjk/six_eyes_overlay.png", cv2.IMREAD_UNCHANGED)


def apply(frame, face):
    if face is None:
        return frame
    angle = face_angle(face)

    # Gojo's eyes are striking bright blue. Overlay a blue-glow iris texture
    # scaled to each eye, plus a subtle cyan additive glow around the sockets.
    for iris_idx, ca, cb in [(468, 33, 133), (473, 362, 263)]:
        eye_center = face[iris_idx, :2].astype(int)
        eye_w = np.linalg.norm(face[ca, :2] - face[cb, :2])
        scale = eye_w / _six_eyes.shape[1] * 1.1
        frame = overlay_rgba(frame, _six_eyes, tuple(eye_center), scale, angle)

    # Optional: cyan rim light on the upper face for the "cursed energy" aura
    return frame
```

> **On "Six Eyes":** Gojo's Six Eyes is a dojutsu ability, not literally six eyeballs — visually it's his piercing bright blue eyes. The asset should be a stylized glowing blue iris with cyan energy, not six literal eyes. If you want the literal meme interpretation (six eyes on the face), that's a different asset but the same compositing logic — just add more overlay points using additional face landmarks.

---

## Phase 4 — Orchestration

`src/main.py`:
```python
"""Main capture loop. Ties gesture detection → style → features together."""
import cv2
import numpy as np

from landmarks import LandmarkExtractor
from gesture.detector import detect_show, Show
from gesture.smoothing import GestureDebouncer
from styles.naruto import NarutoStyle
from styles.atla import ATLAStyle
from styles.jjk import JJKStyle
from features import naruto_features, atla_features, jjk_features
from segmentation import PersonSegmenter


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    extractor = LandmarkExtractor()
    debouncer = GestureDebouncer(confirm_frames=3)
    segmenter = PersonSegmenter()

    styles = {
        Show.NARUTO: NarutoStyle(),
        Show.ATLA: ATLAStyle(w, h),
        Show.JJK: JJKStyle(),
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror for natural interaction

        lm = extractor.process(frame)
        raw_show = detect_show(lm.left_hand, lm.right_hand)
        show = debouncer.update(raw_show)

        if show == Show.NONE:
            output = frame
        else:
            mask = segmenter.mask(frame)
            # 1. Full-frame style
            output = styles[show].render(frame, mask)
            # 2. Character features on top
            if show == Show.NARUTO:
                output = naruto_features.apply(output, lm.face)
            elif show == Show.ATLA:
                output = atla_features.apply(output, lm.face,
                                             lm.left_hand, lm.right_hand)
            elif show == Show.JJK:
                output = jjk_features.apply(output, lm.face)

        # HUD: show active mode
        cv2.putText(output, show.value.upper(), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Anime Transform", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

`src/segmentation.py`:
```python
"""Selfie segmentation → person vs background mask."""
import mediapipe as mp
import numpy as np

class PersonSegmenter:
    def __init__(self):
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def mask(self, frame):
        rgb = frame[:, :, ::-1]
        result = self.seg.process(rgb)
        return result.segmentation_mask  # (H,W) float [0,1]
```

---

## Phase 5 — Asset creation

You need transparent PNG assets. Options ranked by effort:

**Fastest — AI image generation:**
Generate each asset with a transparent background using an image model, then clean up alpha in any editor. Prompts like "Naruto Hidden Leaf headband, front view, transparent background, game asset" work well. You'll need: headband, sage iris, ATLA forehead arrow, ATLA hand arrow, avatar glow, Gojo blue eye overlay.

**Better quality — trace from reference:**
Screenshot the actual anime frames, trace in Figma or Procreate, export as transparent PNG. More faithful but slower.

**Asset requirements:**
- All PNGs with proper alpha channel (RGBA)
- Front-facing / symmetric where possible (they get rotated by the compositor)
- Headband: wide aspect ratio, metal plate centered
- Iris assets: square, centered iris, transparent outside the circle
- Arrow tattoos: the ATLA arrow is a specific shape — pointed at top, stem down the forehead

---

## Phase 6 (optional) — Browser deployment

The high-effort, high-reward path. Port to run at a URL with no install.

- **MediaPipe Tasks for Web** (`@mediapipe/tasks-vision`) — Face Landmarker + Hand Landmarker in JS
- **Effects in WebGL** — port the GLSL shaders directly (they're already GLSL), rewrite the OpenCV NumPy effects as fragment shaders
- **Asset compositing** — canvas 2D or WebGL textured quads at landmark positions
- **Deploy on Vercel** — static site, webcam via `getUserMedia`

This is where the project becomes shareable — someone opens a link, allows webcam, forms the rectangle, transforms. That's the version that goes viral like the wxll.hx post.

---

## Implementation order

| Step | Task | Effort |
|---|---|---|
| 1 | MediaPipe Holistic pipeline + landmark extraction | 2 hr |
| 2 | Gesture detector + debouncer — get show switching working | 3 hr |
| 3 | Naruto style (cel-shade + outline) — prove the style pipeline | 2 hr |
| 4 | JJK + ATLA styles (ATLA needs GLSL setup) | 4 hr |
| 5 | Create/generate the PNG assets | 2 hr |
| 6 | Compositor + Naruto features (headband + eyes) | 3 hr |
| 7 | ATLA + JJK features | 2 hr |
| 8 | Segmentation, polish, HUD, recording | 2 hr |
| 9 | Record demo clips, write README | 2 hr |
| 10 | (Optional) Browser port to WebGL + Vercel | 8+ hr |

**Core (steps 1-9):** ~22 hours across 3-4 sessions.

---

## The hardest parts (where to expect trouble)

**Gesture disambiguation.** Thumb+index, index+middle, and middle+pinky overlap — when your index is extended for Naruto, it's also extended for ATLA. The detector's "cleanest match" scoring handles this but expect to tune the extension thresholds against your own hands on camera. This is the part that'll take the most iteration.

**Feature alignment at angles.** When you tilt your head, the headband and eyes need to rotate and stay glued to the face. The `face_angle` roll compensation handles 2D rotation but not pitch/yaw (looking up/down/side). For a portfolio demo, staying roughly front-facing is fine; full 3D pose is a much harder problem you don't need to solve.

**30fps with everything on.** MediaPipe Holistic + K-means quantization + GLSL + compositing is a lot per frame. If it drops below 24fps, the K-means color quantization in the Naruto style is the first thing to optimize — reduce `k`, downsample before quantizing, or cache the palette.

---

## Why this is a standout portfolio project

The interview story: "I built a real-time anime transformation filter. It runs MediaPipe Holistic to track hands, face mesh with iris refinement, and body pose simultaneously. A rectangle gesture — different finger pairs form the corners — instantly switches between three transformations. Each applies a full-frame art style, one of them through a custom GLSL fragment shader running on the GPU, then composites character-specific features onto facial landmarks in real time: a headband anchored to forehead points, iris textures aligned to the refined iris landmarks, tattoos placed on hand keypoints. The whole pipeline holds 30fps."

That covers: real-time CV, multi-model landmark fusion, GLSL/GPU programming, geometric transforms (rotation/scale from landmark vectors), and a genuinely novel interaction model. Nothing else in a typical new-grad portfolio looks like it — and the demo video is the kind of thing that gets shared.