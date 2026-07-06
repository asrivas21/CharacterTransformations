"""JJK: Gojo's Six Eyes — striking bright-blue glowing iris overlay."""
from __future__ import annotations

import numpy as np

from features.compositor import load_asset, overlay_rgba, face_angle

_six_eyes = load_asset("jjk/six_eyes_overlay.png")

_EYES = [(468, 33, 133), (473, 362, 263)]


def apply(frame: np.ndarray, face: np.ndarray | None) -> np.ndarray:
    if face is None or _six_eyes is None:
        return frame
    if face.shape[0] <= 473:
        return frame
    angle = face_angle(face)

    for iris_idx, ca, cb in _EYES:
        eye_center = tuple(face[iris_idx, :2].astype(int))
        eye_w = np.linalg.norm(face[ca, :2] - face[cb, :2])
        scale = eye_w / _six_eyes.shape[1] * 1.1
        frame = overlay_rgba(frame, _six_eyes, eye_center, scale, angle)

    return frame
