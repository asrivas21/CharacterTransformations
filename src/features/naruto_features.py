"""Naruto: Hidden Leaf headband on forehead, Sage Mode iris on eyes."""
from __future__ import annotations

import numpy as np

from features.compositor import load_asset, overlay_rgba, face_angle, face_width

_headband = load_asset("naruto/headband.png")
_sage_iris = load_asset("naruto/sage_iris.png")

# (iris center, inner corner, outer corner) for left and right eyes.
_EYES = [(468, 33, 133), (473, 362, 263)]


def apply(frame: np.ndarray, face: np.ndarray | None) -> np.ndarray:
    if face is None:
        return frame
    fw = face_width(face)
    angle = face_angle(face)

    # Headband across the forehead (10 = top center, 151 = slightly lower).
    forehead = face[10, :2]
    hairline = face[151, :2]
    hb_center = tuple(((forehead + hairline) / 2).astype(int))
    if _headband is not None:
        hb_scale = fw / _headband.shape[1] * 1.4
        frame = overlay_rgba(frame, _headband, hb_center, hb_scale, angle)

    # Sage Mode eyes on each refined iris center.
    if _sage_iris is not None and face.shape[0] > 473:
        for iris_idx, corner_a, corner_b in _EYES:
            iris_center = tuple(face[iris_idx, :2].astype(int))
            eye_w = np.linalg.norm(face[corner_a, :2] - face[corner_b, :2])
            iris_scale = eye_w / _sage_iris.shape[1] * 0.9
            frame = overlay_rgba(frame, _sage_iris, iris_center, iris_scale, angle)

    return frame
