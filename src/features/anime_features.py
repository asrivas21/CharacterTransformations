"""Anime tracker: traces the actual iris and places a Sharingan there, clipped
to the visible eye opening (so it never covers eyelids/skin and vanishes on a
blink). The camera background is kept as-is.

Built to run on a cropped sheet region, so face landmarks (full-frame pixels)
are translated into region-local coordinates via `offset`.

Requires the refined 478-point mesh (iris landmarks 468-477)."""
from __future__ import annotations

import cv2
import numpy as np

from features.compositor import load_asset, face_angle

_sharingan = load_asset("naruto/sharingan-image.png")

# Refined-mesh iris: (center idx, ring indices) per eye.
_IRIS = [
    (468, (469, 470, 471, 472)),  # left
    (473, (474, 475, 476, 477)),  # right
]

# Eyelid ring (visible eye opening) per eye, for the clip mask.
_LEFT_EYE_RING = (33, 246, 161, 160, 159, 158, 157, 173,
                  133, 155, 154, 153, 145, 144, 163, 7)
_RIGHT_EYE_RING = (263, 466, 388, 387, 386, 385, 384, 398,
                   362, 382, 381, 380, 374, 373, 390, 249)
_EYE_RINGS = [_LEFT_EYE_RING, _RIGHT_EYE_RING]

# Sharingan diameter relative to iris diameter (>1 to fully cover the iris).
_IRIS_SCALE = 1.35


def _iris_radius(face: np.ndarray, center_idx: int,
                 ring: tuple[int, ...]) -> float:
    c = face[center_idx, :2]
    return float(np.mean([np.linalg.norm(face[r, :2] - c) for r in ring]))


def _overlay_clipped(region: np.ndarray, asset: np.ndarray,
                     center: tuple[int, int], diameter: int, angle: float,
                     aperture: np.ndarray) -> None:
    """Scale+rotate `asset` to `diameter`, clip its alpha to the `aperture`
    mask (region-local, uint8 0/255), and alpha-composite onto `region`."""
    if diameter < 2:
        return
    resized = cv2.resize(asset, (diameter, diameter),
                         interpolation=cv2.INTER_AREA)
    m = cv2.getRotationMatrix2D((diameter / 2, diameter / 2), angle, 1.0)
    rot = cv2.warpAffine(resized, m, (diameter, diameter),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0, 0))

    cx, cy = center
    x0, y0 = cx - diameter // 2, cy - diameter // 2
    x1, y1 = x0 + diameter, y0 + diameter
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(region.shape[1], x1), min(region.shape[0], y1)
    if bx0 >= bx1 or by0 >= by1:
        return
    ax0, ay0 = bx0 - x0, by0 - y0
    ax1, ay1 = ax0 + (bx1 - bx0), ay0 + (by1 - by0)

    piece = rot[ay0:ay1, ax0:ax1]
    alpha = piece[:, :, 3:4].astype(np.float32) / 255.0
    # Clip to the visible eye opening.
    ap = aperture[by0:by1, bx0:bx1].astype(np.float32)[:, :, None] / 255.0
    alpha *= ap
    rgb = piece[:, :, :3].astype(np.float32)
    base = region[by0:by1, bx0:bx1].astype(np.float32)
    region[by0:by1, bx0:bx1] = (alpha * rgb + (1 - alpha) * base).astype(np.uint8)


def apply(region: np.ndarray, face: np.ndarray | None,
          offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Overlay iris-traced, eyelid-clipped Sharingan eyes onto `region`
    (mutated and returned). `offset` is the region's (x0, y0) in the frame."""
    if face is None or _sharingan is None or face.shape[0] <= 477:
        return region
    ox, oy = offset
    off = np.array([ox, oy], np.float32)
    angle = face_angle(face)
    h, w = region.shape[:2]
    for (center_idx, ring), eye_ring in zip(_IRIS, _EYE_RINGS):
        # Region-local eye-aperture mask from the eyelid ring. Use the convex
        # hull so the fill is robust to landmark ordering.
        pts = (face[list(eye_ring), :2] - off).astype(np.int32)
        aperture = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(aperture, cv2.convexHull(pts), 255)

        c = face[center_idx, :2] - off
        cx, cy = int(round(c[0])), int(round(c[1]))
        radius = _iris_radius(face, center_idx, ring)
        diameter = int(round(2.0 * radius * _IRIS_SCALE))
        _overlay_clipped(region, _sharingan, (cx, cy), diameter, angle,
                         aperture)
    return region
