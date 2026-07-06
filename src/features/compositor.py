"""Landmark-driven asset compositing. Warps and blends transparent PNG assets
onto facial/hand landmark positions with correct scale and rotation."""
from __future__ import annotations

import os

import cv2
import numpy as np

# Project root = two levels up from this file (src/features/ -> repo root).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ASSETS_DIR = os.path.join(_ROOT, "assets")


def load_asset(rel_path: str) -> np.ndarray | None:
    """Load an RGBA asset relative to the assets/ directory. Returns None if
    the file is missing so callers can degrade gracefully."""
    path = os.path.join(ASSETS_DIR, rel_path)
    asset = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if asset is None:
        print(f"[compositor] missing asset: {path}")
        return None
    if asset.ndim == 3 and asset.shape[2] == 3:  # add opaque alpha
        alpha = np.full(asset.shape[:2] + (1,), 255, dtype=asset.dtype)
        asset = np.concatenate([asset, alpha], axis=2)
    return asset


def overlay_rgba(base: np.ndarray, asset_rgba: np.ndarray | None,
                 center: tuple[int, int], scale: float, angle: float) -> np.ndarray:
    """Rotate + scale an RGBA asset and alpha-composite onto base at center."""
    if asset_rgba is None:
        return base
    ah, aw = asset_rgba.shape[:2]
    new_w, new_h = int(aw * scale), int(ah * scale)
    if new_w < 1 or new_h < 1:
        return base
    resized = cv2.resize(asset_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Rotate around the asset center.
    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
    rotated = cv2.warpAffine(resized, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

    cx, cy = int(center[0]), int(center[1])
    x0, y0 = cx - new_w // 2, cy - new_h // 2
    x1, y1 = x0 + new_w, y0 + new_h

    # Clip to frame bounds.
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
    """Roll angle of the face in degrees, from the eye-to-eye vector."""
    left_eye = face[33, :2]
    right_eye = face[263, :2]
    dx, dy = right_eye - left_eye
    return -np.degrees(np.arctan2(dy, dx))


def face_width(face: np.ndarray) -> float:
    """Pixel width of the face for scale reference (cheek to cheek)."""
    return float(np.linalg.norm(face[454, :2] - face[234, :2]))
