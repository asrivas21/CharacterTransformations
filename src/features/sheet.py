"""Sheet warp/composite for the "stretched sheet" interaction.

Warps a full-frame styled image into an arbitrary quad (perspective), carves a
jittered torn leading edge, lays RGB halftone dots on top, and alpha-composites
sheets in painter order."""
from __future__ import annotations

import cv2
import numpy as np

from styles.risograph import _halftone_mask  # reuse the dot lattice


def _quad_is_degenerate(quad: np.ndarray, min_area: float = 25.0) -> bool:
    """True when the quad is collinear / near-zero-area (invalid perspective)."""
    q = quad.astype(np.float64)
    area = 0.0
    for i in range(4):
        x0, y0 = q[i]
        x1, y1 = q[(i + 1) % 4]
        area += x0 * y1 - x1 * y0
    return abs(area) * 0.5 < min_area


def warp_styled_into_quad(styled_full_frame: np.ndarray, dst_quad: np.ndarray,
                          frame_shape) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-warp a full styled frame into `dst_quad`.

    Source corners are BL, TL, TR, BR so the image stays upright: the left hand
    (dst corners 0/1) gets the image's left edge and the right hand (dst corners
    2/3) its right edge, with tops mapping to tops. Returns (warped, mask)."""
    h, w = frame_shape[:2]
    src = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst_quad.astype(np.float32))
    warped = cv2.warpPerspective(styled_full_frame, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, dst_quad.astype(np.int32), 255)
    return warped, mask


def _torn_edge_alpha(quad: np.ndarray, shape, jitter: int = 8,
                     seed: int = 7) -> np.ndarray:
    """Alpha (0/255) for the quad with a jittered diagonal cut on the leading
    (right) edge — the segment between quad[2] (right_b) and quad[3] (right_a).

    The torn edge is pushed inward (toward the quad centroid), so the torn area
    is always smaller than a plain fillConvexPoly of the same quad."""
    h, w = shape[:2]
    q = quad.astype(np.float32)
    centroid = q.mean(axis=0)
    rng = np.random.default_rng(seed)
    n = 12
    ts = np.linspace(0.0, 1.0, n)
    torn = []
    for t in ts:
        p = (1.0 - t) * q[2] + t * q[3]
        d = centroid - p
        norm = float(np.linalg.norm(d)) + 1e-6
        jit = float(rng.uniform(0.0, jitter))
        torn.append(p + (d / norm) * jit)
    poly = np.array([q[0], q[1], *torn], dtype=np.float32)
    alpha = np.zeros((h, w), np.uint8)
    cv2.fillPoly(alpha, [poly.astype(np.int32)], 255)
    return alpha


def halftone_overlay(img_bgr: np.ndarray, cell: int = 6,
                     strength: float = 0.6) -> np.ndarray:
    """Lay RGB halftone dots over `img_bgr`, blending toward the dotted version
    by `strength`. Dots are built per channel via the risograph dot lattice at 3
    screen angles (15/75/0) to reduce moire."""
    strength = float(np.clip(strength, 0.0, 1.0))
    f = img_bgr.astype(np.float32) / 255.0
    angles = (15.0, 75.0, 0.0)  # B, G, R
    dotted = np.empty_like(f)
    for c in range(3):
        ink = 1.0 - f[:, :, c]  # more ink where the channel is dark
        mask = _halftone_mask(ink, cell, angles[c])
        # White paper where no dot; dark channel where a dot lands.
        dotted[:, :, c] = 1.0 - mask
    out = f * (1.0 - strength) + dotted * strength
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def composite_sheet(base: np.ndarray, warped_bgr: np.ndarray,
                    alpha: np.ndarray) -> np.ndarray:
    """Painter-order alpha blend of `warped_bgr` over `base` using `alpha`
    (0..255). Mutates and returns `base`."""
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    blended = a * warped_bgr.astype(np.float32) + (1.0 - a) * base.astype(np.float32)
    base[:] = np.clip(blended, 0, 255).astype(np.uint8)
    return base
