"""Risograph style: CMY channel separation, per-channel halftone dots via
sine-wave thresholding, and slight registration offsets for the printed look.

`control` (finger spread, 0..1) drives the halftone cell size: a wider spread
prints coarser, chunkier dots."""
from __future__ import annotations

import cv2
import numpy as np

from styles.base import StyleRenderer

# Spot-color inks (BGR) roughly matching real Riso drums.
_CYAN = np.array([180, 100, 0], dtype=np.float32)     # blue-cyan
_MAGENTA = np.array([80, 20, 200], dtype=np.float32)  # fluorescent pink
_YELLOW = np.array([30, 200, 230], dtype=np.float32)  # warm yellow
_PAPER = np.array([245, 244, 238], dtype=np.float32)  # off-white stock

# Registration offsets (dx, dy) per channel — the mis-alignment "error".
_OFFSETS = [(-2, -1), (2, 1), (1, -2)]  # C, M, Y


def _halftone_mask(ink: np.ndarray, cell: int, angle_deg: float) -> np.ndarray:
    """Return a float [0,1] dot-coverage mask for a single ink channel.

    A rotated sine grid produces a smooth dot lattice; thresholding it against
    the (inverted) ink amount grows/shrinks the dots so darker ink => bigger
    dots. Fully vectorized."""
    h, w = ink.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = np.radians(angle_deg)
    u = xx * np.cos(theta) - yy * np.sin(theta)
    v = xx * np.sin(theta) + yy * np.cos(theta)
    freq = np.pi / max(cell, 2)
    # Grid in [0,1]; peaks are dot centers.
    grid = (np.sin(u * freq) * np.sin(v * freq) + 1.0) * 0.5
    # Ink amount in [0,1] (already inverted by caller). Where grid <= ink -> dot.
    return (grid <= ink).astype(np.float32)


class RisographStyle(StyleRenderer):
    def __init__(self, min_cell: int = 4, max_cell: int = 12):
        self.min_cell = min_cell
        self.max_cell = max_cell

    def render(self, frame: np.ndarray, control: float = 0.5) -> np.ndarray:
        control = float(np.clip(control, 0.0, 1.0))
        cell = int(round(self.min_cell + control * (self.max_cell - self.min_cell)))

        f = frame.astype(np.float32) / 255.0
        b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        # CMY ink amounts (subtractive): more ink where the channel is dark.
        cyan = 1.0 - r
        magenta = 1.0 - g
        yellow = 1.0 - b

        # Classic screen angles (C 15, M 75, Y 0) to reduce moire.
        masks = [
            _halftone_mask(cyan, cell, 15.0),
            _halftone_mask(magenta, cell, 75.0),
            _halftone_mask(yellow, cell, 0.0),
        ]
        inks = [_CYAN, _MAGENTA, _YELLOW]

        # Start from paper white, multiply down where each ink dot lands, with a
        # per-channel registration shift.
        out = np.tile(_PAPER, (frame.shape[0], frame.shape[1], 1))
        for mask, ink, (dx, dy) in zip(masks, inks, _OFFSETS):
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            cov = shifted[:, :, None]  # (H,W,1)
            # Multiplicative ink: paper * (1-cov) + ink * cov, but keep it
            # subtractive so overlapping inks darken realistically.
            out = out * (1.0 - cov) + (out * (ink / 255.0)) * cov

        return np.clip(out, 0, 255).astype(np.uint8)
