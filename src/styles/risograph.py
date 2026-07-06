"""Risograph style: a single red spot-color halftone on white paper. The image
is reduced to grayscale and printed as darkness-driven red dots (darker =>
bigger dots) via sine-wave thresholding, for a one-drum Riso print look.

`control` (finger spread, 0..1) drives the halftone cell size: a wider spread
prints coarser, chunkier dots."""
from __future__ import annotations

import cv2
import numpy as np

from styles.base import StyleRenderer

# Single Riso "red" drum (BGR) printed on white paper.
_RISO_RED = np.array([45, 40, 225], dtype=np.float32)
_PAPER = np.array([255, 255, 255], dtype=np.float32)  # white stock
_SCREEN_ANGLE = 15.0  # halftone screen angle (degrees)


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

        # Grayscale -> ink amount (darkness). Darker pixels grow bigger dots.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        ink = 1.0 - gray
        mask = _halftone_mask(ink, cell, _SCREEN_ANGLE)  # (H,W) 0/1

        # White paper everywhere; lay red ink only where a dot lands.
        out = np.tile(_PAPER, (frame.shape[0], frame.shape[1], 1))
        cov = mask[:, :, None]  # (H,W,1)
        out = out * (1.0 - cov) + _RISO_RED * cov

        return np.clip(out, 0, 255).astype(np.uint8)
