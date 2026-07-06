"""Stippling style: black ink dots on white paper, placed with probability
proportional to darkness (inverse brightness). Fully vectorized — no per-pixel
Python loops — and computed on a downsampled grid for speed, then upscaled.

`control` (finger spread, 0..1) drives dot density: a wider spread lays down a
denser field of stipples."""
from __future__ import annotations

import cv2
import numpy as np

from styles.base import StyleRenderer


class StipplingStyle(StyleRenderer):
    def __init__(self, min_density: float = 0.15, max_density: float = 0.6,
                 grid: int = 220, dot_radius: int = 1):
        # `grid` = target width of the sampling lattice; smaller => coarser and
        # faster. Density scales the per-cell dot probability.
        self.min_density = min_density
        self.max_density = max_density
        self.grid = grid
        self.dot_radius = dot_radius
        self._rng = np.random.default_rng(12345)

    def render(self, frame: np.ndarray, control: float = 0.5) -> np.ndarray:
        control = float(np.clip(control, 0.0, 1.0))
        density = self.min_density + control * (self.max_density - self.min_density)

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return frame.copy()

        # Downsample to a lattice for the probability test (keeps it real-time).
        gw = max(8, min(self.grid, w))
        gh = max(8, int(round(gw * h / w)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)

        # Darkness in [0,1]; dot probability grows with darkness * density.
        darkness = 1.0 - small.astype(np.float32) / 255.0
        prob = np.clip(darkness * density, 0.0, 1.0)
        hits = self._rng.random((gh, gw)) < prob  # vectorized Bernoulli draw

        # White paper; paint black dots at the lattice cells that fired.
        out = np.full((h, w, 3), 255, dtype=np.uint8)
        ys, xs = np.nonzero(hits)
        if ys.size == 0:
            return out
        # Map lattice coords back to full-res pixel centers.
        px = (xs.astype(np.float32) * (w / gw)).astype(np.int32)
        py = (ys.astype(np.float32) * (h / gh)).astype(np.int32)

        if self.dot_radius <= 0:
            out[py, px] = (0, 0, 0)
        else:
            # Stamp small filled dots via a vectorized neighborhood splat.
            rr = self.dot_radius
            for oy in range(-rr, rr + 1):
                for ox in range(-rr, rr + 1):
                    if ox * ox + oy * oy > rr * rr:
                        continue
                    yy = np.clip(py + oy, 0, h - 1)
                    xx = np.clip(px + ox, 0, w - 1)
                    out[yy, xx] = (0, 0, 0)
        return out
