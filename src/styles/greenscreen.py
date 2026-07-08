"""Green/white style: an inverted-luminance two-tone map. Dark pixels (hair,
shadows) become white and bright pixels (face, skin) become green, with the
tones blended continuously so the result reads as a tonal outline of the
subject rather than a flat white blob.

`control` (finger spread, 0..1) drives contrast: a wider spread pushes the map
harder toward the two extremes (crisper, more graphic)."""
from __future__ import annotations

import cv2
import numpy as np

from styles.base import StyleRenderer

# BGR fill colors.
_GREEN = np.array([0, 255, 0], dtype=np.float32)
_WHITE = np.array([255, 255, 255], dtype=np.float32)


class GreenScreenStyle(StyleRenderer):
    def render(self, frame: np.ndarray, control: float = 0.5) -> np.ndarray:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return frame.copy()

        # Luminance in [0, 1]; invert so dark => 1 (white), bright => 0 (green).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t = 1.0 - gray

        # Contrast S-curve around mid-gray. Wider finger spread => steeper curve
        # => more graphic (closer to pure two-tone) while keeping tonal detail.
        control = float(np.clip(control, 0.0, 1.0))
        steepness = 6.0 + 8.0 * control
        t = 1.0 / (1.0 + np.exp(-steepness * (t - 0.5)))

        t = t[:, :, None]
        out = _WHITE * t + _GREEN * (1.0 - t)
        return np.clip(out, 0, 255).astype(np.uint8)
