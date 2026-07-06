"""JJK style: crushed blacks, boosted highlights, purple/blue accent grade."""
from __future__ import annotations

import cv2
import numpy as np

from styles.base import StyleRenderer

# High-contrast gamma S-curve LUT, precomputed once.
_CONTRAST_LUT = np.clip(
    255.0 * ((np.arange(256) / 255.0) ** 1.4), 0, 255
).astype(np.uint8)

# Purple/blue tint pushed into the shadows (BGR).
_SHADOW_TINT = np.array([120, 40, 80], dtype=np.uint8)


class JJKStyle(StyleRenderer):
    def render(self, frame: np.ndarray, control: float = 0.5) -> np.ndarray:
        graded = cv2.LUT(frame, _CONTRAST_LUT)

        # Desaturate.
        hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.55
        hsv = np.clip(hsv, 0, 255)
        graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Purple/blue tint in the darks.
        gray = cv2.cvtColor(graded, cv2.COLOR_BGR2GRAY)
        dark_mask = gray < 100
        if np.any(dark_mask):
            tint = np.full_like(graded[dark_mask], _SHADOW_TINT)
            graded[dark_mask] = cv2.addWeighted(
                graded[dark_mask], 0.7, tint, 0.3, 0)
        return graded
