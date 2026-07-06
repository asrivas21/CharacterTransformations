"""Cyanotype (blueprint) style: grayscale -> Prussian-blue LUT with a contrast
boost and a soft vignette.

`control` (inter-hand distance, 0..1) drives exposure: hands far apart => a
brighter, more sun-exposed print; hands close => a darker, deeper blue."""
from __future__ import annotations

import cv2
import numpy as np

from styles.base import StyleRenderer


def _build_blue_luts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel (B, G, R) LUTs mapping luminance -> cyanotype tones.

    Shadows land on deep Prussian blue, highlights on pale blue-white."""
    x = np.arange(256, dtype=np.float32) / 255.0
    # Endpoints (BGR): shadow (60,30,10) -> highlight (250,235,200).
    shadow = np.array([60.0, 30.0, 10.0])
    highlight = np.array([250.0, 235.0, 200.0])
    lut = shadow[None, :] + x[:, None] * (highlight - shadow)[None, :]
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[:, 0].copy(), lut[:, 1].copy(), lut[:, 2].copy()


_LUT_B, _LUT_G, _LUT_R = _build_blue_luts()

# Contrast S-curve LUT applied to the grayscale before toning.
_CONTRAST_LUT = np.clip(
    255.0 / (1.0 + np.exp(-10.0 * (np.arange(256) / 255.0 - 0.5))), 0, 255
).astype(np.uint8)


class CyanotypeStyle(StyleRenderer):
    def __init__(self, min_gain: float = 0.7, max_gain: float = 1.4):
        self.min_gain = min_gain
        self.max_gain = max_gain
        self._vignette_cache: dict[tuple[int, int], np.ndarray] = {}

    def _vignette(self, h: int, w: int) -> np.ndarray:
        """Cached radial falloff mask in [0,1], (H,W,1)."""
        cached = self._vignette_cache.get((h, w))
        if cached is not None:
            return cached
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        r = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
        mask = np.clip(1.0 - 0.6 * np.clip(r - 0.4, 0, None), 0.3, 1.0)
        mask = mask[:, :, None]
        self._vignette_cache[(h, w)] = mask
        return mask

    def render(self, frame: np.ndarray, control: float = 0.5) -> np.ndarray:
        control = float(np.clip(control, 0.0, 1.0))
        gain = self.min_gain + control * (self.max_gain - self.min_gain)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Exposure: scale luminance by the hand-distance gain.
        gray = np.clip(gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        gray = cv2.LUT(gray, _CONTRAST_LUT)

        toned = cv2.merge([
            cv2.LUT(gray, _LUT_B),
            cv2.LUT(gray, _LUT_G),
            cv2.LUT(gray, _LUT_R),
        ])

        h, w = frame.shape[:2]
        vig = self._vignette(h, w)
        out = toned.astype(np.float32) * vig
        return np.clip(out, 0, 255).astype(np.uint8)
