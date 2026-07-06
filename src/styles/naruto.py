"""Naruto style: flat cel-shaded colors, thick black ink outlines,
orange/blue palette push."""
from __future__ import annotations

import cv2
import numpy as np

from styles.base import StyleRenderer

# Warm orange bias applied across the whole frame (BGR).
_WARM_BIAS = np.array([20, 60, 90], dtype=np.uint8)


def _quantize_colors(frame: np.ndarray, k: int = 6) -> np.ndarray:
    """K-means color quantization -> flat cel-shaded regions."""
    data = frame.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # A single attempt keeps this fast enough for real-time.
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(frame.shape).astype(np.uint8)
    return quantized


def _ink_outline(frame: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Thick black outlines via edge detection. Returns a mask where
    white (255) = keep the color, black (0) = ink outline."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=9, C=2,
    )
    kernel = np.ones((thickness, thickness), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges


class NarutoStyle(StyleRenderer):
    def __init__(self, k: int = 6, outline_thickness: int = 2):
        self.k = k
        self.outline_thickness = outline_thickness

    def render(self, frame: np.ndarray, control: float = 0.5) -> np.ndarray:
        # Bilateral filter smooths while preserving edges (cel look).
        smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        quantized = _quantize_colors(smooth, k=self.k)
        edges = _ink_outline(frame, thickness=self.outline_thickness)
        # Composite outlines over the quantized color.
        result = cv2.bitwise_and(quantized, quantized, mask=edges)
        # Warm the whole frame slightly toward orange.
        warm = np.full_like(result, _WARM_BIAS)
        result = cv2.addWeighted(result, 0.85, warm, 0.15, 0)
        return result
