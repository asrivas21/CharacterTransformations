"""StyleRenderer interface. Each show implements a full-frame art treatment."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class StyleRenderer(ABC):
    @abstractmethod
    def render(self, frame: np.ndarray, control: float = 0.5) -> np.ndarray:
        """Apply the art style to the frame. `control` is a normalized value in
        [0, 1] derived from a live hand metric (finger spread or inter-hand
        distance) that modulates a style-specific parameter (dot size, exposure,
        stipple density, ...)."""
        ...
