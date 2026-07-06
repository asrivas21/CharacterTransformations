"""Selfie segmentation -> person vs background mask."""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

mp_selfie = mp.solutions.selfie_segmentation


class PersonSegmenter:
    def __init__(self, model_selection: int = 1):
        self.seg = mp_selfie.SelfieSegmentation(model_selection=model_selection)

    def mask(self, frame: np.ndarray) -> np.ndarray | None:
        """Return an (H, W) float mask in [0, 1], or None if unavailable."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.seg.process(rgb)
        return result.segmentation_mask

    def close(self):
        self.seg.close()
