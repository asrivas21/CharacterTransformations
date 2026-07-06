"""MediaPipe Holistic wrapper. Returns hands, face mesh, and pose landmarks
in a single normalized structure per frame."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

# With refine_face_landmarks=True the face mesh has 478 points (468 + 10 iris).
FACE_LANDMARK_COUNT = 478
HAND_LANDMARK_COUNT = 21
POSE_LANDMARK_COUNT = 33


@dataclass
class FrameLandmarks:
    left_hand: np.ndarray | None   # (21, 3) or None — pixel x, y, relative z
    right_hand: np.ndarray | None  # (21, 3) or None
    face: np.ndarray | None        # (478, 3) or None
    pose: np.ndarray | None        # (33, 4) or None — x, y, z, visibility
    frame_w: int
    frame_h: int


class LandmarkExtractor:
    def __init__(self, model_complexity: int = 1,
                 refine_face_landmarks: bool = True):
        # `model_complexity` (0=fast, 1=balanced, 2=accurate) and
        # `refine_face_landmarks` (iris landmarks) are the main speed/quality
        # knobs. Hands-only consumers can pass 0 / False for a big speed win.
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray) -> FrameLandmarks:
        h, w = frame_bgr.shape[:2]
        # MediaPipe requires a contiguous RGB array; cvtColor copies.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)

        def to_array(landmarks, n, dims=3):
            if landmarks is None:
                return None
            arr = np.zeros((n, dims), dtype=np.float32)
            for i, lm in enumerate(landmarks.landmark):
                if i >= n:
                    break
                arr[i, 0] = lm.x * w
                arr[i, 1] = lm.y * h
                arr[i, 2] = lm.z
                if dims == 4:
                    arr[i, 3] = lm.visibility
            return arr

        return FrameLandmarks(
            left_hand=to_array(results.left_hand_landmarks, HAND_LANDMARK_COUNT),
            right_hand=to_array(results.right_hand_landmarks, HAND_LANDMARK_COUNT),
            face=to_array(results.face_landmarks, FACE_LANDMARK_COUNT),
            pose=to_array(results.pose_landmarks, POSE_LANDMARK_COUNT, dims=4),
            frame_w=w,
            frame_h=h,
        )

    def close(self):
        self.holistic.close()
