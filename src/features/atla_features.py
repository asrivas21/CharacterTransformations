"""ATLA: glowing white Avatar State eyes + blue arrow tattoos on forehead & hands."""
from __future__ import annotations

import numpy as np

from features.compositor import load_asset, overlay_rgba, face_angle, face_width

_arrow_head = load_asset("atla/arrow_head.png")
_avatar_glow = load_asset("atla/avatar_glow.png")
_arrow_hand = load_asset("atla/arrow_hand.png")

_EYES = [(468, 33, 133), (473, 362, 263)]


def apply(frame: np.ndarray, face: np.ndarray | None,
          left_hand: np.ndarray | None = None,
          right_hand: np.ndarray | None = None) -> np.ndarray:
    if face is not None:
        fw = face_width(face)
        angle = face_angle(face)

        # Forehead arrow: from brow center (9) up over the forehead (10).
        brow = face[9, :2]
        top = face[10, :2]
        arrow_center = tuple(((brow + top) / 2).astype(int))
        if _arrow_head is not None:
            arrow_scale = fw / _arrow_head.shape[1] * 0.5
            frame = overlay_rgba(frame, _arrow_head, arrow_center, arrow_scale, angle)

        # Avatar State glow over both eyes.
        if _avatar_glow is not None and face.shape[0] > 473:
            for iris_idx, ca, cb in _EYES:
                eye_center = tuple(face[iris_idx, :2].astype(int))
                eye_w = np.linalg.norm(face[ca, :2] - face[cb, :2])
                glow_scale = eye_w / _avatar_glow.shape[1] * 1.6
                frame = overlay_rgba(frame, _avatar_glow, eye_center, glow_scale, angle)

    # Hand arrows on the back of each hand (9 = middle-finger MCP ~ hand center).
    if _arrow_hand is not None:
        for hand in (left_hand, right_hand):
            if hand is None:
                continue
            hand_center = tuple(hand[9, :2].astype(int))
            hand_w = np.linalg.norm(hand[5, :2] - hand[17, :2])
            hand_scale = hand_w / _arrow_hand.shape[1] * 1.2
            vec = hand[12, :2] - hand[0, :2]
            hand_angle = -np.degrees(np.arctan2(vec[1], vec[0])) - 90
            frame = overlay_rgba(frame, _arrow_hand, hand_center,
                                 hand_scale, hand_angle)
    return frame
