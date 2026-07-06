"""Reference-style HUD: a bottom timecode/scrubber bar plus an optional
top-left monospace project label."""
from __future__ import annotations

import cv2
import numpy as np

_FF = 30  # assumed frames-per-second for the ticking timecode


def _timecode(frame_index: int, fps_assumed: int = _FF) -> str:
    """Render `frame_index` as HH:MM:SS:FF (frames wrap every `fps_assumed`)."""
    ff = frame_index % fps_assumed
    s = frame_index // fps_assumed
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}:{ff:02d}"


def draw_hud(frame: np.ndarray, frame_index: int, fps: float,
             active_style_names) -> np.ndarray:
    """Draw the HUD onto `frame` (mutates & returns it)."""
    h, w = frame.shape[:2]

    # Bottom dark strip.
    bar_h = 28
    y0 = h - bar_h
    strip = frame[y0:h].astype(np.float32) * 0.35
    frame[y0:h] = strip.astype(np.uint8)

    # Timecode at the left.
    tc = _timecode(frame_index)
    cv2.putText(frame, tc, (12, h - 9), cv2.FONT_HERSHEY_PLAIN, 1.2,
                (230, 230, 230), 1, cv2.LINE_AA)

    # Active style names + FPS at the right.
    names = " ".join(str(n).upper() for n in active_style_names) or "-"
    right = f"{names}   {fps:4.1f} FPS"
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)
    cv2.putText(frame, right, (w - tw - 12, h - 9), cv2.FONT_HERSHEY_PLAIN, 1.2,
                (230, 230, 230), 1, cv2.LINE_AA)

    # Thin scrubber line with a moving tick driven by frame_index.
    line_y = y0 + 6
    cv2.line(frame, (110, line_y), (w - 220, line_y), (120, 120, 120), 1,
             cv2.LINE_AA)
    span = max(1, (w - 220) - 110)
    tick_x = 110 + (frame_index * 3) % span
    cv2.line(frame, (tick_x, line_y - 4), (tick_x, line_y + 4),
             (255, 255, 255), 1, cv2.LINE_AA)

    # Optional top-left monospace project label.
    label = f"/project [{w},{h}]"
    cv2.putText(frame, label, (12, 22), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (200, 200, 200), 1, cv2.LINE_AA)
    return frame
