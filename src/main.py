"""Main capture loop. Ties multi-sheet gesture detection -> styles -> warp
composite together for the "stretched sheet" interaction."""
from __future__ import annotations

import time

import cv2
import numpy as np

from landmarks import LandmarkExtractor
from gesture.detector import detect_sheets, extended_labels, SHEET_PAIRS
from features.sheet import (
    composite_sheet, _torn_edge_alpha, _quad_is_degenerate)
from features import anime_features
from hud import draw_hud
from styles.risograph import RisographStyle
from styles.greenscreen import GreenScreenStyle

# Stable string keys shared with gesture.detector.SHEET_PAIRS.
# The "greenwhite" sheet segments the person out as solid white on a solid
# green background. The "cyano" sheet is an anime tracker (Sharingan eyes on
# the camera feed), handled specially below.
STYLES = {
    "riso": RisographStyle(),
    "greenwhite": GreenScreenStyle(),
}

_EMA_ALPHA = 0.5        # corner smoothing (higher => snappier)
_WORK_W, _WORK_H = 960, 540   # working resolution (camera may ignore requests)


def main():
    cap = cv2.VideoCapture(0)
    # Lower capture resolution keeps MediaPipe Holistic responsive on CPU.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _WORK_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _WORK_H)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    # The anime (cyano) sheet traces the actual iris, so iris refinement must
    # be on (468-477). Keep the fastest model to retain most of the speed win.
    extractor = LandmarkExtractor(model_complexity=0,
                                  refine_face_landmarks=True)

    # EMA-smoothed corners per active style key (drop stale keys each frame).
    smoothed: dict[str, np.ndarray] = {}

    prev_t = time.time()
    fps = 0.0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        # The camera may ignore the requested size; enforce the working res so
        # the pipeline cost stays predictable.
        if frame.shape[1] != _WORK_W or frame.shape[0] != _WORK_H:
            frame = cv2.resize(frame, (_WORK_W, _WORK_H))
        h, w = frame.shape[:2]

        lm = extractor.process(frame)
        sheets = detect_sheets(lm.left_hand, lm.right_hand)

        output = frame.copy()
        active_keys = [s.style_key for s in sheets]

        # Drop smoothing state for styles inactive this frame (no ghosts).
        for key in list(smoothed.keys()):
            if key not in active_keys:
                del smoothed[key]

        # Painter order = SHEET_PAIRS key order (later sheets draw on top).
        by_key = {s.style_key: s for s in sheets}
        for key in SHEET_PAIRS:
            sheet = by_key.get(key)
            if sheet is None:
                continue

            # EMA corner smoothing keyed by style_key.
            new_corners = sheet.corners.astype(np.float32)
            prev = smoothed.get(key)
            if prev is None or prev.shape != new_corners.shape:
                quad = new_corners
            else:
                quad = _EMA_ALPHA * new_corners + (1.0 - _EMA_ALPHA) * prev
            smoothed[key] = quad

            if _quad_is_degenerate(quad):
                continue

            # Style the actual pixels inside this section, in place (no warp, no
            # zoom): the region between the finger pair is "highlighted" in the
            # style. The quad (with its torn edge) is only used as the mask, so
            # each section changes just what sits between its own fingers.
            bx, by, bw, bh = cv2.boundingRect(quad.astype(np.int32))
            x0, y0 = max(0, bx), max(0, by)
            x1, y1 = min(w, bx + bw), min(h, by + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            rw, rh = x1 - x0, y1 - y0
            try:
                region = frame[y0:y1, x0:x1]
                if key == "cyano":
                    # Anime tracker: keep the camera background, overlay
                    # Sharingan eyes on the face inside this section.
                    styled = anime_features.apply(region.copy(), lm.face,
                                                  offset=(x0, y0))
                else:
                    styled = STYLES[key].render(region, sheet.control)
                local_quad = quad - np.array([x0, y0], np.float32)
                alpha = _torn_edge_alpha(local_quad, (rh, rw))
                composite_sheet(output[y0:y1, x0:x1], styled, alpha)
            except cv2.error:
                # Defensive fallback: skip this sheet for the frame.
                continue

        # FPS (exponential moving average).
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        draw_hud(output, frame_index, fps, active_keys)
        # Debug readout: extended fingers per hand.
        dbg = (f"L:{extended_labels(lm.left_hand)}"
               f"  R:{extended_labels(lm.right_hand)}")
        cv2.putText(output, dbg, (12, 44), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Anime Transform", output)
        frame_index += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
