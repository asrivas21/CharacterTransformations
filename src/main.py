"""Main capture loop. Ties gesture detection -> style -> features together."""
from __future__ import annotations

import time

import cv2
import numpy as np

from landmarks import LandmarkExtractor
from gesture.detector import (
    detect_show, rectangle_corners, extended_labels, Show)
from gesture.smoothing import GestureDebouncer
from styles.risograph import RisographStyle
from styles.cyanotype import CyanotypeStyle
from styles.stippling import StipplingStyle


def _composite_in_rect(original: np.ndarray, styled: np.ndarray,
                       corners: np.ndarray) -> np.ndarray:
    """Show the styled frame only inside the finger-rectangle; keep the
    original webcam image everywhere else, with a thin boundary outline."""
    poly = corners.astype(np.int32)
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    out = original.copy()
    out[mask == 255] = styled[mask == 255]
    cv2.polylines(out, [poly], isClosed=True, color=(255, 255, 255),
                  thickness=2, lineType=cv2.LINE_AA)
    return out


def _style_control(show: Show, corners: np.ndarray, diag: float) -> float:
    """Map a live hand metric to a normalized [0, 1] control for the style.

    Corners are ordered [left_a, left_b, right_b, right_a] (the two finger tips
    on the left hand, then on the right). Finger spread is the tip-to-tip gap on
    a single hand; inter-hand distance is the gap between the two hands.
    """
    left_a, left_b, right_b, right_a = corners
    if show == Show.ATLA:  # Cyanotype: inter-hand distance -> exposure.
        left_c = (left_a + left_b) * 0.5
        right_c = (right_a + right_b) * 0.5
        metric = np.linalg.norm(right_c - left_c) / diag
        return float(np.clip(metric / 0.6, 0.0, 1.0))
    # Risograph / Stippling: finger spread -> dot size / density.
    spread = 0.5 * (np.linalg.norm(left_b - left_a)
                    + np.linalg.norm(right_b - right_a)) / diag
    return float(np.clip(spread / 0.25, 0.0, 1.0))


def main():
    cap = cv2.VideoCapture(0)
    # Lower capture resolution keeps MediaPipe Holistic responsive on CPU.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    extractor = LandmarkExtractor()
    debouncer = GestureDebouncer(confirm_frames=3)

    # Temporary art styles wired to the existing three gestures:
    #   thumb+index (NARUTO gesture) -> Risograph
    #   index+middle (ATLA gesture)  -> Cyanotype
    #   middle+pinky (JJK gesture)   -> Stippling
    styles = {
        Show.NARUTO: RisographStyle(),
        Show.ATLA: CyanotypeStyle(),
        Show.JJK: StipplingStyle(),
    }
    diag = float(np.hypot(w, h))

    prev_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror for natural interaction

        lm = extractor.process(frame)
        raw_show = detect_show(lm.left_hand, lm.right_hand)
        show = debouncer.update(raw_show)

        # The rectangle needs both hands present to define its four corners.
        corners = None
        if lm.left_hand is not None and lm.right_hand is not None:
            corners = rectangle_corners(lm.left_hand, lm.right_hand, show)

        control = 0.0
        if show == Show.NONE or corners is None:
            output = frame
        else:
            # A live hand metric drives each style's dynamic parameter.
            control = _style_control(show, corners, diag)
            # Style only the rectangle's bounding box (much cheaper than the
            # whole frame), then reveal just the polygon interior.
            bx, by, bw, bh = cv2.boundingRect(corners.astype(np.int32))
            x0, y0 = max(0, bx), max(0, by)
            x1, y1 = min(w, bx + bw), min(h, by + bh)
            styled = frame.copy()
            if x1 > x0 and y1 > y0:
                styled[y0:y1, x0:x1] = styles[show].render(
                    frame[y0:y1, x0:x1], control)
            output = _composite_in_rect(frame, styled, corners)

        # FPS (exponential moving average).
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(output, f"{show.value.upper()}  {fps:4.1f} fps", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        # Debug readout: extended fingers per hand + the raw (pre-debounce) show.
        dbg = (f"L:{extended_labels(lm.left_hand)}  R:{extended_labels(lm.right_hand)}"
               f"  raw:{raw_show.value}  ctrl:{control:4.2f}")
        cv2.putText(output, dbg, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
        cv2.imshow("Anime Transform", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
