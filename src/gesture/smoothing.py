"""Debounce gesture detection. Switch instantly once a gesture is confirmed
across a few frames, but never flicker on single-frame noise."""
from collections import deque

from gesture.detector import Show


class GestureDebouncer:
    def __init__(self, confirm_frames: int = 3):
        self.confirm_frames = confirm_frames
        self.history = deque(maxlen=confirm_frames)
        self.current = Show.NONE

    def update(self, detected: Show) -> Show:
        self.history.append(detected)
        # Switch only when the last N frames unanimously agree
        if len(self.history) == self.confirm_frames and \
           len(set(self.history)) == 1 and self.history[0] != self.current:
            self.current = self.history[0]
        return self.current
