"""Tests for the gesture debouncer."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gesture.detector import Show  # noqa: E402
from gesture.smoothing import GestureDebouncer  # noqa: E402


class TestGestureDebouncer(unittest.TestCase):
    def test_starts_none(self):
        self.assertEqual(GestureDebouncer(3).current, Show.NONE)

    def test_requires_consecutive_frames(self):
        d = GestureDebouncer(confirm_frames=3)
        self.assertEqual(d.update(Show.NARUTO), Show.NONE)
        self.assertEqual(d.update(Show.NARUTO), Show.NONE)
        # Third unanimous frame confirms the switch.
        self.assertEqual(d.update(Show.NARUTO), Show.NARUTO)

    def test_single_frame_noise_does_not_flicker(self):
        d = GestureDebouncer(confirm_frames=3)
        for _ in range(3):
            d.update(Show.NARUTO)
        # A single stray frame must not switch the active show.
        self.assertEqual(d.update(Show.ATLA), Show.NARUTO)
        self.assertEqual(d.update(Show.NARUTO), Show.NARUTO)

    def test_switch_after_confirmed(self):
        d = GestureDebouncer(confirm_frames=3)
        for _ in range(3):
            d.update(Show.NARUTO)
        self.assertEqual(d.update(Show.ATLA), Show.NARUTO)
        self.assertEqual(d.update(Show.ATLA), Show.NARUTO)
        self.assertEqual(d.update(Show.ATLA), Show.ATLA)

    def test_confirm_frames_one_switches_immediately(self):
        d = GestureDebouncer(confirm_frames=1)
        self.assertEqual(d.update(Show.JJK), Show.JJK)


if __name__ == "__main__":
    unittest.main()
