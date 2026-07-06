"""Tests for the rectangle gesture detector."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gesture.detector import (  # noqa: E402
    Show, detect_show, rectangle_corners, _extended_fingers,
    detect_sheets, Sheet, SHEET_PAIRS, _is_clumped, _pair_open,
    _fingertip_spread, _pair_control, _CLUMP_EPS,
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
    INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP,
)

# (mcp index, pip index, tip index, x column) for the four non-thumb fingers.
_FINGERS = [
    (5, INDEX_PIP, INDEX_TIP, 0),
    (9, MIDDLE_PIP, MIDDLE_TIP, 10),
    (13, RING_PIP, RING_TIP, 20),
    (17, PINKY_PIP, PINKY_TIP, 30),
]


def make_hand(extended):
    """Build a plausible (21, 3) hand where the given fingertips are extended.

    Fingers point "up" (negative y) from a wrist at y=0: MCP at y=-30, PIP at
    y=-50. An extended tip reaches well past the PIP (y=-100); a curled tip folds
    back toward the palm (y=-30, i.e. closer to the wrist than its PIP), matching
    the detector's PIP-relative extension test. The thumb tip sits far from the
    index knuckle when extended and near the palm when curled."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[WRIST] = [15, 0, 0]
    for mcp, pip, tip, x in _FINGERS:
        hand[mcp] = [x, -30, 0]
        hand[pip] = [x, -50, 0]
        hand[tip] = [x, -100 if tip in extended else -30, 0]
    # Thumb: index MCP is at (0, -30), pinky MCP at (30, -30) => palm width 30.
    hand[THUMB_TIP] = [-40, -30, 0] if THUMB_TIP in extended else [8, -25, 0]
    return hand


def make_spread_hand():
    """A fully open hand: all tracked fingertips extended and spread wide."""
    return make_hand({THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP})


def make_clumped_hand():
    """A bunched hand: normal palm width but all fingertips clustered together,
    so the fingertip spread falls below the clump threshold."""
    hand = make_hand(set())
    # Keep MCPs (palm width 30) but pull all tracked tips into a tight cluster.
    hand[THUMB_TIP] = [14, -58, 0]
    hand[INDEX_TIP] = [15, -60, 0]
    hand[MIDDLE_TIP] = [16, -60, 0]
    hand[PINKY_TIP] = [17, -59, 0]
    return hand


class TestExtendedFingers(unittest.TestCase):
    def test_all_extended(self):
        hand = make_hand({THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP})
        self.assertEqual(
            _extended_fingers(hand),
            {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP})

    def test_none_extended(self):
        self.assertEqual(_extended_fingers(make_hand(set())), set())

    def test_subset_extended(self):
        hand = make_hand({INDEX_TIP, MIDDLE_TIP})
        self.assertEqual(_extended_fingers(hand), {INDEX_TIP, MIDDLE_TIP})


class TestDetectShow(unittest.TestCase):
    def test_naruto(self):
        h = make_hand({THUMB_TIP, INDEX_TIP})
        self.assertEqual(detect_show(h, h), Show.NARUTO)

    def test_atla(self):
        h = make_hand({INDEX_TIP, MIDDLE_TIP})
        self.assertEqual(detect_show(h, h), Show.ATLA)

    def test_jjk(self):
        h = make_hand({MIDDLE_TIP, PINKY_TIP})
        self.assertEqual(detect_show(h, h), Show.JJK)

    def test_missing_hand_is_none(self):
        h = make_hand({THUMB_TIP, INDEX_TIP})
        self.assertEqual(detect_show(None, h), Show.NONE)
        self.assertEqual(detect_show(h, None), Show.NONE)

    def test_mismatched_hands_is_none(self):
        left = make_hand({THUMB_TIP, INDEX_TIP})
        right = make_hand({INDEX_TIP, MIDDLE_TIP})
        self.assertEqual(detect_show(left, right), Show.NONE)

    def test_no_pair_is_none(self):
        h = make_hand({INDEX_TIP})
        self.assertEqual(detect_show(h, h), Show.NONE)


class TestRectangleCorners(unittest.TestCase):
    def test_none_show_returns_none(self):
        h = make_hand({THUMB_TIP, INDEX_TIP})
        self.assertIsNone(rectangle_corners(h, h, Show.NONE))

    def test_corner_order(self):
        left = make_hand({THUMB_TIP, INDEX_TIP})
        right = make_hand({THUMB_TIP, INDEX_TIP})
        corners = rectangle_corners(left, right, Show.NARUTO)
        self.assertEqual(corners.shape, (4, 2))
        np.testing.assert_allclose(corners[0], left[THUMB_TIP, :2])
        np.testing.assert_allclose(corners[1], left[INDEX_TIP, :2])
        np.testing.assert_allclose(corners[2], right[INDEX_TIP, :2])
        np.testing.assert_allclose(corners[3], right[THUMB_TIP, :2])


class TestSheetHelpers(unittest.TestCase):
    def test_is_clumped(self):
        self.assertTrue(_is_clumped(make_clumped_hand()))
        self.assertFalse(_is_clumped(make_spread_hand()))

    def test_fingertip_spread_ordering(self):
        self.assertGreater(_fingertip_spread(make_spread_hand()),
                           _fingertip_spread(make_clumped_hand()))

    def test_pair_open_true(self):
        h = make_hand({THUMB_TIP, INDEX_TIP})
        self.assertTrue(_pair_open(h, THUMB_TIP, INDEX_TIP))

    def test_pair_open_false_curled_tip(self):
        h = make_hand({INDEX_TIP})  # thumb curled
        self.assertFalse(_pair_open(h, THUMB_TIP, INDEX_TIP))

    def test_pair_open_false_small_gap(self):
        h = make_hand({INDEX_TIP, MIDDLE_TIP})
        h[MIDDLE_TIP] = [1, -100, 0]  # both extended but tips nearly touching
        self.assertFalse(_pair_open(h, INDEX_TIP, MIDDLE_TIP))

    def test_pair_control_in_range(self):
        h = make_spread_hand()
        c = _pair_control(h, h, THUMB_TIP, INDEX_TIP)
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)


class TestDetectSheets(unittest.TestCase):
    def test_missing_hand_returns_empty(self):
        h = make_spread_hand()
        self.assertEqual(detect_sheets(None, h), [])
        self.assertEqual(detect_sheets(h, None), [])

    def test_normal_single_pair(self):
        h = make_hand({THUMB_TIP, INDEX_TIP})
        sheets = detect_sheets(h, h)
        self.assertEqual(len(sheets), 1)
        s = sheets[0]
        self.assertIsInstance(s, Sheet)
        self.assertEqual(s.style_key, "riso")
        self.assertEqual(s.corners.shape, (4, 2))
        np.testing.assert_allclose(s.corners[0], h[THUMB_TIP, :2])
        np.testing.assert_allclose(s.corners[1], h[INDEX_TIP, :2])
        np.testing.assert_allclose(s.corners[2], h[INDEX_TIP, :2])
        np.testing.assert_allclose(s.corners[3], h[THUMB_TIP, :2])

    def test_normal_multiple_pairs(self):
        h = make_spread_hand()
        keys = {s.style_key for s in detect_sheets(h, h)}
        self.assertEqual(keys, set(SHEET_PAIRS.keys()))

    def test_clump_mode_fans_wedges(self):
        left = make_clumped_hand()
        right = make_spread_hand()
        sheets = detect_sheets(left, right)
        self.assertEqual(len(sheets), 3)
        for s in sheets:
            a, b = SHEET_PAIRS[s.style_key]
            # Left corners differ only by +/- _CLUMP_EPS on the y axis.
            self.assertAlmostEqual(s.corners[0][0], s.corners[1][0], places=4)
            self.assertAlmostEqual(s.corners[1][1] - s.corners[0][1],
                                   2 * _CLUMP_EPS, places=4)
            # Right corners equal the right-hand fingertips.
            np.testing.assert_allclose(s.corners[2], right[b, :2])
            np.testing.assert_allclose(s.corners[3], right[a, :2])


if __name__ == "__main__":
    unittest.main()
