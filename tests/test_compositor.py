"""Tests for the compositor math and asset loading."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2  # noqa: E402

from features.compositor import (  # noqa: E402
    overlay_rgba, face_angle, face_width, load_asset,
)
from features.sheet import (  # noqa: E402
    warp_styled_into_quad, _torn_edge_alpha, _quad_is_degenerate,
)
from features import anime_features  # noqa: E402
from hud import _timecode  # noqa: E402


def _face_with(points):
    face = np.zeros((478, 3), dtype=np.float32)
    for idx, (x, y) in points.items():
        face[idx] = [x, y, 0]
    return face


class TestFaceGeometry(unittest.TestCase):
    def test_face_angle_level(self):
        face = _face_with({33: (0, 0), 263: (10, 0)})
        self.assertAlmostEqual(face_angle(face), 0.0, places=5)

    def test_face_angle_rolled(self):
        # Right eye below+right of left eye -> +45deg screen roll -> -45 return.
        face = _face_with({33: (0, 0), 263: (10, 10)})
        self.assertAlmostEqual(face_angle(face), -45.0, places=5)

    def test_face_width(self):
        face = _face_with({234: (0, 0), 454: (100, 0)})
        self.assertAlmostEqual(face_width(face), 100.0, places=5)


class TestOverlayRgba(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((100, 100, 3), dtype=np.uint8)

    def _asset(self, rgb, alpha):
        a = np.zeros((10, 10, 4), dtype=np.uint8)
        a[:, :, :3] = rgb
        a[:, :, 3] = alpha
        return a

    def test_none_asset_is_identity(self):
        out = overlay_rgba(self.base, None, (50, 50), 1.0, 0.0)
        self.assertTrue(np.array_equal(out, self.base))

    def test_opaque_paints_center(self):
        overlay_rgba(self.base, self._asset(200, 255), (50, 50), 1.0, 0.0)
        np.testing.assert_array_equal(self.base[50, 50], [200, 200, 200])
        # Corner far from the asset stays untouched.
        np.testing.assert_array_equal(self.base[0, 0], [0, 0, 0])

    def test_alpha_blends(self):
        overlay_rgba(self.base, self._asset(200, 128), (50, 50), 1.0, 0.0)
        # 128/255 * 200 ~= 100 blended over black.
        val = int(self.base[50, 50, 0])
        self.assertTrue(99 <= val <= 101, f"expected ~100, got {val}")

    def test_out_of_bounds_no_change(self):
        overlay_rgba(self.base, self._asset(255, 255), (500, 500), 1.0, 0.0)
        self.assertTrue(np.array_equal(self.base, np.zeros_like(self.base)))

    def test_zero_scale_no_change(self):
        overlay_rgba(self.base, self._asset(255, 255), (50, 50), 0.0, 0.0)
        self.assertTrue(np.array_equal(self.base, np.zeros_like(self.base)))


class TestLoadAsset(unittest.TestCase):
    def test_missing_returns_none(self):
        self.assertIsNone(load_asset("does/not/exist.png"))

    def test_existing_is_rgba(self):
        asset = load_asset("naruto/headband.png")
        self.assertIsNotNone(asset)
        self.assertEqual(asset.shape[2], 4)


def _rect_quad():
    # Order matches rectangle_corners: [left_a, left_b, right_b, right_a].
    return np.array([[10, 10], [10, 90], [90, 90], [90, 10]], np.float32)


class TestQuadDegenerate(unittest.TestCase):
    def test_normal_quad_not_degenerate(self):
        self.assertFalse(_quad_is_degenerate(_rect_quad()))

    def test_collinear_is_degenerate(self):
        quad = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], np.float32)
        self.assertTrue(_quad_is_degenerate(quad))

    def test_clump_eps_quad_not_degenerate(self):
        # A clump wedge: two nearly-coincident left corners, wide right corners.
        quad = np.array([[50, 46], [50, 54], [400, 100], [400, 300]], np.float32)
        self.assertFalse(_quad_is_degenerate(quad))


class TestWarpStyledIntoQuad(unittest.TestCase):
    def test_shapes_and_mask_area(self):
        shape = (120, 160, 3)
        styled = np.full(shape, 200, np.uint8)
        quad = _rect_quad()
        warped, mask = warp_styled_into_quad(styled, quad, shape)
        self.assertEqual(warped.shape, shape)
        self.assertEqual(mask.shape, shape[:2])
        # Mask nonzero area ~= quad area (80 x 80 = 6400).
        area = float(np.count_nonzero(mask))
        self.assertLess(abs(area - 6400.0) / 6400.0, 0.05)
        # Mask stays within frame bounds.
        ys, xs = np.nonzero(mask)
        self.assertTrue(xs.min() >= 0 and xs.max() < shape[1])
        self.assertTrue(ys.min() >= 0 and ys.max() < shape[0])


class TestTornEdgeAlpha(unittest.TestCase):
    def test_shape_binary_and_reduced_area(self):
        shape = (120, 160, 3)
        quad = _rect_quad()
        alpha = _torn_edge_alpha(quad, shape)
        self.assertEqual(alpha.shape, shape[:2])
        self.assertTrue(np.all(np.isin(np.unique(alpha), [0, 255])))
        # Torn area < plain fillConvexPoly of the same quad.
        plain = np.zeros(shape[:2], np.uint8)
        cv2.fillConvexPoly(plain, quad.astype(np.int32), 255)
        self.assertLess(np.count_nonzero(alpha), np.count_nonzero(plain))


class TestAnimeFeatures(unittest.TestCase):
    # Eye centers in frame space; iris radius ~8px; eyelid box ~28x20.
    _EYES = {"left": (50, 40), "right": (110, 40)}
    _R = 8
    _APER_HW, _APER_HH = 14, 10

    def _eye_face(self, offset=(0, 0), aperture=True):
        ox, oy = offset
        pts: dict[int, tuple[float, float]] = {}
        centers = [(468, (469, 470, 471, 472), self._EYES["left"],
                    anime_features._LEFT_EYE_RING),
                   (473, (474, 475, 476, 477), self._EYES["right"],
                    anime_features._RIGHT_EYE_RING)]
        for cidx, ring, (cx, cy), eye_ring in centers:
            fx, fy = cx + ox, cy + oy
            pts[cidx] = (fx, fy)
            # Iris ring: 4 points at +/-R around the center (drives radius).
            r = self._R
            pts[ring[0]] = (fx + r, fy)
            pts[ring[1]] = (fx - r, fy)
            pts[ring[2]] = (fx, fy + r)
            pts[ring[3]] = (fx, fy - r)
            # Eyelid ring: a box around the eye (aperture). If aperture is
            # False, collapse it to a point so nothing shows (blink).
            hw = self._APER_HW if aperture else 0
            hh = self._APER_HH if aperture else 0
            for i, ridx in enumerate(eye_ring):
                sx = hw if (i % 2 == 0) else -hw
                sy = hh if (i % 4 < 2) else -hh
                pts[ridx] = (fx + sx, fy + sy)
        return _face_with(pts)

    def test_none_face_is_identity(self):
        region = np.zeros((80, 160, 3), np.uint8)
        out = anime_features.apply(region, None)
        self.assertTrue(np.array_equal(out, region))

    @unittest.skipIf(anime_features._sharingan is None,
                     "sharingan asset missing")
    def test_paints_over_iris(self):
        region = np.zeros((80, 160, 3), np.uint8)
        anime_features.apply(region, self._eye_face())
        # A point just off each iris center lands on bright Sharingan (non-zero).
        for cx, cy in self._EYES.values():
            self.assertGreater(int(region[cy, cx + 4].sum()), 0)

    @unittest.skipIf(anime_features._sharingan is None,
                     "sharingan asset missing")
    def test_clipped_outside_eye_opening(self):
        region = np.zeros((80, 160, 3), np.uint8)
        anime_features.apply(region, self._eye_face())
        # Well outside the eyelid box (aperture) the Sharingan is clipped away.
        for cx, cy in self._EYES.values():
            self.assertEqual(int(region[cy, cx + self._APER_HW + 6].sum()), 0)

    @unittest.skipIf(anime_features._sharingan is None,
                     "sharingan asset missing")
    def test_blink_shrinks_overlay(self):
        # A collapsed eyelid aperture (blink) shows far less Sharingan than an
        # open eye, since the overlay is clipped to the eye opening.
        open_r = np.zeros((80, 160, 3), np.uint8)
        anime_features.apply(open_r, self._eye_face(aperture=True))
        blink_r = np.zeros((80, 160, 3), np.uint8)
        anime_features.apply(blink_r, self._eye_face(aperture=False))
        open_px = int(np.count_nonzero(open_r.sum(axis=2)))
        blink_px = int(np.count_nonzero(blink_r.sum(axis=2)))
        self.assertLess(blink_px, open_px * 0.1)

    @unittest.skipIf(anime_features._sharingan is None,
                     "sharingan asset missing")
    def test_offset_shifts_placement(self):
        region = np.zeros((80, 160, 3), np.uint8)
        # Landmarks shifted +30/+10; offset cancels it so the overlay still
        # lands at the same region-local iris positions.
        anime_features.apply(region, self._eye_face(offset=(30, 10)),
                             offset=(30, 10))
        for cx, cy in self._EYES.values():
            self.assertGreater(int(region[cy, cx + 4].sum()), 0)


class TestTimecode(unittest.TestCase):
    def test_values(self):
        self.assertEqual(_timecode(0), "00:00:00:00")
        self.assertEqual(_timecode(30), "00:00:01:00")
        self.assertEqual(_timecode(31), "00:00:01:01")


if __name__ == "__main__":
    unittest.main()
