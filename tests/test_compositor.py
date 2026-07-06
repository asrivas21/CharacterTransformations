"""Tests for the compositor math and asset loading."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from features.compositor import (  # noqa: E402
    overlay_rgba, face_angle, face_width, load_asset,
)


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


if __name__ == "__main__":
    unittest.main()
