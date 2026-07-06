"""Generate simple stylized placeholder RGBA assets so the pipeline runs.
Replace these with real traced/generated art later (see INSTRUCTIONS Phase 5)."""
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")


def _save(rel, bgra):
    path = os.path.join(ASSETS, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bgra)
    print("wrote", path)


def _blank(h, w):
    return np.zeros((h, w, 4), dtype=np.uint8)


def radial_glow(size, color_bgr, softness=1.0):
    img = _blank(size, size)
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c
    a = np.clip(1.0 - d, 0, 1) ** (2.0 * softness)
    for i in range(3):
        img[:, :, i] = color_bgr[i]
    img[:, :, 3] = (a * 255).astype(np.uint8)
    return img


def iris(size, inner_bgr, outer_bgr):
    img = _blank(size, size)
    c = size // 2
    cv2.circle(img, (c, c), int(size * 0.46), (*outer_bgr, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(size * 0.30), (*inner_bgr, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(size * 0.12), (10, 10, 10, 255), -1, cv2.LINE_AA)
    # tiny highlight
    cv2.circle(img, (int(c * 0.8), int(c * 0.8)), max(2, size // 20),
               (255, 255, 255, 230), -1, cv2.LINE_AA)
    return img


def headband(w=420, h=130):
    img = _blank(h, w)
    band = (60, 40, 25)  # dark navy (BGR)
    cv2.rectangle(img, (0, int(h * 0.28)), (w, int(h * 0.72)), (*band, 255), -1)
    # metal plate
    pw, ph = int(w * 0.34), int(h * 0.6)
    x0, y0 = (w - pw) // 2, (h - ph) // 2
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (185, 180, 170, 255), -1)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (120, 115, 110, 255), 3)
    # leaf swirl glyph
    cx, cy = w // 2, h // 2
    cv2.circle(img, (cx, cy), int(ph * 0.28), (70, 70, 70, 255), 3, cv2.LINE_AA)
    cv2.line(img, (cx, cy - int(ph * 0.28)), (cx, cy + int(ph * 0.4)),
             (70, 70, 70, 255), 3, cv2.LINE_AA)
    return img


def metal_plate(w=170, h=110):
    img = _blank(h, w)
    cv2.rectangle(img, (4, 4), (w - 4, h - 4), (185, 180, 170, 255), -1)
    cv2.rectangle(img, (4, 4), (w - 4, h - 4), (120, 115, 110, 255), 3)
    return img


def arrow(w=140, h=230, color=(210, 120, 40)):
    """Upward-pointing arrow: triangle head + stem (ATLA style)."""
    img = _blank(h, w)
    cx = w // 2
    tri = np.array([[cx, int(h * 0.05)],
                    [int(w * 0.12), int(h * 0.42)],
                    [int(w * 0.88), int(h * 0.42)]], np.int32)
    cv2.fillPoly(img, [tri], (*color, 255), cv2.LINE_AA)
    cv2.rectangle(img, (int(w * 0.38), int(h * 0.42)),
                  (int(w * 0.62), int(h * 0.95)), (*color, 255), -1)
    return img


def blindfold(w=440, h=90):
    img = _blank(h, w)
    cv2.rectangle(img, (0, int(h * 0.2)), (w, int(h * 0.8)), (25, 25, 25, 255), -1)
    return img


def main():
    _save("naruto/headband.png", headband())
    _save("naruto/headband_metal.png", metal_plate())
    _save("naruto/sage_iris.png", iris(140, (30, 140, 240), (20, 90, 200)))
    _save("atla/arrow_head.png", arrow())
    _save("atla/arrow_hand.png", arrow(120, 170))
    _save("atla/avatar_glow.png", radial_glow(140, (255, 245, 235), softness=0.8))
    _save("jjk/six_eyes_overlay.png", iris(140, (255, 220, 90), (255, 150, 40)))
    _save("jjk/blindfold.png", blindfold())


if __name__ == "__main__":
    main()
