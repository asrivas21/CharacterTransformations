"""ATLA style: watercolor ink-wash via a GLSL fragment shader (ModernGL).

Falls back to a CPU approximation when a standalone GL context is unavailable
so the rest of the pipeline keeps running."""
from __future__ import annotations

import os

import cv2
import numpy as np

from styles.base import StyleRenderer

_SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shaders")
_WATERCOLOR_PATH = os.path.join(_SHADER_DIR, "watercolor.glsl")

_VERT = """
#version 330
in vec2 in_pos; out vec2 uv;
void main() { uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos,0,1); }
"""


class ATLAStyle(StyleRenderer):
    def __init__(self):
        self.ctx = None
        self._fbos = {}  # (w, h) -> framebuffer, reused across frames
        try:
            import moderngl
            self.moderngl = moderngl
            self.ctx = moderngl.create_standalone_context()
            with open(_WATERCOLOR_PATH) as f:
                frag = f.read()
            self.prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=frag)
            quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
            self.vbo = self.ctx.buffer(quad.tobytes())
            self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_pos")
        except Exception as exc:  # pragma: no cover - depends on GL availability
            print(f"[ATLAStyle] GPU shader unavailable ({exc}); using CPU fallback.")
            self.ctx = None

    def _fbo_for(self, size):
        fbo = self._fbos.get(size)
        if fbo is None:
            fbo = self.ctx.simple_framebuffer(size)
            self._fbos[size] = fbo
        return fbo

    def render(self, frame: np.ndarray, control: float = 0.5) -> np.ndarray:
        if self.ctx is None:
            return self._render_cpu(frame)

        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
        tex = self.ctx.texture((w, h), 3, rgb.tobytes())
        tex.use(0)
        fbo = self._fbo_for((w, h))
        fbo.use()
        self.prog["tex"] = 0
        self.vao.render(self.moderngl.TRIANGLE_STRIP)
        data = fbo.read(components=3)
        out = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        tex.release()
        return np.ascontiguousarray(out[:, :, ::-1])  # RGB -> BGR

    def _render_cpu(self, frame: np.ndarray) -> np.ndarray:
        """OpenCV approximation of the watercolor wash."""
        smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=60, sigmaSpace=60)
        # Gentle posterize into flat wash regions.
        posterized = (np.floor(smooth.astype(np.float32) / 255.0 * 6.0)
                      / 6.0 * 255.0)
        # Warm desaturation.
        lum = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lum = lum[:, :, None]
        col = lum + (posterized - lum) * 0.65
        col *= np.array([0.9, 1.0, 1.08], dtype=np.float32)  # warm bias (BGR)
        # Paper grain.
        grain = np.random.rand(*frame.shape[:2], 1).astype(np.float32) * 15.0
        col += grain - 7.5
        return np.clip(col, 0, 255).astype(np.uint8)

    def close(self):
        if self.ctx is not None:
            self.ctx.release()
            self.ctx = None
