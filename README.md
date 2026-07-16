# PyBitEngine ⚡

[![Documentation](https://img.shields.io/badge/docs-online-brightgreen)](https://hakerbituf.github.io/PyBitEngine/)
[![License](https://img.shields.io/badge/license-MPL--2.0-green)](https://github.com/HakerBituf/PyBitEngine/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A GPU-accelerated 2D game engine for Python, built on SDL2, ModernGL, NumPy and Numba.

PyBitEngine helps developers draw thousands of 2D objects efficiently while keeping the API simple. It uses instanced GPU rendering, texture atlases, and Numba kernels to reduce runtime overhead.

---

## 📦 Version: 0.1.2

This is a stability and correctness release. It focuses on fixing rendering-order
bugs, camera-shake artifacts, geometric collision edge cases and a few packaging
rough edges reported against 0.1.1. No public API was removed or renamed:
upgrading from 0.1.1 requires no code changes.

### What is new in 0.1.2

- **Painter's algorithm fixed for textures + primitives.** `DrawTexture` now
  participates in the same `_use_batch` batching used by every other primitive.
  Mixing `DrawTexture`, `DrawRect`, `DrawRoundedRect`, etc. now produces draw
  calls in the exact order of the `Draw*` calls — no more textures rendered
  after rectangles regardless of call order.
- **Screen-shake bias removed** on both `CameraGPU` and `CameraCPU`. The shake
  offset is now a zero-mean oscillation (`sin/cos * factor`) instead of the
  previous `2*sin - 1` formula that produced a net drift toward the
  bottom-left.
- **Polygon vs Ellipse collision** now also tests polygon edges against the
  ellipse boundary. Cases where an ellipse crosses a polygon edge without
  containing any vertex and without having its center inside are correctly
  reported as colliding (previously false negatives).
- **PE_PAKER** — `pack()` injects the absolute directory of the source script
  into `sys.path` of the generated `setup.py`, instead of `Path.cwd()` at
  generation time. Packaging now works reliably when `pack()` is invoked from
  a directory other than the project root.
- Minor documentation and comment cleanups across `PE_DRAW`, `PE_CAMERA` and
  `PE_PAKER` explaining the invariants behind the fixes above.

---

## ✨ Highlights

- Instanced GPU rendering for rectangles, rounded rectangles, lines, triangles, rounded triangles, ellipses and sprites.
- Outline variants for every filled primitive, with a shared AA formula for consistent borders.
- Texture atlas packing with MaxRects allocation.
- 2D camera support with CPU and GPU-backed variants.
- Font rendering for TTF/OTF files with caching.
- SDL2 input handling for keyboard, mouse, drag and wheel events.
- Geometry helpers for points, rectangles, triangles, ellipses, circles and OBB collision checks.

---

## 🚀 Installation

```bash
pip install pybitengine
```

## 🧪 Quick usage

```python
from PyBitEngine import WINDOW

window = WINDOW(title="PyBitEngine", geometry=("center", "center", 800, 600))
window.Loop()
```

## 🎨 Correct layering (fixed in 0.1.2)

```python
# Draw calls now render strictly in the order they are issued,
# even when mixing textures and primitives.
draw.DrawTexture("background", 0, 0, 800, 600)          # layer 0
draw.DrawRect(100, 100, 50, 50, color=(255, 0, 0))      # layer 1 (above bg)
draw.DrawTexture("player", 200, 200, 32, 32)            # layer 2 (above rect)
```

## 🟦 Rounded shapes (from 0.1.1)

```python
# Immediate
draw.DrawRoundedRect(100, 100, 200, 80, radius=16, color=(255, 100, 50))
draw.DrawRoundedRectOutline(100, 100, 200, 80, radius=16, thickness=3,
                            color=(255, 255, 255))
draw.DrawRoundedTriangle(100, 50, 200, 200, 50, 200, radius=12,
                         color=(0, 255, 150))

# Batch (GPU-instanced, one draw call per chunk)
import numpy as np
pos    = np.array([[10, 10], [120, 10], [230, 10]], dtype="f4")
size   = np.array([[80, 40], [80, 40],  [80, 40]],  dtype="f4")
radius = np.array([6, 12, 20], dtype="f4")           # per-instance radius
draw.DrawRoundedRectsBatch(pos, size, radius, colors=(255, 0, 128))
```

## 📦 Packaging a game

```python
from PyBitEngine import pack

pack("main.py", name="MyGame", output_dir="dist")
```

## 📝 Notes

PyBitEngine is still evolving, but the core rendering and packaging APIs are
increasingly stable and suitable for real projects. The 0.1.2 release is a
maintenance drop on top of 0.1.1: same features, more correct behavior when
mixing textures with primitives, using camera shake, or packaging from a
non-standard working directory.
