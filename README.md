# PyBitEngine ⚡

[![Documentation](https://img.shields.io/badge/docs-online-brightgreen)](https://hakerbituf.github.io/PyBitEngine/)
[![License](https://img.shields.io/badge/license-MPL--2.0-green)](https://github.com/HakerBituf/PyBitEngine/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A GPU-accelerated 2D game engine for Python, built on SDL2, ModernGL, NumPy and Numba.

PyBitEngine helps developers draw thousands of 2D objects efficiently while keeping the API simple. It uses instanced GPU rendering, texture atlases, and Numba kernels to reduce runtime overhead.

---

## 📦 Version: 0.1.1

This release expands the primitive set with rounded shapes and their fully-instanced batch counterparts, keeping the same SDF + AA pipeline used by every other primitive in the engine.

### What is new in 0.1.1
- New immediate primitives: `DrawRoundedRect`, `DrawRoundedRectOutline`, `DrawRoundedTriangle`, `DrawRoundedTriangleOutline`.
- New GPU-instanced batch primitives: `DrawRoundedRectsBatch`, `DrawRoundedRectsOutlineBatch`, `DrawRoundedTrianglesBatch`, `DrawRoundedTrianglesOutlineBatch`.
- Rounded shapes share the same SDF fragment shader and `fwidth`-based anti-aliasing used by rectangles, triangles and ellipses, so outlines stay visually consistent across every primitive (identical to the straight-edge version when `radius = 0`).
- Numba-parallel instance packing for both rounded rects and rounded triangles: one draw call per chunk, ~0 CPU overhead per shape.
- `radius`, `rotation` and `thickness` accept either a scalar or a per-instance array of length `n` in every rounded batch call.
- Documentation site updated to v0.1.1 with full coverage of the new shapes (signatures, parameters, examples).
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

## 🟦 Rounded shapes (new in 0.1.1)

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

PyBitEngine is still evolving, but the core rendering and packaging APIs are increasingly stable and suitable for real projects. The 0.1.1 release rounds out (pun intended) the primitive set so UI panels, buttons and stylized shapes can be drawn with the same batched performance as everything else.
