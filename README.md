# PyBitEngine ⚡

[![Documentation](https://img.shields.io/badge/docs-online-brightgreen)](https://hakerbituf.github.io/PyBitEngine/)
[![License](https://img.shields.io/badge/license-MPL--2.0-green)](https://github.com/HakerBituf/PyBitEngine/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A GPU-accelerated 2D game engine for Python, built on SDL2, ModernGL, NumPy and Numba.

PyBitEngine helps developers draw thousands of 2D objects efficiently while keeping the API simple. It uses instanced GPU rendering, texture atlases, and Numba kernels to reduce runtime overhead.

---

## 📦 Version: 0.1.0

This release focuses on packaging readiness and runtime performance.

### What is new in 0.1.0
- Added a standalone packaging helper via `pack()` for exporting games with cx_Freeze.
- Bundles runtime files and license notices into the output folder.
- Improved public imports so `from PyBitEngine import *` works cleanly.
- Reduced overhead in font resolution and text layout paths.
- Tightened frame pacing to avoid unnecessary CPU/GPU saturation.

---

## ✨ Highlights

- Instanced GPU rendering for rectangles, lines, triangles, ellipses and sprites.
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

## 📦 Packaging a game

```python
from PyBitEngine import pack

pack("main.py", name="MyGame", output_dir="dist")
```

## 📝 Notes

PyBitEngine is still evolving, but the core rendering and packaging APIs are now more stable and more suitable for real projects.