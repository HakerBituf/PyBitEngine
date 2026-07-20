# PyBitEngine ⚡

[![Documentation](https://img.shields.io/badge/docs-online-brightgreen)](https://hakerbituf.github.io/PyBitEngine/)
[![License](https://img.shields.io/badge/license-MPL--2.0-green)](https://github.com/HakerBituf/PyBitEngine/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A GPU-accelerated 2D game engine for Python, built on SDL2, ModernGL, NumPy and Numba.

PyBitEngine helps developers draw thousands of 2D objects efficiently while keeping the API simple. It uses instanced GPU rendering, texture atlases, and Numba kernels to reduce runtime overhead.

---

## 📦 Version: 0.1.3

This release brings major new features: a unified collision system, mouse helpers, rounded shapes, pixel‑perfect texture collisions, a texture atlas, depth sorting, performance optimizations, and a complete timer module.

### What is new in 0.1.3

- **Unified collision system** — `CheckCollision` works with any combination of immutable shapes (`Rect`, `Circle`, `Ellipse`, `Triangle`, `Line`, `RotRect`, `RoundedRect`, `Polygon`, `TextureCollider`). Zero object creation overhead.
- **Mouse helpers** — `MouseOver`, `MousePressed`, `MouseClicked`, `MouseHeld`, `MouseDragging`, `MouseWheelOn` work directly with any shape and optionally with a camera for world-space picking.
- **Full rounded shapes** — `DrawRoundedRect`, `DrawRoundedTriangle` and their outline variants, with GPU-instanced batch versions (`DrawRoundedRectsBatch`, `DrawRoundedTrianglesBatch`).
- **Pixel‑perfect texture collisions** — `CollidePointTexture` and `CollidePointTextureBatch` use the alpha channel stored on the CPU (no GPU readback, no disk I/O at runtime).
- **Texture atlas** — `LoadTextureAtlas` inserts an image into the shared atlas; `DrawSpritesBatch` can then render thousands of sprites from the same atlas in one draw call.
- **Easy‑to‑use batch wrappers** — `DrawRects`, `DrawLines`, `DrawCircles`, `DrawEllipses`, `DrawSprites`, `DrawTexts` accept lists of tuples with the same parameters as their immediate counterparts. No NumPy knowledge required.
- **Early‑Z / depth sorting** — `enable_early_z()` and the `begin_opaque_pass()` / `begin_transparent_pass()` helpers let you sort opaque objects front‑to‑back and transparent objects back‑to‑front, dramatically reducing overdraw.
- **Performance boost** — color packing in uint32, pre‑computed cos/sin on the CPU, optimized Numba kernels, and fewer allocations make batch rendering faster than ever.
- **Complete key constants** — all keyboard and mouse constants are now exported, including previously missing ones like `PE_K_CAPSLOCK`, `PE_K_NUMLOCK`, `PE_K_PRINTSCREEN`, `PE_K_LSUPER`, etc.
- **PE_TIME module** — non‑blocking timers: `Scheduler`, `After`, `Every`, `Countdown`, `Cooldown`, `Stopwatch`, plus `AsyncTimer`/`AsyncAfter` for background work with `RunOnMainThread` and `PumpMainThread`.
- **PE_PAKER module** — `pack()` creates standalone executables using `cx_Freeze`, with automatic inclusion of assets and dependencies.
- **CameraGPU and CameraCPU** — both now support smooth follow, screen‑shake, world bounds, coordinate conversion (`screen_to_world` / `world_to_screen`), and batch culling (`is_visible_batch`). The shake offset is now zero‑mean (no net drift).
- **Painter’s algorithm fixed** — `DrawTexture` now participates in the same batching system as primitives, so draw calls appear in the exact order they are issued.

---

## 🚀 Installation

```bash
pip install pybitengine