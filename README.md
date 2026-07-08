# PyBitEngine ⚡

**Motore grafico 2D GPU-accelerato per Python — basato su SDL2, ModernGL, NumPy e Numba.**

PyBitEngine è una libreria per sviluppatori che vogliono disegnare **migliaia di oggetti 2D** a 60 FPS, senza impazzire con OpenGL. 
Usa rendering instanced sulla GPU, texture atlas, e kernel Numba per zero overhead a runtime.

---

## 📦 Versione: **0.1.0-alpha**

API in evoluzione. Feedback e contributi sono benvenuti!

---

## ✨ Cosa fa davvero (dal codice)

- **Rendering GPU instanced** — una draw call per migliaia di rettangoli, linee, triangoli, ellissi o sprite
- **Texture Atlas** — carica le immagini in un unico grande texture, con allocatore MaxRects
- **Camera 2D** — due versioni: 
  - `CameraGPU`: FBO offscreen + blit shader (zero overhead CPU, 1 draw call extra)
  - `CameraCPU`: trasformazioni esplicite (controllo totale, debug)
- **Font rendering** — TTF/OTF su texture atlas (Pillow + caching)
- **Eventi SDL2** — tastiera, mouse, drag, scroll (tutti i tasti `PE_K_*`)
- **Collisioni geometriche** — punto, rettangolo, triangolo, ellisse, cerchio, OBB (con SAT)
- **Zero allocazioni a runtime** — buffer preallocati, Numba per il packing

---

## 🚀 Installazione

```bash
pip install PyBitEngine