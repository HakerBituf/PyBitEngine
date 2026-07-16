# File: PE_CAMERA.py
"""
PE_CAMERA — Telecamere 2D per PyEngine
=======================================

Due implementazioni ottimizzate per casi d'uso opposti:

┌──────────────────────────────────────────────────────────────────────────┐
│  CameraGPU  — "Muovi tutto d'un colpo, lascia fare la GPU"              │
│  ─────────────────────────────────────────────────────────────────────── │
│  Tecnica  : FBO offscreen → quad fullscreen con UV shiftati dal blit    │
│  Costo CPU: ~0 per oggetto — nessuna coordinata da trasformare          │
│  Costo GPU: 1 draw call extra al frame (il blit finale)                 │
│  Ideale   : scene dense (migliaia di sprite), scrolling fluido          │
│  Limite   : alloca una texture FBO grande quanto il viewport            │
├──────────────────────────────────────────────────────────────────────────┤
│  CameraCPU — "Massimo controllo, zero overhead GPU nascosto"            │
│  ─────────────────────────────────────────────────────────────────────── │
│  Tecnica  : world_to_screen() / screen_to_world() puri in Python+NumPy │
│  Costo CPU: O(n) — un mul+add per coordinata (NumPy vectorizable)       │
│  Costo GPU: zero overhead nascosto                                       │
│  Ideale   : scene rade, frustum culling custom, debug, editor           │
│  Limite   : ogni coordinata va trasformata manualmente dal programmatore │
└──────────────────────────────────────────────────────────────────────────┘

Uso rapido CameraGPU
--------------------
    cam = CameraGPU(window)

    def update(dt, events):
        cam.follow(player.x, player.y, speed=250)
        cam.update(dt)

    def draw():
        cam.begin()          # da qui tutto va nell'FBO
        draw_world()
        cam.end()            # blit GPU → schermo con pan/zoom applicati

Uso rapido CameraCPU
--------------------
    cam = CameraCPU(window)

    def update(dt, events):
        cam.follow(player.x, player.y, speed=250)
        cam.update(dt)

    def draw():
        # Trasforma manualmente prima di disegnare
        sx, sy = cam.world_to_screen(player.x, player.y)
        sw, sh = cam.scale(player.w, player.h)
        if cam.is_visible(player.x, player.y, player.w, player.h):
            window.DrawRect(sx, sy, sw, sh, color=(255, 0, 0))
"""

import math
import numpy as np
import moderngl

# Numba è opzionale: se assente, fallback puro-NumPy (già rapido su N grandi).
try:
    from numba import njit  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False
    def njit(*a, **kw):  # noqa: D401
        def _wrap(fn): return fn
        return _wrap if not a or not callable(a[0]) else a[0]


# ──────────────────────────────────────────────────────────────────────────────
# Costanti interne
# ──────────────────────────────────────────────────────────────────────────────
_ZOOM_MIN = 0.01
_ZOOM_MAX = 64.0


# ──────────────────────────────────────────────────────────────────────────────
# Frustum / AABB culling — kernel Numba (fallback NumPy se numba assente)
# ──────────────────────────────────────────────────────────────────────────────
@njit(fastmath=True, cache=True)
def _numba_cull_aabb(rects, cam_x, cam_y, vw, vh, out):
    """
    rects: (N,4) float32 [wx, wy, ww, wh]  →  out: (N,) uint8 mask (0/1).
    Test AABB-vs-AABB standard: visibile se i due rettangoli si sovrappongono.
    Chiamato UNA volta per frame su tutte le entità del mondo: la GPU riceverà
    poi solo ciò che è effettivamente dentro (o a filo) del frustum 2D.
    """
    n = rects.shape[0]
    cx2 = cam_x + vw
    cy2 = cam_y + vh
    for i in range(n):
        x = rects[i, 0]
        y = rects[i, 1]
        w = rects[i, 2]
        h = rects[i, 3]
        vis = (x + w >= cam_x) and (x <= cx2) and (y + h >= cam_y) and (y <= cy2)
        out[i] = 1 if vis else 0


def _cull_aabb_numpy(rects, cam_x, cam_y, vw, vh):
    """Fallback vettoriale NumPy (usato quando numba non è disponibile)."""
    r = np.asarray(rects, dtype="f4")
    cx2 = cam_x + vw
    cy2 = cam_y + vh
    m = (
        (r[:, 0] + r[:, 2] >= cam_x) &
        (r[:, 0] <= cx2) &
        (r[:, 1] + r[:, 3] >= cam_y) &
        (r[:, 1] <= cy2)
    )
    return m.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# CAMERA GPU — FBO-based, zero overhead CPU per oggetto
# ──────────────────────────────────────────────────────────────────────────────
class CameraGPU:
    """
    Telecamera 2D completamente GPU-accelerata tramite Framebuffer Object.

    La scena viene disegnata in un FBO offscreen alle coordinate mondo originali.
    Al termine del frame, il blit shader legge l'FBO e lo ritaglia/zooma in base
    alla posizione e allo zoom della telecamera, il tutto in un singolo draw call.

    Parametri
    ---------
    draw : WINDOW | DRAW
        L'istanza del motore (deve avere `.ctx`, `.size`, e i metodi Flush/FlushAll).
    x : float
        Posizione iniziale X della telecamera nel mondo (angolo superiore sinistro).
    y : float
        Posizione iniziale Y della telecamera nel mondo (angolo superiore sinistro).
    zoom : float
        Zoom iniziale (1.0 = nessuno zoom, >1 = ingrandisce, <1 = rimpicciolisce).
    """

    # ── Shader blit ──────────────────────────────────────────────────────────
    # Il vertex shader mappa il quad fullscreen [-1,1] in UV che tengono conto
    # di posizione telecamera e zoom. Il fragment shader campiona l'FBO texture.
    _VERT = """
        #version 330
        in  vec2 in_vert;
        out vec2 v_uv;

        uniform vec2  u_cam_uv;    // (cam_x / W,  cam_y / H)
        uniform float u_inv_zoom;  // 1.0 / zoom

        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            // NDC [-1,1] → UV [0,1] → UV cam-spaziate
            // x: u = cam_uv.x + (vert.x + 1) * 0.5 * inv_zoom
            // y: v = cam_uv.y + (1 - vert.y) * 0.5 * inv_zoom  (asse Y invertito)
            v_uv = vec2(
                u_cam_uv.x + (in_vert.x + 1.0) * 0.5 * u_inv_zoom,
                u_cam_uv.y + (1.0 - in_vert.y) * 0.5 * u_inv_zoom
            );
        }
    """
    _FRAG = """
        #version 330
        uniform sampler2D u_fbo_tex;
        in  vec2 v_uv;
        out vec4 f_color;
        void main() {
            f_color = texture(u_fbo_tex, v_uv);
        }
    """

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, draw, x: float = 0.0, y: float = 0.0, zoom: float = 1.0, samples: int = 0):
        self._draw    = draw
        self._ctx: moderngl.Context = draw.ctx

        # Posizione e zoom correnti
        self._x    = float(x)
        self._y    = float(y)
        self._zoom = float(max(_ZOOM_MIN, min(_ZOOM_MAX, zoom)))

        # FIX MSAA: 0 = disabilitato (compat), 2/4/8/16 = campioni per pixel
        # sul framebuffer offscreen. Il resolve verso una texture non-MSAA
        # avviene dentro end(), la pipeline blit resta identica.
        self._samples = int(samples)

        # Shake
        self._shake_intensity  = 0.0
        self._shake_remaining  = 0.0

        # Smooth-follow
        self._follow_x      : float | None = None
        self._follow_y      : float | None = None
        self._follow_speed  : float | None = None  # pixel/s; None = snap
        self._deadzone_w    = 0.0
        self._deadzone_h    = 0.0

        # Bounds clamping (None = illimitato)
        self._bounds: tuple | None = None  # (min_x, min_y, max_x, max_y) in world

        # Inizializza FBO e shader GPU
        self._size = tuple(draw.size)   # (W, H)
        self._fbo_tex : moderngl.Texture       | None = None
        self._fbo     : moderngl.Framebuffer   | None = None
        self._msaa_rb : moderngl.Renderbuffer  | None = None  # FIX MSAA
        self._resolve_fbo: moderngl.Framebuffer | None = None  # FIX MSAA
        self._prog    : moderngl.Program     | None = None
        self._vao     : moderngl.VertexArray | None = None
        self._vbo     : moderngl.Buffer      | None = None

        self._build_gpu_resources()

    # ── Costruzione risorse GPU ───────────────────────────────────────────────
    def _build_gpu_resources(self):
        """Crea/ricrea FBO, shader e VAO del blit.

        FIX MSAA: se self._samples > 1, crea un renderbuffer multisample
        come target di disegno (self._fbo) e una texture RGBA di resolve
        (self._fbo_tex) usata dal blit. In end() si esegue copy_framebuffer
        da MSAA -> resolve prima del blit a schermo.
        Se samples == 0/1, comportamento identico alla versione originale
        (una sola texture, zero overhead).
        """
        self._destroy_gpu_resources()

        w, h = self._size
        ctx   = self._ctx

        # Texture di resolve (sempre non-MSAA: quella che il blit shader campiona)
        self._fbo_tex = ctx.texture((w, h), 4)
        self._fbo_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        if self._samples and self._samples > 1:
            # Renderbuffer multisample come target dei draw call
            self._msaa_rb  = ctx.renderbuffer((w, h), components=4,
                                              samples=self._samples)
            self._fbo      = ctx.framebuffer(color_attachments=[self._msaa_rb])
            # Framebuffer di resolve (wrappa la texture non-MSAA)
            self._resolve_fbo = ctx.framebuffer(color_attachments=[self._fbo_tex])
        else:
            # Path originale: la texture E' anche il target
            self._msaa_rb     = None
            self._resolve_fbo = None
            self._fbo = ctx.framebuffer(color_attachments=[self._fbo_tex])

        # Shader blit
        self._prog = ctx.program(
            vertex_shader   = self._VERT,
            fragment_shader = self._FRAG,
        )

        # Fullscreen quad: due triangoli che coprono tutto l'NDC [-1,1]
        quad = np.array(
            [-1.0, -1.0,   1.0, -1.0,   1.0, 1.0,
             -1.0, -1.0,   1.0,  1.0,  -1.0, 1.0],
            dtype="f4"
        )
        self._vbo = ctx.buffer(quad.tobytes())
        self._vao = ctx.vertex_array(self._prog, [(self._vbo, "2f", "in_vert")])

    def _destroy_gpu_resources(self):
        """Rilascia le risorse GPU allocate."""
        for attr in ("_vao", "_vbo", "_prog", "_resolve_fbo", "_fbo", "_msaa_rb", "_fbo_tex"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
                setattr(self, attr, None)

    # ── Ciclo di vita ─────────────────────────────────────────────────────────
    def begin(self):
        """
        Reindirizza il rendering verso l'FBO interno.

        Chiama questo metodo PRIMA di disegnare qualsiasi elemento del mondo.
        Effettua automaticamente il flush dei buffer pendenti del motore.
        """
        # Aggiorna la dimensione FBO se la finestra è stata ridimensionata
        current_size = tuple(self._draw.size)
        if current_size != self._size:
            self._size = current_size
            self._build_gpu_resources()

        # Svuota i buffer del motore prima di cambiare framebuffer
        self._draw.FlushAll()

        # Attiva l'FBO e puliscilo col colore di sfondo corrente
        self._fbo.use()
        bg = getattr(self._draw, "color", (0.0, 0.0, 0.0, 1.0))
        self._ctx.clear(*bg)

    def end(self):
        """
        Termina il rendering nell'FBO ed esegue il blit a schermo.

        Chiama questo metodo DOPO aver disegnato tutti gli elementi del mondo,
        PRIMA di `FlushAll()` / `SDL_GL_SwapWindow` del loop principale.
        """
        # Svuota i draw call pendenti nell'FBO
        self._draw.FlushAll()

        # FIX MSAA: resolve renderbuffer multisample -> texture non-MSAA
        if self._msaa_rb is not None:
            self._ctx.copy_framebuffer(self._resolve_fbo, self._fbo)

        # Torna al framebuffer di default (schermo)
        self._ctx.screen.use()

        # Calcola offset UV in base alla posizione e allo zoom correnti
        cx, cy = self._effective_position()
        w, h   = self._size
        cam_uv = (cx / w, cy / h)
        inv_z  = 1.0 / self._zoom

        self._prog["u_cam_uv"].value   = cam_uv
        self._prog["u_inv_zoom"].value = inv_z

        # Blit: campiona l'FBO e disegna il quad fullscreen
        self._fbo_tex.use(location=0)
        self._prog["u_fbo_tex"].value = 0
        self._vao.render(moderngl.TRIANGLES)

    def release(self):
        """Rilascia tutte le risorse GPU. Chiama questo se crei/distruggi la camera dinamicamente."""
        self._destroy_gpu_resources()

    # ── Update — logica telecamera ────────────────────────────────────────────
    def update(self, dt: float):
        """
        Aggiorna smooth-follow e screen-shake. Chiama ogni frame prima di begin().

        Parametri
        ---------
        dt : float
            Delta time in secondi (come fornito da `WINDOW.Loop` via `update(dt, events)`).
        """
        self._update_follow(dt)
        self._update_shake(dt)

    def _update_follow(self, dt: float):
        if self._follow_x is None:
            return

        # Centro del viewport in coordinate mondo
        vw = self._size[0] / self._zoom
        vh = self._size[1] / self._zoom
        center_x = self._x + vw * 0.5
        center_y = self._y + vh * 0.5

        tx = self._follow_x
        ty = self._follow_y

        # Deadzone: non muoversi se il target è dentro il rettangolo centrale
        dx = tx - center_x
        dy = ty - center_y
        dz_hw = self._deadzone_w * 0.5
        dz_hh = self._deadzone_h * 0.5
        if abs(dx) <= dz_hw and abs(dy) <= dz_hh:
            return

        # Target della camera (angolo superiore sinistro del viewport centrato sul target)
        target_cam_x = tx - vw * 0.5
        target_cam_y = ty - vh * 0.5

        if self._follow_speed is None:
            # Snap immediato
            self._x = target_cam_x
            self._y = target_cam_y
        else:
            # Lerp con velocità limitata
            dist = math.hypot(target_cam_x - self._x, target_cam_y - self._y)
            if dist > 0.0:
                step = self._follow_speed * dt
                t    = min(1.0, step / dist)
                self._x = self._x + (target_cam_x - self._x) * t
                self._y = self._y + (target_cam_y - self._y) * t

        self._apply_bounds()

    def _update_shake(self, dt: float):
        # BUG FIX (bug 5): decrementa PRIMA di controllare, cosi'
        # _shake_intensity viene azzerata nello stesso frame in cui il timer
        # scade (allinea il comportamento a CameraCPU._update_shake).
        self._shake_remaining = max(0.0, self._shake_remaining - dt)
        if self._shake_remaining <= 0.0:
            self._shake_intensity = 0.0
            self._shake_remaining = 0.0

    def _effective_position(self) -> tuple[float, float]:
        x, y = self._x, self._y
        if self._shake_remaining > 0.0 and self._shake_intensity > 0.0:
            factor = self._shake_remaining * self._shake_intensity
            # FIX: media zero – senza il "- 1.0"
            x += math.sin(self._shake_remaining * 97.3) * factor
            y += math.cos(self._shake_remaining * 73.1) * factor
        return x, y

    def _apply_bounds(self):
        """Clampla la posizione della camera all'interno dei bound del mondo."""
        if self._bounds is None:
            return
        min_x, min_y, max_x, max_y = self._bounds
        vw = self._size[0] / self._zoom
        vh = self._size[1] / self._zoom
        self._x = max(min_x, min(max_x - vw, self._x))
        self._y = max(min_y, min(max_y - vh, self._y))

    # ── API pubblica — posizione ──────────────────────────────────────────────
    @property
    def x(self) -> float:
        """Coordinata X dell'angolo superiore sinistro del viewport nel mondo."""
        return self._x

    @x.setter
    def x(self, value: float):
        self._x = float(value)
        self._apply_bounds()

    @property
    def y(self) -> float:
        """Coordinata Y dell'angolo superiore sinistro del viewport nel mondo."""
        return self._y

    @y.setter
    def y(self, value: float):
        self._y = float(value)
        self._apply_bounds()

    @property
    def zoom(self) -> float:
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        self._zoom = float(max(_ZOOM_MIN, min(_ZOOM_MAX, value)))
        self._apply_bounds()

    def move(self, dx: float, dy: float):
        """Sposta la camera di (dx, dy) in coordinate mondo."""
        self._x += dx
        self._y += dy
        self._apply_bounds()

    def center_on(self, wx: float, wy: float):
        """Centra immediatamente la camera sul punto mondo (wx, wy)."""
        vw = self._size[0] / self._zoom
        vh = self._size[1] / self._zoom
        self._x = wx - vw * 0.5
        self._y = wy - vh * 0.5
        self._apply_bounds()

    # ── API pubblica — smooth follow ──────────────────────────────────────────
    def follow(self, target_x: float, target_y: float,
               speed: float | None = None,
               deadzone_w: float = 0.0,
               deadzone_h: float = 0.0):
        """
        Imposta il target che la telecamera deve inseguire.

        Chiama questo ogni frame (o quando vuoi aggiornare il target);
        il movimento effettivo avviene in `update()`.

        Parametri
        ---------
        target_x, target_y : float
            Posizione del target nel mondo.
        speed : float | None
            Pixel al secondo. None = snap istantaneo.
        deadzone_w, deadzone_h : float
            Dimensione della zona morta centrale (pixel). Il target deve
            uscire da questa zona prima che la camera inizi a muoversi.
        """
        self._follow_x     = float(target_x)
        self._follow_y     = float(target_y)
        self._follow_speed = float(speed) if speed is not None else None
        self._deadzone_w   = float(deadzone_w)
        self._deadzone_h   = float(deadzone_h)

    def stop_follow(self):
        """Interrompe il smooth-follow."""
        self._follow_x = None
        self._follow_y = None

    # ── API pubblica — shake ──────────────────────────────────────────────────
    def shake(self, intensity: float, duration: float):
        """
        Avvia un effetto screen-shake.

        Parametri
        ---------
        intensity : float
            Ampiezza massima dello shake in pixel mondo.
        duration : float
            Durata in secondi.
        """
        self._shake_intensity = max(0.0, float(intensity))
        self._shake_remaining = max(0.0, float(duration))

    # ── API pubblica — bounds ─────────────────────────────────────────────────
    def set_bounds(self, min_x: float, min_y: float, max_x: float, max_y: float):
        """
        Limita il movimento della camera all'interno del rettangolo mondo specificato.
        La camera non potrà mai mostrare aree fuori da questo rettangolo.
        """
        self._bounds = (float(min_x), float(min_y), float(max_x), float(max_y))
        self._apply_bounds()

    def clear_bounds(self):
        """Rimuove i limiti di movimento della camera."""
        self._bounds = None

    # ── Conversione coordinate ────────────────────────────────────────────────
    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """
        Converte coordinate schermo (pixel) → coordinate mondo.

        Utile per sapere su quale punto del mondo si trova il cursore.
        """
        # BUG FIX (bug 4): usa la posizione effettiva (con shake) per
        # allineare le conversioni al blit realmente eseguito in end().
        cx, cy = self._effective_position()
        wx = cx + sx / self._zoom
        wy = cy + sy / self._zoom
        return wx, wy

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        """
        Converte coordinate mondo → coordinate schermo (pixel).

        Utile per posizionare elementi UI sopra oggetti mondo.
        """
        # BUG FIX (bug 4): vedi screen_to_world.
        cx, cy = self._effective_position()
        sx = (wx - cx) * self._zoom
        sy = (wy - cy) * self._zoom
        return sx, sy

    # ── Frustum culling (2D AABB) ─────────────────────────────────────────────
    def is_visible(self, wx: float, wy: float,
                   ww: float = 0.0, wh: float = 0.0,
                   margin: float = 0.0) -> bool:
        """
        True se il rettangolo mondo (wx, wy, ww, wh) è (parzialmente) visibile
        nella viewport corrente della CameraGPU. Usa `_effective_position()`
        per essere coerente con il blit finale (incluso lo shake).
        """
        cx, cy = self._effective_position()
        vw = self._size[0] / self._zoom + margin * 2.0
        vh = self._size[1] / self._zoom + margin * 2.0
        cam_x = cx - margin
        cam_y = cy - margin
        return (wx + ww >= cam_x and wx <= cam_x + vw and
                wy + wh >= cam_y and wy <= cam_y + vh)

    def is_visible_batch(self, rects: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Maschera booleana NumPy (vettoriale) per un array (N,4) di rettangoli."""
        r = np.asarray(rects, dtype="f4")
        cx, cy = self._effective_position()
        vw = self._size[0] / self._zoom + margin * 2.0
        vh = self._size[1] / self._zoom + margin * 2.0
        cam_x = cx - margin
        cam_y = cy - margin
        return (
            (r[:, 0] + r[:, 2] >= cam_x) & (r[:, 0] <= cam_x + vw) &
            (r[:, 1] + r[:, 3] >= cam_y) & (r[:, 1] <= cam_y + vh)
        )

    def is_visible_batch_numba(self, rects: np.ndarray,
                               margin: float = 0.0,
                               out: np.ndarray | None = None) -> np.ndarray:
        """
        Variante Numba-JIT del culling batch: ideale per N >> 10k.
        Ritorna una maschera uint8 (0/1) — usabile come bool per l'indexing.
        Se `out` è fornito (uint8, shape (N,)) viene riusato per zero-alloc.
        """
        r = np.asarray(rects, dtype="f4")
        cx, cy = self._effective_position()
        vw = self._size[0] / self._zoom + margin * 2.0
        vh = self._size[1] / self._zoom + margin * 2.0
        cam_x = cx - margin
        cam_y = cy - margin
        if _HAS_NUMBA:
            if out is None or out.shape[0] != r.shape[0] or out.dtype != np.uint8:
                out = np.empty(r.shape[0], dtype=np.uint8)
            _numba_cull_aabb(r, np.float32(cam_x), np.float32(cam_y),
                             np.float32(vw), np.float32(vh), out)
            return out
        return _cull_aabb_numpy(r, cam_x, cam_y, vw, vh)

    # ── Info ──────────────────────────────────────────────────────────────────
    @property
    def viewport_rect(self) -> tuple[float, float, float, float]:
        """Ritorna il rettangolo visibile nel mondo (x, y, w, h)."""
        vw = self._size[0] / self._zoom
        vh = self._size[1] / self._zoom
        return self._x, self._y, vw, vh

    def __repr__(self) -> str:
        return (
            f"CameraGPU(x={self._x:.1f}, y={self._y:.1f}, "
            f"zoom={self._zoom:.3f}, size={self._size})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# CAMERA CPU — trasformazioni pure in Python / NumPy
# ──────────────────────────────────────────────────────────────────────────────
class CameraCPU:
    """
    Telecamera 2D CPU-side: trasformazione esplicita delle coordinate.

    Ogni coordinata mondo viene trasformata in coordinata schermo tramite
    `world_to_screen()` prima di essere passata alle funzioni Draw del motore.
    Il costo è O(n) sul numero di coordinate, ma è completamente CPU-bound
    e senza alcun overhead GPU nascosto.

    Parametri
    ---------
    draw : WINDOW | DRAW
        L'istanza del motore (serve solo per leggere `.size`).
    x : float
        Posizione iniziale X della telecamera (angolo sup. sinistro del viewport).
    y : float
        Posizione iniziale Y della telecamera.
    zoom : float
        Zoom iniziale (1.0 = nessuno zoom).
    """

    def __init__(self, draw, x: float = 0.0, y: float = 0.0, zoom: float = 1.0):
        self._draw  = draw
        self._x     = float(x)
        self._y     = float(y)
        self._zoom  = float(max(_ZOOM_MIN, min(_ZOOM_MAX, zoom)))

        # Shake
        self._shake_intensity = 0.0
        self._shake_remaining = 0.0

        # Smooth-follow
        self._follow_x     : float | None = None
        self._follow_y     : float | None = None
        self._follow_speed : float | None = None
        self._deadzone_w   = 0.0
        self._deadzone_h   = 0.0

        # Bounds
        self._bounds: tuple | None = None

    # ── Ciclo di vita ─────────────────────────────────────────────────────────
    def update(self, dt: float):
        """
        Aggiorna smooth-follow e screen-shake. Chiama ogni frame in `update()`.

        Parametri
        ---------
        dt : float
            Delta time in secondi.
        """
        self._update_follow(dt)
        self._update_shake(dt)

    def _update_follow(self, dt: float):
        if self._follow_x is None:
            return

        w, h = self._draw.size
        vw   = w / self._zoom
        vh   = h / self._zoom
        cx   = self._x + vw * 0.5
        cy   = self._y + vh * 0.5
        tx   = self._follow_x
        ty   = self._follow_y

        # Deadzone
        dx = tx - cx; dy = ty - cy
        if abs(dx) <= self._deadzone_w * 0.5 and abs(dy) <= self._deadzone_h * 0.5:
            return

        target_cam_x = tx - vw * 0.5
        target_cam_y = ty - vh * 0.5

        if self._follow_speed is None:
            self._x = target_cam_x
            self._y = target_cam_y
        else:
            dist = math.hypot(target_cam_x - self._x, target_cam_y - self._y)
            if dist > 0.0:
                t = min(1.0, self._follow_speed * dt / dist)
                self._x += (target_cam_x - self._x) * t
                self._y += (target_cam_y - self._y) * t

        self._apply_bounds()

    def _update_shake(self, dt: float):
        self._shake_remaining = max(0.0, self._shake_remaining - dt)
        if self._shake_remaining <= 0.0:
            self._shake_intensity = 0.0
            self._shake_remaining = 0.0

    def _effective_offset(self) -> tuple[float, float]:
        if self._shake_remaining > 0.0 and self._shake_intensity > 0.0:
            f = self._shake_remaining * self._shake_intensity
            # FIX: media zero
            ox = math.sin(self._shake_remaining * 97.3) * f
            oy = math.cos(self._shake_remaining * 73.1) * f
            return ox, oy
        return 0.0, 0.0

    def _apply_bounds(self):
        if self._bounds is None:
            return
        min_x, min_y, max_x, max_y = self._bounds
        w, h = self._draw.size
        vw = w / self._zoom
        vh = h / self._zoom
        self._x = max(min_x, min(max_x - vw, self._x))
        self._y = max(min_y, min(max_y - vh, self._y))

    # ── API pubblica — posizione ──────────────────────────────────────────────
    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, value: float):
        self._x = float(value)
        self._apply_bounds()

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, value: float):
        self._y = float(value)
        self._apply_bounds()

    @property
    def zoom(self) -> float:
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        self._zoom = float(max(_ZOOM_MIN, min(_ZOOM_MAX, value)))
        self._apply_bounds()

    def move(self, dx: float, dy: float):
        """Sposta la camera di (dx, dy) in coordinate mondo."""
        self._x += dx
        self._y += dy
        self._apply_bounds()

    def center_on(self, wx: float, wy: float):
        """Centra immediatamente la camera sul punto mondo (wx, wy)."""
        w, h = self._draw.size
        self._x = wx - w / (2.0 * self._zoom)
        self._y = wy - h / (2.0 * self._zoom)
        self._apply_bounds()

    # ── API pubblica — smooth follow ──────────────────────────────────────────
    def follow(self, target_x: float, target_y: float,
               speed: float | None = None,
               deadzone_w: float = 0.0,
               deadzone_h: float = 0.0):
        """
        Imposta il target che la telecamera deve inseguire.

        Parametri
        ---------
        target_x, target_y : float
            Posizione del target nel mondo (es. il centro del player).
        speed : float | None
            Velocità di inseguimento in pixel/s. None = snap istantaneo.
        deadzone_w, deadzone_h : float
            Zona morta centrale (pixel). Il target deve uscire dalla zona
            prima che la camera inizi a muoversi.
        """
        self._follow_x     = float(target_x)
        self._follow_y     = float(target_y)
        self._follow_speed = float(speed) if speed is not None else None
        self._deadzone_w   = float(deadzone_w)
        self._deadzone_h   = float(deadzone_h)

    def stop_follow(self):
        """Interrompe il smooth-follow."""
        self._follow_x = None
        self._follow_y = None

    # ── API pubblica — shake ──────────────────────────────────────────────────
    def shake(self, intensity: float, duration: float):
        """
        Avvia un effetto screen-shake.

        Parametri
        ---------
        intensity : float
            Ampiezza massima dello shake in pixel.
        duration : float
            Durata in secondi.
        """
        self._shake_intensity = max(0.0, float(intensity))
        self._shake_remaining = max(0.0, float(duration))

    # ── API pubblica — bounds ─────────────────────────────────────────────────
    def set_bounds(self, min_x: float, min_y: float, max_x: float, max_y: float):
        """
        Limita il movimento della camera all'interno del rettangolo specificato.
        La camera non mostrerà mai aree fuori da questo rettangolo.
        """
        self._bounds = (float(min_x), float(min_y), float(max_x), float(max_y))
        self._apply_bounds()

    def clear_bounds(self):
        """Rimuove i limiti di movimento."""
        self._bounds = None

    # ── Conversione coordinate — singolo punto ────────────────────────────────
    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        """
        Trasforma un punto mondo in coordinate schermo.

        Parametri
        ---------
        wx, wy : float
            Coordinate mondo.

        Ritorna
        -------
        (sx, sy) : tuple[float, float]
            Coordinate schermo in pixel.
        """
        # FIX: segno dell'offset di shake allineato a CameraGPU._effective_position()
        # (che fa `x + ox`). Prima qui si faceva l'opposto (`- ox` sulla
        # posizione effettiva), quindi lo shake scuoteva la scena in direzione
        # opposta rispetto a CameraGPU a parita' di intensity/duration.
        ox, oy = self._effective_offset()
        sx = (wx - self._x - ox) * self._zoom
        sy = (wy - self._y - oy) * self._zoom
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """
        Converte coordinate schermo → coordinate mondo.

        Utile per trasformare la posizione del mouse in coordinate mondo
        (es. per il picking di oggetti).
        """
        # FIX: vedi world_to_screen qui sopra — stesso allineamento di segno.
        ox, oy = self._effective_offset()
        wx = self._x + sx / self._zoom + ox
        wy = self._y + sy / self._zoom + oy
        return wx, wy

    def scale(self, world_w: float, world_h: float) -> tuple[float, float]:
        """
        Scala dimensioni mondo → dimensioni schermo.

        Parametri
        ---------
        world_w, world_h : float
            Dimensioni in coordinate mondo.

        Ritorna
        -------
        (screen_w, screen_h) : tuple[float, float]
        """
        return world_w * self._zoom, world_h * self._zoom

    def apply_rect(self, wx: float, wy: float,
                   ww: float, wh: float
                   ) -> tuple[float, float, float, float]:
        """
        Trasforma un rettangolo mondo (x, y, w, h) in coordinate schermo.

        Comodo per passare direttamente a DrawRect/DrawTexture:

            sx, sy, sw, sh = cam.apply_rect(wx, wy, ww, wh)
            window.DrawRect(sx, sy, sw, sh)

        Ritorna
        -------
        (sx, sy, sw, sh) : tuple[float, float, float, float]
        """
        sx, sy = self.world_to_screen(wx, wy)
        sw     = ww * self._zoom
        sh     = wh * self._zoom
        return sx, sy, sw, sh

    # ── Conversione coordinate — batch NumPy ──────────────────────────────────
    def world_to_screen_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Trasforma un array di punti mondo in coordinate schermo via NumPy.

        Parametri
        ---------
        points : np.ndarray, shape (N, 2)
            Array di N punti [wx, wy].

        Ritorna
        -------
        np.ndarray, shape (N, 2)
            Array di N punti [sx, sy] in coordinate schermo.

        Esempio
        -------
            pts  = np.array([[100, 200], [300, 400]], dtype='f4')
            spts = cam.world_to_screen_batch(pts)
        """
        # FIX: stesso allineamento di segno di world_to_screen (vedi sopra).
        pts   = np.asarray(points, dtype="f4")
        ox, oy = self._effective_offset()
        offset = np.array([self._x + ox, self._y + oy], dtype="f4")
        return (pts - offset) * self._zoom

    def screen_to_world_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Trasforma un array di punti schermo in coordinate mondo via NumPy.

        Parametri
        ---------
        points : np.ndarray, shape (N, 2)
            Array di N punti [sx, sy].

        Ritorna
        -------
        np.ndarray, shape (N, 2)
        """
        # FIX: stesso allineamento di segno di screen_to_world (vedi sopra).
        pts    = np.asarray(points, dtype="f4")
        ox, oy = self._effective_offset()
        offset = np.array([self._x + ox, self._y + oy], dtype="f4")
        return pts / self._zoom + offset

    # ── Frustum culling ───────────────────────────────────────────────────────
    def is_visible(self, wx: float, wy: float,
                   ww: float = 0.0, wh: float = 0.0,
                   margin: float = 0.0) -> bool:
        """
        Ritorna True se il rettangolo mondo (wx, wy, ww, wh) è (parzialmente)
        visibile nella viewport corrente.

        Parametri
        ---------
        wx, wy : float
            Angolo superiore sinistro dell'oggetto nel mondo.
        ww, wh : float
            Dimensioni dell'oggetto (0 per un singolo punto).
        margin : float
            Margine extra in pixel mondo da aggiungere alla viewport
            (utile per oggetti con effetti che sconfinano).

        Uso tipico nel draw():
            if cam.is_visible(enemy.x, enemy.y, 32, 32):
                sx, sy, sw, sh = cam.apply_rect(enemy.x, enemy.y, 32, 32)
                window.DrawRect(sx, sy, sw, sh, color=RED)
        """
        w, h   = self._draw.size
        vw     = w / self._zoom + margin * 2.0
        vh     = h / self._zoom + margin * 2.0
        cam_x  = self._x - margin
        cam_y  = self._y - margin
        return (
            wx + ww >= cam_x and wx <= cam_x + vw and
            wy + wh >= cam_y and wy <= cam_y + vh
        )

    def is_visible_batch(self, rects: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """
        Maschera booleana di visibilità per un array di rettangoli.

        Parametri
        ---------
        rects : np.ndarray, shape (N, 4)
            Ogni riga è [wx, wy, ww, wh].
        margin : float
            Margine extra in pixel mondo.

        Ritorna
        -------
        np.ndarray di bool, shape (N,)

        Esempio
        -------
            mask = cam.is_visible_batch(enemy_rects)
            for rect in enemy_rects[mask]:
                ...
        """
        rects  = np.asarray(rects, dtype="f4")
        w, h   = self._draw.size
        vw     = w / self._zoom + margin * 2.0
        vh     = h / self._zoom + margin * 2.0
        cam_x  = self._x - margin
        cam_y  = self._y - margin
        return (
            (rects[:, 0] + rects[:, 2] >= cam_x) &
            (rects[:, 0] <= cam_x + vw)           &
            (rects[:, 1] + rects[:, 3] >= cam_y) &
            (rects[:, 1] <= cam_y + vh)
        )

    def is_visible_batch_numba(self, rects: np.ndarray,
                               margin: float = 0.0,
                               out: np.ndarray | None = None) -> np.ndarray:
        """
        Variante Numba-JIT del culling AABB (ideale per N >> 10k oggetti).
        Ritorna una maschera uint8 (0/1) usabile come bool per l'indexing NumPy.
        Se `out` è fornito e ha shape/dtype giusti viene riusato (zero-alloc).

        Fallback automatico a NumPy se numba non è installato.
        """
        r = np.asarray(rects, dtype="f4")
        w, h = self._draw.size
        vw = w / self._zoom + margin * 2.0
        vh = h / self._zoom + margin * 2.0
        cam_x = self._x - margin
        cam_y = self._y - margin
        if _HAS_NUMBA:
            if out is None or out.shape[0] != r.shape[0] or out.dtype != np.uint8:
                out = np.empty(r.shape[0], dtype=np.uint8)
            _numba_cull_aabb(r, np.float32(cam_x), np.float32(cam_y),
                             np.float32(vw), np.float32(vh), out)
            return out
        return _cull_aabb_numpy(r, cam_x, cam_y, vw, vh)



    # ── Info ──────────────────────────────────────────────────────────────────
    @property
    def viewport_rect(self) -> tuple[float, float, float, float]:
        """Ritorna il rettangolo visibile nel mondo (x, y, w, h)."""
        w, h = self._draw.size
        return self._x, self._y, w / self._zoom, h / self._zoom

    def __repr__(self) -> str:
        return (
            f"CameraCPU(x={self._x:.1f}, y={self._y:.1f}, "
            f"zoom={self._zoom:.3f})"
        )
