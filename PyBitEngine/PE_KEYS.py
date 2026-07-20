# File: PE_KEYS.py
import sdl2 as _sdl2
import math as _math
import numpy as _np

# ============================================================================ #
# Helper geometrici privati per le collisioni mouse su forme "compound" che
# PE_DRAW non espone come singolo Collide* pronto all'uso (linee, rettangoli/
# triangoli arrotondati, curve di Bezier, testo). Nessuno di questi importa
# PE_DRAW: ricevono sempre l'istanza `draw` come parametro a runtime (stesso
# pattern del resto del file), quindi zero rischio di import circolare.
#
# _pe_rtri_shrink / _pe_sd_triangle sono un porting Python puro (no numba,
# un solo punto — il mouse — per frame, non serve JIT) della stessa identica
# matematica usata da PE_DRAW._rtri_shrink_and_aabb + la funzione GLSL
# sdTriangle del fragment shader dei triangoli arrotondati: la hitbox
# combacia ESATTAMENTE con la forma renderizzata, angoli inclusi.
# ============================================================================ #
def _pe_rtri_shrink(ax, ay, bx, by, cx, cy, r):
    """Ritorna i 3 vertici del triangolo 'shrunk' (spinti verso l'interno di
    r_eff) + r_eff stesso — identica logica di PE_DRAW._rtri_shrink_and_aabb."""
    len_a = _math.hypot(bx - cx, by - cy)
    len_b = _math.hypot(cx - ax, cy - ay)
    len_c = _math.hypot(ax - bx, ay - by)
    perim = len_a + len_b + len_c
    cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    area = 0.5 * (cross if cross >= 0.0 else -cross)
    inradius = 2.0 * area / (perim if perim > 1e-6 else 1e-6)
    r_eff = r
    if r_eff > inradius * 0.99:
        r_eff = inradius * 0.99
    if r_eff < 0.0:
        r_eff = 0.0
    sgn = 1.0 if cross > 0.0 else -1.0

    inv_c = 1.0 / (len_c if len_c > 1e-6 else 1e-6)
    inv_a = 1.0 / (len_a if len_a > 1e-6 else 1e-6)
    inv_b = 1.0 / (len_b if len_b > 1e-6 else 1e-6)

    nab_x = -(by - ay) * sgn * inv_c
    nab_y = (bx - ax) * sgn * inv_c
    d_ab = nab_x * ax + nab_y * ay + r_eff

    nbc_x = -(cy - by) * sgn * inv_a
    nbc_y = (cx - bx) * sgn * inv_a
    d_bc = nbc_x * bx + nbc_y * by + r_eff

    nca_x = -(ay - cy) * sgn * inv_b
    nca_y = (ax - cx) * sgn * inv_b
    d_ca = nca_x * cx + nca_y * cy + r_eff

    det_a = nca_x * nab_y - nca_y * nab_x
    if -1e-6 < det_a < 1e-6:
        sax, say = ax, ay
    else:
        inv_da = 1.0 / det_a
        sax = (d_ca * nab_y - d_ab * nca_y) * inv_da
        say = (nca_x * d_ab - nab_x * d_ca) * inv_da

    det_b = nab_x * nbc_y - nab_y * nbc_x
    if -1e-6 < det_b < 1e-6:
        sbx, sby = bx, by
    else:
        inv_db = 1.0 / det_b
        sbx = (d_ab * nbc_y - d_bc * nab_y) * inv_db
        sby = (nab_x * d_bc - nbc_x * d_ab) * inv_db

    det_c = nbc_x * nca_y - nbc_y * nca_x
    if -1e-6 < det_c < 1e-6:
        scx, scy = cx, cy
    else:
        inv_dc = 1.0 / det_c
        scx = (d_bc * nca_y - d_ca * nbc_y) * inv_dc
        scy = (nbc_x * d_ca - nca_x * d_bc) * inv_dc

    return sax, say, sbx, sby, scx, scy, r_eff


def _pe_sd_triangle(px, py, p0x, p0y, p1x, p1y, p2x, p2y):
    """Signed distance point-triangolo (Inigo Quilez) — porting Python 1:1
    della funzione GLSL sdTriangle usata dal fragment shader dei triangoli
    (arrotondati e non). Negativa dentro, positiva fuori."""
    e0x, e0y = p1x - p0x, p1y - p0y
    e1x, e1y = p2x - p1x, p2y - p1y
    e2x, e2y = p0x - p2x, p0y - p2y
    w0x, w0y = px - p0x, py - p0y
    w1x, w1y = px - p1x, py - p1y
    w2x, w2y = px - p2x, py - p2y

    def _clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    t0 = _clamp01((w0x * e0x + w0y * e0y) / max(e0x * e0x + e0y * e0y, 1e-12))
    pq0x, pq0y = w0x - e0x * t0, w0y - e0y * t0
    t1 = _clamp01((w1x * e1x + w1y * e1y) / max(e1x * e1x + e1y * e1y, 1e-12))
    pq1x, pq1y = w1x - e1x * t1, w1y - e1y * t1
    t2 = _clamp01((w2x * e2x + w2y * e2y) / max(e2x * e2x + e2y * e2y, 1e-12))
    pq2x, pq2y = w2x - e2x * t2, w2y - e2y * t2

    s = 1.0 if (e0x * e2y - e0y * e2x) >= 0.0 else -1.0

    dx = min(pq0x * pq0x + pq0y * pq0y,
             pq1x * pq1x + pq1y * pq1y,
             pq2x * pq2x + pq2y * pq2y)
    dy = min(s * (w0x * e0y - w0y * e0x),
             s * (w1x * e1y - w1y * e1x),
             s * (w2x * e2y - w2y * e2x))
    sign_dy = 1.0 if dy >= 0.0 else -1.0
    return -_math.sqrt(dx) * sign_dy


# Definiamo un nostro Oggetto Evento facile da leggere
class PE_Event:
    # __slots__: ogni evento SDL (mouse, tastiera, rotellina...) crea una nuova
    # istanza di questa classe, anche piu' volte per frame. Senza __slots__,
    # ogni istanza si porta dietro un __dict__ generico (allocazione extra).
    # Con __slots__ riserviamo uno slot fisso per attributo: creazione e
    # accesso agli attributi piu' rapidi, meno memoria. L'interfaccia pubblica
    # (event.x, event.type, ...) resta identica.
    __slots__ = ("type", "key", "button", "x", "y", "dx", "dy", "clicks", "wheel_x", "wheel_y")

    def __init__(self, type, key=None, button=None, x=0, y=0, dx=0, dy=0, clicks=1, wheel_x=0, wheel_y=0):
        self.type = type
        self.key = key
        # Proprietà Mouse
        self.button = button
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.clicks = clicks
        # Proprietà Rotellina Mouse
        self.wheel_x = wheel_x
        self.wheel_y = wheel_y

    # ================================================================
    # Collisioni mouse ↔ DRAW
    # ------------------------------------------------------------------
    # Perche' qui e non in un modulo a parte: le funzioni CollideXxx sono
    # GIA' tutte in PE_DRAW.py (unica fonte di verita' per la geometria,
    # coerenza + velocita' come richiesto). PE_Event non duplica NESSUNA
    # logica di collisione: ogni metodo qui sotto e' solo un adattatore
    # sottile che (1) decide quale punto testare — schermo o mondo — e
    # (2) chiama il Collide* corrispondente su `draw`.
    #
    # Coordinate camera: le funzioni di collisione di PE_DRAW restano
    # agnostiche rispetto alla camera (stesso identico costo O(1) di
    # sempre, zero accoppiamento PE_DRAW<->PE_CAMERA). Se e' in uso una
    # CameraCPU/CameraGPU, passa `camera=cam`: la conversione screen->world
    # (gia' presente e ottimizzata in PE_CAMERA) viene fatta qui, una sola
    # volta per evento, PRIMA del test — mai a monte per ogni singola
    # shape. E' l'opzione piu' veloce disponibile e resta opt-in: se non
    # serve (HUD, UI in screen-space) non si paga nulla in piu'.
    # ================================================================

    def _mouse_point(self, camera=None):
        if camera is None:
            return self.x, self.y
        return camera.screen_to_world(self.x, self.y)

    # --- Point-vs-shape singola -------------------------------------------------
    def CollideRect(self, draw, x, y, w, h, camera=None):
        px, py = self._mouse_point(camera)
        return draw.CollidePointRect(px, py, x, y, w, h)

    def CollideRotatedRect(self, draw, x, y, w, h, rotation=0.0, camera=None):
        px, py = self._mouse_point(camera)
        return draw.CollidePointRotatedRect(px, py, x, y, w, h, rotation)

    def CollideCircle(self, draw, cx, cy, r, camera=None):
        px, py = self._mouse_point(camera)
        return draw.CollidePointCircle(px, py, cx, cy, r)

    def CollideEllipse(self, draw, cx, cy, rx, ry, rotation=0.0, camera=None):
        px, py = self._mouse_point(camera)
        return draw.CollidePointEllipse(px, py, cx, cy, rx, ry, rotation)

    def CollideTriangle(self, draw, x1, y1, x2, y2, x3, y3, camera=None):
        px, py = self._mouse_point(camera)
        return draw.CollidePointTriangle(px, py, x1, y1, x2, y2, x3, y3)

    def CollideTexture(self, draw, name, x, y, w=None, h=None, rotation=0.0,
                       flip_x=False, flip_y=False, alpha_threshold=1, camera=None):
        """Pixel-perfect: delega a DRAW.CollidePointTexture (early-out
        rotated-rect + lookup alpha, vedi PE_DRAW.py)."""
        px, py = self._mouse_point(camera)
        return draw.CollidePointTexture(px, py, name, x, y, w, h, rotation,
                                        flip_x, flip_y, alpha_threshold)

    # --- Point-vs-molte-shape (batch NumPy) --------------------------------------
    # Un solo punto (questo evento) contro N shape in un colpo solo: picking
    # su liste di sprite/hitbox senza un ciclo Python per elemento.
    def CollideRectBatch(self, draw, x_arr, y_arr, w_arr, h_arr, camera=None):
        px, py = self._mouse_point(camera)
        return draw.CollidePointRectBatch(px, py, x_arr, y_arr, w_arr, h_arr)

    def CollideRotatedRectBatch(self, draw, x_arr, y_arr, w_arr, h_arr, rotation_arr=0.0, camera=None):
        px, py = self._mouse_point(camera)
        return draw.CollidePointRotatedRectBatch(px, py, x_arr, y_arr, w_arr, h_arr, rotation_arr)

    def CollideCircleBatch(self, draw, cx_arr, cy_arr, r_arr, camera=None):
        px, py = self._mouse_point(camera)
        return draw.CollidePointCircleBatch(px, py, cx_arr, cy_arr, r_arr)

    def CollideEllipseBatch(self, draw, cx_arr, cy_arr, rx_arr, ry_arr, rotation_arr=0.0, camera=None):
        px, py = self._mouse_point(camera)
        return draw.CollidePointEllipseBatch(px, py, cx_arr, cy_arr, rx_arr, ry_arr, rotation_arr)

    def CollideTriangleBatch(self, draw, x1, y1, x2, y2, x3, y3, camera=None):
        px, py = self._mouse_point(camera)
        return draw.CollidePointTriangleBatch(px, py, x1, y1, x2, y2, x3, y3)

    def CollideTextureBatch(self, draw, name, x_arr, y_arr, w_arr=None, h_arr=None,
                            rotation_arr=0.0, flip_x=False, flip_y=False,
                            alpha_threshold=1, camera=None):
        """Pixel-perfect batch: N istanze della STESSA texture (tile
        picking, selezione unita' in un RTS...). Delega a
        DRAW.CollidePointTextureBatch — vedi PE_DRAW.py."""
        px, py = self._mouse_point(camera)
        return draw.CollidePointTextureBatch(px, py, name, x_arr, y_arr, w_arr, h_arr,
                                             rotation_arr, flip_x, flip_y, alpha_threshold)

    # ================================================================
    # Collisioni mouse per le forme che PE_DRAW disegna ma che finora
    # non avevano un adattatore qui: DrawLine(sBatch), DrawRoundedRect
    # (sBatch), DrawRoundedTriangle(sBatch), DrawBezierCurve(sBatch),
    # DrawText(Batch). Cosi' TUTTE le primitive di PE_DRAW sono
    # cliccabili/hoverabili dal mouse, non solo rect/rotrect/cerchio/
    # ellisse/triangolo/texture di prima.
    #
    # Stesso principio del resto del file: dove PE_DRAW espone gia' un
    # Collide* pronto (PointInLine) lo si usa; dove non esiste un test
    # "a punto singolo" per la forma (RoundedRect/RoundedTriangle non
    # hanno un CollidePointXxx in PE_DRAW), si usa CheckCollision/i
    # metodi di PE_COLLISION integrati in DRAW (RoundedRect) o si
    # replica ESATTAMENTE la stessa matematica del rendering GPU in
    # Python puro (RoundedTriangle, via _pe_rtri_shrink/_pe_sd_triangle
    # qui sopra) cosi' la hitbox combacia col disegno, arrotondamenti
    # inclusi — non solo un'approssimazione col rettangolo/triangolo
    # "spigoloso".
    # ================================================================

    # --- Linee -------------------------------------------------------------
    def CollideLine(self, draw, x1, y1, x2, y2, thickness=1.0, rotation=0.0, camera=None):
        px, py = self._mouse_point(camera)
        return draw.PointInLine(px, py, x1, y1, x2, y2, thickness=thickness, rotation=rotation)

    def CollideLineBatch(self, draw, x1_arr, y1_arr, x2_arr, y2_arr, thickness=1.0, camera=None):
        """Punto (mouse) contro N segmenti in un colpo solo (numpy)."""
        px, py = self._mouse_point(camera)
        x1 = _np.asarray(x1_arr, dtype='f8'); y1 = _np.asarray(y1_arr, dtype='f8')
        x2 = _np.asarray(x2_arr, dtype='f8'); y2 = _np.asarray(y2_arr, dtype='f8')
        dx = x2 - x1; dy = y2 - y1
        len_sq = dx * dx + dy * dy
        len_sq_safe = _np.where(len_sq > 1e-12, len_sq, 1.0)
        t = _np.clip(((px - x1) * dx + (py - y1) * dy) / len_sq_safe, 0.0, 1.0)
        near_x = x1 + t * dx; near_y = y1 + t * dy
        dist = _np.hypot(px - near_x, py - near_y)
        half_t = _np.asarray(thickness, dtype='f8') * 0.5
        return dist <= half_t

    # --- Rettangoli arrotondati ---------------------------------------------
    def CollideRoundedRect(self, draw, x, y, w, h, radius, rotation=0.0, camera=None):
        """Point-vs-RoundedRect diretto: nessun Point/RoundedRect temporaneo."""
        px, py = self._mouse_point(camera)
        return draw.CollidePointRoundedRect(px, py, x, y, w, h, radius, rotation)

    def CollideRoundedRectBatch(self, draw, x_arr, y_arr, w_arr, h_arr, radius_arr,
                                rotation_arr=0.0, camera=None):
        """Punto (mouse) contro N rettangoli arrotondati in un colpo solo:
        stessa identica SDF (sdRoundedBox) del fragment shader di
        DrawRoundedRectsBatch, vettorizzata in numpy."""
        px, py = self._mouse_point(camera)
        x = _np.asarray(x_arr, dtype='f8'); y = _np.asarray(y_arr, dtype='f8')
        w = _np.asarray(w_arr, dtype='f8'); h = _np.asarray(h_arr, dtype='f8')
        n = x.shape[0]
        rad = _np.asarray(radius_arr, dtype='f8').reshape(-1)
        if rad.size == 1:
            rad = _np.full(n, rad[0])
        rot = _np.asarray(rotation_arr, dtype='f8').reshape(-1)
        if rot.size == 1:
            rot = _np.full(n, rot[0])

        cx = x + w * 0.5; cy = y + h * 0.5
        ang = -_np.radians(rot)
        cs = _np.cos(ang); sn = _np.sin(ang)
        dx = px - cx; dy = py - cy
        lx = dx * cs - dy * sn
        ly = dx * sn + dy * cs

        hw = w * 0.5; hh = h * 0.5
        r = _np.minimum(rad, _np.minimum(hw, hh))
        qx = _np.abs(lx) - hw + r
        qy = _np.abs(ly) - hh + r
        d = (_np.minimum(_np.maximum(qx, qy), 0.0)
             + _np.hypot(_np.maximum(qx, 0.0), _np.maximum(qy, 0.0)) - r)
        return d <= 0.0

    # --- Triangoli arrotondati -----------------------------------------------
    def CollideRoundedTriangle(self, draw, x1, y1, x2, y2, x3, y3, radius,
                               rotation=0.0, camera=None):
        """Point-vs-RoundedTriangle ESATTO: stessa matematica (shrink dei
        lati + SDF triangolo) usata dal fragment shader di
        DrawRoundedTriangle(sBatch), riportata in Python puro."""
        px, py = self._mouse_point(camera)
        if rotation != 0.0:
            cx = (x1 + x2 + x3) / 3.0; cy = (y1 + y2 + y3) / 3.0
            ang = rotation * 0.017453292519943295
            cs = _math.cos(ang); sn = _math.sin(ang)
            dx1 = x1 - cx; dy1 = y1 - cy
            dx2 = x2 - cx; dy2 = y2 - cy
            dx3 = x3 - cx; dy3 = y3 - cy
            x1 = cx + dx1 * cs - dy1 * sn; y1 = cy + dx1 * sn + dy1 * cs
            x2 = cx + dx2 * cs - dy2 * sn; y2 = cy + dx2 * sn + dy2 * cs
            x3 = cx + dx3 * cs - dy3 * sn; y3 = cy + dx3 * sn + dy3 * cs
        sax, say, sbx, sby, scx, scy, r_eff = _pe_rtri_shrink(x1, y1, x2, y2, x3, y3, radius)
        sd = _pe_sd_triangle(px, py, sax, say, sbx, sby, scx, scy) - r_eff
        return sd <= 0.0

    def CollideRoundedTriangleBatch(self, draw, vertices, radius, camera=None):
        """Punto (mouse) contro N triangoli arrotondati. `vertices` come in
        DrawRoundedTrianglesBatch: (n,3,2) oppure (3*n,2). Nessun parametro
        di rotazione (DrawRoundedTrianglesBatch non ne ha)."""
        px, py = self._mouse_point(camera)
        verts = _np.asarray(vertices, dtype='f8')
        if verts.ndim == 2:
            n = verts.shape[0] // 3
            verts = verts.reshape(n, 3, 2)
        n = verts.shape[0]
        rad = _np.asarray(radius, dtype='f8').reshape(-1)
        if rad.size == 1:
            rad = _np.full(n, rad[0])
        out = _np.zeros(n, dtype=bool)
        for i in range(n):
            ax, ay = verts[i, 0]; bx, by = verts[i, 1]; cx, cy = verts[i, 2]
            sax, say, sbx, sby, scx, scy, r_eff = _pe_rtri_shrink(
                float(ax), float(ay), float(bx), float(by), float(cx), float(cy), float(rad[i]))
            sd = _pe_sd_triangle(px, py, sax, say, sbx, sby, scx, scy) - r_eff
            out[i] = sd <= 0.0
        return out

    # --- Curve di Bezier -----------------------------------------------------
    def CollideBezierCurve(self, draw, p0, p1, p2, thickness=2.0, segments=None, camera=None):
        """Campiona la curva quadratica in `segments` punti e testa il mouse
        contro ogni segmento con draw.PointInLine — stessa tassellazione
        usata da DrawBezierCurve per il rendering."""
        px, py = self._mouse_point(camera)
        if segments is None:
            segments = 24
        x0, y0 = p0; x1, y1 = p1; x2, y2 = p2
        prev_x, prev_y = x0, y0
        for i in range(1, segments + 1):
            t = i / segments
            u = 1.0 - t
            bx = u * u * x0 + 2.0 * u * t * x1 + t * t * x2
            by = u * u * y0 + 2.0 * u * t * y1 + t * t * y2
            if draw.PointInLine(px, py, prev_x, prev_y, bx, by, thickness=thickness):
                return True
            prev_x, prev_y = bx, by
        return False

    def CollideBezierCurveBatch(self, draw, p0s, p1s, p2s, thickness=2.0, segments=None, camera=None):
        """Punto (mouse) contro N curve di Bezier quadratiche, vettorizzato:
        stessa tassellazione di DrawBezierCurvesBatch, poi distanza
        punto-segmento minima per curva confrontata con lo spessore."""
        px, py = self._mouse_point(camera)
        if segments is None:
            segments = 24
        p0 = _np.asarray(p0s, dtype='f8'); p1 = _np.asarray(p1s, dtype='f8'); p2 = _np.asarray(p2s, dtype='f8')
        t = _np.linspace(0.0, 1.0, segments + 1)[_np.newaxis, :, _np.newaxis]
        u = 1.0 - t
        pts = (u * u) * p0[:, _np.newaxis, :] + (2.0 * u * t) * p1[:, _np.newaxis, :] + (t * t) * p2[:, _np.newaxis, :]
        starts = pts[:, :-1, :]; ends = pts[:, 1:, :]
        dx = ends[..., 0] - starts[..., 0]; dy = ends[..., 1] - starts[..., 1]
        len_sq = dx * dx + dy * dy
        len_sq_safe = _np.where(len_sq > 1e-12, len_sq, 1.0)
        tt = _np.clip(((px - starts[..., 0]) * dx + (py - starts[..., 1]) * dy) / len_sq_safe, 0.0, 1.0)
        near_x = starts[..., 0] + tt * dx; near_y = starts[..., 1] + tt * dy
        dist = _np.hypot(px - near_x, py - near_y)
        min_dist = dist.min(axis=1)
        half_t = _np.asarray(thickness, dtype='f8') * 0.5
        return min_dist <= half_t

    # --- Testo ---------------------------------------------------------------
    def CollideText(self, draw, text, x, y, font="arial", size=24, rotation=0.0, camera=None):
        """Point-vs-Text diretto: riusa il layout di DRAW senza wrapper shape."""
        px, py = self._mouse_point(camera)
        return draw.CollidePointText(px, py, text, x, y, font, size, rotation)


# --- EVENTI TASTIERA ---
PE_KEYDOWN = "KEYDOWN"  # Tasto appena premuto
PE_KEYUP   = "KEYUP"    # Tasto appena rilasciato

# --- TUTTI I TASTI DELLA TASTIERA ---
# Alfabeto
PE_K_a = _sdl2.SDLK_a; PE_K_b = _sdl2.SDLK_b; PE_K_c = _sdl2.SDLK_c; PE_K_d = _sdl2.SDLK_d
PE_K_e = _sdl2.SDLK_e; PE_K_f = _sdl2.SDLK_f; PE_K_g = _sdl2.SDLK_g; PE_K_h = _sdl2.SDLK_h
PE_K_i = _sdl2.SDLK_i; PE_K_j = _sdl2.SDLK_j; PE_K_k = _sdl2.SDLK_k; PE_K_l = _sdl2.SDLK_l
PE_K_m = _sdl2.SDLK_m; PE_K_n = _sdl2.SDLK_n; PE_K_o = _sdl2.SDLK_o; PE_K_p = _sdl2.SDLK_p
PE_K_q = _sdl2.SDLK_q; PE_K_r = _sdl2.SDLK_r; PE_K_s = _sdl2.SDLK_s; PE_K_t = _sdl2.SDLK_t
PE_K_u = _sdl2.SDLK_u; PE_K_v = _sdl2.SDLK_v; PE_K_w = _sdl2.SDLK_w; PE_K_x = _sdl2.SDLK_x
PE_K_y = _sdl2.SDLK_y; PE_K_z = _sdl2.SDLK_z

# Numeri
PE_K_0 = _sdl2.SDLK_0; PE_K_1 = _sdl2.SDLK_1; PE_K_2 = _sdl2.SDLK_2; PE_K_3 = _sdl2.SDLK_3
PE_K_4 = _sdl2.SDLK_4; PE_K_5 = _sdl2.SDLK_5; PE_K_6 = _sdl2.SDLK_6; PE_K_7 = _sdl2.SDLK_7
PE_K_8 = _sdl2.SDLK_8; PE_K_9 = _sdl2.SDLK_9

# Tasti Direzionali
PE_K_UP    = _sdl2.SDLK_UP
PE_K_DOWN  = _sdl2.SDLK_DOWN
PE_K_LEFT  = _sdl2.SDLK_LEFT
PE_K_RIGHT = _sdl2.SDLK_RIGHT

# Tasti di Azione e Controllo
PE_K_SPACE     = _sdl2.SDLK_SPACE
PE_K_ESCAPE    = _sdl2.SDLK_ESCAPE
PE_K_RETURN    = _sdl2.SDLK_RETURN     # Tasto Invio
PE_K_BACKSPACE = _sdl2.SDLK_BACKSPACE
PE_K_TAB       = _sdl2.SDLK_TAB
PE_K_INSERT    = _sdl2.SDLK_INSERT
PE_K_DELETE    = _sdl2.SDLK_DELETE
PE_K_HOME      = _sdl2.SDLK_HOME
PE_K_END       = _sdl2.SDLK_END
PE_K_PAGEUP    = _sdl2.SDLK_PAGEUP
PE_K_PAGEDOWN  = _sdl2.SDLK_PAGEDOWN

# Lock / sistema (FIX 12: costanti mancanti)
PE_K_CAPSLOCK    = _sdl2.SDLK_CAPSLOCK
PE_K_NUMLOCK     = _sdl2.SDLK_NUMLOCKCLEAR
PE_K_SCROLLLOCK  = _sdl2.SDLK_SCROLLLOCK
PE_K_PRINTSCREEN = _sdl2.SDLK_PRINTSCREEN
PE_K_PRINT       = _sdl2.SDLK_PRINTSCREEN  # alias
PE_K_PAUSE       = _sdl2.SDLK_PAUSE
PE_K_LSUPER      = _sdl2.SDLK_LGUI          # tasto Windows / Command sinistro
PE_K_RSUPER      = _sdl2.SDLK_RGUI          # tasto Windows / Command destro
PE_K_MENU        = _sdl2.SDLK_MENU

# Modificatori (Shift, Ctrl, Alt)
PE_K_LCTRL  = _sdl2.SDLK_LCTRL
PE_K_RCTRL  = _sdl2.SDLK_RCTRL
PE_K_LSHIFT = _sdl2.SDLK_LSHIFT
PE_K_RSHIFT = _sdl2.SDLK_RSHIFT
PE_K_LALT   = _sdl2.SDLK_LALT
PE_K_RALT   = _sdl2.SDLK_RALT

# Tasti Funzione
PE_K_F1 = _sdl2.SDLK_F1; PE_K_F2 = _sdl2.SDLK_F2; PE_K_F3 = _sdl2.SDLK_F3; PE_K_F4 = _sdl2.SDLK_F4
PE_K_F5 = _sdl2.SDLK_F5; PE_K_F6 = _sdl2.SDLK_F6; PE_K_F7 = _sdl2.SDLK_F7; PE_K_F8 = _sdl2.SDLK_F8
PE_K_F9 = _sdl2.SDLK_F9; PE_K_F10 = _sdl2.SDLK_F10; PE_K_F11 = _sdl2.SDLK_F11; PE_K_F12 = _sdl2.SDLK_F12

# Simboli
PE_K_MINUS = _sdl2.SDLK_MINUS; PE_K_EQUALS = _sdl2.SDLK_EQUALS
PE_K_LEFTBRACKET = _sdl2.SDLK_LEFTBRACKET; PE_K_RIGHTBRACKET = _sdl2.SDLK_RIGHTBRACKET
PE_K_BACKSLASH = _sdl2.SDLK_BACKSLASH; PE_K_SEMICOLON = _sdl2.SDLK_SEMICOLON
PE_K_QUOTE = _sdl2.SDLK_QUOTE; PE_K_BACKQUOTE = _sdl2.SDLK_BACKQUOTE
PE_K_COMMA = _sdl2.SDLK_COMMA; PE_K_PERIOD = _sdl2.SDLK_PERIOD; PE_K_SLASH = _sdl2.SDLK_SLASH

# Tastierino Numerico (Numpad)
PE_K_KP_0 = _sdl2.SDLK_KP_0; PE_K_KP_1 = _sdl2.SDLK_KP_1; PE_K_KP_2 = _sdl2.SDLK_KP_2
PE_K_KP_3 = _sdl2.SDLK_KP_3; PE_K_KP_4 = _sdl2.SDLK_KP_4; PE_K_KP_5 = _sdl2.SDLK_KP_5
PE_K_KP_6 = _sdl2.SDLK_KP_6; PE_K_KP_7 = _sdl2.SDLK_KP_7; PE_K_KP_8 = _sdl2.SDLK_KP_8
PE_K_KP_9 = _sdl2.SDLK_KP_9
PE_K_KP_DIVIDE = _sdl2.SDLK_KP_DIVIDE; PE_K_KP_MULTIPLY = _sdl2.SDLK_KP_MULTIPLY
PE_K_KP_MINUS = _sdl2.SDLK_KP_MINUS; PE_K_KP_PLUS = _sdl2.SDLK_KP_PLUS
PE_K_KP_ENTER = _sdl2.SDLK_KP_ENTER; PE_K_KP_PERIOD = _sdl2.SDLK_KP_PERIOD


# --- EVENTI MOUSE ---
PE_MOUSEMOTION     = "MOUSEMOTION"
PE_MOUSEBUTTONDOWN = "MOUSEBUTTONDOWN"
PE_MOUSEBUTTONUP   = "MOUSEBUTTONUP"
# NOTA (bug 8 documentale): PE_MOUSEDRAG viene emesso al posto di
# PE_MOUSEMOTION quando almeno un pulsante e' premuto durante il movimento.
# I due eventi sono MUTUALMENTE ESCLUSIVI per frame: se serve tracciare
# sempre la posizione del cursore, ascoltare ENTRAMBI gli eventi.
PE_MOUSEDRAG  = "MOUSEDRAG"
PE_MOUSEWHEEL = "MOUSEWHEEL"

# --- PULSANTI MOUSE ---
PE_MOUSE_LEFT   = _sdl2.SDL_BUTTON_LEFT
PE_MOUSE_MIDDLE = _sdl2.SDL_BUTTON_MIDDLE
PE_MOUSE_RIGHT  = _sdl2.SDL_BUTTON_RIGHT
PE_MOUSE_X1     = _sdl2.SDL_BUTTON_X1   # Tasto laterale 1 (Indietro)
PE_MOUSE_X2     = _sdl2.SDL_BUTTON_X2   # Tasto laterale 2 (Avanti)


# FIX 7: __all__ esplicito. Senza questo, `from .PE_KEYS import *` (usato in
# __init__.py) porterebbe nel namespace del package anche il modulo `sdl2`
# (rinominato qui _sdl2 come difesa aggiuntiva). Ora l'export e' controllato.
__all__ = [
    "PE_Event",
    # Eventi tastiera
    "PE_KEYDOWN", "PE_KEYUP",
    # Alfabeto
    "PE_K_a", "PE_K_b", "PE_K_c", "PE_K_d", "PE_K_e", "PE_K_f", "PE_K_g",
    "PE_K_h", "PE_K_i", "PE_K_j", "PE_K_k", "PE_K_l", "PE_K_m", "PE_K_n",
    "PE_K_o", "PE_K_p", "PE_K_q", "PE_K_r", "PE_K_s", "PE_K_t", "PE_K_u",
    "PE_K_v", "PE_K_w", "PE_K_x", "PE_K_y", "PE_K_z",
    # Numeri
    "PE_K_0", "PE_K_1", "PE_K_2", "PE_K_3", "PE_K_4",
    "PE_K_5", "PE_K_6", "PE_K_7", "PE_K_8", "PE_K_9",
    # Direzionali
    "PE_K_UP", "PE_K_DOWN", "PE_K_LEFT", "PE_K_RIGHT",
    # Azione / controllo
    "PE_K_SPACE", "PE_K_ESCAPE", "PE_K_RETURN", "PE_K_BACKSPACE",
    "PE_K_TAB", "PE_K_INSERT", "PE_K_DELETE",
    "PE_K_HOME", "PE_K_END", "PE_K_PAGEUP", "PE_K_PAGEDOWN",
    # Lock / sistema
    "PE_K_CAPSLOCK", "PE_K_NUMLOCK", "PE_K_SCROLLLOCK",
    "PE_K_PRINTSCREEN", "PE_K_PRINT", "PE_K_PAUSE",
    "PE_K_LSUPER", "PE_K_RSUPER", "PE_K_MENU",
    # Modificatori
    "PE_K_LCTRL", "PE_K_RCTRL", "PE_K_LSHIFT", "PE_K_RSHIFT",
    "PE_K_LALT", "PE_K_RALT",
    # Funzione
    "PE_K_F1", "PE_K_F2", "PE_K_F3", "PE_K_F4", "PE_K_F5", "PE_K_F6",
    "PE_K_F7", "PE_K_F8", "PE_K_F9", "PE_K_F10", "PE_K_F11", "PE_K_F12",
    # Simboli
    "PE_K_MINUS", "PE_K_EQUALS", "PE_K_LEFTBRACKET", "PE_K_RIGHTBRACKET",
    "PE_K_BACKSLASH", "PE_K_SEMICOLON", "PE_K_QUOTE", "PE_K_BACKQUOTE",
    "PE_K_COMMA", "PE_K_PERIOD", "PE_K_SLASH",
    # Numpad
    "PE_K_KP_0", "PE_K_KP_1", "PE_K_KP_2", "PE_K_KP_3", "PE_K_KP_4",
    "PE_K_KP_5", "PE_K_KP_6", "PE_K_KP_7", "PE_K_KP_8", "PE_K_KP_9",
    "PE_K_KP_DIVIDE", "PE_K_KP_MULTIPLY", "PE_K_KP_MINUS",
    "PE_K_KP_PLUS", "PE_K_KP_ENTER", "PE_K_KP_PERIOD",
    # Eventi mouse
    "PE_MOUSEMOTION", "PE_MOUSEBUTTONDOWN", "PE_MOUSEBUTTONUP",
    "PE_MOUSEDRAG", "PE_MOUSEWHEEL",
    # Pulsanti mouse
    "PE_MOUSE_LEFT", "PE_MOUSE_MIDDLE", "PE_MOUSE_RIGHT",
    "PE_MOUSE_X1", "PE_MOUSE_X2",
]