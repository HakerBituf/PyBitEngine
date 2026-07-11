import sdl2, moderngl, time
import sdl2.sdlimage as img
import os, numpy as np, math
import ctypes
import warnings
from collections import OrderedDict
from numba import njit, prange

# ---------------------------------------------------------------------- #
# PE_TEXT — costanti precomputate riusate da tutto il modulo
# ---------------------------------------------------------------------- #
_INV_255           = 1.0 / 255.0
_DEG2RAD_CONST     = 0.017453292519943295
_RAD2DEG_CONST     = 57.29577951308232
_TAU_CONST         = 6.283185307179586
_HALF              = 0.5
_MAX_FONT_CACHE    = 64
_MAX_GLYPH_CACHE   = 8192

try:
    from PIL import Image, ImageFont, ImageDraw
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------- #
# NUMBA KERNELS aggiuntivi (batch primitivi + testo)
# ---------------------------------------------------------------------- #
@njit(fastmath=True, parallel=True, cache=True)
def _numba_pack_rect_instances(pos, size, cos_a, sin_a, rgba, out):
    """Impacchetta (N,10) instance buffer per DrawRectsBatch: pos2+size2+dir2+rgba4."""
    n = pos.shape[0]
    for i in prange(n):
        out[i, 0] = pos[i, 0]
        out[i, 1] = pos[i, 1]
        out[i, 2] = size[i, 0]
        out[i, 3] = size[i, 1]
        out[i, 4] = cos_a[i]
        out[i, 5] = sin_a[i]
        out[i, 6] = rgba[i, 0]
        out[i, 7] = rgba[i, 1]
        out[i, 8] = rgba[i, 2]
        out[i, 9] = rgba[i, 3]


@njit(fastmath=True, parallel=True, cache=True)
def _numba_pack_ellipse_instances(centers, radii, cos_a, sin_a, rgba, out):
    n = centers.shape[0]
    for i in prange(n):
        out[i, 0] = centers[i, 0]
        out[i, 1] = centers[i, 1]
        out[i, 2] = radii[i, 0]
        out[i, 3] = radii[i, 1]
        out[i, 4] = cos_a[i]
        out[i, 5] = sin_a[i]
        out[i, 6] = rgba[i, 0]
        out[i, 7] = rgba[i, 1]
        out[i, 8] = rgba[i, 2]
        out[i, 9] = rgba[i, 3]


@njit(fastmath=True, parallel=True, cache=True)
def _numba_pack_ellipse_outline_instances(centers, radii, thickness, cos_a, sin_a, rgba, out):
    """Impacchetta (N,11) instance buffer per DrawEllipsesOutlineBatch/
    DrawCircleOutlineBatch: center2+radius2+thickness1+dir2+rgba4.
    Stessa architettura/prestazioni di _numba_pack_ellipse_instances
    (kernel Numba parallelo, zero overhead Python per istanza)."""
    n = centers.shape[0]
    for i in prange(n):
        out[i, 0] = centers[i, 0]
        out[i, 1] = centers[i, 1]
        out[i, 2] = radii[i, 0]
        out[i, 3] = radii[i, 1]
        out[i, 4] = thickness[i]
        out[i, 5] = cos_a[i]
        out[i, 6] = sin_a[i]
        out[i, 7] = rgba[i, 0]
        out[i, 8] = rgba[i, 1]
        out[i, 9] = rgba[i, 2]
        out[i, 10] = rgba[i, 3]


@njit(fastmath=True, parallel=True, cache=True)
def _numba_pack_rect_outline_instances(pos, size, thickness, cos_a, sin_a, rgba, out):
    """Impacchetta (N,11) instance buffer per DrawRectsOutlineBatch:
    pos2+size2+thickness1+dir2+rgba4. Stessa architettura/prestazioni di
    _numba_pack_rect_instances."""
    n = pos.shape[0]
    for i in prange(n):
        out[i, 0] = pos[i, 0]
        out[i, 1] = pos[i, 1]
        out[i, 2] = size[i, 0]
        out[i, 3] = size[i, 1]
        out[i, 4] = thickness[i]
        out[i, 5] = cos_a[i]
        out[i, 6] = sin_a[i]
        out[i, 7] = rgba[i, 0]
        out[i, 8] = rgba[i, 1]
        out[i, 9] = rgba[i, 2]
        out[i, 10] = rgba[i, 3]


@njit(fastmath=True, parallel=True, cache=True)
def _numba_pack_line_instances(p1, p2, thick, rgba, out):
    n = p1.shape[0]
    for i in prange(n):
        out[i, 0] = p1[i, 0]
        out[i, 1] = p1[i, 1]
        out[i, 2] = p2[i, 0]
        out[i, 3] = p2[i, 1]
        out[i, 4] = thick[i]
        out[i, 5] = rgba[i, 0]
        out[i, 6] = rgba[i, 1]
        out[i, 7] = rgba[i, 2]
        out[i, 8] = rgba[i, 3]


@njit(fastmath=True, parallel=True, cache=True)
def _numba_pack_tri_instances(v0, v1, v2, rgba, out):
    n = v0.shape[0]
    for i in prange(n):
        out[i, 0] = v0[i, 0]; out[i, 1] = v0[i, 1]
        out[i, 2] = v1[i, 0]; out[i, 3] = v1[i, 1]
        out[i, 4] = v2[i, 0]; out[i, 5] = v2[i, 1]
        out[i, 6] = rgba[i, 0]; out[i, 7] = rgba[i, 1]
        out[i, 8] = rgba[i, 2]; out[i, 9] = rgba[i, 3]


@njit(fastmath=True, parallel=True, cache=True)
def _numba_rotate_lines(x1, y1, x2, y2, cs, sn):
    """Ruota N segmenti attorno al proprio punto medio (in-place style)."""
    n = x1.shape[0]
    for i in prange(n):
        mx = (x1[i] + x2[i]) * 0.5
        my = (y1[i] + y2[i]) * 0.5
        dx1 = x1[i] - mx; dy1 = y1[i] - my
        dx2 = x2[i] - mx; dy2 = y2[i] - my
        x1[i] = mx + dx1*cs - dy1*sn
        y1[i] = my + dx1*sn + dy1*cs
        x2[i] = mx + dx2*cs - dy2*sn
        y2[i] = my + dx2*sn + dy2*cs

# INCOERENZA FIX 10 + PERF FIX 18: rotate-per-line con angoli per-segmento e
# pack che accetta x/y separati (niente column_stack di appoggio).
@njit(fastmath=True, parallel=True, cache=True)
def _numba_rotate_lines_arr(x1, y1, x2, y2, cs_arr, sn_arr):
    """Ruota N segmenti attorno al proprio punto medio, angolo per segmento."""
    n = x1.shape[0]
    for i in prange(n):
        mx = (x1[i] + x2[i]) * 0.5
        my = (y1[i] + y2[i]) * 0.5
        dx1 = x1[i] - mx; dy1 = y1[i] - my
        dx2 = x2[i] - mx; dy2 = y2[i] - my
        cs = cs_arr[i]; sn = sn_arr[i]
        x1[i] = mx + dx1*cs - dy1*sn
        y1[i] = my + dx1*sn + dy1*cs
        x2[i] = mx + dx2*cs - dy2*sn
        y2[i] = my + dx2*sn + dy2*cs


@njit(fastmath=True, parallel=True, cache=True)
def _numba_pack_line_instances_xy(x1, y1, x2, y2, thick, rgba, out):
    """Come _numba_pack_line_instances ma legge x/y da 4 array 1D
    (evita np.column_stack: risparmia 2 allocazioni (N,2) per chiamata)."""
    n = x1.shape[0]
    for i in prange(n):
        out[i, 0] = x1[i]
        out[i, 1] = y1[i]
        out[i, 2] = x2[i]
        out[i, 3] = y2[i]
        out[i, 4] = thick[i]
        out[i, 5] = rgba[i, 0]
        out[i, 6] = rgba[i, 1]
        out[i, 7] = rgba[i, 2]
        out[i, 8] = rgba[i, 3]



@njit(fastmath=True, parallel=True, cache=True)
def _numba_clip_rgba(rgba_in, alpha_scalar, use_alpha_scalar, alpha_arr, out):
    """
    Normalizza colori: clip 0-255 e sovrascrittura alpha.
    use_alpha_scalar=1 usa alpha_scalar; altrimenti usa alpha_arr (len=n).
    """
    n = rgba_in.shape[0]
    for i in prange(n):
        r = rgba_in[i, 0]
        g = rgba_in[i, 1]
        b = rgba_in[i, 2]
        a = rgba_in[i, 3]
        if r < 0.0: r = 0.0
        elif r > 255.0: r = 255.0
        if g < 0.0: g = 0.0
        elif g > 255.0: g = 255.0
        if b < 0.0: b = 0.0
        elif b > 255.0: b = 255.0
        if use_alpha_scalar == 1:
            a = alpha_scalar
        elif use_alpha_scalar == 2:
            a = alpha_arr[i]
        if a < 0.0: a = 0.0
        elif a > 255.0: a = 255.0
        out[i, 0] = r
        out[i, 1] = g
        out[i, 2] = b
        out[i, 3] = a


@njit(fastmath=True, parallel=True, cache=True)
def _numba_layout_glyphs(gx, gy, gw, gh, guv,
                         origin_x, origin_y, cos_r, sin_r,
                         alpha_final, out):
    """Layout+rotazione+packing dei glifi per DrawTextBatch.
    Output (N,11): pos2+size2+dir2+uv4+alpha1 — stesso layout dello sprite batch."""
    n = gx.shape[0]
    for i in prange(n):
        lx = gx[i]
        ly = gy[i]
        wx = origin_x + lx * cos_r - ly * sin_r
        wy = origin_y + lx * sin_r + ly * cos_r
        out[i, 0]  = wx
        out[i, 1]  = wy
        out[i, 2]  = gw[i]
        out[i, 3]  = gh[i]
        out[i, 4]  = cos_r
        out[i, 5]  = sin_r
        out[i, 6]  = guv[i, 0]
        out[i, 7]  = guv[i, 1]
        out[i, 8]  = guv[i, 2]
        out[i, 9]  = guv[i, 3]
        out[i, 10] = alpha_final


# ---------------------------------------------------------------------- #
# FONT MANAGER + GLYPH ATLAS (usati dai metodi DrawText* di DRAW)
# ---------------------------------------------------------------------- #
class FontManager:
    """Cache LRU di font PIL indicizzata per (family_or_path, size)."""

    _SYSTEM_DIRS = [
        "/usr/share/fonts", "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        "C:/Windows/Fonts",
        "/System/Library/Fonts", "/Library/Fonts",
    ]

    def __init__(self, capacity=_MAX_FONT_CACHE):
        if not _HAS_PIL:
            raise RuntimeError("PE_DRAW.DrawText richiede Pillow (pip install pillow)")
        self._cache = OrderedDict()
        self._capacity = capacity
        self._registered = {}
        self._resolve_cache = {}
        self._font_file_index = {}
        self._font_dirs = [base for base in self._SYSTEM_DIRS if os.path.isdir(base)]

    def register(self, alias: str, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Font non trovato: {path}")
        self._registered[alias] = path
        self._resolve_cache[alias.lower()] = path

    def get(self, family: str, size: int):
        if size <= 0:
            raise ValueError("size deve essere > 0")
        key = (family, int(size))
        c = self._cache
        if key in c:
            c.move_to_end(key)
            return c[key]
        path = self._resolve(family)
        try:
            font = ImageFont.truetype(path, size)
        except Exception as e:
            raise RuntimeError(f"Impossibile caricare font {family!r}: {e}")
        c[key] = font
        c.move_to_end(key)
        while len(c) > self._capacity:
            c.popitem(last=False)
        return font

    def _build_font_file_index(self):
        index = {}
        for base in self._font_dirs:
            if base in index:
                continue
            font_files = []
            for root, _, files in os.walk(base):
                for file_name in files:
                    lowered = file_name.lower()
                    if lowered.endswith((".ttf", ".otf")):
                        font_files.append(os.path.join(root, file_name))
            index[base] = font_files
        self._font_file_index = index
        return index

    def _resolve(self, family: str) -> str:
        normalized = family.lower()
        cached = self._resolve_cache.get(normalized)
        if cached is not None:
            return cached

        if family in self._registered:
            resolved = self._registered[family]
            self._resolve_cache[normalized] = resolved
            return resolved
        if os.path.isfile(family):
            self._resolve_cache[normalized] = family
            return family

        if not self._font_file_index:
            self._build_font_file_index()

        for base in self._font_dirs:
            font_files = self._font_file_index.get(base, [])
            for path in font_files:
                file_name = os.path.basename(path).lower()
                if normalized in file_name:
                    self._resolve_cache[normalized] = path
                    return path

        default = ImageFont.load_default()
        resolved = getattr(default, "path", family)
        self._resolve_cache[normalized] = resolved
        return resolved


class _GlyphAtlas:
    """On-demand rasterizer che riusa self.atlas di DRAW per i glifi."""

    def __init__(self, draw_ref, font_mgr, capacity=_MAX_GLYPH_CACHE):
        self._draw = draw_ref
        self._fonts = font_mgr
        self._entries = OrderedDict()
        self._capacity = capacity
        # BUG FIX (bug 2): quando TextureAtlas si espande, tutte le UV
        # memorizzate qui diventano stale (le UV normalizzate cambiano al
        # cambiare della dimensione dell'atlas). Traccia la dimensione
        # corrente e invalida la cache al cambio.
        self._last_atlas_size = None

    def get(self, family, size, ch):
        # BUG FIX (bug 2): invalida cache se l'atlas si e' ridimensionato.
        atlas = getattr(self._draw, "atlas", None)
        current_size = getattr(atlas, "size", None) if atlas is not None else None
        if current_size != self._last_atlas_size:
            self._entries.clear()
            self._last_atlas_size = current_size

        key = (family, int(size), ch)
        e = self._entries.get(key)
        if e is not None:
            self._entries.move_to_end(key)
            return e

        font = self._fonts.get(family, size)
        try:
            l, t, r, b = font.getbbox(ch)
        except Exception:
            l, t, r, b = 0, 0, size, size
        gw = max(1, r - l)
        gh = max(1, b - t)
        try:
            adv = float(font.getlength(ch))
        except Exception:
            adv = float(gw)

        img = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((-l, -t), ch, font=font, fill=(255, 255, 255, 255))
        arr = np.asarray(img, dtype=np.uint8)

        atlas_name = f"__glyph__|{family}|{size}|{ord(ch)}"
        # BUG FIX 4: se atlas.add fallisce (atlas pieno, MemoryError, ecc.)
        # NON inserire un'entry con UV degeneri (0,0,0,0) nella cache: quel
        # glifo diventerebbe invisibile per sempre senza alcun messaggio
        # d'errore, e la cache impedirebbe qualunque retry successivo.
        # Preferiamo propagare il fallimento in modo esplicito con warning e
        # NON popolare _entries: il chiamante puo' scegliere di riprovare
        # dopo aver liberato spazio o ampliato l'atlas.
        try:
            self._draw.atlas.add(atlas_name, arr, gw, gh)
            u0, v0, u1, v1 = self._draw.atlas.get_uv(atlas_name)
        except (ValueError, RuntimeError, MemoryError) as exc:
            warnings.warn(
                f"_GlyphAtlas: impossibile inserire il glifo {ch!r} "
                f"(family={family!r}, size={size}) nell'atlas: {exc}. "
                "Glifo saltato (nessun caching di UV degeneri).",
                RuntimeWarning,
                stacklevel=2,
            )
            # Ritorna un'entry "vuota" NON in cache: gli offset locali
            # saranno zero e advance = 0, cosi' il layout continua senza
            # rompersi ma il glifo non lascia buchi persistenti.
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(adv), 0.0, 0.0)

        entry = (float(u0), float(v0), float(u1), float(v1),
                 float(gw), float(gh), adv, float(l), float(t))
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
        return entry



# BUG FIX (bug 7): _numba_prepare_rects rimosso (dead code, mai chiamato;
# _numba_pack_rect_instances lo sostituisce).

class TextureAtlas:
    def __init__(self, ctx, initial_size=4096, max_size=16384):
        if initial_size <= 0:
            raise ValueError("TextureAtlas initial_size must be > 0")
        if max_size < initial_size:
            raise ValueError("max_size must be >= initial_size")
        self.ctx = ctx
        self.size = initial_size
        self.max_size = max_size
        self.tex = ctx.texture((self.size, self.size), 4)
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # --- Allocatore MaxRects ---
        self.free_rects = [(0, 0, self.size, self.size)]   # (x, y, w, h)
        self.uv_map = {}          # name -> (u0, v0, u1, v1)
        self.texture_data = {}    # name -> (numpy_array, w, h)  per poter ricostruire l'atlas
        self.name_to_rect = {}    # name -> (x, y, w, h)  occupato nell'atlas (fix bug 1/2)

    def add(self, name, img_data, w, h):
        """Aggiunge un'immagine all'atlas. Se non c'è spazio, espande l'atlas."""
        # --- Validazioni input ---

        # Fix bug 6: nomi vuoti/None venivano accettati silenziosamente
        # (il dizionario non si lamenta), producendo poi UV "anomale" in
        # get_uv(""), get_uv(None), ecc. Una libreria pubblica deve
        # rifiutare questi input subito, al momento dell'inserimento.
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError(f"name must be a non-empty string, got {name!r}")

        if w <= 0 or h <= 0:
            raise ValueError("w and h must be > 0")
        if w > self.max_size or h > self.max_size:
            raise ValueError(f"Image {w}x{h} bigger than max atlas size {self.max_size}")

        # Fix bug 3: con immagini enormi (es. 30000x30000) np.asarray può
        # esaurire la RAM molto prima di qualunque altro controllo. Non si
        # può evitare l'allocazione, ma si può intercettare il MemoryError
        # e restituire un messaggio chiaro invece di un crash "muto".
        try:
            img_arr = np.asarray(img_data, dtype=np.uint8)
        except MemoryError:
            raise MemoryError(
                f"Not enough memory to allocate a {w}x{h} RGBA image "
                f"(~{(w * h * 4) / (1024 ** 2):.1f} MB requested)"
            ) from None

        if img_arr.ndim != 3 or img_arr.shape[0] != h or img_arr.shape[1] != w or img_arr.shape[2] != 4:
            raise ValueError(f"img_data must have shape ({h},{w},4), got {img_arr.shape}")

        # Fix bug 1/2: se il nome esiste già, la vecchia regione dell'atlas
        # NON veniva mai liberata (space leak: occupa spazio per sempre, e
        # in più _expand_atlas() ricompattava solo l'ultima versione,
        # ignorando i rettangoli "fantasma" lasciati dalle texture vecchie).
        # Ora liberiamo esplicitamente il rettangolo precedente prima di
        # reinserire la nuova texture con lo stesso nome.
        if name in self.uv_map:
            self._remove(name)

        # Prova ad allocare con MaxRects
        rect = self._find_rect(w, h)
        if rect is None:
            # Se fallisce, espandi l'atlas e riprova
            self._expand_atlas()
            rect = self._find_rect(w, h)
            if rect is None:
                raise RuntimeError(f"Unable to insert {w}x{h} even after expanding atlas")

        x, y, rw, rh = rect
        # Scrivi i pixel
        self.tex.write(img_arr.tobytes(), viewport=(x, y, w, h))
        # Calcola UV
        u0 = x / self.size
        v0 = y / self.size
        u1 = (x + w) / self.size
        v1 = (y + h) / self.size
        self.uv_map[name] = (u0, v0, u1, v1)
        # Conserva i dati originali per future espansioni
        self.texture_data[name] = (img_arr.copy(), w, h)
        self.name_to_rect[name] = (x, y, w, h)
        return True

    def _remove(self, name):
        """Rimuove una texture già presente e libera il rettangolo che
        occupava nell'atlas, restituendolo alla lista dei rettangoli liberi
        (fix bug 1/2: prima questo passaggio non esisteva affatto)."""
        old_rect = self.name_to_rect.pop(name, None)
        self.uv_map.pop(name, None)
        self.texture_data.pop(name, None)
        if old_rect is not None:
            self.free_rects.append(old_rect)
            self._prune_contained_rects()
            self._merge_free_rects()

    def _find_rect(self, w, h):
        """Cerca un rettangolo libero con algoritmo MaxRects (Best Short Side Fit)."""
        best_score = None
        best_rect = None
        for i, (rx, ry, rw, rh) in enumerate(self.free_rects):
            if rw >= w and rh >= h:
                # Fix bug 4: il punteggio "Best Short Side Fit" reale
                # confronta prima il lato corto residuo e usa quello lungo
                # solo come spareggio, invece di un semplice
                # leftover_w * leftover_h (che è in realtà un punteggio
                # "Best Area Fit", diverso da quanto dichiarato nel commento).
                leftover_w = rw - w
                leftover_h = rh - h
                short_side = min(leftover_w, leftover_h)
                long_side = max(leftover_w, leftover_h)
                score = (short_side, long_side)
                if best_score is None or score < best_score:
                    best_score = score
                    best_rect = (i, rx, ry, rw, rh)
        if best_rect is None:
            return None
        idx, x, y, rw, rh = best_rect
        used_rect = (x, y, w, h)

        # Fix bug 4: il vero MaxRects non si limita a splittare SOLO il
        # rettangolo scelto in "destra"/"sotto". Ogni rettangolo libero che
        # interseca la regione appena occupata (non solo quello scelto)
        # deve essere tagliato, generando fino a 4 pezzi residui
        # (sinistra/destra/sopra/sotto). La versione precedente frammentava
        # molto più rapidamente perché ignorava le intersezioni con gli
        # altri rettangoli liberi.
        new_free_rects = []
        for free in self.free_rects:
            new_free_rects.extend(self._split_free_rect(free, used_rect))
        self.free_rects = new_free_rects

        # Fix bug 5: elimina i rettangoli liberi completamente contenuti in
        # un altro rettangolo libero, poi unisci quelli adiacenti.
        self._prune_contained_rects()
        self._merge_free_rects()
        return used_rect

    @staticmethod
    def _rects_intersect(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax >= bx + bw or ax + aw <= bx or ay >= by + bh or ay + ah <= by)

    @staticmethod
    def _split_free_rect(free, used):
        """Divide un rettangolo libero `free` rispetto al rettangolo
        `used` appena occupato, generando fino a 4 rettangoli residui
        (sinistra, destra, sopra, sotto). Se non c'è intersezione, il
        rettangolo libero resta invariato. Questo è il vero split MaxRects
        a 4 rettangoli citato nel commento originale (fix bug 4)."""
        if not TextureAtlas._rects_intersect(free, used):
            return [free]
        fx, fy, fw, fh = free
        ux, uy, uw, uh = used
        pieces = []
        if ux > fx:                                  # striscia a sinistra
            pieces.append((fx, fy, ux - fx, fh))
        if ux + uw < fx + fw:                         # striscia a destra
            pieces.append((ux + uw, fy, (fx + fw) - (ux + uw), fh))
        if uy > fy:                                   # striscia sopra
            pieces.append((fx, fy, fw, uy - fy))
        if uy + uh < fy + fh:                          # striscia sotto
            pieces.append((fx, uy + uh, fw, (fy + fh) - (uy + uh)))
        return [p for p in pieces if p[2] > 0 and p[3] > 0]

    @staticmethod
    def _rect_contains(a, b):
        """True se il rettangolo `b` è completamente contenuto in `a`."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return bx >= ax and by >= ay and bx + bw <= ax + aw and by + bh <= ay + ah

    def _prune_contained_rects(self):
        """Elimina i rettangoli liberi completamente contenuti in un altro
        rettangolo libero (fix bug 5: prima `_merge_free_rects` univa solo
        rettangoli perfettamente adiacenti e lasciava accumulare per sempre
        i rettangoli ridondanti completamente contenuti in altri, il che
        peggiora progressivamente il packing)."""
        # Rimuovi prima i duplicati esatti (altrimenti due rettangoli
        # identici si "contengono" a vicenda e finirebbero entrambi
        # eliminati, perdendo dello spazio libero davvero disponibile).
        unique_rects = list(dict.fromkeys(self.free_rects))
        keep = []
        for i, r in enumerate(unique_rects):
            contained = False
            for j, other in enumerate(unique_rects):
                if i != j and self._rect_contains(other, r):
                    contained = True
                    break
            if not contained:
                keep.append(r)
        self.free_rects = keep

    def _merge_free_rects(self):
        """Unisce rettangoli adiacenti (semplice ottimizzazione).

        PERF FIX 15: precedentemente l'algoritmo era O(N^2) iterato fino a
        stabilita' (while merged), invocato per ogni atlas.add. Con atlas
        molto frammentati (centinaia di texture) diventava il collo di
        bottiglia del caricamento. Ora eseguiamo UNA SOLA passata O(N^2) e
        accettiamo una frammentazione leggermente maggiore: il MaxRects
        _find_rect resta comunque efficiente perche' i rettangoli liberi
        troppo piccoli vengono ignorati o riassorbiti al prossimo split.
        """
        n = len(self.free_rects)
        if n < 2:
            return
        rects = list(self.free_rects)
        used = [False] * n
        merged_list = []
        for i in range(n):
            if used[i]:
                continue
            ax, ay, aw, ah = rects[i]
            for j in range(i + 1, n):
                if used[j]:
                    continue
                bx, by, bw, bh = rects[j]
                # Stessa altezza e adiacenti orizzontali
                if ay == by and ah == bh and (ax + aw == bx or bx + bw == ax):
                    ax = min(ax, bx)
                    aw = aw + bw
                    used[j] = True
                    continue
                # Stessa larghezza e adiacenti verticali
                if ax == bx and aw == bw and (ay + ah == by or by + bh == ay):
                    ay = min(ay, by)
                    ah = ah + bh
                    used[j] = True
                    continue
            merged_list.append((ax, ay, aw, ah))
            used[i] = True
        self.free_rects = merged_list

    def _expand_atlas(self):
        """Raddoppia la dimensione dell'atlas e ricopia tutte le texture esistenti."""
        new_size = min(self.size * 2, self.max_size)
        if new_size == self.size:
            raise RuntimeError("Atlas already at maximum size, cannot expand")
        old_tex = self.tex

        # Crea nuova texture più grande
        self.size = new_size
        self.tex = self.ctx.texture((self.size, self.size), 4)
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # BUG FIX: saturazione GPU inutile.
        # Il codice precedente faceva un readback GPU->CPU completo della
        # vecchia texture (old_tex.read()), poi allocava un secondo array
        # 'new_pixels' grande quanto la NUOVA texture e ci copiava dentro i
        # pixel letti — per poi non usare mai il risultato: il blocco sotto
        # ricostruisce comunque l'intero atlas ripescando ogni immagine da
        # self.texture_data, la cache CPU che teniamo già per ogni texture
        # inserita (vedi add()). Quel readback era quindi puro lavoro
        # sprecato ad ogni espansione dell'atlas:
        #   - una vera sincronizzazione GPU->CPU, cioè la CPU che si blocca
        #     ad aspettare che la GPU finisca TUTTO il lavoro pendente prima
        #     di restituire i pixel — tra le operazioni OpenGL più costose
        #     che esistano;
        #   - un'allocazione extra inutile, fino a centinaia di MB con
        #     atlas grandi (es. 16384x16384x4 byte = 1 GB) buttata via
        #     subito dopo senza che nessuno la leggesse.
        # self.texture_data contiene già tutti i pixel originali: li
        # ripacchettiamo direttamente più sotto, senza più toccare la
        # vecchia texture se non per rilasciarla a fine funzione.

        # Resetta la lista dei rettangoli liberi: tutto lo spazio nuovo è libero
        self.free_rects = [(0, 0, self.size, self.size)]

        # Rimuovi il rettangolo occupato dalla vecchia texture (se c'era)
        # Dobbiamo reinserire TUTTE le texture dalla cache self.texture_data
        # Poiché la vecchia texture è già stata copiata, possiamo ricostruire le UV
        # e aggiornare la mappa UV.
        self.uv_map.clear()
        # Fix bug 1/2: name_to_rect deve essere ricostruita insieme a uv_map,
        # altrimenti dopo un'espansione contiene posizioni dell'atlas vecchio
        # e _remove() finirebbe per liberare rettangoli sbagliati.
        self.name_to_rect.clear()

        # Ordina le texture per area (decrescente) per un packing efficiente
        sorted_names = sorted(self.texture_data.keys(),
                              key=lambda n: self.texture_data[n][1] * self.texture_data[n][2],
                              reverse=True)

        for name in sorted_names:
            img_arr, w, h = self.texture_data[name]
            rect = self._find_rect(w, h)
            if rect is None:
                # Dovrebbe sempre funzionare dato che l'atlas è più grande
                raise RuntimeError(f"Failed to repack {name} during expansion")
            x, y, _, _ = rect
            self.tex.write(img_arr.tobytes(), viewport=(x, y, w, h))
            u0 = x / self.size
            v0 = y / self.size
            u1 = (x + w) / self.size
            v1 = (y + h) / self.size
            self.uv_map[name] = (u0, v0, u1, v1)
            self.name_to_rect[name] = (x, y, w, h)

        # Rilascia la vecchia texture
        old_tex.release()

    def get_uv(self, name):
        """Restituisce le coordinate UV (u0, v0, u1, v1) di una texture.

        Fix bug 7: in precedenza un nome sbagliato (es. "plyaer" invece di
        "player") restituiva silenziosamente la texture intera (0,0,1,1),
        un fallback che rendeva l'errore di battitura completamente
        invisibile a runtime (si disegnava tutta la texture senza alcun
        avviso). Ora un nome non trovato alza un KeyError esplicito, come
        ci si aspetta da una API pubblica quando l'errore è quasi sempre
        del programmatore."""
        try:
            return self.uv_map[name]
        except KeyError:
            raise KeyError(
                f"Texture '{name}' not found in atlas. "
                f"Available textures: {sorted(self.uv_map.keys())}"
            ) from None

    def release(self):
        """Rilascia la texture GPU dell'atlas. Va chiamato quando l'atlas
        non serve più (es. in DRAW._release_draw), altrimenti la texture
        resta allocata sulla GPU per sempre (resource leak)."""
        if self.tex is not None:
            try:
                self.tex.release()
            except Exception:
                pass
            self.tex = None


class DRAW:
    _DEG2RAD = 0.017453292519943295
    _TAU = 6.283185307179586

    _EPSILON = 1e-9

    # ------------------------------------------------------------------ #
    # HELPER CENTRALIZZATI (BUG 2, 6, 12)
    # ------------------------------------------------------------------ #
    # BUG 12 fix: unico punto in cui cos/sin vengono calcolati a partire
    # da un angolo in gradi. Se in futuro si vuole una lookup table o SIMD,
    # basta modificare questa funzione.
    @classmethod
    def _cos_sin_deg(cls, angle_deg):
        # Fix bug 9: il fast-path originale scattava solo per angle_deg
        # esattamente 0.0 (o -0.0), ma 360, 720, -360, ecc. sono angoli
        # equivalenti che finivano comunque per chiamare math.cos/math.sin.
        # Normalizziamo modulo 360 così il fast-path vale per tutti gli
        # infiniti angoli equivalenti a zero.
        if angle_deg % 360.0 == 0.0:
            return 1.0, 0.0
        ang = angle_deg * cls._DEG2RAD
        return math.cos(ang), math.sin(ang)

    @classmethod
    def _rotated_quad_corners(cls, x, y, w, h, rotation):
        """Vertici (x,y) dei 4 angoli di un rettangolo x,y,w,h ruotato di
        `rotation` gradi attorno al proprio centro.

        Fix duplicazione: questa identica formula (fast-path rotation==0 +
        formula di rotazione attorno al centro) era scritta due volte,
        parola per parola, in DrawRect e DrawTexture. Centralizzarla qui
        evita che le due copie possano divergere in futuro (es. se si
        corregge un bug nella formula in un punto e si dimentica l'altro)."""
        if rotation == 0.0:
            return x, y, x + w, y, x + w, y + h, x, y + h
        cx = x + w * 0.5; cy = y + h * 0.5
        cs, sn = cls._cos_sin_deg(rotation)
        hw = w * 0.5; hh = h * 0.5
        return (
            cx - hw*cs + hh*sn, cy - hw*sn - hh*cs,
            cx + hw*cs + hh*sn, cy + hw*sn - hh*cs,
            cx + hw*cs - hh*sn, cy + hw*sn + hh*cs,
            cx - hw*cs - hh*sn, cy - hw*sn + hh*cs,
        )

    # BUG 6 fix: blocca NaN/Inf prima che finiscano nel buffer GPU.
    @staticmethod
    def _check_finite(*values, names=None):
        for idx, v in enumerate(values):
            if not math.isfinite(v):
                label = names[idx] if names and idx < len(names) else f"arg{idx}"
                raise ValueError(f"{label} must be a finite number (got {v!r})")

    @staticmethod
    def _check_finite_array(arr, name="value"):
        a = np.asarray(arr)
        # Fix bug 8: con input non numerici (es. ["ciao", "test"]) NumPy
        # crea un array con dtype object/str e np.isfinite() alza un
        # TypeError criptico ("ufunc 'isfinite' not supported..."), prima
        # ancora di arrivare al nostro controllo/messaggio. Validiamo il
        # dtype esplicitamente per dare un errore comprensibile.
        if not np.issubdtype(a.dtype, np.number):
            raise TypeError(f"{name} must contain only numeric values, got dtype {a.dtype}")
        if a.size and not np.all(np.isfinite(a)):
            raise ValueError(f"{name} contains NaN or Inf")

    # BUG 7 fix: cache LRU generica (usata per angoli ellisse e bezier).
    @staticmethod
    def _lru_cache_get(cache: OrderedDict, key, capacity, factory):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        value = factory()
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > capacity:
            cache.popitem(last=False)
        return value


    def _init_sprite_batch(self):
        self.sprite_inst_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_corner;
                in vec2 i_pos;
                in vec2 i_size;
                // PERF FIX: cos/sin precalcolati sulla CPU (NumPy vettorizzato)
                // e passati come vec2 — elimina 2 funzioni trigonometriche per
                // vertice (= 8 per quad) eseguite sulla GPU.
                in vec2 i_dir;  // (cos(rotation), sin(rotation))
                in vec4 i_uv;        // u0, v0, u1, v1
                in float i_alpha;
                uniform vec2 u_resolution;
                out vec2 v_uv;
                out float v_alpha;

                void main() {
                    float c = i_dir.x;
                    float s = i_dir.y;
                    // BUG FIX (bug 1): in_corner e' in [-1,+1] quindi
                    // in_corner*i_size produce un quad di 2w x 2h. Usiamo
                    // half-extent (* 0.5) e trasliamo di i_size*0.5 per
                    // rispettare la convenzione top-left del resto dell'API
                    // (DrawRect, DrawTexture).
                    vec2 half_size = i_size * 0.5;
                    vec2 local = in_corner * half_size;
                    vec2 rotated = vec2(local.x * c - local.y * s,
                                        local.x * s + local.y * c);
                    vec2 world = i_pos + half_size + rotated;
                    vec2 norm = world / u_resolution * 2.0 - 1.0;
                    norm.y = -norm.y;
                    gl_Position = vec4(norm, 0.0, 1.0);
                    // UV mapping: in_corner va da (-1,-1) a (1,1)
                    vec2 uv_bl = i_uv.xy;
                    vec2 uv_tr = i_uv.zw;
                    v_uv = mix(uv_bl, uv_tr, (in_corner + 1.0) * 0.5);
                    // BUG FIX: i_alpha arriva in [0,255] (stesso range di i_color
                    // in rect_inst_prog), ma veniva passato diretto al fragment
                    // shader che lo usa come moltiplicatore [0,1].  Valore 255
                    // saturava il blending; qualsiasi valore > 1 produceva sprite
                    // completamente opachi o corrotti.
                    v_alpha = i_alpha / 255.0;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_tex;
                in vec2 v_uv;
                in float v_alpha;
                out vec4 f_color;
                void main() {
                    vec4 tex_color = texture(u_tex, v_uv);
                    f_color = vec4(tex_color.rgb, tex_color.a * v_alpha);
                }
            """
        )

        quad = np.array([-1,-1, 1,-1, 1,1, -1,1], dtype='f4')
        indices = np.array([0,1,2, 0,2,3], dtype='i4')
        self.sprite_inst_vbo = self.ctx.buffer(quad.tobytes())
        self.sprite_inst_ibo = self.ctx.buffer(indices.tobytes())

        # 11 float/istanza: pos(2)+size(2)+dir(2)+uv(4)+alpha(1)
        self.sprite_inst_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 11 * 4,
            dynamic=True
        )

        self.sprite_inst_vao = self.ctx.vertex_array(
            self.sprite_inst_prog,
            [
                (self.sprite_inst_vbo, "2f", "in_corner"),
                (self.sprite_inst_instance_vbo, "2f 2f 2f 4f 1f/i",
                "i_pos", "i_size", "i_dir", "i_uv", "i_alpha"),
            ],
            index_buffer=self.sprite_inst_ibo
        )

        self.atlas = TextureAtlas(self.ctx, 4096)
        self.sprite_count = 0
        if hasattr(self, "size"):
            self.sprite_inst_prog["u_resolution"].value = (self.size[0], self.size[1])

    def _init_ellipse_gpu(self):
        # Shader per ellissi/circonferenze instanced
        self.ellipse_prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec2 in_corner;     // quad unitario: (-1,-1) (1,-1) (1,1) (-1,1)

                in vec2 i_center;      // centro ellisse
                in vec2 i_radius;      // rx, ry
                // PERF FIX: cos/sin precalcolati su CPU e passati come vec2.
                in vec2 i_dir;         // (cos(rotation), sin(rotation))
                in vec4 i_color;       // 0..255

                uniform vec2 u_resolution;

                out vec2 v_local;      // coordinate locali normalizzate [-1,1]
                out vec4 v_color;

                void main() {
                    float c = i_dir.x;
                    float s = i_dir.y;

                    // scala il quad unitario sui raggi
                    vec2 local = vec2(in_corner.x * i_radius.x,
                                    in_corner.y * i_radius.y);

                    // rotazione attorno al centro
                    vec2 rotated = vec2(
                        local.x * c - local.y * s,
                        local.x * s + local.y * c
                    );

                    vec2 world = i_center + rotated;

                    vec2 norm = world / u_resolution * 2.0 - 1.0;
                    norm.y = -norm.y;

                    gl_Position = vec4(norm, 0.0, 1.0);

                    // coordinate locali del quad, usate nel fragment per il test ellisse
                    v_local = in_corner;
                    v_color = i_color / 255.0;
                }
            """,
            fragment_shader="""
                #version 330

                in vec2 v_local;
                in vec4 v_color;

                out vec4 f_color;

                void main() {
                    // ellisse/circonferenza dentro il quad
                    float d = dot(v_local, v_local);

                    // BUG FIX 1: aa=0.02 fisso dipendeva dalla scala: a zoom 10x
                    // il bordo diventava enorme, a zoom 0.2x quasi invisibile.
                    // fwidth(d) calcola la derivata del valore d nello spazio
                    // schermo, restituendo automaticamente la larghezza di un
                    // pixel in unità normalizzate — indipendente dallo zoom.
                    // max(..., 0.004) evita artefatti su GPU con derivate quasi-zero.
                    float aa = max(fwidth(d), 0.004);
                    float alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, d);

                    // BUG FIX 2: l'istruzione discard disabilita l'Early-Z
                    // hardware su GPU moderne (non può sapere in anticipo se il
                    // fragment scriverà il depth buffer), annullando una delle
                    // principali ottimizzazioni del rasterizzatore.
                    // Poiché questo renderer 2D non usa depth test, restituiamo
                    // semplicemente un colore con alpha=0 per i pixel esterni:
                    // il blending hardware con alpha=0 è una no-op visiva senza
                    // spezzare la pipeline.
                    f_color = vec4(v_color.rgb, v_color.a * alpha);
                }
            """
        )

        # Quad base: un solo quadrato per tutte le ellissi
        quad = np.array([
            -1.0, -1.0,
            1.0, -1.0,
            1.0,  1.0,
            -1.0,  1.0,
        ], dtype='f4')
        indices = np.array([0,1,2, 0,2,3], dtype='i4')
        self.ellipse_vbo = self.ctx.buffer(quad.tobytes())
        self.ellipse_ibo = self.ctx.buffer(indices.tobytes())

        # 10 float/istanza: center(2)+radius(2)+dir(2)+color(4)
        self.ellipse_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 10 * 4,
            dynamic=True
        )

        self.ellipse_vao = self.ctx.vertex_array(
            self.ellipse_prog,
            [
                (self.ellipse_vbo, "2f", "in_corner"),
                (self.ellipse_instance_vbo, "2f 2f 2f 4f/i",
                "i_center", "i_radius", "i_dir", "i_color"),
            ],
            index_buffer=self.ellipse_ibo
        )

        if hasattr(self, "size"):
            self.SetResolution(self.size[0], self.size[1])

    def _init_ellipse_outline_gpu(self):
        # Pipeline gemella di _init_ellipse_gpu: stessa architettura instanced
        # (1 quad + N istanze + kernel Numba di packing), così DrawCircleOutlineBatch/
        # DrawEllipsesOutlineBatch hanno esattamente le stesse prestazioni GPU-batch
        # di DrawCircle/DrawEllipse in versione Batch (DrawEllipsesBatch).
        self.ellipse_outline_prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec2 in_corner;     // quad unitario: (-1,-1) (1,-1) (1,1) (-1,1)

                in vec2 i_center;
                in vec2 i_radius;      // rx, ry (raggio ESTERNO)
                in float i_thickness;  // spessore anello, in unità mondo
                in vec2 i_dir;         // (cos(rotation), sin(rotation))
                in vec4 i_color;

                uniform vec2 u_resolution;

                out vec2 v_local;
                out vec2 v_inner_ratio;
                out vec4 v_color;

                void main() {
                    float c = i_dir.x;
                    float s = i_dir.y;

                    vec2 local = vec2(in_corner.x * i_radius.x,
                                    in_corner.y * i_radius.y);

                    vec2 rotated = vec2(
                        local.x * c - local.y * s,
                        local.x * s + local.y * c
                    );

                    vec2 world = i_center + rotated;

                    vec2 norm = world / u_resolution * 2.0 - 1.0;
                    norm.y = -norm.y;

                    gl_Position = vec4(norm, 0.0, 1.0);

                    v_local = in_corner;
                    // raggio interno normalizzato sul raggio esterno per asse:
                    // consente al fragment shader di testare il bordo interno
                    // con la stessa equazione parametrica del bordo esterno.
                    vec2 inner_r = max(i_radius - vec2(i_thickness), vec2(0.0));
                    v_inner_ratio = inner_r / max(i_radius, vec2(1e-6));
                    v_color = i_color / 255.0;
                }
            """,
            fragment_shader="""
                #version 330

                in vec2 v_local;
                in vec2 v_inner_ratio;
                in vec4 v_color;

                out vec4 f_color;

                void main() {
                    // Bordo esterno: stessa equazione/AA di ellipse_prog.
                    float d = dot(v_local, v_local);
                    float aa = max(fwidth(d), 0.004);
                    float outer = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, d);

                    // FIX perf: niente branch — divergenza wavefront eliminata.
                    // Quando inv≈0 (anello degenerato in ellisse piena) di
                    // esplode -> inner=0 -> alpha=outer, comportamento identico.
                    vec2 inv = max(v_inner_ratio, vec2(1e-6));
                    vec2 inner_local = v_local / inv;
                    float di = dot(inner_local, inner_local);
                    float aai = max(fwidth(di), 0.004);
                    float inner = 1.0 - smoothstep(1.0 - aai, 1.0 + aai, di);
                    float alpha = clamp(outer - inner, 0.0, 1.0);

                    f_color = vec4(v_color.rgb, v_color.a * alpha);
                }
            """
        )

        quad = np.array([
            -1.0, -1.0,
            1.0, -1.0,
            1.0,  1.0,
            -1.0,  1.0,
        ], dtype='f4')
        indices = np.array([0,1,2, 0,2,3], dtype='i4')
        self.ellipse_outline_vbo = self.ctx.buffer(quad.tobytes())
        self.ellipse_outline_ibo = self.ctx.buffer(indices.tobytes())

        # 11 float/istanza: center(2)+radius(2)+thickness(1)+dir(2)+color(4)
        self.ellipse_outline_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 11 * 4,
            dynamic=True
        )

        self.ellipse_outline_vao = self.ctx.vertex_array(
            self.ellipse_outline_prog,
            [
                (self.ellipse_outline_vbo, "2f", "in_corner"),
                (self.ellipse_outline_instance_vbo, "2f 2f 1f 2f 4f/i",
                "i_center", "i_radius", "i_thickness", "i_dir", "i_color"),
            ],
            index_buffer=self.ellipse_outline_ibo
        )

        if hasattr(self, "size"):
            self.ellipse_outline_prog["u_resolution"].value = (self.size[0], self.size[1])

    def _init_rect_gpu(self):
        # Shader per rettangoli instanced
        self.rect_inst_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_corner;
                in vec2 i_pos;
                in vec2 i_size;
                // PERF FIX: invece di passare l'angolo e calcolare cos/sin
                // nel vertex shader (eseguito 4× per istanza × milioni di frame),
                // passiamo direttamente il vettore direzione (cos, sin) precalcolato
                // una sola volta sulla CPU con NumPy vettorizzato.
                in vec2 i_dir;  // (cos(rotation), sin(rotation))
                in vec4 i_color;
                uniform vec2 u_resolution;
                out vec4 v_color;
                out vec2 v_local;       // FIX AA: coordinate locali in pixel, centrate
                out vec2 v_half_size;   // FIX AA: mezza-dimensione in pixel

                void main() {
                    float c = i_dir.x;
                    float s = i_dir.y;
                    vec2 half_size = i_size * 0.5;
                    vec2 local = in_corner * half_size;
                    vec2 rotated = vec2(local.x * c - local.y * s,
                                        local.x * s + local.y * c);
                    vec2 world = i_pos + half_size + rotated;
                    vec2 norm = world / u_resolution * 2.0 - 1.0;
                    norm.y = -norm.y;
                    gl_Position = vec4(norm, 0.0, 1.0);
                    v_color = i_color / 255.0;
                    v_local = local;
                    v_half_size = half_size;
                }
            """,
            fragment_shader="""
                #version 330
                in vec4 v_color;
                in vec2 v_local;
                in vec2 v_half_size;
                out vec4 f_color;
                void main() {
                    // FIX AA: SDF ai 4 bordi + smoothstep con fwidth per
                    // AA indipendente da zoom/rotazione. Bordi lisci come
                    // le ellissi, senza costo aggiuntivo apprezzabile.
                    vec2 d = v_half_size - abs(v_local);
                    float dist = min(d.x, d.y);
                    float aa = max(fwidth(dist), 0.5);
                    float alpha = smoothstep(-aa, aa, dist);
                    f_color = vec4(v_color.rgb, v_color.a * alpha);
                }
            """
        )

        quad = np.array([-1,-1, 1,-1, 1,1, -1,1], dtype='f4')
        indices = np.array([0,1,2, 0,2,3], dtype='i4')
        self.rect_inst_vbo = self.ctx.buffer(quad.tobytes())
        self.rect_inst_ibo = self.ctx.buffer(indices.tobytes())

        # Buffer dinamico standard — ora 10 float/istanza: pos(2)+size(2)+dir(2)+color(4)
        self.rect_inst_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 10 * 4,
            dynamic=True
        )

        self.rect_inst_vao = self.ctx.vertex_array(
            self.rect_inst_prog,
            [
                (self.rect_inst_vbo, "2f", "in_corner"),
                (self.rect_inst_instance_vbo, "2f 2f 2f 4f/i",
                "i_pos", "i_size", "i_dir", "i_color"),
            ],
            index_buffer=self.rect_inst_ibo
        )

        if hasattr(self, "size"):
            self.rect_inst_prog["u_resolution"].value = (self.size[0], self.size[1])

    def _init_rect_outline_gpu(self):
        # Pipeline gemella di _init_rect_gpu: stessa architettura instanced,
        # così DrawRectsOutlineBatch ha le stesse prestazioni GPU-batch di
        # DrawRectsBatch (1 draw call instanced per chunk, packing via kernel
        # Numba parallelo, nessun overhead Python per rettangolo).
        self.rect_outline_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_corner;
                in vec2 i_pos;
                in vec2 i_size;
                in float i_thickness;
                in vec2 i_dir;
                in vec4 i_color;
                uniform vec2 u_resolution;

                out vec2 v_local;       // coordinate locali in pixel, centrate
                out vec2 v_half_size;
                out float v_thickness;
                out vec4 v_color;

                void main() {
                    float c = i_dir.x;
                    float s = i_dir.y;
                    vec2 half_size = i_size * 0.5;
                    vec2 local = in_corner * half_size;
                    vec2 rotated = vec2(local.x * c - local.y * s,
                                        local.x * s + local.y * c);
                    vec2 world = i_pos + half_size + rotated;
                    vec2 norm = world / u_resolution * 2.0 - 1.0;
                    norm.y = -norm.y;
                    gl_Position = vec4(norm, 0.0, 1.0);

                    v_local = local;
                    v_half_size = half_size;
                    v_thickness = i_thickness;
                    v_color = i_color / 255.0;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_local;
                in vec2 v_half_size;
                in float v_thickness;
                in vec4 v_color;
                out vec4 f_color;

                void main() {
                    // distanza (in pixel) dal bordo più vicino del rettangolo
                    vec2 d = v_half_size - abs(v_local);
                    float dist = min(d.x, d.y);
                    // FIX AA: floor 1.0 -> 0.5 (il bordo era sempre ~1px sfocato a zoom alto)
                    float aa = max(fwidth(dist), 0.5);

                    // dentro il rettangolo (bordo esterno con AA)
                    float outerA = smoothstep(-aa, aa, dist);
                    // taglio della zona interna oltre lo spessore (bordo interno con AA)
                    float innerA = smoothstep(v_thickness - aa, v_thickness + aa, dist);

                    float alpha = outerA * (1.0 - innerA);
                    f_color = vec4(v_color.rgb, v_color.a * alpha);
                }
            """
        )

        quad = np.array([-1,-1, 1,-1, 1,1, -1,1], dtype='f4')
        indices = np.array([0,1,2, 0,2,3], dtype='i4')
        self.rect_outline_vbo = self.ctx.buffer(quad.tobytes())
        self.rect_outline_ibo = self.ctx.buffer(indices.tobytes())

        # 11 float/istanza: pos(2)+size(2)+thickness(1)+dir(2)+color(4)
        self.rect_outline_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 11 * 4,
            dynamic=True
        )

        self.rect_outline_vao = self.ctx.vertex_array(
            self.rect_outline_prog,
            [
                (self.rect_outline_vbo, "2f", "in_corner"),
                (self.rect_outline_instance_vbo, "2f 2f 1f 2f 4f/i",
                "i_pos", "i_size", "i_thickness", "i_dir", "i_color"),
            ],
            index_buffer=self.rect_outline_ibo
        )

        if hasattr(self, "size"):
            self.rect_outline_prog["u_resolution"].value = (self.size[0], self.size[1])


    def _init_triangle_gpu(self):
        self.tri_inst_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 i_v1;
                in vec2 i_v2;
                in vec2 i_v3;
                in vec4 i_color;
                uniform vec2 u_resolution;
                out vec4 v_color;

                void main() {
                    vec2 pos;
                    int id = gl_VertexID;
                    if (id == 0) pos = i_v1;
                    else if (id == 1) pos = i_v2;
                    else pos = i_v3;

                    vec2 norm = pos / u_resolution * 2.0 - 1.0;
                    norm.y = -norm.y;
                    gl_Position = vec4(norm, 0.0, 1.0);
                    v_color = i_color / 255.0;
                }
            """,
            fragment_shader="""
                #version 330
                in vec4 v_color;
                out vec4 f_color;
                void main() {
                    f_color = v_color;
                }
            """
        )

        # Buffer istanze persistente (10 float per istanza)
        self.tri_inst_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 10 * 4,
            dynamic=True
        )

        self.tri_inst_vao = self.ctx.vertex_array(
            self.tri_inst_prog,
            [
                (self.tri_inst_instance_vbo, "2f 2f 2f 4f/i",
                "i_v1", "i_v2", "i_v3", "i_color")
            ]
        )

        if hasattr(self, "size"):
            self.tri_inst_prog["u_resolution"].value = (self.size[0], self.size[1])


    def _init_line_gpu(self):
        self.line_inst_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_corner;
                in vec2 i_p1;
                in vec2 i_p2;
                in float i_thickness;
                in vec4 i_color;
                uniform vec2 u_resolution;
                out vec4 v_color;
                out float v_edge;      // FIX AA: -1..+1 sull'asse trasversale
                out float v_halfpx;    // FIX AA: 1px in unità v_edge

                void main() {
                    vec2 dir = i_p2 - i_p1;
                    float len = length(dir);
                    vec2 pos;
                    float half_t = max(i_thickness * 0.5, 0.5);
                    if (len < 1e-6) {
                        pos = i_p1;
                    } else {
                        vec2 n = vec2(-dir.y, dir.x) / len;
                        vec2 offset = n * half_t * in_corner.y;
                        float t = (in_corner.x + 1.0) * 0.5;
                        pos = mix(i_p1, i_p2, t) + offset;
                    }
                    vec2 norm = pos / u_resolution * 2.0 - 1.0;
                    norm.y = -norm.y;
                    gl_Position = vec4(norm, 0.0, 1.0);
                    v_color = i_color / 255.0;
                    v_edge = in_corner.y;                 // -1..+1
                    v_halfpx = 1.0 / max(half_t, 0.5);    // pixel size in unita' v_edge
                }
            """,
            fragment_shader="""
                #version 330
                in vec4 v_color;
                in float v_edge;
                in float v_halfpx;
                out vec4 f_color;
                void main() {
                    // FIX AA: distanza dal centro lungo l'asse trasversale (0=centro, 1=bordo).
                    // fwidth(v_edge) da' automaticamente la larghezza di un pixel
                    // nello spazio di v_edge -> AA indipendente da spessore/zoom.
                    float d = 1.0 - abs(v_edge);
                    float aa = max(fwidth(v_edge), v_halfpx * 0.5);
                    float alpha = smoothstep(0.0, aa, d);
                    f_color = vec4(v_color.rgb, v_color.a * alpha);
                }
            """
        )

        quad = np.array([-1,-1, 1,-1, 1,1, -1,1], dtype='f4')
        indices = np.array([0,1,2, 0,2,3], dtype='i4')
        self.line_inst_vbo = self.ctx.buffer(quad.tobytes())
        self.line_inst_ibo = self.ctx.buffer(indices.tobytes())

        self.line_inst_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 9 * 4,
            dynamic=True
        )

        self.line_inst_vao = self.ctx.vertex_array(
            self.line_inst_prog,
            [
                (self.line_inst_vbo, "2f", "in_corner"),
                (self.line_inst_instance_vbo, "2f 2f 1f 4f/i",
                "i_p1", "i_p2", "i_thickness", "i_color"),
            ],
            index_buffer=self.line_inst_ibo
        )

        if hasattr(self, "size"):
            self.line_inst_prog["u_resolution"].value = (self.size[0], self.size[1])

    def _init_draw(self, max_elements: int = 99999):
        # BUG 10 fix: con max_elements <= 0 tutti i controlli
        # 'if self.rect_count >= self.max_rects' diventano sempre veri (o il
        # buffer numpy ha dimensione 0/negativa), portando a comportamento
        # indefinito. Falliamo subito con un errore chiaro.
        if max_elements <= 0:
            raise ValueError(f"max_elements must be > 0 (got {max_elements})")
        # ------------------------------------------------------------------ #
        # 1. SHADER PRIMITIVI (rettangoli, linee, triangoli, ellissi…)
        # ------------------------------------------------------------------ #
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert;
                in vec4 in_color;
                uniform vec2 u_resolution;
                out vec4 v_color;

                void main() {
                    vec2 norm = in_vert / u_resolution * 2.0 - 1.0;
                    norm.y = -norm.y;
                    gl_Position = vec4(norm, 0.0, 1.0);
                    v_color = in_color / 255.0;
                }
            """,
            fragment_shader="""
                #version 330
                in vec4 v_color;
                out vec4 f_color;

                void main() {
                    f_color = v_color;
                }
            """
        )

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # ------------------------------------------------------------------ #
        # 2. BATCH PRIMITIVI — buffer numpy pre-allocati (zero alloc a runtime)
        # ------------------------------------------------------------------ #
        self.max_rects = max_elements

        self.rect_inst_data = np.empty((self.max_rects, 10), dtype='f4')
        self.line_inst_data = np.empty((self.max_rects, 9), dtype='f4')
        self.tri_inst_data  = np.empty((self.max_rects, 10), dtype='f4')
        self.ellipse_instance_data = np.empty((self.max_rects, 10), dtype='f4')
        self.sprite_inst_data = np.empty((self.max_rects, 11), dtype='f4')
        # Buffer instance CPU per le versioni *Outline* Batch (stesso principio
        # zero-alloc-a-runtime dei buffer sopra): 11 float/istanza.
        self.ellipse_outline_instance_data = np.empty((self.max_rects, 11), dtype='f4')
        self.rect_outline_instance_data = np.empty((self.max_rects, 11), dtype='f4')

        # Buffer CPU: (N, 4 vertici, 6 float: x,y,r,g,b,a)
        self._np_batch_buffer = np.empty((self.max_rects, 4, 6), dtype='f4')
        # Buffer CPU texture: (N, 4 vertici, 5 float: x,y,u,v,alpha)
        self._np_tex_buffer   = np.empty((self.max_rects, 4, 5), dtype='f4')

        # IBO condiviso quad: ogni quad → 2 triangoli (0,1,2) + (0,2,3)
        v = np.arange(self.max_rects, dtype='i4') * 4
        indices = np.empty((self.max_rects, 6), dtype='i4')
        indices[:, 0] = v;     indices[:, 1] = v + 1; indices[:, 2] = v + 2
        indices[:, 3] = v;     indices[:, 4] = v + 2; indices[:, 5] = v + 3

        self.rect_ibo = self.ctx.buffer(indices.tobytes())
        self.rect_vbo = self.ctx.buffer(
            reserve=self.max_rects * 4 * 6 * 4, dynamic=True)
        self.rect_vao = self.ctx.vertex_array(
            self.prog,
            [(self.rect_vbo, "2f 4f", "in_vert", "in_color")],
            index_buffer=self.rect_ibo
        )
        self.rect_count = 0

        # Cache angoli per ellissi e curve di Bezier — evita np.linspace ogni frame
        # BUG 7 fix: OrderedDict + _lru_cache_get implementano una vera LRU,
        # invece di svuotare tutta la cache quando si supera la capacità
        # (che causava ricostruzioni continue con segments variabili).
        self._ellipse_cache = OrderedDict()
        self._bezier_cache  = OrderedDict()   # key: segments → np.linspace(0,1,seg+1)

        if hasattr(self, "size"):
            self.SetResolution(self.size[0], self.size[1])

        # ------------------------------------------------------------------ #
        # 3. SHADER & BATCH TEXTURE
        # ------------------------------------------------------------------ #
        self.tex_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert;
                in vec2 in_uv;
                in float in_alpha;
                uniform vec2 u_resolution;
                out vec2 v_uv;
                out float v_alpha;

                void main() {
                    vec2 norm = in_vert / u_resolution * 2.0 - 1.0;
                    norm.y = -norm.y;
                    gl_Position = vec4(norm, 0.0, 1.0);
                    v_uv = in_uv;
                    v_alpha = in_alpha;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_tex;
                in vec2 v_uv;
                in float v_alpha;
                out vec4 f_color;

                void main() {
                    vec4 tex_color = texture(u_tex, v_uv);
                    f_color = vec4(tex_color.rgb, tex_color.a * v_alpha);
                }
            """
        )

        self.tex_vbo = self.ctx.buffer(
            reserve=self.max_rects * 4 * 5 * 4, dynamic=True)
        self.tex_vao = self.ctx.vertex_array(
            self.tex_prog,
            [(self.tex_vbo, "2f 2f 1f", "in_vert", "in_uv", "in_alpha")],
            index_buffer=self.rect_ibo
        )
        self.tex_rect_count  = 0
        self.current_texture = None
        self._texture_cache  = {}

        self._init_ellipse_gpu()
        self._init_ellipse_outline_gpu()
        self._init_rect_gpu()
        self._init_rect_outline_gpu()
        self._init_triangle_gpu()
        self._init_line_gpu()
        # Fix bug critico: _init_sprite_batch() non veniva mai chiamata da
        # nessuna parte del file. Di conseguenza self.atlas (TextureAtlas),
        # self.sprite_inst_prog e tutti i buffer/VAO degli sprite NON
        # esistevano mai: ogni chiamata a DrawSpritesBatch falliva con
        # AttributeError ("'DRAW' object has no attribute 'atlas'"), e non
        # esisteva alcun modo per caricare texture nell'atlas (vedi anche
        # il nuovo metodo LoadTextureAtlas più sotto).
        self._init_sprite_batch()
        # PE_TEXT: init lazy — il font manager si crea alla prima DrawText/DrawTextBatch
        self._text_ready = False

    # ------------------------------------------------------------------ #
    # TEXTURES: CARICAMENTO E RENDERING
    # ------------------------------------------------------------------ #

    def DrawSpritesBatch(self, sprites):
        """
        sprites: array numpy di shape (N, 10) oppure lista di tuple.
        Ogni sprite: (x, y, w, h, rot, u0, v0, u1, v1, alpha)
        `rot` è in GRADI, come ogni altro parametro `rotation`/`rot`
        dell'API (DrawRect, DrawEllipse, DrawRectsBatch, ...).
        """
        if self.rect_count > 0:
            self.Flush()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        data = np.asarray(sprites, dtype='f4')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Fix: a differenza di DrawRectsBatch/DrawLinesBatch/
        # DrawTrianglesBatch/DrawEllipsesBatch, questa funzione non
        # validava affatto la forma dei dati in ingresso né i valori
        # (niente controllo NaN/Inf, niente clamp dell'alpha). Un singolo
        # NaN/Inf finiva diretto nel buffer GPU producendo sprite invisibili
        # o artefatti, senza alcun errore che indicasse la causa.
        if data.ndim != 2 or data.shape[1] != 10:
            raise ValueError(
                "sprites must have shape (n, 10): "
                "(x, y, w, h, rot, u0, v0, u1, v1, alpha), got "
                f"{data.shape}"
            )
        n = data.shape[0]
        if n == 0:
            return
        self._check_finite_array(data, "sprites")

        # PERF FIX: precalcola cos/sin in gradi sulla CPU; lo shader riceve
        # vec2 i_dir invece di float i_rotation — elimina 2 costose funzioni
        # trigonometriche per vertice direttamente nel vertex shader.
        rot_rad = data[:, 4] * self._DEG2RAD
        cos_arr = np.cos(rot_rad).astype('f4')
        sin_arr = np.sin(rot_rad).astype('f4')

        # L'alpha (colonna 9 nell'input) viene clampato a [0,255].
        alpha_arr = np.clip(data[:, 9], 0.0, 255.0).astype('f4')

        # FIX 3: u_resolution gestito da SetResolution, non serve per-chiamata.
        self.atlas.tex.use(location=0)
        self.sprite_inst_prog["u_tex"].value = 0

        # Layout VBO: pos(2)+size(2)+dir(2)+uv(4)+alpha(1) = 11 float
        i = 0
        while i < n:
            chunk = min(self.max_rects, n - i)
            inst = self.sprite_inst_data[:chunk]
            sl = slice(i, i + chunk)
            inst[:, 0:2] = data[sl, 0:2]   # pos x,y
            inst[:, 2:4] = data[sl, 2:4]   # size w,h
            inst[:, 4]   = cos_arr[sl]      # cos(rot)
            inst[:, 5]   = sin_arr[sl]      # sin(rot)
            inst[:, 6:10] = data[sl, 5:9]  # uv u0,v0,u1,v1
            inst[:, 10]  = alpha_arr[sl]    # alpha

            self.sprite_inst_instance_vbo.write(memoryview(inst))
            self.sprite_inst_vao.render(instances=chunk)
            i += chunk

    # ------------------------------------------------------------------ #
    # TESTO — sistema font avanzato integrato nel modulo DRAW
    # ------------------------------------------------------------------ #
    def _ensure_text_system(self):
        if self._text_ready:
            return
        self.fonts = FontManager()
        self._glyphs = _GlyphAtlas(self, self.fonts)
        self._text_batch_buf = np.empty((4096, 11), dtype=np.float32)
        # PERF FIX 13: buffer di layout preallocati. In precedenza
        # _layout_string allocava 5 array NumPy per ogni chiamata (=> per
        # ogni DrawText, cioe' potenzialmente ogni frame per FPS counter/UI).
        # Ora vengono riusati e ridimensionati solo se la stringa cresce.
        self._layout_gx  = np.empty(256, dtype=np.float32)
        self._layout_gy  = np.empty(256, dtype=np.float32)
        self._layout_gw  = np.empty(256, dtype=np.float32)
        self._layout_gh  = np.empty(256, dtype=np.float32)
        self._layout_guv = np.empty((256, 4), dtype=np.float32)
        self._text_ready = True

    def RegisterFont(self, alias: str, path: str):
        """Registra un font TTF/OTF con un alias breve (usabile poi come
        parametro `font=` di DrawText / DrawTextBatch)."""
        self._ensure_text_system()
        self.fonts.register(alias, path)

    def _layout_string(self, text, family, size, reuse=False):
        """Ritorna (gx, gy, gw, gh, guv) — offset locali dei glifi (top-left).

        PERF FIX 13: quando `reuse=True` (path DrawText), usa i buffer
        preallocati in _ensure_text_system, evitando 5 allocazioni NumPy per
        chiamata. DrawTextBatch tiene invece piu' layout in vita nella lista
        `chunks`, quindi passa reuse=False (fresh allocations, come prima).
        """
        self._ensure_text_system()
        n = len(text)
        if n == 0:
            return None

        if reuse:
            if self._layout_gx.shape[0] < n:
                new_cap = max(n, self._layout_gx.shape[0] * 2)
                self._layout_gx  = np.empty(new_cap, dtype=np.float32)
                self._layout_gy  = np.empty(new_cap, dtype=np.float32)
                self._layout_gw  = np.empty(new_cap, dtype=np.float32)
                self._layout_gh  = np.empty(new_cap, dtype=np.float32)
                self._layout_guv = np.empty((new_cap, 4), dtype=np.float32)
            gx  = self._layout_gx
            gy  = self._layout_gy
            gw  = self._layout_gw
            gh  = self._layout_gh
            guv = self._layout_guv
        else:
            gx  = np.empty(n, dtype=np.float32)
            gy  = np.empty(n, dtype=np.float32)
            gw  = np.empty(n, dtype=np.float32)
            gh  = np.empty(n, dtype=np.float32)
            guv = np.empty((n, 4), dtype=np.float32)

        pen_x = 0.0
        line_y = 0.0
        idx = 0
        get = self._glyphs.get
        for ch in text:
            if ch == "\n":
                pen_x = 0.0
                line_y += size
                continue
            u0, v0, u1, v1, cw, chgt, adv, lft, top = get(family, size, ch)
            gx[idx] = pen_x + lft
            gy[idx] = line_y + top
            gw[idx] = cw
            gh[idx] = chgt
            guv[idx, 0] = u0
            guv[idx, 1] = v0
            guv[idx, 2] = u1
            guv[idx, 3] = v1
            pen_x += adv
            idx += 1

        if idx == 0:
            return None
        return gx[:idx], gy[:idx], gw[:idx], gh[:idx], guv[:idx]

    def DrawText(self, text, x, y,
                 font="arial", size=24,
                 color=(255, 255, 255, 255), alpha=255, rotation=0.0):
        """Versione CPU: layout in Python, render via sprite VAO (una draw call).
        Non usa Numba — pensata per pochi testi statici / debug / editor."""
        self._ensure_text_system()
        if not text:
            return
        # PERF FIX 13: reuse=True => layout scritto nei buffer preallocati.
        laid = self._layout_string(str(text), font, int(size), reuse=True)
        if laid is None:
            return
        gx, gy, gw, gh, guv = laid

        if len(color) == 4:
            r, g, b, a = color
        else:
            r, g, b = color; a = 255
        # costante _INV_255 precomputata a modulo — niente divisione qui
        col_a = max(0.0, min(255.0, (a * alpha) * _INV_255))

        cos_r, sin_r = self._cos_sin_deg(rotation)

        if self.rect_count > 0:
            self.Flush()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        self.atlas.tex.use(location=0)
        self.sprite_inst_prog["u_tex"].value = 0

        n = gx.shape[0]
        buf = self._text_batch_buf
        if buf.shape[0] < n:
            self._text_batch_buf = buf = np.empty((n, 11), dtype=np.float32)

        # pos world = origin + rotate(local)
        buf[:n, 0] = x + gx * cos_r - gy * sin_r
        buf[:n, 1] = y + gx * sin_r + gy * cos_r
        buf[:n, 2] = gw
        buf[:n, 3] = gh
        buf[:n, 4] = cos_r
        buf[:n, 5] = sin_r
        buf[:n, 6:10] = guv
        buf[:n, 10] = col_a

        self.sprite_inst_instance_vbo.write(memoryview(buf[:n]))
        self.sprite_inst_vao.render(instances=n)

    def DrawTextBatch(self, items):
        """
        items: iterable di tuple
            (text, x, y, font, size, color, alpha, rotation)
        Layout+packing con Numba (_numba_layout_glyphs, parallel). Una sola
        pipeline sprite riusata; N draw calls solo se si sfora max_rects.
        """
        self._ensure_text_system()
        if not items:
            return

        chunks = []
        for entry in items:
            text, x, y, family, size, color, alpha, rotation = entry
            if not text:
                continue
            laid = self._layout_string(str(text), family, int(size))
            if laid is None:
                continue
            gx, gy, gw, gh, guv = laid
            if len(color) == 4:
                r, g, b, a = color
            else:
                r, g, b = color; a = 255
            col_a = max(0.0, min(255.0, (a * float(alpha)) * _INV_255))
            # INCOERENZA FIX 9: usa _cos_sin_deg (fast-path per angoli
            # multipli di 360°) invece di calcolare cos/sin diretti come
            # faceva prima; ora DrawText e DrawTextBatch condividono la
            # stessa primitiva trigonometrica.
            cos_r, sin_r = self._cos_sin_deg(float(rotation))
            chunks.append((gx, gy, gw, gh, guv,
                           float(x), float(y),
                           np.float32(cos_r),
                           np.float32(sin_r),
                           np.float32(col_a)))

        if not chunks:
            return

        total = sum(c[0].shape[0] for c in chunks)
        if self._text_batch_buf.shape[0] < total:
            self._text_batch_buf = np.empty((total, 11), dtype=np.float32)
        out = self._text_batch_buf[:total]

        offset = 0
        for (gx, gy, gw, gh, guv, ox, oy, cos_r, sin_r, col_a) in chunks:
            n = gx.shape[0]
            _numba_layout_glyphs(gx, gy, gw, gh, guv,
                                 ox, oy, cos_r, sin_r, col_a,
                                 out[offset:offset + n])
            offset += n

        if self.rect_count > 0:
            self.Flush()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        self.atlas.tex.use(location=0)
        self.sprite_inst_prog["u_tex"].value = 0

        i = 0
        while i < total:
            chunk = min(self.max_rects, total - i)
            self.sprite_inst_instance_vbo.write(memoryview(out[i:i + chunk]))
            self.sprite_inst_vao.render(instances=chunk)
            i += chunk

    def LoadTexture(self, name: str, filepath: str, filter_mode="LINEAR"):
        surf_ptr = img.IMG_Load(filepath.encode("utf-8"))
        if not surf_ptr:
            print(f"[PyEngine] Errore texture: Impossibile trovare {filepath}")
            return False

        format_ptr = sdl2.SDL_AllocFormat(sdl2.SDL_PIXELFORMAT_ABGR8888)
        conv_surf  = sdl2.SDL_ConvertSurface(surf_ptr, format_ptr, 0)

        if not conv_surf:
            print(f"[PyEngine] Errore texture: impossibile convertire {filepath}")
            sdl2.SDL_FreeFormat(format_ptr)
            sdl2.SDL_FreeSurface(surf_ptr)
            return False

        w, h   = conv_surf.contents.w, conv_surf.contents.h
        pitch  = conv_surf.contents.pitch
        raw    = ctypes.string_at(conv_surf.contents.pixels, pitch * h)

        # FIX 5: LoadTextureAtlas rimuoveva già il padding di riga (pitch > w*4),
        # ma LoadTexture non lo faceva: su hardware che aggiunge padding la
        # texture veniva caricata con pixel sfasati producendo immagini
        # corrotte in modo silenzioso. Ora il comportamento è identico.
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, pitch))
        if pitch != w * 4:
            arr = arr[:, :w * 4]
        pixels = arr.reshape((h, w, 4)).tobytes()

        tex = self.ctx.texture((w, h), 4, pixels)
        if filter_mode.upper() == "NEAREST":
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        if name in self._texture_cache:
            self._texture_cache[name][0].release()

        self._texture_cache[name] = (tex, w, h)

        sdl2.SDL_FreeFormat(format_ptr)
        sdl2.SDL_FreeSurface(conv_surf)
        sdl2.SDL_FreeSurface(surf_ptr)
        return True
    
    def UnloadTexture(self, name):
        if name in self._texture_cache:
            tex, _, _ = self._texture_cache.pop(name)
            tex.release()

    def LoadTextureAtlas(self, name: str, filepath: str):
        """Carica un'immagine da disco e la inserisce in self.atlas, per
        poterla poi disegnare in batch con DrawSpritesBatch (recuperando le
        UV con self.atlas.get_uv(name)).

        Fix: prima di questo metodo non esisteva ALCUN modo pubblico per
        popolare l'atlas (TextureAtlas.add() era raggiungibile solo
        manipolando self.atlas direttamente), il che rendeva l'intero
        sistema di sprite batching inutilizzabile dall'esterno. La logica
        di caricamento ricalca LoadTexture() per restare consistente
        (stesso formato pixel ABGR8888) con l'unica differenza che i pixel
        vengono inseriti nell'atlas condiviso invece che in una texture
        OpenGL dedicata.
        """
        surf_ptr = img.IMG_Load(filepath.encode("utf-8"))
        if not surf_ptr:
            print(f"[PyEngine] Errore texture atlas: Impossibile trovare {filepath}")
            return False

        format_ptr = sdl2.SDL_AllocFormat(sdl2.SDL_PIXELFORMAT_ABGR8888)
        conv_surf  = sdl2.SDL_ConvertSurface(surf_ptr, format_ptr, 0)

        if not conv_surf:
            print(f"[PyEngine] Errore texture atlas: impossibile convertire {filepath}")
            sdl2.SDL_FreeFormat(format_ptr)
            sdl2.SDL_FreeSurface(surf_ptr)
            return False

        w, h  = conv_surf.contents.w, conv_surf.contents.h
        pitch = conv_surf.contents.pitch
        raw   = ctypes.string_at(conv_surf.contents.pixels, pitch * h)

        # Rimuove l'eventuale padding di riga (pitch > w*4) prima di
        # interpretare il buffer come (h, w, 4): per superfici a 32 bit
        # per pixel pitch è quasi sempre w*4, ma non è garantito da SDL.
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, pitch))
        if pitch != w * 4:
            arr = arr[:, :w * 4]
        img_arr = arr.reshape((h, w, 4)).copy()

        try:
            self.atlas.add(name, img_arr, w, h)
            ok = True
        except (ValueError, RuntimeError, MemoryError) as e:
            print(f"[PyEngine] Errore texture atlas: {name}: {e}")
            ok = False

        sdl2.SDL_FreeFormat(format_ptr)
        sdl2.SDL_FreeSurface(conv_surf)
        sdl2.SDL_FreeSurface(surf_ptr)
        return ok

    def DrawTexture(self, name: str, x: float, y: float,
                    w=None, h=None, rotation=0.0, alpha=255,
                    flip_x=False, flip_y=False):
        # Svuota i primitivi pendenti PRIMA di passare alle texture
        if self.rect_count > 0:
            self.Flush()

        if name not in self._texture_cache:
            return

        tex, orig_w, orig_h = self._texture_cache[name]

        if self.current_texture is not None and self.current_texture != tex:
            self.RefreshTextures()

        self.current_texture = tex

        if w is None: w = orig_w
        if h is None: h = orig_h

        self._check_finite(x, y, w, h, rotation, alpha,
                            names=("x", "y", "w", "h", "rotation", "alpha"))
        a_norm = max(0.0, min(255.0, float(alpha))) / 255.0

        u0, v0 = (1.0 if flip_x else 0.0), (1.0 if flip_y else 0.0)
        u1, v1 = (0.0 if flip_x else 1.0), (0.0 if flip_y else 1.0)

        if self.tex_rect_count >= self.max_rects:
            self.RefreshTextures()

        buf = self._np_tex_buffer[self.tex_rect_count]

        x0, y0, x1, y1, x2, y2, x3, y3 = self._rotated_quad_corners(x, y, w, h, rotation)
        buf[0] = [x0, y0, u0, v0, a_norm]
        buf[1] = [x1, y1, u1, v0, a_norm]
        buf[2] = [x2, y2, u1, v1, a_norm]
        buf[3] = [x3, y3, u0, v1, a_norm]

        self.tex_rect_count += 1
        if self.tex_rect_count >= self.max_rects:
            self.RefreshTextures()

    # ------------------------------------------------------------------ #
    # PRIMITIVE SINGOLE
    # ------------------------------------------------------------------ #

    def DrawRect(self, x, y, w, h,
                 color=(255, 255, 255, 255), rotation=0.0, alpha=255):
        self._check_finite(x, y, w, h, rotation, names=("x", "y", "w", "h", "rotation"))
        r, g, b, a = self._parse_color(color, alpha)

        if self.rect_count >= self.max_rects:
            self.Flush()

        buf = self._np_batch_buffer[self.rect_count]

        x0, y0, x1, y1, x2, y2, x3, y3 = self._rotated_quad_corners(x, y, w, h, rotation)
        buf[0] = [x0, y0, r, g, b, a]
        buf[1] = [x1, y1, r, g, b, a]
        buf[2] = [x2, y2, r, g, b, a]
        buf[3] = [x3, y3, r, g, b, a]

        self.rect_count += 1

    def DrawRectOutline(self, x, y, w, h, thickness=1.0,
                        color=(255, 255, 255, 255), rotation=0.0, alpha=255):
        """Versione 'solo bordo' di DrawRect. Path CPU immediate: costruisce
        4 quad (una cornice) usando lo stesso _rotated_quad_corners e lo
        stesso _np_batch_buffer/Flush() di DrawRect — stessa fascia di
        prestazioni delle altre primitive non-Batch."""
        self._check_finite(x, y, w, h, thickness, rotation,
                            names=("x", "y", "w", "h", "thickness", "rotation"))
        if thickness <= 0:
            raise ValueError(f"thickness must be > 0 (got {thickness})")
        r, g, b, a = self._parse_color(color, alpha)

        t = min(thickness, w * 0.5, h * 0.5)
        ox0, oy0, ox1, oy1, ox2, oy2, ox3, oy3 = self._rotated_quad_corners(x, y, w, h, rotation)
        ix, iy = x + t, y + t
        iw, ih = max(w - 2*t, 0.0), max(h - 2*t, 0.0)
        ix0, iy0, ix1, iy1, ix2, iy2, ix3, iy3 = self._rotated_quad_corners(ix, iy, iw, ih, rotation)

        if self.rect_count + 4 > self.max_rects:
            self.Flush()

        data = self._np_batch_buffer[self.rect_count:self.rect_count + 4]
        data[0, 0] = [ox0, oy0, r, g, b, a]; data[0, 1] = [ox1, oy1, r, g, b, a]
        data[0, 2] = [ix1, iy1, r, g, b, a]; data[0, 3] = [ix0, iy0, r, g, b, a]

        data[1, 0] = [ox1, oy1, r, g, b, a]; data[1, 1] = [ox2, oy2, r, g, b, a]
        data[1, 2] = [ix2, iy2, r, g, b, a]; data[1, 3] = [ix1, iy1, r, g, b, a]

        data[2, 0] = [ox2, oy2, r, g, b, a]; data[2, 1] = [ox3, oy3, r, g, b, a]
        data[2, 2] = [ix3, iy3, r, g, b, a]; data[2, 3] = [ix2, iy2, r, g, b, a]

        data[3, 0] = [ox3, oy3, r, g, b, a]; data[3, 1] = [ox0, oy0, r, g, b, a]
        data[3, 2] = [ix0, iy0, r, g, b, a]; data[3, 3] = [ix3, iy3, r, g, b, a]

        self.rect_count += 4

    def DrawLine(self, x1, y1, x2, y2,
                 thickness=1.0, color=(255, 255, 255, 255),
                 rotation=0.0, alpha=255):
        self._check_finite(x1, y1, x2, y2, thickness, rotation,
                            names=("x1", "y1", "x2", "y2", "thickness", "rotation"))
        if rotation != 0.0:
            mx  = (x1 + x2) * 0.5; my  = (y1 + y2) * 0.5
            cs, sn = self._cos_sin_deg(rotation)
            dx1 = x1 - mx; dy1 = y1 - my
            dx2 = x2 - mx; dy2 = y2 - my
            x1 = mx + dx1*cs - dy1*sn; y1 = my + dx1*sn + dy1*cs
            x2 = mx + dx2*cs - dy2*sn; y2 = my + dx2*sn + dy2*cs

        dx = x2 - x1; dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0: return

        nx = dx / length; ny = dy / length
        ht = thickness * 0.5
        px = -ny * ht;   py = nx * ht

        r, g, b, a = self._parse_color(color, alpha)

        if self.rect_count >= self.max_rects:
            self.Flush()

        buf = self._np_batch_buffer[self.rect_count]
        buf[0] = [x1 + px, y1 + py, r, g, b, a]
        buf[1] = [x2 + px, y2 + py, r, g, b, a]
        buf[2] = [x2 - px, y2 - py, r, g, b, a]
        buf[3] = [x1 - px, y1 - py, r, g, b, a]
        self.rect_count += 1

    def DrawTriangle(self, x1, y1, x2, y2, x3, y3,
                     color=(255, 255, 255, 255), rotation=0.0, alpha=255):
        self._check_finite(x1, y1, x2, y2, x3, y3, rotation,
                            names=("x1", "y1", "x2", "y2", "x3", "y3", "rotation"))
        if rotation != 0.0:
            cx  = (x1 + x2 + x3) / 3.0; cy = (y1 + y2 + y3) / 3.0
            cs, sn = self._cos_sin_deg(rotation)
            dx1 = x1 - cx; dy1 = y1 - cy
            dx2 = x2 - cx; dy2 = y2 - cy
            dx3 = x3 - cx; dy3 = y3 - cy
            x1 = cx + dx1*cs - dy1*sn; y1 = cy + dx1*sn + dy1*cs
            x2 = cx + dx2*cs - dy2*sn; y2 = cy + dx2*sn + dy2*cs
            x3 = cx + dx3*cs - dy3*sn; y3 = cy + dx3*sn + dy3*cs

        r, g, b, a = self._parse_color(color, alpha)

        if self.rect_count >= self.max_rects:
            self.Flush()

        buf = self._np_batch_buffer[self.rect_count]
        buf[0] = [x1, y1, r, g, b, a]
        buf[1] = [x2, y2, r, g, b, a]
        buf[2] = [x3, y3, r, g, b, a]
        buf[3] = [x3, y3, r, g, b, a]  # degenere: secondo triangolo area=0
        self.rect_count += 1

    def DrawTriangleOutline(self, x1, y1, x2, y2, x3, y3, thickness=1.0,
                            color=(255, 255, 255, 255), rotation=0.0, alpha=255):
        """Versione 'solo bordo' di DrawTriangle: 3 lati, ognuno costruito
        come un quad sottile (stesso principio di DrawLine), scritti nello
        stesso _np_batch_buffer/Flush() — stessa fascia di prestazioni
        delle altre primitive non-Batch. Nota: gli angoli non sono
        mitrati (come DrawLine), coerente con il resto del motore."""
        self._check_finite(x1, y1, x2, y2, x3, y3, thickness, rotation,
                            names=("x1", "y1", "x2", "y2", "x3", "y3", "thickness", "rotation"))
        if thickness <= 0:
            raise ValueError(f"thickness must be > 0 (got {thickness})")
        if rotation != 0.0:
            cx = (x1 + x2 + x3) / 3.0; cy = (y1 + y2 + y3) / 3.0
            cs, sn = self._cos_sin_deg(rotation)
            dx1 = x1 - cx; dy1 = y1 - cy
            dx2 = x2 - cx; dy2 = y2 - cy
            dx3 = x3 - cx; dy3 = y3 - cy
            x1 = cx + dx1*cs - dy1*sn; y1 = cy + dx1*sn + dy1*cs
            x2 = cx + dx2*cs - dy2*sn; y2 = cy + dx2*sn + dy2*cs
            x3 = cx + dx3*cs - dy3*sn; y3 = cy + dx3*sn + dy3*cs

        r, g, b, a = self._parse_color(color, alpha)

        if self.rect_count + 3 > self.max_rects:
            self.Flush()

        ht = thickness * 0.5
        data = self._np_batch_buffer[self.rect_count:self.rect_count + 3]
        edges = ((x1, y1, x2, y2), (x2, y2, x3, y3), (x3, y3, x1, y1))
        for idx, (ax, ay, bx, by) in enumerate(edges):
            dx = bx - ax; dy = by - ay
            length = math.hypot(dx, dy)
            if length == 0:
                nx = ny = 0.0
            else:
                nx = -dy / length * ht; ny = dx / length * ht
            data[idx, 0] = [ax + nx, ay + ny, r, g, b, a]
            data[idx, 1] = [bx + nx, by + ny, r, g, b, a]
            data[idx, 2] = [bx - nx, by - ny, r, g, b, a]
            data[idx, 3] = [ax - nx, ay - ny, r, g, b, a]

        self.rect_count += 3

    def DrawCircle(self, cx, cy, r,
                   segments=None, color=(255, 255, 255, 255),
                   smooth=True, alpha=255):
        """Alias comodo per DrawEllipse con rx == ry."""
        self.DrawEllipse(cx, cy, r, r,
                         segments=segments, color=color,
                         smooth=smooth, rotation=0.0, alpha=alpha)

    def DrawEllipse(self, cx, cy, rx, ry,
                    segments=None, color=(255, 255, 255, 255),
                    smooth=True, rotation=0.0, alpha=255):
        self._check_finite(cx, cy, rx, ry, rotation,
                            names=("cx", "cy", "rx", "ry", "rotation"))
        if segments is None:
            segments = 64 if smooth else 12

        # BUG 9 fix: un'ellisse con meno di 3 segmenti è degenere/non disegnabile.
        if segments < 3:
            raise ValueError(f"segments must be >= 3 (got {segments})")

        r, g, b, a = self._parse_color(color, alpha)

        # BUG 7 fix: cache LRU vera invece di un clear() totale quando supera
        # i 30 elementi (che annullava ogni beneficio con segments variabili).
        cos_t, sin_t = self._lru_cache_get(
            self._ellipse_cache, segments, 30,
            lambda: (lambda angles: (np.cos(angles), np.sin(angles)))(
                np.linspace(0, 2.0 * np.pi, segments + 1, dtype='f4')
            )
        )

        if rotation == 0.0:
            xs = cx + rx * cos_t
            ys = cy + ry * sin_t
        else:
            cs, sn = self._cos_sin_deg(rotation)
            lx  = rx * cos_t;    ly = ry * sin_t
            xs  = cx + lx*cs - ly*sn
            ys  = cy + lx*sn + ly*cs

        # Fast-path: tutta l'ellisse entra nel buffer restante senza flush intermedi
        if self.rect_count + segments <= self.max_rects:
            start = self.rect_count
            data  = self._np_batch_buffer[start:start + segments]
            data[:, 0, 0] = cx;            data[:, 0, 1] = cy
            data[:, 1, 0] = xs[:segments]; data[:, 1, 1] = ys[:segments]
            data[:, 2, 0] = xs[1:];        data[:, 2, 1] = ys[1:]
            data[:, 3, 0] = cx;            data[:, 3, 1] = cy
            data[:, :, 2:6] = (r, g, b, a)
            self.rect_count += segments
            return

        # Slow-path: scrittura per chunk (buffer quasi pieno)
        i = 0
        while i < segments:
            disponibili = self.max_rects - self.rect_count
            if disponibili <= 0:
                self.Flush()
                disponibili = self.max_rects

            n     = min(segments - i, disponibili)
            start = self.rect_count
            data  = self._np_batch_buffer[start:start + n]

            data[:, 0, 0] = cx;              data[:, 0, 1] = cy
            data[:, 1, 0] = xs[i:i+n];      data[:, 1, 1] = ys[i:i+n]
            data[:, 2, 0] = xs[i+1:i+1+n]; data[:, 2, 1] = ys[i+1:i+1+n]
            data[:, 3, 0] = cx;              data[:, 3, 1] = cy
            data[:, :, 2:6] = (r, g, b, a)

            self.rect_count += n
            i += n

    def DrawCircleOutline(self, cx, cy, r, thickness=1.0,
                          segments=None, color=(255, 255, 255, 255),
                          smooth=True, alpha=255):
        """Alias comodo per DrawEllipseOutline con rx == ry.
        Stessa fascia di prestazioni (path CPU immediate, stesso buffer
        condiviso _np_batch_buffer + Flush()) di DrawCircle."""
        self.DrawEllipseOutline(cx, cy, r, r, thickness=thickness,
                                segments=segments, color=color,
                                smooth=smooth, rotation=0.0, alpha=alpha)

    def DrawEllipseOutline(self, cx, cy, rx, ry, thickness=1.0,
                           segments=None, color=(255, 255, 255, 255),
                           smooth=True, rotation=0.0, alpha=255):
        """Versione 'anello' (solo bordo) di DrawEllipse. Path CPU immediate:
        tassella il bordo in `segments` quad (anziché in un fan verso il
        centro come DrawEllipse) e li scrive nello stesso _np_batch_buffer
        condiviso da DrawRect/DrawLine/DrawTriangle/DrawEllipse, con lo
        stesso schema fast-path/slow-path e Flush() — identica fascia di
        prestazioni delle altre primitive non-Batch."""
        self._check_finite(cx, cy, rx, ry, thickness, rotation,
                            names=("cx", "cy", "rx", "ry", "thickness", "rotation"))
        if thickness <= 0:
            raise ValueError(f"thickness must be > 0 (got {thickness})")
        if segments is None:
            segments = 64 if smooth else 12
        if segments < 3:
            raise ValueError(f"segments must be >= 3 (got {segments})")

        r, g, b, a = self._parse_color(color, alpha)

        cos_t, sin_t = self._lru_cache_get(
            self._ellipse_cache, segments, 30,
            lambda: (lambda angles: (np.cos(angles), np.sin(angles)))(
                np.linspace(0, 2.0 * np.pi, segments + 1, dtype='f4')
            )
        )

        irx = max(rx - thickness, 0.0)
        iry = max(ry - thickness, 0.0)

        if rotation == 0.0:
            oxs = cx + rx * cos_t;  oys = cy + ry * sin_t
            ixs = cx + irx * cos_t; iys = cy + iry * sin_t
        else:
            cs, sn = self._cos_sin_deg(rotation)
            olx = rx * cos_t;  oly = ry * sin_t
            ilx = irx * cos_t; ily = iry * sin_t
            oxs = cx + olx*cs - oly*sn; oys = cy + olx*sn + oly*cs
            ixs = cx + ilx*cs - ily*sn; iys = cy + ilx*sn + ily*cs

        def _write_ring(start, n, i0):
            data = self._np_batch_buffer[start:start + n]
            data[:, 0, 0] = oxs[i0:i0+n];     data[:, 0, 1] = oys[i0:i0+n]
            data[:, 1, 0] = oxs[i0+1:i0+1+n]; data[:, 1, 1] = oys[i0+1:i0+1+n]
            data[:, 2, 0] = ixs[i0+1:i0+1+n]; data[:, 2, 1] = iys[i0+1:i0+1+n]
            data[:, 3, 0] = ixs[i0:i0+n];     data[:, 3, 1] = iys[i0:i0+n]
            data[:, :, 2:6] = (r, g, b, a)

        # Fast-path: tutto l'anello entra nel buffer restante senza flush intermedi
        if self.rect_count + segments <= self.max_rects:
            _write_ring(self.rect_count, segments, 0)
            self.rect_count += segments
            return

        # Slow-path: scrittura per chunk (buffer quasi pieno)
        i = 0
        while i < segments:
            disponibili = self.max_rects - self.rect_count
            if disponibili <= 0:
                self.Flush()
                disponibili = self.max_rects
            n = min(segments - i, disponibili)
            _write_ring(self.rect_count, n, i)
            self.rect_count += n
            i += n

    # ------------------------------------------------------------------ #
    # BATCH VETTORIALI (numpy ottimizzati)
    # ------------------------------------------------------------------ #

    def _prepare_rgba_batch(self, colors, alpha, n):
        """
        Normalizza e VALIDA colori/alpha per tutte le funzioni *Batch.
        Il clip 0-255 e la sovrascrittura alpha sono fatti da un kernel
        Numba parallelo (_numba_clip_rgba), che elimina 3-4 passaggi
        NumPy (clip x3, broadcast) su array grandi.
        """
        col = np.asarray(colors, dtype='f4')
        if col.ndim == 1:
            col = col[np.newaxis, :]
        elif col.ndim != 2:
            raise ValueError(
                "colors must be a sequence like (r,g,b)/(r,g,b,a) or an "
                "array with shape (n,3) / (n,4)"
            )
        if col.shape[1] not in (3, 4):
            raise ValueError("colors must have exactly 3 or 4 channels")
        if col.shape[0] == 1 and n > 1:
            col = np.broadcast_to(col, (n, col.shape[1])).astype('f4', copy=False)
        elif col.shape[0] != n:
            raise ValueError(f"colors must contain either 1 color or {n} colors")
        if not np.all(np.isfinite(col)):
            raise ValueError("colors contains NaN or Inf")

        # espandi a 4 canali (rgba) — alpha di default 255
        rgba_in = np.empty((n, 4), dtype='f4')
        rgba_in[:, :3] = col[:, :3]
        rgba_in[:, 3] = col[:, 3] if col.shape[1] == 4 else 255.0

        # scelta modalità alpha per il kernel
        alpha_arr_np = np.zeros(1, dtype='f4')
        alpha_scalar = 255.0
        mode = 0                                   # nessuna sovrascrittura
        if np.isscalar(alpha):
            if not math.isfinite(alpha):
                raise ValueError("alpha must be finite")
            if alpha != 255:
                alpha_scalar = float(alpha)
                mode = 1
        else:
            a = np.asarray(alpha, dtype='f4').reshape(-1)
            if not np.all(np.isfinite(a)):
                raise ValueError("alpha contains NaN or Inf")
            if a.size == 1:
                alpha_scalar = float(a[0])
                mode = 1
            elif a.size == n:
                alpha_arr_np = a
                mode = 2
            else:
                raise ValueError("alpha must be scalar or have length n")

        out = np.empty((n, 4), dtype='f4')
        if alpha_arr_np.size < n:
            alpha_arr_np = np.zeros(n, dtype='f4')
        _numba_clip_rgba(rgba_in, np.float32(alpha_scalar), np.int32(mode),
                         alpha_arr_np, out)
        return out

    @staticmethod
    def _parse_color(color, alpha=255):
        """
        Converte un colore nel formato RGBA validato.

        Accetta:

            (r,g,b)

            (r,g,b,a)

        alpha sovrascrive il quarto canale se diverso da 255.

        Restituisce:
            r, g, b, a
        """

        if not hasattr(color, "__len__"):
            raise TypeError(
                "color must be a tuple/list/ndarray"
            )

        if len(color) == 3:

            r, g, b = color
            a = 255

        elif len(color) == 4:

            r, g, b, a = color

        else:

            raise ValueError(
                "color must contain 3 or 4 components"
            )

        if alpha != 255:
            a = alpha

        rgba = (r, g, b, a)

        # PERF FIX 17: math.isfinite lavora direttamente sullo scalare Python;
        # np.isfinite crea un numpy.bool_ 0-dimensionale per ogni valore.
        # _parse_color viene chiamata per ogni primitiva singola => hot path.
        for value in rgba:
            if not math.isfinite(value):
                raise ValueError(
                    "color contains NaN or Inf"
                )

        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        a = max(0, min(255, int(a)))

        return r, g, b, a

    @staticmethod
    def _ensure_f4(arr):
        """Converte in array f4 solo se necessario."""
        if isinstance(arr, np.ndarray) and arr.dtype == np.dtype('f4'):
            return arr
        return np.asarray(arr, dtype='f4')

    @staticmethod
    def _apply_color_alpha(data, col, alpha_arr, i, n):
        """Scrive colore e alpha nel chunk di data[:,:,2:6]."""
        c_chunk = col[i:i+n] if col.shape[0] > 1 else col

        if c_chunk.shape[1] >= 4:
            data[:, :, 2:6] = c_chunk[:, np.newaxis, :4]
            if alpha_arr.ndim == 0:
                if alpha_arr != 255.0:
                    data[:, :, 5] = alpha_arr
            else:
                a_chunk = alpha_arr[i:i+n] if alpha_arr.shape[0] > 1 else alpha_arr
                data[:, :, 5] = a_chunk[:, np.newaxis]
        else:
            data[:, :, 2:5] = c_chunk[:, np.newaxis, :3]
            if alpha_arr.ndim == 0:
                data[:, :, 5] = alpha_arr
            else:
                a_chunk = alpha_arr[i:i+n] if alpha_arr.shape[0] > 1 else alpha_arr
                data[:, :, 5] = a_chunk[:, np.newaxis]


    def DrawRectsBatch(self, positions, sizes, colors, alpha=255, rotation=0.0):
        if self.rect_count > 0:
            self.Flush()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        pos = np.asarray(positions, dtype='f4')
        size = np.asarray(sizes, dtype='f4')
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions must be (n,2)")
        if size.shape != pos.shape:
            raise ValueError("sizes must have same shape as positions")
        n = pos.shape[0]
        if n == 0:
            return
        self._check_finite_array(pos, "positions")
        self._check_finite_array(size, "sizes")

        rgba = self._prepare_rgba_batch(colors, alpha, n)
        rot = np.asarray(rotation, dtype='f4').reshape(-1)
        if rot.size == 1:
            rot = np.full(n, rot[0], dtype='f4')
        elif rot.size != n:
            raise ValueError("rotation must be scalar or array of length n")
        self._check_finite_array(rot, "rotation")
        # PERF FIX: precalcola cos/sin in gradi sulla CPU con NumPy vettorizzato
        # (un'unica chiamata np.cos/sin su tutto l'array) invece di delegare la
        # trigonometria al vertex shader (4 chiamate per rettangolo × N istanze).
        rot_rad = rot * self._DEG2RAD
        cos_arr = np.cos(rot_rad).astype('f4')
        sin_arr = np.sin(rot_rad).astype('f4')

        # FIX 3: u_resolution aggiornato una sola volta in SetResolution;
        # aggiornare lo uniform ad ogni chiamata batch è ridondante e spreca
        # un round-trip driver per frame senza alcun beneficio.
        # Packing dell'instance buffer via kernel Numba parallelo.
        i = 0
        while i < n:
            chunk = min(self.max_rects, n - i)
            inst = self.rect_inst_data[:chunk]
            _numba_pack_rect_instances(
                pos[i:i+chunk], size[i:i+chunk],
                cos_arr[i:i+chunk], sin_arr[i:i+chunk],
                rgba[i:i+chunk], inst
            )
            self.rect_inst_instance_vbo.write(memoryview(inst))
            self.rect_inst_vao.render(instances=chunk)
            i += chunk

    def DrawRectsOutlineBatch(self, positions, sizes, colors, thickness=1.0, alpha=255, rotation=0.0):
        """Versione 'solo bordo' di DrawRectsBatch. Stessa identica
        architettura GPU-instanced (packing vettoriale via kernel Numba
        parallelo + un'unica render(instances=chunk) per chunk): stessa
        fascia di prestazioni Batch di DrawRectsBatch."""
        if self.rect_count > 0:
            self.Flush()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        pos = np.asarray(positions, dtype='f4')
        size = np.asarray(sizes, dtype='f4')
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions must be (n,2)")
        if size.shape != pos.shape:
            raise ValueError("sizes must have same shape as positions")
        n = pos.shape[0]
        if n == 0:
            return
        self._check_finite_array(pos, "positions")
        self._check_finite_array(size, "sizes")

        rgba = self._prepare_rgba_batch(colors, alpha, n)
        rot = np.asarray(rotation, dtype='f4').reshape(-1)
        if rot.size == 1:
            rot = np.full(n, rot[0], dtype='f4')
        elif rot.size != n:
            raise ValueError("rotation must be scalar or array of length n")
        self._check_finite_array(rot, "rotation")
        rot_rad = rot * self._DEG2RAD
        cos_arr = np.cos(rot_rad).astype('f4')
        sin_arr = np.sin(rot_rad).astype('f4')

        thick = np.asarray(thickness, dtype='f4').reshape(-1)
        if thick.size == 1:
            thick = np.full(n, thick[0], dtype='f4')
        elif thick.size != n:
            raise ValueError("thickness must be scalar or array of length n")
        self._check_finite_array(thick, "thickness")
        if np.any(thick <= 0):
            raise ValueError("thickness must be > 0")

        i = 0
        while i < n:
            chunk = min(self.max_rects, n - i)
            inst = self.rect_outline_instance_data[:chunk]
            _numba_pack_rect_outline_instances(
                pos[i:i+chunk], size[i:i+chunk], thick[i:i+chunk],
                cos_arr[i:i+chunk], sin_arr[i:i+chunk],
                rgba[i:i+chunk], inst
            )
            self.rect_outline_instance_vbo.write(memoryview(inst))
            self.rect_outline_vao.render(instances=chunk)
            i += chunk


    def DrawLinesBatch(self, x1, y1, x2, y2, colors, thickness=1.0, alpha=255, rotation=0.0):
        if self.rect_count > 0:
            self.Flush()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        # BUG FIX (bug 3): np.asarray non copia se l'input e' gia' un ndarray
        # float32; il successivo _numba_rotate_lines (con rotation != 0) scrive
        # in-place, mutando gli array dell'utente. np.array forza la copia.
        x1 = np.array(x1, dtype='f4').ravel()
        y1 = np.array(y1, dtype='f4').ravel()
        x2 = np.array(x2, dtype='f4').ravel()
        y2 = np.array(y2, dtype='f4').ravel()
        # FIX 4: verifica che i quattro array abbiano la stessa lunghezza.
        # Senza questo controllo, shape diverse producevano un errore numpy
        # criptico dentro column_stack invece di un messaggio chiaro.
        if not (x1.shape == y1.shape == x2.shape == y2.shape):
            raise ValueError(
                "x1, y1, x2, y2 must all have the same length; "
                f"got shapes {x1.shape}, {y1.shape}, {x2.shape}, {y2.shape}"
            )
        n = x1.shape[0]
        if n == 0:
            return
        for name, arr in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
            self._check_finite_array(arr, name)

        # INCOERENZA FIX 10: rotation ora accetta anche array di lunghezza n
        # (allineato a DrawRectsBatch). Un array veniva collassato a un
        # singolo float via float(rotation), producendo comportamento
        # silenziosamente errato.
        rot_arr = np.asarray(rotation, dtype='f4').reshape(-1)
        if rot_arr.size == 1:
            r0 = float(rot_arr[0])
            if r0 != 0.0:
                ang = r0 * _DEG2RAD_CONST
                _numba_rotate_lines(x1, y1, x2, y2,
                                    np.float32(math.cos(ang)),
                                    np.float32(math.sin(ang)))
        elif rot_arr.size == n:
            self._check_finite_array(rot_arr, "rotation")
            rad = rot_arr * self._DEG2RAD
            cs_arr = np.cos(rad).astype('f4')
            sn_arr = np.sin(rad).astype('f4')
            _numba_rotate_lines_arr(x1, y1, x2, y2, cs_arr, sn_arr)
        else:
            raise ValueError("rotation must be scalar or array of length n")

        thick = np.asarray(thickness, dtype='f4').reshape(-1)
        if thick.size == 1:
            thick = np.full(n, thick[0], dtype='f4')
        elif thick.size != n:
            raise ValueError("thickness must be scalar or array of length n")
        self._check_finite_array(thick, "thickness")

        rgba = self._prepare_rgba_batch(colors, alpha, n)

        # PERF FIX 18: niente np.column_stack (2 allocazioni (N,2) per
        # chiamata anche con input gia' f4). Il nuovo kernel legge da 4
        # array 1D contigui, con lo stesso costo GPU.
        i = 0
        while i < n:
            chunk = min(self.max_rects, n - i)
            inst = self.line_inst_data[:chunk]
            _numba_pack_line_instances_xy(
                x1[i:i+chunk], y1[i:i+chunk],
                x2[i:i+chunk], y2[i:i+chunk],
                thick[i:i+chunk], rgba[i:i+chunk], inst
            )
            self.line_inst_instance_vbo.write(memoryview(inst))
            self.line_inst_vao.render(instances=chunk)
            i += chunk


    def DrawTrianglesBatch(self, vertices, colors, alpha=255):
        if self.rect_count > 0:
            self.Flush()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        verts = np.asarray(vertices, dtype='f4')
        if verts.ndim == 2:
            if verts.shape[1] != 2:
                raise ValueError("vertices shape must be (n,3,2) or (3*n,2)")
            n = verts.shape[0] // 3
            verts = verts.reshape(n, 3, 2)
        elif verts.ndim == 3 and verts.shape[1] == 3 and verts.shape[2] == 2:
            n = verts.shape[0]
        else:
            raise ValueError("vertices must be (n,3,2) or (3*n,2)")

        if n == 0:
            return

        self._check_finite_array(verts, "vertices")
        rgba = self._prepare_rgba_batch(colors, alpha, n)

        # PERF FIX 14: gli slice verts[:, 0, :] su un (N,3,2) NON sono
        # contigui in memoria (stride non unitari). Prima np.ascontiguousarray
        # veniva chiamato DENTRO il loop dei chunk, allocando 3 copie ad ogni
        # iterazione. Estraiamo qui una volta 3 array (N,2) contigui, poi lo
        # slicing per chunk resta contiguo => zero copie nel loop.
        v0 = np.ascontiguousarray(verts[:, 0, :])
        v1 = np.ascontiguousarray(verts[:, 1, :])
        v2 = np.ascontiguousarray(verts[:, 2, :])

        i = 0
        while i < n:
            chunk = min(self.max_rects, n - i)
            inst = self.tri_inst_data[:chunk]
            _numba_pack_tri_instances(
                v0[i:i+chunk],
                v1[i:i+chunk],
                v2[i:i+chunk],
                rgba[i:i+chunk], inst
            )
            self.tri_inst_instance_vbo.write(memoryview(inst))
            self.tri_inst_vao.render(vertices=3, instances=chunk)
            i += chunk

    def DrawTrianglesOutlineBatch(self, vertices, colors, thickness=1.0, alpha=255):
        """Versione 'solo bordo' di DrawTrianglesBatch. Ogni triangolo ha 3
        lati: costruiamo gli array p1/p2 dei lati in modo interamente
        vettoriale (nessun loop Python per triangolo) e li mandiamo alla
        stessa pipeline GPU-instanced di DrawLinesBatch (packing via
        _numba_pack_line_instances + render instanced) — stessa fascia di
        prestazioni Batch del resto del motore. Nota: gli angoli non sono
        mitrati, coerente con DrawLinesBatch/DrawTriangleOutline."""
        if self.rect_count > 0:
            self.Flush()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        verts = np.asarray(vertices, dtype='f4')
        if verts.ndim == 2:
            if verts.shape[1] != 2:
                raise ValueError("vertices shape must be (n,3,2) or (3*n,2)")
            n = verts.shape[0] // 3
            verts = verts.reshape(n, 3, 2)
        elif verts.ndim == 3 and verts.shape[1] == 3 and verts.shape[2] == 2:
            n = verts.shape[0]
        else:
            raise ValueError("vertices must be (n,3,2) or (3*n,2)")

        if n == 0:
            return
        self._check_finite_array(verts, "vertices")

        rgba = self._prepare_rgba_batch(colors, alpha, n)
        rgba3 = np.repeat(rgba, 3, axis=0)

        thick = np.asarray(thickness, dtype='f4').reshape(-1)
        if thick.size == 1:
            thick3 = np.full(n * 3, thick[0], dtype='f4')
        elif thick.size == n:
            thick3 = np.repeat(thick, 3).astype('f4', copy=False)
        else:
            raise ValueError("thickness must be scalar or array of length n")
        self._check_finite_array(thick3, "thickness")
        if np.any(thick3 <= 0):
            raise ValueError("thickness must be > 0")

        v0 = verts[:, 0, :]; v1 = verts[:, 1, :]; v2 = verts[:, 2, :]
        p1 = np.empty((n * 3, 2), dtype='f4')
        p2 = np.empty((n * 3, 2), dtype='f4')
        p1[0::3] = v0; p2[0::3] = v1
        p1[1::3] = v1; p2[1::3] = v2
        p1[2::3] = v2; p2[2::3] = v0

        m = n * 3
        i = 0
        while i < m:
            chunk = min(self.max_rects, m - i)
            inst = self.line_inst_data[:chunk]
            _numba_pack_line_instances(
                p1[i:i+chunk], p2[i:i+chunk],
                thick3[i:i+chunk], rgba3[i:i+chunk], inst
            )
            self.line_inst_instance_vbo.write(memoryview(inst))
            self.line_inst_vao.render(instances=chunk)
            i += chunk

    def DrawEllipsesBatch(self, centers, radii, colors, segments=None, alpha=255, rotation=0.0):
        """
        NOTA (BUG 11): questo path usa il path GPU instanced (SDF nel
        fragment shader), che disegna un'ellisse matematicamente esatta e
        NON usa una mesh poligonale: il parametro `segments`, a differenza
        di DrawEllipse/DrawCircle (path CPU), non ha alcun effetto qui.
        Viene mantenuto solo per compatibilità di firma con le funzioni
        CPU equivalenti; se viene passato esplicitamente, emettiamo un
        warning per evitare di confondere chi si aspetta lo stesso
        comportamento del path CPU.
        """
        if segments is not None:
            warnings.warn(
                "DrawEllipsesBatch: 'segments' is ignored — the GPU instanced "
                "ellipse path renders an exact ellipse via SDF, not a polygon. "
                "Use DrawEllipse()/DrawCircle() if you need an explicit segment count.",
                stacklevel=2
            )

        if self.rect_count > 0:
            self.Flush()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        c = DRAW._ensure_f4(centers)
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError("centers must have shape (n, 2)")

        n = c.shape[0]
        if n == 0:
            return
        self._check_finite_array(c, "centers")

        rgba = self._prepare_rgba_batch(colors, alpha, n)

        rad = np.asarray(radii, dtype='f4')
        if rad.ndim == 0:
            rad = np.full((n, 2), float(rad), dtype='f4')
        elif rad.ndim == 1:
            if rad.size == 1:
                rad = np.full((n, 2), float(rad[0]), dtype='f4')
            elif rad.size == 2:
                rad = np.broadcast_to(rad[np.newaxis, :], (n, 2)).astype('f4', copy=False)
            elif rad.size == n:
                rad = np.column_stack((rad, rad)).astype('f4', copy=False)
            else:
                raise ValueError("radii must be scalar, (n,), (1,), (2,) or (n,2)")
        elif rad.ndim == 2 and rad.shape[1] == 2:
            if rad.shape[0] != n:
                raise ValueError("radii must have the same length as centers")
        else:
            raise ValueError("radii must be scalar, (n,), (1,), (2,) or (n,2)")
        self._check_finite_array(rad, "radii")

        rot = np.asarray(rotation, dtype='f4')
        if rot.ndim == 0:
            rot = np.full((n, 1), float(rot), dtype='f4')
        else:
            rot = rot.reshape(-1, 1).astype('f4', copy=False)
            if rot.shape[0] == 1:
                rot = np.full((n, 1), float(rot[0, 0]), dtype='f4')
            elif rot.shape[0] != n:
                raise ValueError("rotation must be scalar or have length n")
        self._check_finite_array(rot, "rotation")
        # PERF FIX: precalcola cos/sin in gradi sulla CPU; lo shader riceve vec2 i_dir.
        rot_rad = rot.ravel() * self._DEG2RAD
        cos_arr = np.cos(rot_rad).astype('f4')
        sin_arr = np.sin(rot_rad).astype('f4')

        # FIX 3: u_resolution gestito da SetResolution, non serve per-chiamata.
        i = 0
        while i < n:
            chunk = min(self.max_rects, n - i)
            inst = self.ellipse_instance_data[:chunk]
            _numba_pack_ellipse_instances(
                c[i:i+chunk], rad[i:i+chunk],
                cos_arr[i:i+chunk], sin_arr[i:i+chunk],
                rgba[i:i+chunk], inst
            )
            self.ellipse_instance_vbo.write(memoryview(inst))
            self.ellipse_vao.render(mode=moderngl.TRIANGLES, instances=chunk)
            i += chunk

    def DrawCirclesBatch(self, centers, radius, colors=(255,255,255,255),
                         segments=None, alpha=255):
        """Alias di DrawEllipsesBatch con rx == ry — simmetrico a
        DrawCircleOutlineBatch. Stessa pipeline instanced di DrawEllipsesBatch,
        stessa fascia di prestazioni."""
        c_arr = np.asarray(centers)
        if c_arr.ndim != 2 or c_arr.shape[1] != 2:
            raise ValueError("centers must have shape (n, 2)")
        n = c_arr.shape[0]
        r = np.asarray(radius, dtype='f4').reshape(-1)
        if r.size == 1:
            radii = np.full((n, 2), float(r[0]), dtype='f4')
        elif r.size == n:
            radii = np.column_stack((r, r)).astype('f4', copy=False)
        else:
            raise ValueError("radius must be scalar or array of length n")
        self.DrawEllipsesBatch(centers, radii, colors,
                               segments=segments, alpha=alpha, rotation=0.0)

    def DrawBezierCurvesBatch(self, p0s, p1s, p2s, thickness=2.0,
                              colors=(255,255,255,255), segments=None,
                              smooth=True, alpha=255):
        """Batch di curve di Bezier quadratiche.
        Ogni curva viene tassellata in `segments` segmenti; tutti i segmenti
        di tutte le curve vengono inviati con una singola DrawLinesBatch,
        quindi la pipeline instanced della linea. Stessa fascia di prestazioni
        di DrawLinesBatch(n*segments)."""
        if segments is None:
            segments = 40 if smooth else 8
        if segments < 1:
            raise ValueError(f"segments must be >= 1 (got {segments})")

        p0 = np.asarray(p0s, dtype='f4')
        p1 = np.asarray(p1s, dtype='f4')
        p2 = np.asarray(p2s, dtype='f4')
        for name, arr in (("p0s", p0), ("p1s", p1), ("p2s", p2)):
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"{name} must have shape (n, 2)")
            self._check_finite_array(arr, name)
        if not (p0.shape == p1.shape == p2.shape):
            raise ValueError("p0s, p1s, p2s must have the same shape")
        n = p0.shape[0]
        if n == 0:
            return

        # t: (seg+1,) — cache LRU condivisa con DrawBezierCurve
        t_1d = self._lru_cache_get(
            self._bezier_cache, segments, 20,
            lambda: np.linspace(0, 1, segments + 1, dtype='f4')
        )
        t = t_1d[np.newaxis, :, np.newaxis]     # (1, seg+1, 1)
        u = 1.0 - t
        P0 = p0[:, np.newaxis, :]               # (n, 1, 2)
        P1 = p1[:, np.newaxis, :]
        P2 = p2[:, np.newaxis, :]
        pts = (u * u) * P0 + (2.0 * u * t) * P1 + (t * t) * P2  # (n, seg+1, 2)

        # Costruisci N*segments segmenti (p_i -> p_{i+1})
        starts = pts[:, :-1, :].reshape(-1, 2)
        ends   = pts[:,  1:, :].reshape(-1, 2)
        total  = starts.shape[0]

        # Espandi thickness e colors su total = n*segments
        thk = np.asarray(thickness, dtype='f4').reshape(-1)
        if thk.size == 1:
            thk_full = np.full(total, thk[0], dtype='f4')
        elif thk.size == n:
            thk_full = np.repeat(thk, segments)
        else:
            raise ValueError("thickness must be scalar or array of length n")

        col = np.asarray(colors)
        if col.ndim == 1:
            col_full = colors  # scalare/tuple → _prepare_rgba_batch broadcast
        elif col.ndim == 2 and col.shape[0] == n:
            col_full = np.repeat(col, segments, axis=0)
        else:
            raise ValueError("colors must be a single RGBA or (n, 4)")

        self.DrawLinesBatch(starts[:, 0], starts[:, 1],
                            ends[:, 0],   ends[:, 1],
                            colors=col_full, thickness=thk_full,
                            alpha=alpha, rotation=0.0)

    def DrawCircleOutlineBatch(self, centers, radius, thickness=1.0, colors=(255,255,255,255), alpha=255):
        """Alias comodo per DrawEllipsesOutlineBatch con rx == ry."""
        self.DrawEllipsesOutlineBatch(centers, radius, thickness=thickness,
                                       colors=colors, alpha=alpha, rotation=0.0)

    def DrawEllipsesOutlineBatch(self, centers, radii, thickness=1.0, colors=(255,255,255,255),
                                 alpha=255, rotation=0.0):
        """
        Versione 'anello' (solo bordo) di DrawEllipsesBatch. Path GPU
        instanced con SDF nel fragment shader (ellisse/anello matematicamente
        esatti, non una mesh poligonale) — stessa identica architettura e
        fascia di prestazioni Batch di DrawEllipsesBatch.
        """
        if self.rect_count > 0:
            self.Flush()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        c = DRAW._ensure_f4(centers)
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError("centers must have shape (n, 2)")

        n = c.shape[0]
        if n == 0:
            return
        self._check_finite_array(c, "centers")

        rgba = self._prepare_rgba_batch(colors, alpha, n)

        rad = np.asarray(radii, dtype='f4')
        if rad.ndim == 0:
            rad = np.full((n, 2), float(rad), dtype='f4')
        elif rad.ndim == 1:
            if rad.size == 1:
                rad = np.full((n, 2), float(rad[0]), dtype='f4')
            elif rad.size == 2:
                rad = np.broadcast_to(rad[np.newaxis, :], (n, 2)).astype('f4', copy=False)
            elif rad.size == n:
                rad = np.column_stack((rad, rad)).astype('f4', copy=False)
            else:
                raise ValueError("radii must be scalar, (n,), (1,), (2,) or (n,2)")
        elif rad.ndim == 2 and rad.shape[1] == 2:
            if rad.shape[0] != n:
                raise ValueError("radii must have the same length as centers")
        else:
            raise ValueError("radii must be scalar, (n,), (1,), (2,) or (n,2)")
        self._check_finite_array(rad, "radii")

        thick = np.asarray(thickness, dtype='f4').reshape(-1)
        if thick.size == 1:
            thick = np.full(n, thick[0], dtype='f4')
        elif thick.size != n:
            raise ValueError("thickness must be scalar or array of length n")
        self._check_finite_array(thick, "thickness")
        if np.any(thick <= 0):
            raise ValueError("thickness must be > 0")

        rot = np.asarray(rotation, dtype='f4')
        if rot.ndim == 0:
            rot = np.full((n, 1), float(rot), dtype='f4')
        else:
            rot = rot.reshape(-1, 1).astype('f4', copy=False)
            if rot.shape[0] == 1:
                rot = np.full((n, 1), float(rot[0, 0]), dtype='f4')
            elif rot.shape[0] != n:
                raise ValueError("rotation must be scalar or have length n")
        self._check_finite_array(rot, "rotation")
        rot_rad = rot.ravel() * self._DEG2RAD
        cos_arr = np.cos(rot_rad).astype('f4')
        sin_arr = np.sin(rot_rad).astype('f4')

        i = 0
        while i < n:
            chunk = min(self.max_rects, n - i)
            inst = self.ellipse_outline_instance_data[:chunk]
            _numba_pack_ellipse_outline_instances(
                c[i:i+chunk], rad[i:i+chunk], thick[i:i+chunk],
                cos_arr[i:i+chunk], sin_arr[i:i+chunk],
                rgba[i:i+chunk], inst
            )
            self.ellipse_outline_instance_vbo.write(memoryview(inst))
            self.ellipse_outline_vao.render(mode=moderngl.TRIANGLES, instances=chunk)
            i += chunk


    def DrawBezierCurve(self, p0, p1, p2,
                        segments=None, thickness=2.0,
                        color=(255, 255, 255, 255),
                        smooth=True, rotation=0.0, alpha=255):
        self._check_finite(thickness, rotation, names=("thickness", "rotation"))
        if segments is None:
            segments = 40 if smooth else 8
        if segments < 1:
            raise ValueError(f"segments must be >= 1 (got {segments})")

        # BUG 7 fix: vera cache LRU (vedi _lru_cache_get) invece di un
        # clear() totale quando la cache supera la capacità.
        t_1d = self._lru_cache_get(
            self._bezier_cache, segments, 20,
            lambda: np.linspace(0, 1, segments + 1, dtype='f4')
        )
        t    = t_1d[:, np.newaxis]   # (seg+1, 1)
        u    = 1.0 - t

        p0a = np.array(p0, dtype='f4')
        p1a = np.array(p1, dtype='f4')
        p2a = np.array(p2, dtype='f4')
        self._check_finite_array(p0a, "p0")
        self._check_finite_array(p1a, "p1")
        self._check_finite_array(p2a, "p2")
        points = (u**2) * p0a + (2.0 * u * t) * p1a + (t**2) * p2a  # (seg+1, 2)

        if rotation != 0.0:
            ocx = (p0a[0] + p1a[0] + p2a[0]) / 3.0
            ocy = (p0a[1] + p1a[1] + p2a[1]) / 3.0
            cs, sn = self._cos_sin_deg(rotation)
            dx  = points[:, 0] - ocx; dy = points[:, 1] - ocy
            points[:, 0] = ocx + dx*cs - dy*sn
            points[:, 1] = ocy + dx*sn + dy*cs

        dx = points[1:, 0] - points[:-1, 0]
        dy = points[1:, 1] - points[:-1, 1]
        lengths = np.hypot(dx, dy)
        lengths[lengths == 0] = 1.0

        nx = dx / lengths; ny = dy / lengths
        ht = thickness * 0.5
        px = -ny * ht; py = nx * ht

        # BUG 1/4/5/6 fix: stessa validazione usata dalle funzioni singole,
        # applicata anche qui invece di passare colore/alpha grezzi al buffer.
        r0, g0, b0, a0 = self._parse_color(color, alpha)
        col       = np.asarray([r0, g0, b0, a0], dtype='f4')[np.newaxis, :]
        alpha_arr = np.asarray(a0, dtype='f4')

        i = 0
        while i < segments:
            disponibili = self.max_rects - self.rect_count
            if disponibili <= 0:
                self.Flush()
                disponibili = self.max_rects

            n     = min(segments - i, disponibili)
            start = self.rect_count
            data  = self._np_batch_buffer[start:start + n]

            data[:, 0, 0] = points[i:i+n,   0] + px[i:i+n]
            data[:, 0, 1] = points[i:i+n,   1] + py[i:i+n]
            data[:, 1, 0] = points[i+1:i+1+n, 0] + px[i:i+n]
            data[:, 1, 1] = points[i+1:i+1+n, 1] + py[i:i+n]
            data[:, 2, 0] = points[i+1:i+1+n, 0] - px[i:i+n]
            data[:, 2, 1] = points[i+1:i+1+n, 1] - py[i:i+n]
            data[:, 3, 0] = points[i:i+n,   0] - px[i:i+n]
            data[:, 3, 1] = points[i:i+n,   1] - py[i:i+n]

            self._apply_color_alpha(data, col, alpha_arr, i, n)
            self.rect_count += n
            i += n

    # ------------------------------------------------------------------ #
    # FLUSH / SINCRONIZZAZIONE ALLA GPU
    # ------------------------------------------------------------------ #

    def ClearBuffers(self):
        """4. COMPITO: Resetta/Annulla il disegno corrente svuotando i contatori senza mandare nulla alla GPU."""
        self.rect_count = 0
        self.tex_rect_count = 0


    def RefreshTextures(self):
        if self.tex_rect_count == 0 or self.current_texture is None:
            return

        self.current_texture.use(location=0)
        self.tex_prog["u_tex"].value = 0

        view = self._np_tex_buffer[:self.tex_rect_count]
        self.tex_vbo.write(memoryview(view))
        self.tex_vao.render(vertices=self.tex_rect_count * 6)
        self.tex_rect_count = 0

    def Flush(self):
        if self.rect_count == 0:
            return

        view = self._np_batch_buffer[:self.rect_count]
        self.rect_vbo.write(memoryview(view))
        self.rect_vao.render(vertices=self.rect_count * 6)
        self.rect_count = 0

    def FlushAll(self):
        """3. COMPITO: Disegna tutto. Coordina i buffer nell'ordine corretto per i layer."""
        if self.tex_rect_count > 0:
            self.RefreshTextures()
        if self.rect_count > 0:
            self.Flush()

    def SetResolution(self, width, height):
        self.size = (width, height)
        self.prog["u_resolution"].value = (width, height)
        if hasattr(self, "tex_prog"):
            self.tex_prog["u_resolution"].value = (width, height)
        if hasattr(self, "rect_inst_prog"):
            self.rect_inst_prog["u_resolution"].value = (width, height)
        if hasattr(self, "tri_inst_prog"):
            self.tri_inst_prog["u_resolution"].value = (width, height)
        if hasattr(self, "line_inst_prog"):
            self.line_inst_prog["u_resolution"].value = (width, height)
        # FIX 2: i due shader instanced mancanti causavano coordinate NDC
        # sbagliate dopo qualsiasi resize perché u_resolution restava alla
        # risoluzione precedente. I workaround per-chiamata in
        # DrawEllipsesBatch / DrawSpritesBatch (rimossi in FIX 3) erano
        # parziali: fallivano in presenza di Flush/RefreshTextures intermedi
        # che non aggiornano la risoluzione di questi shader.
        if hasattr(self, "ellipse_prog"):
            self.ellipse_prog["u_resolution"].value = (width, height)
        if hasattr(self, "sprite_inst_prog"):
            self.sprite_inst_prog["u_resolution"].value = (width, height)
        if hasattr(self, "ellipse_outline_prog"):
            self.ellipse_outline_prog["u_resolution"].value = (width, height)
        if hasattr(self, "rect_outline_prog"):
            self.rect_outline_prog["u_resolution"].value = (width, height)

    def _release_draw(self):
        # Pulizia cache Python (questi hanno .clear())
        self._ellipse_cache.clear()
        self._bezier_cache.clear()

        if hasattr(self, "_texture_cache"):
            for tex_obj, _, _ in self._texture_cache.values():
                try:
                    tex_obj.release()
                except Exception:
                    pass
            self._texture_cache.clear()

        # Fix: questo elenco controllava attributi "*_mapped"
        # (rect_inst_mapped, line_inst_mapped, ...) che non sono MAI stati
        # assegnati da nessuna parte del file (verificato con grep: zero
        # occorrenze al di fuori di questo ciclo). getattr(...) restituiva
        # quindi sempre None e il blocco non liberava nulla: ogni VBO, IBO,
        # VAO, program e texture creati in _init_draw/_init_*_gpu restavano
        # allocati sulla GPU anche dopo il rilascio del DRAW — un resource
        # leak completo a ogni shutdown o reinizializzazione del motore.
        # Qui elenchiamo i nomi reali degli oggetti moderngl creati.
        # BUG FIX (bug 6): rilasciare i VAO PRIMA dei buffer che referenziano.
        # rect_ibo e' condiviso da rect_vao e tex_vao: rilasciarlo prima dei
        # VAO produce GL_INVALID_OPERATION con i debug layer OpenGL (KHR_debug).
        gpu_resource_names = (
            # 1) VAO prima (dipendono dai buffer)
            "rect_vao", "tex_vao",
            "rect_inst_vao", "line_inst_vao", "tri_inst_vao",
            "ellipse_vao", "sprite_inst_vao",
            "ellipse_outline_vao", "rect_outline_vao",
            # 2) IBO/VBO condivisi e specifici
            "rect_vbo", "rect_ibo",
            "tex_vbo",
            "rect_inst_vbo", "rect_inst_ibo", "rect_inst_instance_vbo",
            "line_inst_vbo", "line_inst_ibo", "line_inst_instance_vbo",
            "tri_inst_instance_vbo",
            "ellipse_vbo", "ellipse_ibo", "ellipse_instance_vbo",
            "sprite_inst_vbo", "sprite_inst_ibo", "sprite_inst_instance_vbo",
            "ellipse_outline_vbo", "ellipse_outline_ibo", "ellipse_outline_instance_vbo",
            "rect_outline_vbo", "rect_outline_ibo", "rect_outline_instance_vbo",
            # 3) Program per ultimi
            "prog", "tex_prog", "rect_inst_prog", "line_inst_prog",
            "tri_inst_prog", "ellipse_prog", "sprite_inst_prog",
            "ellipse_outline_prog", "rect_outline_prog",
        )
        for name in gpu_resource_names:
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
                setattr(self, name, None)

        # L'atlas delle texture (self.atlas) ha la propria texture GPU
        # indipendente, da rilasciare separatamente.
        if hasattr(self, "atlas") and self.atlas is not None:
            try:
                self.atlas.release()
            except Exception:
                pass

        # FIX 7: libera esplicitamente i 45+ MB di buffer numpy allocati in
        # _init_draw. Il garbage collector li ripulirà comunque, MA se l'app
        # ricrea WINDOW (reset motore) senza uscire dal processo, con GC
        # disabilitato (gc_auto=True) la memoria rimane indefinitamente.
        for attr_name in ("rect_inst_data", "line_inst_data", "tri_inst_data",
                          "ellipse_instance_data", "sprite_inst_data",
                          "ellipse_outline_instance_data", "rect_outline_instance_data",
                          "_np_batch_buffer", "_np_tex_buffer"):
            if hasattr(self, attr_name):
                setattr(self, attr_name, None)


    # ============================================================
    # COLLISIONI GEOMETRICHE
    # ============================================================

    def PointInRect(self, px, py, x, y, w, h):
        return x <= px <= x + w and y <= py <= y + h

    def PointInEllipse(self, px, py, cx, cy, rx, ry):
        if rx == 0 or ry == 0: return False
        return ((px - cx)**2) / (rx**2) + ((py - cy)**2) / (ry**2) <= 1.0

    def PointInTriangle(self, px, py, x1, y1, x2, y2, x3, y3):
        return self._point_in_triangle(px, py, x1, y1, x2, y2, x3, y3)

    def PointInLine(self, px, py, x1, y1, x2, y2,
                    thickness=1.0, rotation=0.0):
        if rotation != 0.0:
            mx  = (x1 + x2) * 0.5; my = (y1 + y2) * 0.5
            ang = -rotation * self._DEG2RAD
            cs  = math.cos(ang); sn = math.sin(ang)
            dxp = px - mx; dyp = py - my
            px  = mx + dxp*cs - dyp*sn
            py  = my + dxp*sn + dyp*cs

        dx = x2 - x1; dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1) <= thickness / 2.0

        t = ((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (x1 + t*dx), py - (y1 + t*dy)) <= thickness / 2.0

    def _aabb_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2):
        return not (x1+w1 < x2 or x1 > x2+w2 or y1+h1 < y2 or y1 > y2+h2)

    def _ellipse_aabb(self, cx, cy, rx, ry, rotation=0.0):
        if rotation == 0.0: return cx-rx, cy-ry, rx*2.0, ry*2.0
        ang = rotation * self._DEG2RAD
        cs  = math.cos(ang); sn = math.sin(ang)
        hw  = math.sqrt((rx*cs)**2 + (ry*sn)**2)
        hh  = math.sqrt((rx*sn)**2 + (ry*cs)**2)
        return cx-hw, cy-hh, hw*2.0, hh*2.0

    def _triangle_aabb(self, x1, y1, x2, y2, x3, y3):
        return (min(x1,x2,x3), min(y1,y2,y3),
                max(x1,x2,x3)-min(x1,x2,x3),
                max(y1,y2,y3)-min(y1,y2,y3))

    def _point_in_triangle(self, px, py, x1, y1, x2, y2, x3, y3):
        d1 = (px-x2)*(y1-y2) - (x1-x2)*(py-y2)
        d2 = (px-x3)*(y2-y3) - (x2-x3)*(py-y3)
        d3 = (px-x1)*(y3-y1) - (x3-x1)*(py-y1)
        has_neg = (d1 < 0.0) or (d2 < 0.0) or (d3 < 0.0)
        has_pos = (d1 > 0.0) or (d2 > 0.0) or (d3 > 0.0)
        return not (has_neg and has_pos)

    def _point_in_rotated_ellipse(self, px, py, cx, cy, rx, ry, rotation=0.0):
        if rx <= 0.0 or ry <= 0.0: return False
        if rotation != 0.0:
            ang = -rotation * self._DEG2RAD
            cs  = math.cos(ang); sn = math.sin(ang)
            dx  = px - cx; dy = py - cy
            px  = dx*cs - dy*sn
            py  = dx*sn + dy*cs
        else:
            px -= cx; py -= cy
        return (px*px)/(rx*rx) + (py*py)/(ry*ry) <= 1.0

    def _orient(self, ax, ay, bx, by, cx, cy):
        return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)

    def _on_segment(self, ax, ay, bx, by, px, py):
        return (min(ax,bx) <= px <= max(ax,bx) and
                min(ay,by) <= py <= max(ay,by))

    def _segments_intersect(self, ax, ay, bx, by, cx, cy, dx, dy):
        o1 = self._orient(ax, ay, bx, by, cx, cy)
        o2 = self._orient(ax, ay, bx, by, dx, dy)
        o3 = self._orient(cx, cy, dx, dy, ax, ay)
        o4 = self._orient(cx, cy, dx, dy, bx, by)

        # FIX: uso di epsilon invece di == 0.0
        if abs(o1) < self._EPSILON and self._on_segment(ax, ay, bx, by, cx, cy):
            return True
        if abs(o2) < self._EPSILON and self._on_segment(ax, ay, bx, by, dx, dy):
            return True
        if abs(o3) < self._EPSILON and self._on_segment(cx, cy, dx, dy, ax, ay):
            return True
        if abs(o4) < self._EPSILON and self._on_segment(cx, cy, dx, dy, bx, by):
            return True

        return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)

    def _segment_intersects_rect(self, x1, y1, x2, y2, rx, ry, rw, rh):
        if self.PointInRect(x1, y1, rx, ry, rw, rh) or \
        self.PointInRect(x2, y2, rx, ry, rw, rh):
            return True

        dx = x2 - x1
        dy = y2 - y1
        t0 = 0.0
        t1 = 1.0

        def clip(p, q, t0, t1):
            # FIX: epsilon invece di == 0.0
            if abs(p) < 1e-9:
                return (q >= 0.0), t0, t1
            r = q / p
            if p < 0.0:
                if r > t1: return False, t0, t1
                if r > t0: t0 = r
            else:
                if r < t0: return False, t0, t1
                if r < t1: t1 = r
            return True, t0, t1

        ok, t0, t1 = clip(-dx, x1 - rx,      t0, t1)
        if not ok: return False
        ok, t0, t1 = clip( dx, rx + rw - x1, t0, t1)
        if not ok: return False
        ok, t0, t1 = clip(-dy, y1 - ry,      t0, t1)
        if not ok: return False
        ok, t0, t1 = clip( dy, ry + rh - y1, t0, t1)
        if not ok: return False
        return t0 <= t1


    def _segment_intersects_triangle(self, x1,y1,x2,y2,tx1,ty1,tx2,ty2,tx3,ty3):
        if self._point_in_triangle(x1,y1,tx1,ty1,tx2,ty2,tx3,ty3): return True
        if self._point_in_triangle(x2,y2,tx1,ty1,tx2,ty2,tx3,ty3): return True
        if self._segments_intersect(x1,y1,x2,y2,tx1,ty1,tx2,ty2): return True
        if self._segments_intersect(x1,y1,x2,y2,tx2,ty2,tx3,ty3): return True
        if self._segments_intersect(x1,y1,x2,y2,tx3,ty3,tx1,ty1): return True
        return False

    def _segment_intersects_ellipse(self, x1,y1,x2,y2,cx,cy,rx,ry,rotation=0.0):
        if rx <= 0.0 or ry <= 0.0: return False
        if rotation != 0.0:
            ang = -rotation * self._DEG2RAD
            cs  = math.cos(ang); sn = math.sin(ang)
            dx1 = x1-cx; dy1 = y1-cy; dx2 = x2-cx; dy2 = y2-cy
            x1  = dx1*cs - dy1*sn; y1 = dx1*sn + dy1*cs
            x2  = dx2*cs - dy2*sn; y2 = dx2*sn + dy2*cs
        else:
            x1 -= cx; y1 -= cy; x2 -= cx; y2 -= cy
        dx = x2-x1; dy = y2-y1
        irx2 = 1.0/(rx*rx); iry2 = 1.0/(ry*ry)
        a = dx*dx*irx2 + dy*dy*iry2
        b = 2.0*(x1*dx*irx2 + y1*dy*iry2)
        c = x1*x1*irx2 + y1*y1*iry2 - 1.0
        if c <= 0.0: return True
        if x2*x2*irx2 + y2*y2*iry2 - 1.0 <= 0.0: return True
        disc = b*b - 4.0*a*c
        if disc < 0.0: return False
        sq  = math.sqrt(disc); i2a = 0.5/a
        t1  = (-b - sq)*i2a; t2 = (-b + sq)*i2a
        return (0.0<=t1<=1.0) or (0.0<=t2<=1.0)

    # --- Collisioni pubbliche (invariate) ---

    def CollideRectRect(self, x1,y1,w1,h1,x2,y2,w2,h2):
        return self._aabb_overlap(x1,y1,w1,h1,x2,y2,w2,h2)
    
    def CollidePointRectBatch(self, px_arr, py_arr, x_arr, y_arr, w_arr, h_arr):
        """Verifica le collisioni punto-rettangolo su interi array NumPy."""
        return (x_arr <= px_arr) & (px_arr <= x_arr + w_arr) & (y_arr <= py_arr) & (py_arr <= y_arr + h_arr)

    def CollidePointRect(self, px,py,x,y,w,h):
        return x<=px<=x+w and y<=py<=y+h

    def CollidePointTriangle(self, px,py,x1,y1,x2,y2,x3,y3):
        return self._point_in_triangle(px,py,x1,y1,x2,y2,x3,y3)

    def CollidePointEllipse(self, px,py,cx,cy,rx,ry,rotation=0.0):
        return self._point_in_rotated_ellipse(px,py,cx,cy,rx,ry,rotation)

    def CollideLineLine(self, x1,y1,x2,y2,x3,y3,x4,y4):
        mn1x=min(x1,x2); mn1y=min(y1,y2)
        mx1x=max(x1,x2); mx1y=max(y1,y2)
        mn2x=min(x3,x4); mn2y=min(y3,y4)
        mx2x=max(x3,x4); mx2y=max(y3,y4)
        if not self._aabb_overlap(mn1x,mn1y,mx1x-mn1x,mx1y-mn1y,
                                   mn2x,mn2y,mx2x-mn2x,mx2y-mn2y):
            return False
        return self._segments_intersect(x1,y1,x2,y2,x3,y3,x4,y4)

    def CollideLineRect(self, x1,y1,x2,y2,rx,ry,rw,rh):
        mnx=min(x1,x2); mny=min(y1,y2)
        mxx=max(x1,x2); mxy=max(y1,y2)
        if not self._aabb_overlap(mnx,mny,mxx-mnx,mxy-mny,rx,ry,rw,rh):
            return False
        return self._segment_intersects_rect(x1,y1,x2,y2,rx,ry,rw,rh)

    def CollideLineTriangle(self, x1,y1,x2,y2,tx1,ty1,tx2,ty2,tx3,ty3):
        tmnx,tmny,tww,thh = self._triangle_aabb(tx1,ty1,tx2,ty2,tx3,ty3)
        lmnx=min(x1,x2); lmny=min(y1,y2)
        lmxx=max(x1,x2); lmxy=max(y1,y2)
        if not self._aabb_overlap(lmnx,lmny,lmxx-lmnx,lmxy-lmny,
                                   tmnx,tmny,tww,thh): return False
        return self._segment_intersects_triangle(x1,y1,x2,y2,
                                                  tx1,ty1,tx2,ty2,tx3,ty3)

    def CollideLineEllipse(self, x1,y1,x2,y2,cx,cy,rx,ry,rotation=0.0):
        ex,ey,ew,eh = self._ellipse_aabb(cx,cy,rx,ry,rotation)
        lmnx=min(x1,x2); lmny=min(y1,y2)
        lmxx=max(x1,x2); lmxy=max(y1,y2)
        if not self._aabb_overlap(lmnx,lmny,lmxx-lmnx,lmxy-lmny,
                                   ex,ey,ew,eh): return False
        return self._segment_intersects_ellipse(x1,y1,x2,y2,cx,cy,rx,ry,rotation)

    def CollideRectTriangle(self, rx,ry,rw,rh,tx1,ty1,tx2,ty2,tx3,ty3):
        tmnx,tmny,tww,thh = self._triangle_aabb(tx1,ty1,tx2,ty2,tx3,ty3)
        if not self._aabb_overlap(rx,ry,rw,rh,tmnx,tmny,tww,thh): return False
        if self.CollidePointRect(tx1,ty1,rx,ry,rw,rh): return True
        if self.CollidePointRect(tx2,ty2,rx,ry,rw,rh): return True
        if self.CollidePointRect(tx3,ty3,rx,ry,rw,rh): return True
        if self._point_in_triangle(rx,   ry,   tx1,ty1,tx2,ty2,tx3,ty3): return True
        if self._point_in_triangle(rx+rw,ry,   tx1,ty1,tx2,ty2,tx3,ty3): return True
        if self._point_in_triangle(rx+rw,ry+rh,tx1,ty1,tx2,ty2,tx3,ty3): return True
        if self._point_in_triangle(rx,   ry+rh,tx1,ty1,tx2,ty2,tx3,ty3): return True
        for (ax,ay,bx,by) in [(rx,ry,rx+rw,ry),(rx+rw,ry,rx+rw,ry+rh),
                               (rx+rw,ry+rh,rx,ry+rh),(rx,ry+rh,rx,ry)]:
            if self._segments_intersect(ax,ay,bx,by,tx1,ty1,tx2,ty2): return True
            if self._segments_intersect(ax,ay,bx,by,tx2,ty2,tx3,ty3): return True
            if self._segments_intersect(ax,ay,bx,by,tx3,ty3,tx1,ty1): return True
        return False

    def CollideRectEllipse(self, rx,ry,rw,rh,cx,cy,erx,ery,rotation=0.0):
        ex,ey,ew,eh = self._ellipse_aabb(cx,cy,erx,ery,rotation)
        if not self._aabb_overlap(rx,ry,rw,rh,ex,ey,ew,eh): return False
        if self._point_in_rotated_ellipse(rx,   ry,   cx,cy,erx,ery,rotation): return True
        if self._point_in_rotated_ellipse(rx+rw,ry,   cx,cy,erx,ery,rotation): return True
        if self._point_in_rotated_ellipse(rx+rw,ry+rh,cx,cy,erx,ery,rotation): return True
        if self._point_in_rotated_ellipse(rx,   ry+rh,cx,cy,erx,ery,rotation): return True
        if self.CollidePointRect(cx,cy,rx,ry,rw,rh): return True
        if self._segment_intersects_ellipse(rx,ry,rx+rw,ry,      cx,cy,erx,ery,rotation): return True
        if self._segment_intersects_ellipse(rx+rw,ry,rx+rw,ry+rh,cx,cy,erx,ery,rotation): return True
        if self._segment_intersects_ellipse(rx+rw,ry+rh,rx,ry+rh,cx,cy,erx,ery,rotation): return True
        if self._segment_intersects_ellipse(rx,ry+rh,rx,ry,      cx,cy,erx,ery,rotation): return True
        return False

    def CollideTriangleTriangle(self, a1x,a1y,a2x,a2y,a3x,a3y,
                                 b1x,b1y,b2x,b2y,b3x,b3y):
        aminx,aminy,aw,ah = self._triangle_aabb(a1x,a1y,a2x,a2y,a3x,a3y)
        bminx,bminy,bw,bh = self._triangle_aabb(b1x,b1y,b2x,b2y,b3x,b3y)
        if not self._aabb_overlap(aminx,aminy,aw,ah,bminx,bminy,bw,bh): return False
        if self._point_in_triangle(a1x,a1y,b1x,b1y,b2x,b2y,b3x,b3y): return True
        if self._point_in_triangle(a2x,a2y,b1x,b1y,b2x,b2y,b3x,b3y): return True
        if self._point_in_triangle(a3x,a3y,b1x,b1y,b2x,b2y,b3x,b3y): return True
        if self._point_in_triangle(b1x,b1y,a1x,a1y,a2x,a2y,a3x,a3y): return True
        if self._point_in_triangle(b2x,b2y,a1x,a1y,a2x,a2y,a3x,a3y): return True
        if self._point_in_triangle(b3x,b3y,a1x,a1y,a2x,a2y,a3x,a3y): return True
        for (ax,ay,bx,by) in [(a1x,a1y,a2x,a2y),(a2x,a2y,a3x,a3y),(a3x,a3y,a1x,a1y)]:
            for (cx,cy,dx,dy) in [(b1x,b1y,b2x,b2y),(b2x,b2y,b3x,b3y),(b3x,b3y,b1x,b1y)]:
                if self._segments_intersect(ax,ay,bx,by,cx,cy,dx,dy): return True
        return False

    def CollideEllipseEllipse(self, c1x, c1y, r1x, r1y,
                           c2x, c2y, r2x, r2y,
                           rot1=0.0, rot2=0.0, samples=20):
        ex1, ey1, ew1, eh1 = self._ellipse_aabb(c1x, c1y, r1x, r1y, rot1)
        ex2, ey2, ew2, eh2 = self._ellipse_aabb(c2x, c2y, r2x, r2y, rot2)
        if not self._aabb_overlap(ex1, ey1, ew1, eh1, ex2, ey2, ew2, eh2):
            return False
        if self._point_in_rotated_ellipse(c1x, c1y, c2x, c2y, r2x, r2y, rot2):
            return True
        if self._point_in_rotated_ellipse(c2x, c2y, c1x, c1y, r1x, r1y, rot1):
            return True

        if samples is None:
            samples = max(20, int(max(r1x, r1y) * 0.4))

        ang1 = rot1 * self._DEG2RAD
        cs1  = math.cos(ang1)
        sn1  = math.sin(ang1)
        step = self._TAU / samples
        sc   = math.cos(step)
        ss   = math.sin(step)
        ca   = 1.0
        sa   = 0.0

        # BUG FIX 5: prima campionavamo solo il perimetro dell'ellisse 1 vs
        # ellisse 2. Se due ellissi allungate si sfiorano in un tratto molto
        # piccolo del perimetro dell'ellisse 2, nessun campione fisso su
        # ellisse 1 poteva cadere dentro ellisse 2 => falso negativo. Ora
        # campioniamo il perimetro di ENTRAMBE le ellissi, coprendo
        # simmetricamente il caso di tangenze / intersezioni sottili.
        ang2 = rot2 * self._DEG2RAD
        cs2  = math.cos(ang2)
        sn2  = math.sin(ang2)

        ca = 1.0
        sa = 0.0
        for _ in range(samples):
            # Punto locale sul perimetro dell'ellisse 1 (r1x*ca, r1y*sa)
            # ruotato e traslato in world.
            lx = r1x * ca
            ly = r1y * sa
            px = c1x + lx * cs1 - ly * sn1
            py = c1y + lx * sn1 + ly * cs1
            if self._point_in_rotated_ellipse(px, py, c2x, c2y, r2x, r2y, rot2):
                return True
            nca = ca * sc - sa * ss
            sa  = ca * ss + sa * sc
            ca  = nca

        # Secondo giro: perimetro dell'ellisse 2 vs ellisse 1.
        ca = 1.0
        sa = 0.0
        for _ in range(samples):
            lx = r2x * ca
            ly = r2y * sa
            px = c2x + lx * cs2 - ly * sn2
            py = c2y + lx * sn2 + ly * cs2
            if self._point_in_rotated_ellipse(px, py, c1x, c1y, r1x, r1y, rot1):
                return True
            nca = ca * sc - sa * ss
            sa  = ca * ss + sa * sc
            ca  = nca

        return False


    def CollidePointCircle(self, px,py,cx,cy,r):
        dx=px-cx; dy=py-cy
        return dx*dx+dy*dy <= r*r

    def CollideCircleCircle(self, c1x,c1y,r1,c2x,c2y,r2):
        dx=c2x-c1x; dy=c2y-c1y; rs=r1+r2
        return dx*dx+dy*dy <= rs*rs

    def CollideRotatedRectCircle(self, rx,ry,rw,rh,rotation,cx,cy,cr):
        rcx=rx+rw*0.5; rcy=ry+rh*0.5
        if rotation != 0.0:
            ang=(-rotation)*self._DEG2RAD
            cs=math.cos(ang); sn=math.sin(ang)
            dx=cx-rcx; dy=cy-rcy
            local_cx=rcx+(dx*cs-dy*sn)
            local_cy=rcy+(dx*sn+dy*cs)
        else:
            local_cx=cx; local_cy=cy
        closest_x=max(rx,min(local_cx,rx+rw))
        closest_y=max(ry,min(local_cy,ry+rh))
        dx=local_cx-closest_x; dy=local_cy-closest_y
        return dx*dx+dy*dy <= cr*cr

    def CollidePointRotatedRect(self, px,py,rx,ry,rw,rh,rotation):
        if rotation==0.0: return self.PointInRect(px,py,rx,ry,rw,rh)
        cx=rx+rw*0.5; cy=ry+rh*0.5
        ang=(-rotation)*self._DEG2RAD
        cs=math.cos(ang); sn=math.sin(ang)
        dx=px-cx; dy=py-cy
        local_x=cx+dx*cs-dy*sn
        local_y=cy+dx*sn+dy*cs
        return self.PointInRect(local_x,local_y,rx,ry,rw,rh)

    # ================================================================
    # FIX 6: collisioni mancanti (tabella completa) + batch PointIn*
    # ================================================================

    # --- Batch dei PointIn* (vettorizzati NumPy, stesso stile di
    # CollidePointRectBatch). Ognuno accetta array (N,) di px,py e i
    # parametri della forma singola (scalari) oppure array della stessa lunghezza.
    def CollidePointCircleBatch(self, px_arr, py_arr, cx, cy, r):
        px = np.asarray(px_arr, dtype='f4'); py = np.asarray(py_arr, dtype='f4')
        dx = px - np.asarray(cx, dtype='f4')
        dy = py - np.asarray(cy, dtype='f4')
        return (dx*dx + dy*dy) <= (np.asarray(r, dtype='f4') ** 2)

    def CollidePointEllipseBatch(self, px_arr, py_arr, cx, cy, rx, ry, rotation=0.0):
        px = np.asarray(px_arr, dtype='f4'); py = np.asarray(py_arr, dtype='f4')
        cx = np.asarray(cx, dtype='f4'); cy = np.asarray(cy, dtype='f4')
        rx = np.asarray(rx, dtype='f4'); ry = np.asarray(ry, dtype='f4')
        dx = px - cx; dy = py - cy
        rot = np.asarray(rotation, dtype='f4')
        if np.any(rot != 0.0):
            ang = -rot * np.float32(_DEG2RAD_CONST)
            cs = np.cos(ang); sn = np.sin(ang)
            lx = dx * cs - dy * sn
            ly = dx * sn + dy * cs
            dx, dy = lx, ly
        return (dx*dx) / np.maximum(rx*rx, 1e-12) + (dy*dy) / np.maximum(ry*ry, 1e-12) <= 1.0

    def CollidePointTriangleBatch(self, px_arr, py_arr, x1,y1,x2,y2,x3,y3):
        px = np.asarray(px_arr, dtype='f4'); py = np.asarray(py_arr, dtype='f4')
        d1 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
        d2 = (px - x3) * (y2 - y3) - (x2 - x3) * (py - y3)
        d3 = (px - x1) * (y3 - y1) - (x3 - x1) * (py - y1)
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return ~(has_neg & has_pos)

    def CollidePointRotatedRectBatch(self, px_arr, py_arr, rx, ry, rw, rh, rotation):
        px = np.asarray(px_arr, dtype='f4'); py = np.asarray(py_arr, dtype='f4')
        cx = rx + rw * 0.5; cy = ry + rh * 0.5
        ang = -float(rotation) * _DEG2RAD_CONST
        cs = math.cos(ang); sn = math.sin(ang)
        dx = px - cx; dy = py - cy
        lx = dx * cs - dy * sn
        ly = dx * sn + dy * cs
        hw = rw * 0.5; hh = rh * 0.5
        return (np.abs(lx) <= hw) & (np.abs(ly) <= hh)

    # --- 8 collisioni forma-forma mancanti dalla tabella ---

    def CollideLineCircle(self, x1,y1,x2,y2,cx,cy,cr):
        """Line-vs-Circle: caso specializzato di CollideLineEllipse."""
        return self._segment_intersects_ellipse(x1,y1,x2,y2, cx,cy, cr,cr, 0.0) \
            or self.CollidePointCircle(x1,y1,cx,cy,cr) \
            or self.CollidePointCircle(x2,y2,cx,cy,cr)

    def CollideLineRotatedRect(self, x1,y1,x2,y2, rx,ry,rw,rh, rotation):
        """Line-vs-RotatedRect: trasforma la linea nel frame locale del rect."""
        cx = rx + rw * 0.5; cy = ry + rh * 0.5
        ang = -float(rotation) * _DEG2RAD_CONST
        cs = math.cos(ang); sn = math.sin(ang)
        def to_local(px, py):
            dx = px - cx; dy = py - cy
            return (cx + dx*cs - dy*sn, cy + dx*sn + dy*cs)
        lx1, ly1 = to_local(x1, y1)
        lx2, ly2 = to_local(x2, y2)
        return self._segment_intersects_rect(lx1, ly1, lx2, ly2, rx, ry, rw, rh)

    def CollideRectCircle(self, rx,ry,rw,rh, cx,cy,cr):
        """Rect (AABB) vs Circle: caso specializzato di CollideRectEllipse."""
        return self.CollideRectEllipse(rx, ry, rw, rh, cx, cy, cr, cr, 0.0)

    def CollideRectRotatedRect(self, ax,ay,aw,ah, bx,by,bw,bh, b_rotation):
        """AABB vs RotatedRect via SAT (4 assi = 2 dell'AABB + 2 del rotato)."""
        return self.CollideRotatedRectRotatedRect(
            ax, ay, aw, ah, 0.0,
            bx, by, bw, bh, b_rotation
        )

    def CollideCircleEllipse(self, cx,cy,cr, ecx,ecy,erx,ery, rotation=0.0):
        """Circle vs Ellipse: cerchio = ellisse degenere, delega a EllipseEllipse.

        BUG CRITICO FIX 1: i kwargs erano `a_rotation=/b_rotation=`, che non
        esistono nella firma di CollideEllipseEllipse (i suoi parametri si
        chiamano `rot1`/`rot2`). Ogni chiamata sollevava TypeError."""
        return self.CollideEllipseEllipse(cx, cy, cr, cr,
                                          ecx, ecy, erx, ery,
                                          rot1=0.0, rot2=rotation)

    def CollideCircleTriangle(self, cx,cy,cr, tx1,ty1,tx2,ty2,tx3,ty3):
        """Circle vs Triangle: centro dentro triangolo OR uno dei 3 lati
        interseca il cerchio (test lato-cerchio = segment-vs-ellipse rx=ry)."""
        if self._point_in_triangle(cx, cy, tx1,ty1,tx2,ty2,tx3,ty3):
            return True
        for (ax, ay, bx, by) in ((tx1,ty1,tx2,ty2),
                                 (tx2,ty2,tx3,ty3),
                                 (tx3,ty3,tx1,ty1)):
            if self._segment_intersects_ellipse(ax, ay, bx, by, cx, cy, cr, cr, 0.0):
                return True
            if self.CollidePointCircle(ax, ay, cx, cy, cr):
                return True
        return False

    def CollideEllipseTriangle(self, cx,cy,rx,ry, tx1,ty1,tx2,ty2,tx3,ty3, rotation=0.0):
        """Ellipse vs Triangle: centro dentro OR uno dei 3 lati interseca l'ellisse."""
        if self._point_in_triangle(cx, cy, tx1,ty1,tx2,ty2,tx3,ty3):
            return True
        for (ax, ay, bx, by) in ((tx1,ty1,tx2,ty2),
                                 (tx2,ty2,tx3,ty3),
                                 (tx3,ty3,tx1,ty1)):
            if self._segment_intersects_ellipse(ax, ay, bx, by, cx, cy, rx, ry, rotation):
                return True
            if self._point_in_rotated_ellipse(ax, ay, cx, cy, rx, ry, rotation):
                return True
        return False

    def CollideEllipseRotatedRect(self, cx,cy,rx,ry, e_rotation,
                                  bx,by,bw,bh, r_rotation):
        """Ellipse vs RotatedRect: porta l'ellisse nel frame locale del rect,
        poi test contro AABB (con rotazione locale = e_rotation - r_rotation)."""
        rcx = bx + bw * 0.5; rcy = by + bh * 0.5
        ang = -float(r_rotation) * _DEG2RAD_CONST
        cs = math.cos(ang); sn = math.sin(ang)
        dx = cx - rcx; dy = cy - rcy
        lcx = rcx + dx*cs - dy*sn
        lcy = rcy + dx*sn + dy*cs
        return self.CollideRectEllipse(
            bx, by, bw, bh,
            lcx, lcy, rx, ry,
            rotation=(e_rotation - r_rotation)
        )

    def CollideTriangleRotatedRect(self, tx1,ty1,tx2,ty2,tx3,ty3,
                                   rx,ry,rw,rh, rotation):
        """Triangle vs RotatedRect: trasforma il triangolo nel frame locale
        del rect e delega a CollideRectTriangle (AABB vs triangolo)."""
        cx = rx + rw * 0.5; cy = ry + rh * 0.5
        ang = -float(rotation) * _DEG2RAD_CONST
        cs = math.cos(ang); sn = math.sin(ang)
        def to_local(px, py):
            dx = px - cx; dy = py - cy
            return (cx + dx*cs - dy*sn, cy + dx*sn + dy*cs)
        lx1, ly1 = to_local(tx1, ty1)
        lx2, ly2 = to_local(tx2, ty2)
        lx3, ly3 = to_local(tx3, ty3)
        return self.CollideRectTriangle(rx, ry, rw, rh,
                                        lx1, ly1, lx2, ly2, lx3, ly3)

    def CollideRotatedRectRotatedRect(self, ax,ay,aw,ah,a_rotation,
                                      bx,by,bw,bh,b_rotation):
        """OBB vs OBB via SAT (Separating Axis Theorem) su 4 assi."""
        def corners(x, y, w, h, rot):
            cx = x + w * 0.5; cy = y + h * 0.5
            hw = w * 0.5; hh = h * 0.5
            ang = float(rot) * _DEG2RAD_CONST
            cs = math.cos(ang); sn = math.sin(ang)
            pts = []
            for lx, ly in ((-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)):
                pts.append((cx + lx*cs - ly*sn, cy + lx*sn + ly*cs))
            return pts

        A = corners(ax, ay, aw, ah, a_rotation)
        B = corners(bx, by, bw, bh, b_rotation)

        def axes(poly):
            out = []
            for i in range(4):
                x1, y1 = poly[i]; x2, y2 = poly[(i+1) % 4]
                # normale al lato
                out.append((-(y2 - y1), x2 - x1))
            return out

        def project(poly, axis):
            ax_, ay_ = axis
            dots = [px*ax_ + py*ay_ for (px, py) in poly]
            return min(dots), max(dots)

        for axis in axes(A) + axes(B):
            # normalizza per stabilita' numerica (non necessario per SAT,
            # ma evita overflow con OBB grandi)
            ax_, ay_ = axis
            L = math.hypot(ax_, ay_)
            if L < 1e-12:
                continue
            n = (ax_ / L, ay_ / L)
            aMin, aMax = project(A, n)
            bMin, bMax = project(B, n)
            if aMax < bMin or bMax < aMin:
                return False
        return True

    def CollidePointTexture(self, px,py,tx,ty,tw,th,rotation=0.0):
        return self.CollidePointRotatedRect(px,py,tx,ty,tw,th,rotation)
