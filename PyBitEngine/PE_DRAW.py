import sdl2, moderngl, time
import sdl2.sdlimage as img
import os, numpy as np, math
import ctypes
import warnings
from math import hypot as _math_hypot
from collections import OrderedDict
from numba import njit

# Costanti tipo-evento/pulsante mouse, usate dai metodi PE_COLLISION di DRAW
# (UpdateMouseState, MousePressed, MouseReleased, MouseHeld, MouseDragging...).
# PE_KEYS non dipende da PE_DRAW (importa solo sdl2), quindi nessun import
# circolare.
from .PE_KEYS import (
    PE_MOUSEMOTION, PE_MOUSEBUTTONDOWN, PE_MOUSEBUTTONUP,
    PE_MOUSEDRAG, PE_MOUSEWHEEL, PE_MOUSE_LEFT,
)

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
# PERF FIX: rimosso parallel=True/prange da questi kernel. Sono chiamati
# UNA VOLTA PER FRAME per ogni Draw*Batch, con N tipico da poche decine a
# poche migliaia di istanze, e fanno solo copie di pochi float per elemento
# (nessun calcolo pesante): sono quindi puramente memory-bound. Con
# parallel=True, ogni chiamata paga il costo fisso di dispatch/join del
# thread pool di Numba (decine di microsecondi), che per questi carichi
# tipici e' piu' grande del tempo risparmiato parallelizzando una manciata
# di assegnazioni: il risultato netto era un rallentamento per-frame, non
# un'accelerazione. Kernel sequenziale + cache=True (compilato una sola
# volta, poi sempre veloce) e' la scelta giusta qui.
@njit(fastmath=True, cache=True)
def _numba_pack_rect_instances(pos, size, cos_a, sin_a, color_bits, out):
    """Impacchetta (N,7) instance buffer per DrawRectsBatch: pos2+size2+dir2+color(1, uint32 packed)."""
    n = pos.shape[0]
    for i in range(n):
        out[i, 0] = pos[i, 0]
        out[i, 1] = pos[i, 1]
        out[i, 2] = size[i, 0]
        out[i, 3] = size[i, 1]
        out[i, 4] = cos_a[i]
        out[i, 5] = sin_a[i]
        out[i, 6] = color_bits[i]


@njit(fastmath=True, cache=True)
def _numba_pack_ellipse_instances(centers, radii, cos_a, sin_a, color_bits, out):
    """Impacchetta (N,7) instance buffer: center2+radius2+dir2+color(1, uint32 packed)."""
    n = centers.shape[0]
    for i in range(n):
        out[i, 0] = centers[i, 0]
        out[i, 1] = centers[i, 1]
        out[i, 2] = radii[i, 0]
        out[i, 3] = radii[i, 1]
        out[i, 4] = cos_a[i]
        out[i, 5] = sin_a[i]
        out[i, 6] = color_bits[i]


@njit(fastmath=True, cache=True)
def _numba_pack_ellipse_outline_instances(centers, radii, thickness, cos_a, sin_a, color_bits, out):
    """Impacchetta (N,8) instance buffer per DrawEllipsesOutlineBatch/
    DrawCircleOutlineBatch: center2+radius2+thickness1+dir2+color(1, uint32 packed).
    Stessa architettura/prestazioni di _numba_pack_ellipse_instances
    (kernel Numba compilato, zero overhead Python per istanza)."""
    n = centers.shape[0]
    for i in range(n):
        out[i, 0] = centers[i, 0]
        out[i, 1] = centers[i, 1]
        out[i, 2] = radii[i, 0]
        out[i, 3] = radii[i, 1]
        out[i, 4] = thickness[i]
        out[i, 5] = cos_a[i]
        out[i, 6] = sin_a[i]
        out[i, 7] = color_bits[i]


@njit(fastmath=True, cache=True)
def _numba_pack_rect_outline_instances(pos, size, thickness, cos_a, sin_a, color_bits, out):
    """Impacchetta (N,8) instance buffer per DrawRectsOutlineBatch:
    pos2+size2+thickness1+dir2+color(1, uint32 packed). Stessa architettura/
    prestazioni di _numba_pack_rect_instances."""
    n = pos.shape[0]
    for i in range(n):
        out[i, 0] = pos[i, 0]
        out[i, 1] = pos[i, 1]
        out[i, 2] = size[i, 0]
        out[i, 3] = size[i, 1]
        out[i, 4] = thickness[i]
        out[i, 5] = cos_a[i]
        out[i, 6] = sin_a[i]
        out[i, 7] = color_bits[i]


@njit(fastmath=True, cache=True)
def _numba_pack_line_instances(p1, p2, thick, color_bits, out):
    """Impacchetta (N,6) instance buffer: p1(2)+p2(2)+thickness(1)+color(1, uint32 packed)."""
    n = p1.shape[0]
    for i in range(n):
        out[i, 0] = p1[i, 0]
        out[i, 1] = p1[i, 1]
        out[i, 2] = p2[i, 0]
        out[i, 3] = p2[i, 1]
        out[i, 4] = thick[i]
        out[i, 5] = color_bits[i]


@njit(fastmath=True, cache=True)
def _numba_pack_tri_instances(v0, v1, v2, color_bits, out):
    """Impacchetta (N,7) instance buffer: v0(2)+v1(2)+v2(2)+color(1, uint32 packed)."""
    n = v0.shape[0]
    for i in range(n):
        out[i, 0] = v0[i, 0]; out[i, 1] = v0[i, 1]
        out[i, 2] = v1[i, 0]; out[i, 3] = v1[i, 1]
        out[i, 4] = v2[i, 0]; out[i, 5] = v2[i, 1]
        out[i, 6] = color_bits[i]


@njit(fastmath=True, cache=True)
def _numba_rotate_lines(x1, y1, x2, y2, cs, sn):
    """Ruota N segmenti attorno al proprio punto medio (in-place style)."""
    n = x1.shape[0]
    for i in range(n):
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
@njit(fastmath=True, cache=True)
def _numba_rotate_lines_arr(x1, y1, x2, y2, cs_arr, sn_arr):
    """Ruota N segmenti attorno al proprio punto medio, angolo per segmento."""
    n = x1.shape[0]
    for i in range(n):
        mx = (x1[i] + x2[i]) * 0.5
        my = (y1[i] + y2[i]) * 0.5
        dx1 = x1[i] - mx; dy1 = y1[i] - my
        dx2 = x2[i] - mx; dy2 = y2[i] - my
        cs = cs_arr[i]; sn = sn_arr[i]
        x1[i] = mx + dx1*cs - dy1*sn
        y1[i] = my + dx1*sn + dy1*cs
        x2[i] = mx + dx2*cs - dy2*sn
        y2[i] = my + dx2*sn + dy2*cs


@njit(fastmath=True, cache=True)
def _numba_pack_line_instances_xy(x1, y1, x2, y2, thick, color_bits, out):
    """Come _numba_pack_line_instances ma legge x/y da 4 array 1D
    (evita np.column_stack: risparmia 2 allocazioni (N,2) per chiamata).
    Output (N,6): p1(2)+p2(2)+thickness(1)+color(1, uint32 packed)."""
    n = x1.shape[0]
    for i in range(n):
        out[i, 0] = x1[i]
        out[i, 1] = y1[i]
        out[i, 2] = x2[i]
        out[i, 3] = y2[i]
        out[i, 4] = thick[i]
        out[i, 5] = color_bits[i]



@njit(fastmath=True, cache=True)
def _numba_clip_rgba(rgba_in, alpha_scalar, use_alpha_scalar, alpha_arr, out):
    """
    Normalizza colori: clip 0-255 e sovrascrittura alpha.
    use_alpha_scalar=1 usa alpha_scalar; altrimenti usa alpha_arr (len=n).
    """
    n = rgba_in.shape[0]
    for i in range(n):
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


# ---------------------------------------------------------------------- #
# COLOR PACKING (RGBA8 -> singolo uint32 per istanza)
# ---------------------------------------------------------------------- #
# PERF FIX (packing colori in UINT): i vecchi instance buffer portavano
# i_color come vec4 (16 byte/istanza). Impacchettando R,G,B,A in un solo
# uint32 (4 byte) il vertex shader lo decodifica con pochi bitshift (vedi
# sostituzione "in uint i_color" + unpack manuale nei vertex shader sopra:
# GLSL 330 core non garantisce unpackUnorm4x8, che e' core solo da GLSL
# 4.20/GL_ARB_shading_language_packing, quindi lo shader fa lo shift a
# mano invece di affidarsi a quella funzione). Risultato: -12 byte/istanza
# su ogni pipeline instanced (rect/rrect/rtri/ellipse/linee/triangoli/
# sprite/testo), quindi meno traffico VBO per frame, specialmente evidente
# con 100k+ istanze.
#
# Il trucco per restare a costo zero: la funzione ritorna un array float32
# il cui pattern di bit e' IDENTICO all'uint32 impacchettato (via
# numpy .view(), zero conversioni/copie ulteriori). Questo permette di
# scrivere il valore direttamente in una colonna degli stessi buffer
# instance float32 pre-allocati (nessun cambio di dtype/struct necessario)
# — moderngl scrive i byte grezzi sulla GPU e la VAO li reinterpreta come
# `uint` in base alla format string ("...1u/i"), indipendentemente dal
# dtype numpy usato per costruirli lato CPU.
def _pack_rgba_u32_as_f4(rgba):
    """rgba: array (N,4) di float 0..255 -> array (N,) float32 il cui
    bit-pattern e' il colore RGBA8 impacchettato (R nel byte basso, A nel
    byte alto), pronto per essere scritto in una colonna 'float32' di un
    instance buffer e letto lato GPU come uint (i_color)."""
    u8 = np.clip(np.asarray(rgba), 0.0, 255.0).astype(np.uint8)
    packed = (u8[:, 0].astype(np.uint32)
              | (u8[:, 1].astype(np.uint32) << 8)
              | (u8[:, 2].astype(np.uint32) << 16)
              | (u8[:, 3].astype(np.uint32) << 24))
    return packed.view(np.float32)


def _pack_rgba_u32_scalar_as_f4(r, g, b, a):
    """Variante scalare di _pack_rgba_u32_as_f4, per i path 'immediate'
    (DrawRoundedRect, DrawRoundedTriangle, ...) che scrivono una singola
    riga alla volta in un instance buffer preallocato."""
    r = 0 if r < 0.0 else (255 if r > 255.0 else int(r))
    g = 0 if g < 0.0 else (255 if g > 255.0 else int(g))
    b = 0 if b < 0.0 else (255 if b > 255.0 else int(b))
    a = 0 if a < 0.0 else (255 if a > 255.0 else int(a))
    packed = np.uint32(r | (g << 8) | (b << 16) | (a << 24))
    return packed.view(np.float32).item()


@njit(fastmath=True, cache=True)
def _numba_layout_glyphs(gx, gy, gw, gh, guv,
                         origin_x, origin_y, cos_r, sin_r,
                         color_bits, out):
    """Layout+rotazione+packing dei glifi per DrawTextBatch.
    Output (N,11): pos2+size2+dir2+uv4+color(1, uint32 packed) — stesso
    layout dello sprite batch.

    BUG FIX (rotazione testo): lo shader condiviso sprite_inst_prog calcola
    world = i_pos + half_size(NON ruotato) + rotate(corner_offset), perche'
    ogni istanza e' pensata per ruotare attorno al proprio centro FISSO
    (i_pos + half_size resta costante al variare dell'angolo: e' il
    comportamento giusto per sprite/rettangoli indipendenti). Prima di
    questo fix, qui si ruotava solo l'angolo top-left del glifo (lx,ly)
    attorno all'origine della stringa e si passava quello come i_pos: lo
    shader ci sommava sopra un half_size NON ruotato, introducendo un
    offset che cresce con l'angolo -> a rotation!=0 i glifi finivano
    disallineati (sovrapposti o spostati in verticale), tanto piu' quanto
    piu' la rotazione era marcata. Ora ruotiamo il CENTRO del glifo attorno
    all'origine e pre-sottraiamo lo stesso half_size non ruotato che lo
    shader ri-aggiungera': i due si cancellano e il centro del glifo finisce
    esattamente su origin + rotate(local_center), come deve essere per un
    blocco di testo che ruota rigidamente attorno alla propria origine."""
    n = gx.shape[0]
    for i in range(n):
        half_w = gw[i] * 0.5
        half_h = gh[i] * 0.5
        cx = gx[i] + half_w
        cy = gy[i] + half_h
        rot_cx = cx * cos_r - cy * sin_r
        rot_cy = cx * sin_r + cy * cos_r
        out[i, 0]  = origin_x + rot_cx - half_w
        out[i, 1]  = origin_y + rot_cy - half_h
        out[i, 2]  = gw[i]
        out[i, 3]  = gh[i]
        out[i, 4]  = cos_r
        out[i, 5]  = sin_r
        out[i, 6]  = guv[i, 0]
        out[i, 7]  = guv[i, 1]
        out[i, 8]  = guv[i, 2]
        out[i, 9]  = guv[i, 3]
        out[i, 10] = color_bits




@njit(fastmath=True, cache=True)
def _numba_pack_rrect_instances(pos, size, radius, cos_a, sin_a, color_bits, out):
    """Impacchetta (N,8) instance buffer per DrawRoundedRectsBatch:
    pos2+size2+radius1+dir2+color(1, uint32 packed). Stessa architettura
    Numba compilata (sequenziale, cache=True) di _numba_pack_rect_instances."""
    n = pos.shape[0]
    for i in range(n):
        out[i,  0] = pos[i, 0]
        out[i,  1] = pos[i, 1]
        out[i,  2] = size[i, 0]
        out[i,  3] = size[i, 1]
        out[i,  4] = radius[i]
        out[i,  5] = cos_a[i]
        out[i,  6] = sin_a[i]
        out[i,  7] = color_bits[i]


@njit(fastmath=True, cache=True)
def _numba_pack_rrect_outline_instances(pos, size, radius, thickness,
                                        cos_a, sin_a, color_bits, out):
    """Impacchetta (N,9) instance buffer per DrawRoundedRectsOutlineBatch:
    pos2+size2+radius1+thickness1+dir2+color(1, uint32 packed)."""
    n = pos.shape[0]
    for i in range(n):
        out[i,  0] = pos[i, 0]
        out[i,  1] = pos[i, 1]
        out[i,  2] = size[i, 0]
        out[i,  3] = size[i, 1]
        out[i,  4] = radius[i]
        out[i,  5] = thickness[i]
        out[i,  6] = cos_a[i]
        out[i,  7] = sin_a[i]
        out[i,  8] = color_bits[i]


@njit(fastmath=True, cache=True)
def _rtri_shrink_and_aabb(ax, ay, bx, by, cx, cy, r):
    """Ritorna (sax,say, sbx,sby, scx,scy, r_eff, min_x,min_y,max_x,max_y).
    Calcola i 3 vertici 'shrunk' del triangolo dopo aver spinto ogni lato
    verso l'interno di r_eff (clampato all'incentro). r_eff <= 0.99*inradius
    per evitare vertici degeneri quando r >= inradius."""
    len_a = _math_hypot(bx - cx, by - cy)
    len_b = _math_hypot(cx - ax, cy - ay)
    len_c = _math_hypot(ax - bx, ay - by)
    perim = len_a + len_b + len_c
    cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    area  = 0.5 * (cross if cross >= 0.0 else -cross)
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
    nab_y =  (bx - ax) * sgn * inv_c
    d_ab  = nab_x * ax + nab_y * ay + r_eff

    nbc_x = -(cy - by) * sgn * inv_a
    nbc_y =  (cx - bx) * sgn * inv_a
    d_bc  = nbc_x * bx + nbc_y * by + r_eff

    nca_x = -(ay - cy) * sgn * inv_b
    nca_y =  (ax - cx) * sgn * inv_b
    d_ca  = nca_x * cx + nca_y * cy + r_eff

    det_a = nca_x * nab_y - nca_y * nab_x
    if det_a > -1e-6 and det_a < 1e-6:
        sax = ax; say = ay
    else:
        inv_da = 1.0 / det_a
        sax = (d_ca * nab_y - d_ab * nca_y) * inv_da
        say = (nca_x * d_ab - nab_x * d_ca) * inv_da

    det_b = nab_x * nbc_y - nab_y * nbc_x
    if det_b > -1e-6 and det_b < 1e-6:
        sbx = bx; sby = by
    else:
        inv_db = 1.0 / det_b
        sbx = (d_ab * nbc_y - d_bc * nab_y) * inv_db
        sby = (nab_x * d_bc - nbc_x * d_ab) * inv_db

    det_c = nbc_x * nca_y - nbc_y * nca_x
    if det_c > -1e-6 and det_c < 1e-6:
        scx = cx; scy = cy
    else:
        inv_dc = 1.0 / det_c
        scx = (d_bc * nca_y - d_ca * nbc_y) * inv_dc
        scy = (nbc_x * d_ca - nca_x * d_bc) * inv_dc

    pad = 2.0
    min_x = ax if ax < bx else bx
    if cx < min_x: min_x = cx
    min_y = ay if ay < by else by
    if cy < min_y: min_y = cy
    max_x = ax if ax > bx else bx
    if cx > max_x: max_x = cx
    max_y = ay if ay > by else by
    if cy > max_y: max_y = cy
    return (sax, say, sbx, sby, scx, scy, r_eff,
            min_x - pad, min_y - pad, max_x + pad, max_y + pad)


@njit(fastmath=True, cache=True)
def _numba_pack_rtri_instances(v0, v1, v2, radius, color_bits, out):
    """Impacchetta (N,12) instance buffer per DrawRoundedTrianglesBatch:
    shrunkV0(2)+shrunkV1(2)+shrunkV2(2)+r_eff(1)+aabb_min(2)+aabb_max(2)+
    color(1, uint32 packed)."""
    n = v0.shape[0]
    for i in range(n):
        (sax, say, sbx, sby, scx, scy, r_eff,
         mnx, mny, mxx, mxy) = _rtri_shrink_and_aabb(
            v0[i, 0], v0[i, 1],
            v1[i, 0], v1[i, 1],
            v2[i, 0], v2[i, 1],
            radius[i])
        out[i,  0] = sax; out[i,  1] = say
        out[i,  2] = sbx; out[i,  3] = sby
        out[i,  4] = scx; out[i,  5] = scy
        out[i,  6] = r_eff
        out[i,  7] = mnx; out[i,  8] = mny
        out[i,  9] = mxx; out[i, 10] = mxy
        out[i, 11] = color_bits[i]


@njit(fastmath=True, cache=True)
def _numba_pack_rtri_outline_instances(v0, v1, v2, radius, thickness, color_bits, out):
    """Impacchetta (N,13) instance buffer per DrawRoundedTrianglesOutlineBatch:
    shrunkV0(2)+shrunkV1(2)+shrunkV2(2)+r_eff(1)+aabb_min(2)+aabb_max(2)+
    thickness(1)+color(1, uint32 packed)."""
    n = v0.shape[0]
    for i in range(n):
        (sax, say, sbx, sby, scx, scy, r_eff,
         mnx, mny, mxx, mxy) = _rtri_shrink_and_aabb(
            v0[i, 0], v0[i, 1],
            v1[i, 0], v1[i, 1],
            v2[i, 0], v2[i, 1],
            radius[i])
        out[i,  0] = sax; out[i,  1] = say
        out[i,  2] = sbx; out[i,  3] = sby
        out[i,  4] = scx; out[i,  5] = scy
        out[i,  6] = r_eff
        out[i,  7] = mnx; out[i,  8] = mny
        out[i,  9] = mxx; out[i, 10] = mxy
        out[i, 11] = thickness[i]
        out[i, 12] = color_bits[i]


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

    # FIX: mappa di alias comuni -> file di sistema tipici per piattaforma.
    # Serve come rete di sicurezza quando l'utente chiede "arial" su Linux/mac.
    _FALLBACK_ALIASES = {
        "arial":     ("Arial.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Helvetica.ttc"),
        "sans":      ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf", "arial.ttf", "Helvetica.ttc"),
        "sans-serif":("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf", "arial.ttf", "Helvetica.ttc"),
        "serif":     ("DejaVuSerif.ttf", "LiberationSerif-Regular.ttf", "Times New Roman.ttf", "times.ttf"),
        "mono":      ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Consolas.ttf", "consola.ttf", "Menlo.ttc"),
        "monospace": ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Consolas.ttf", "consola.ttf", "Menlo.ttc"),
    }

    def __init__(self, capacity=_MAX_FONT_CACHE):
        if not _HAS_PIL:
            raise RuntimeError("PE_DRAW.DrawText richiede Pillow (pip install pillow)")
        self._cache = OrderedDict()
        self._capacity = capacity
        self._registered = {}
        # FIX: cache degli esiti di _resolve. Prima ogni family sconosciuta
        # scatenava un os.walk su tutti i _SYSTEM_DIRS (migliaia di file) a
        # ogni miss. Ora il risultato viene memoizzato per family.
        self._resolved = {}

    def register(self, alias: str, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Font non trovato: {path}")
        self._registered[alias] = path

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

    def _resolve(self, family: str) -> str:
        if family in self._registered:
            return self._registered[family]
        if os.path.isfile(family):
            return family

        # FIX: cache dei lookup per family (evita os.walk ripetuti).
        cached = self._resolved.get(family)
        if cached is not None:
            return cached

        target = family.lower()

        # 1) match diretto per nome file nelle cartelle di sistema.
        for base in self._SYSTEM_DIRS:
            if not os.path.isdir(base):
                continue
            for root, _, files in os.walk(base):
                for f in files:
                    lf = f.lower()
                    if lf.endswith((".ttf", ".otf", ".ttc")) and target in lf:
                        path = os.path.join(root, f)
                        self._resolved[family] = path
                        return path

        # 2) FIX: fallback funzionante. Prima veniva restituito il "path"
        # (inesistente) del bitmap font di default, che poi faceva esplodere
        # ImageFont.truetype(). Ora proviamo alias comuni per piattaforma:
        aliases = self._FALLBACK_ALIASES.get(target, ())
        for alias in aliases:
            for base in self._SYSTEM_DIRS:
                if not os.path.isdir(base):
                    continue
                for root, _, files in os.walk(base):
                    for f in files:
                        if f.lower() == alias.lower():
                            path = os.path.join(root, f)
                            self._resolved[family] = path
                            return path

        # 3) Ultima chance: qualunque TTF/OTF trovato nel sistema.
        for base in self._SYSTEM_DIRS:
            if not os.path.isdir(base):
                continue
            for root, _, files in os.walk(base):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf", ".ttc")):
                        path = os.path.join(root, f)
                        self._resolved[family] = path
                        warnings.warn(
                            f"FontManager: font {family!r} non trovato, "
                            f"uso fallback {f!r}.",
                            RuntimeWarning, stacklevel=2,
                        )
                        return path

        raise FileNotFoundError(
            f"FontManager: impossibile risolvere il font {family!r}: "
            f"nessun .ttf/.otf trovato in {self._SYSTEM_DIRS}. "
            "Registra un font esplicito con RegisterFont(alias, path)."
        )


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
    # BUG FIX (atlas bleeding): l'atlas usa filtro LINEAR (bilineare) ma le
    # sub-texture venivano impacchettate perfettamente a contatto l'una con
    # l'altra, senza alcun margine. Con LINEAR, campionare un texel vicino al
    # bordo di una regione fa sì che la GPU interpoli anche pixel della
    # regione ADIACENTE nell'atlas: per i glifi di testo questo produce
    # sottili "lineette" colorate sopra/sotto/accanto alle lettere, perché
    # si vede un filo del glifo impacchettato subito accanto. PADDING px di
    # bordo trasparente attorno a ogni immagine inserita eliminano il
    # bleeding: le UV restituite continuano a puntare solo al contenuto
    # reale (il padding non è mai visibile), ma lo spazio vuoto impedisce
    # il campionamento di texel estranei.
    PADDING = 2

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
        self.name_to_rect = {}    # name -> (x, y, w, h)  occupato nell'atlas, PADDING incluso (fix bug 1/2)

    def _packed_rect_for(self, w, h):
        """Calcola la dimensione (con padding) da richiedere all'allocatore
        per un'immagine w x h, con fallback a 0 padding se anche il solo
        contenuto reale non ci starebbe (immagini enormi vicine a max_size)."""
        pad = self.PADDING
        pw, ph = w + 2 * pad, h + 2 * pad
        if pw > self.max_size or ph > self.max_size:
            pad = 0
            pw, ph = w, h
        return pad, pw, ph

    def _write_padded(self, img_arr, x, y, w, h, pad):
        """Scrive img_arr nell'atlas centrato in un riquadro (w+2*pad, h+2*pad)
        con bordo trasparente, e ritorna (inner_x, inner_y) del contenuto reale."""
        if pad == 0:
            self.tex.write(img_arr.tobytes(), viewport=(x, y, w, h))
            return x, y
        padded = np.zeros((h + 2 * pad, w + 2 * pad, 4), dtype=np.uint8)
        padded[pad:pad + h, pad:pad + w] = img_arr
        self.tex.write(padded.tobytes(), viewport=(x, y, w + 2 * pad, h + 2 * pad))
        return x + pad, y + pad

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

        # Prova ad allocare con MaxRects, riservando anche il padding
        # anti-bleeding attorno all'immagine (vedi PADDING sopra).
        pad, pw, ph = self._packed_rect_for(w, h)
        rect = self._find_rect(pw, ph)
        if rect is None:
            # Se fallisce, espandi l'atlas e riprova
            self._expand_atlas()
            pad, pw, ph = self._packed_rect_for(w, h)
            rect = self._find_rect(pw, ph)
            if rect is None:
                raise RuntimeError(f"Unable to insert {w}x{h} even after expanding atlas")

        x, y, rw, rh = rect
        # Scrivi i pixel (con bordo trasparente di padding attorno)
        inner_x, inner_y = self._write_padded(img_arr, x, y, w, h, pad)
        # Calcola UV: puntano SOLO al contenuto reale, mai al padding
        u0 = inner_x / self.size
        v0 = inner_y / self.size
        u1 = (inner_x + w) / self.size
        v1 = (inner_y + h) / self.size
        self.uv_map[name] = (u0, v0, u1, v1)
        # Conserva i dati originali per future espansioni
        self.texture_data[name] = (img_arr.copy(), w, h)
        self.name_to_rect[name] = (x, y, pw, ph)
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
            pad, pw, ph = self._packed_rect_for(w, h)
            rect = self._find_rect(pw, ph)
            if rect is None:
                # Dovrebbe sempre funzionare dato che l'atlas è più grande
                raise RuntimeError(f"Failed to repack {name} during expansion")
            x, y, _, _ = rect
            inner_x, inner_y = self._write_padded(img_arr, x, y, w, h, pad)
            u0 = inner_x / self.size
            v0 = inner_y / self.size
            u1 = (inner_x + w) / self.size
            v1 = (inner_y + h) / self.size
            self.uv_map[name] = (u0, v0, u1, v1)
            self.name_to_rect[name] = (x, y, pw, ph)

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


# ============================================================================ #
# PE_COLLISION — integrato in DRAW
# ----------------------------------------------------------------------------
# Tutto quello che segue proviene dal modulo PE_COLLISION (sistema unificato
# di collisioni + interazione mouse): forme immutabili (Point, Rect, RotRect,
# Circle, Ellipse, Line, Triangle, RoundedRect, Polygon, TextureCollider) e il
# dispatcher polimorfico O(1) via tabella (_TABLE/_dispatch). Le funzioni
# _xx_yy(a, b, draw) provano PRIMA a delegare ai metodi Collide* gia'
# presenti piu' sotto in DRAW (via _use_draw), che sono le versioni piu'
# ottimizzate (numpy/numba); solo se `draw` non le espone (o e' None) si
# ricade sul fallback puro Python qui sotto. Le API pubbliche (che nel
# modulo originale erano funzioni sciolte con un singleton globale
# _RUNTIME + bind_draw/bind_window) sono diventate METODI di DRAW
# (CheckCollision, MouseOver, MousePressed, ...): `self` e' sempre il
# contesto draw/mouse, quindi ogni istanza DRAW/WINDOW ha il proprio stato
# mouse indipendente, senza bisogno di registrarsi da nessuna parte.
# ============================================================================ #
_HAS_NUMPY = True  # numpy e' una dipendenza obbligatoria di questo file

# --------------------------------------------------------------------------- #
# Costanti tipo forma (int per branch prediction migliore delle stringhe)
# --------------------------------------------------------------------------- #
_T_POINT    = 0
_T_RECT     = 1
_T_ROTRECT  = 2
_T_CIRCLE   = 3
_T_ELLIPSE  = 4
_T_LINE     = 5
_T_TRIANGLE = 6
_T_POLYGON  = 7
_T_RRECT    = 8   # rounded rect
_T_TEXCOL   = 9   # TextureCollider (compound)


# --------------------------------------------------------------------------- #
# Forme (immutabili, __slots__)
# --------------------------------------------------------------------------- #
class _Shape:
    __slots__ = ("_t",)
    def __repr__(self):
        vals = ", ".join(f"{s}={getattr(self, s)!r}"
                         for s in self.__slots__ if s != "_t")
        return f"{self.__class__.__name__}({vals})"


class Point(_Shape):
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self._t = _T_POINT
        self.x = float(x); self.y = float(y)


class Rect(_Shape):
    """Rettangolo allineato agli assi. (x,y) = angolo top-left, (w,h) = dimensioni."""
    __slots__ = ("x", "y", "w", "h")
    def __init__(self, x, y, w, h):
        self._t = _T_RECT
        self.x = float(x); self.y = float(y)
        self.w = float(w); self.h = float(h)


class RotRect(_Shape):
    """Rettangolo ruotato (OBB). rotation in gradi, centro = (x+w/2, y+h/2)."""
    __slots__ = ("x", "y", "w", "h", "rotation")
    def __init__(self, x, y, w, h, rotation=0.0):
        self._t = _T_ROTRECT
        self.x = float(x); self.y = float(y)
        self.w = float(w); self.h = float(h)
        self.rotation = float(rotation)


class Circle(_Shape):
    __slots__ = ("cx", "cy", "r")
    def __init__(self, cx, cy, r):
        self._t = _T_CIRCLE
        self.cx = float(cx); self.cy = float(cy); self.r = float(r)


class Ellipse(_Shape):
    __slots__ = ("cx", "cy", "rx", "ry", "rotation")
    def __init__(self, cx, cy, rx, ry, rotation=0.0):
        self._t = _T_ELLIPSE
        self.cx = float(cx); self.cy = float(cy)
        self.rx = float(rx); self.ry = float(ry)
        self.rotation = float(rotation)


class Line(_Shape):
    __slots__ = ("x1", "y1", "x2", "y2", "thickness")
    def __init__(self, x1, y1, x2, y2, thickness=1.0):
        self._t = _T_LINE
        self.x1 = float(x1); self.y1 = float(y1)
        self.x2 = float(x2); self.y2 = float(y2)
        self.thickness = float(thickness)


class Triangle(_Shape):
    __slots__ = ("x1", "y1", "x2", "y2", "x3", "y3")
    def __init__(self, x1, y1, x2, y2, x3, y3):
        self._t = _T_TRIANGLE
        self.x1 = float(x1); self.y1 = float(y1)
        self.x2 = float(x2); self.y2 = float(y2)
        self.x3 = float(x3); self.y3 = float(y3)


class RoundedRect(_Shape):
    """Rectangle con angoli arrotondati. Per collisione: unione di rect + 4 cerchi."""
    __slots__ = ("x", "y", "w", "h", "radius")
    def __init__(self, x, y, w, h, radius):
        self._t = _T_RRECT
        self.x = float(x); self.y = float(y)
        self.w = float(w); self.h = float(h)
        self.radius = max(0.0, float(radius))


class Polygon(_Shape):
    """Poligono convesso o concavo. `points` = lista [(x,y), ...]."""
    __slots__ = ("points", "_aabb", "_convex")
    def __init__(self, points, convex=None):
        self._t = _T_POLYGON
        if len(points) < 3:
            raise ValueError("Polygon requires >= 3 points")
        self.points = tuple((float(x), float(y)) for x, y in points)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        self._aabb = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
        self._convex = _polygon_is_convex(self.points) if convex is None else bool(convex)


# --------------------------------------------------------------------------- #
# Utility geometriche pure (fallback e supporto polygon)
# --------------------------------------------------------------------------- #
_EPS = 1e-9

def _polygon_is_convex(pts):
    n = len(pts)
    sign = 0
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i+1) % n]
        cx, cy = pts[(i+2) % n]
        cross = (bx-ax)*(cy-by) - (by-ay)*(cx-bx)
        if cross > _EPS:
            if sign < 0: return False
            sign = 1
        elif cross < -_EPS:
            if sign > 0: return False
            sign = -1
    return True


def _point_in_polygon(px, py, pts):
    """Ray casting - funziona anche per poligoni concavi."""
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]; xj, yj = pts[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi + _EPS) + xi):
            inside = not inside
        j = i
    return inside


def _seg_seg_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    eps = _EPS
    def orient(px, py, qx, qy, rx, ry):
        return (qx-px)*(ry-py) - (qy-py)*(rx-px)
    def on_segment(px, py, qx, qy, rx, ry):
        return (min(px, qx) - eps <= rx <= max(px, qx) + eps and
                min(py, qy) - eps <= ry <= max(py, qy) + eps)
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    if abs(o1) <= eps and on_segment(ax, ay, bx, by, cx, cy):
        return True
    if abs(o2) <= eps and on_segment(ax, ay, bx, by, dx, dy):
        return True
    if abs(o3) <= eps and on_segment(cx, cy, dx, dy, ax, ay):
        return True
    if abs(o4) <= eps and on_segment(cx, cy, dx, dy, bx, by):
        return True
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    return False


def _segment_intersects_ellipse_pure(x1, y1, x2, y2, cx, cy, rx, ry, rotation=0.0):
    if rx <= 0.0 or ry <= 0.0:
        return False
    if rotation != 0.0:
        ang = -rotation * _DEG2RAD_CONST
        cs = math.cos(ang); sn = math.sin(ang)
        dx1 = x1 - cx; dy1 = y1 - cy
        dx2 = x2 - cx; dy2 = y2 - cy
        x1 = dx1 * cs - dy1 * sn; y1 = dx1 * sn + dy1 * cs
        x2 = dx2 * cs - dy2 * sn; y2 = dx2 * sn + dy2 * cs
    else:
        x1 -= cx; y1 -= cy; x2 -= cx; y2 -= cy
    dx = x2 - x1; dy = y2 - y1
    irx2 = 1.0 / (rx * rx); iry2 = 1.0 / (ry * ry)
    c = x1 * x1 * irx2 + y1 * y1 * iry2 - 1.0
    if c <= 0.0:
        return True
    if x2 * x2 * irx2 + y2 * y2 * iry2 - 1.0 <= 0.0:
        return True
    a = dx * dx * irx2 + dy * dy * iry2
    if a <= _EPS:
        return False
    b = 2.0 * (x1 * dx * irx2 + y1 * dy * iry2)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    i2a = 0.5 / a
    t1 = (-b - sq) * i2a
    t2 = (-b + sq) * i2a
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def _point_in_rounded_rect_direct(px, py, x, y, w, h, radius, rotation=0.0):
    if w <= 0.0 or h <= 0.0:
        return False
    cx = x + w * 0.5; cy = y + h * 0.5
    if rotation != 0.0:
        ang = -rotation * _DEG2RAD_CONST
        cs = math.cos(ang); sn = math.sin(ang)
        dx = px - cx; dy = py - cy
        px = dx * cs - dy * sn
        py = dx * sn + dy * cs
    else:
        px -= cx; py -= cy
    hw = w * 0.5; hh = h * 0.5
    r = max(0.0, min(float(radius), hw, hh))
    qx = abs(px) - hw + r
    qy = abs(py) - hh + r
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return inside + outside - r <= 0.0


def _sd_triangle_direct(px, py, ax, ay, bx, by, cx, cy):
    e0x, e0y = bx - ax, by - ay
    e1x, e1y = cx - bx, cy - by
    e2x, e2y = ax - cx, ay - cy
    v0x, v0y = px - ax, py - ay
    v1x, v1y = px - bx, py - by
    v2x, v2y = px - cx, py - cy

    def closest_sq(vx, vy, ex, ey):
        den = ex * ex + ey * ey
        t = 0.0 if den <= _EPS else max(0.0, min(1.0, (vx * ex + vy * ey) / den))
        qx = vx - ex * t; qy = vy - ey * t
        return qx * qx + qy * qy

    s = 1.0 if (e0x * e2y - e0y * e2x) >= 0.0 else -1.0
    dx = min(closest_sq(v0x, v0y, e0x, e0y),
             closest_sq(v1x, v1y, e1x, e1y),
             closest_sq(v2x, v2y, e2x, e2y))
    dy = min(s * (v0x * e0y - v0y * e0x),
             s * (v1x * e1y - v1y * e1x),
             s * (v2x * e2y - v2y * e2x))
    return -math.sqrt(dx) * (1.0 if dy >= 0.0 else -1.0)


def _point_in_rounded_triangle_direct(px, py, x1, y1, x2, y2, x3, y3, radius, rotation=0.0):
    if rotation != 0.0:
        cx = (x1 + x2 + x3) / 3.0; cy = (y1 + y2 + y3) / 3.0
        cs = math.cos(rotation * _DEG2RAD_CONST); sn = math.sin(rotation * _DEG2RAD_CONST)
        dx1 = x1 - cx; dy1 = y1 - cy
        dx2 = x2 - cx; dy2 = y2 - cy
        dx3 = x3 - cx; dy3 = y3 - cy
        x1 = cx + dx1 * cs - dy1 * sn; y1 = cy + dx1 * sn + dy1 * cs
        x2 = cx + dx2 * cs - dy2 * sn; y2 = cy + dx2 * sn + dy2 * cs
        x3 = cx + dx3 * cs - dy3 * sn; y3 = cy + dx3 * sn + dy3 * cs
    sax, say, sbx, sby, scx, scy, r_eff, _mnx, _mny, _mxx, _mxy = _rtri_shrink_and_aabb(
        float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(radius)
    )
    return (_sd_triangle_direct(px, py, sax, say, sbx, sby, scx, scy) - r_eff) <= 0.0


def _polygon_aabb_overlap(a_aabb, b_aabb):
    ax, ay, aw, ah = a_aabb
    bx, by, bw, bh = b_aabb
    return not (ax+aw < bx or ax > bx+bw or ay+ah < by or ay > by+bh)


def _polygon_vs_polygon(A, B):
    # AABB pre-check
    if not _polygon_aabb_overlap(A._aabb, B._aabb):
        return False
    # Convesso vs convesso -> SAT
    if A._convex and B._convex:
        return _sat_convex(A.points, B.points)
    # Almeno uno concavo -> test punti + intersezione lati
    for p in A.points:
        if _point_in_polygon(p[0], p[1], B.points):
            return True
    for p in B.points:
        if _point_in_polygon(p[0], p[1], A.points):
            return True
    na, nb = len(A.points), len(B.points)
    for i in range(na):
        ax, ay = A.points[i]; bx, by = A.points[(i+1) % na]
        for j in range(nb):
            cx, cy = B.points[j]; dx, dy = B.points[(j+1) % nb]
            if _seg_seg_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                return True
    return False


def _sat_convex(A, B):
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
            nx, ny = -(y2 - y1), (x2 - x1)
            L = math.hypot(nx, ny)
            if L < _EPS: continue
            nx /= L; ny /= L
            aMin = aMax = A[0][0]*nx + A[0][1]*ny
            for px, py in A[1:]:
                d = px*nx + py*ny
                if d < aMin: aMin = d
                elif d > aMax: aMax = d
            bMin = bMax = B[0][0]*nx + B[0][1]*ny
            for px, py in B[1:]:
                d = px*nx + py*ny
                if d < bMin: bMin = d
                elif d > bMax: bMax = d
            if aMax < bMin or bMax < aMin:
                return False
    return True


def _rect_corners(x, y, w, h, rotation):
    cx = x + w * 0.5; cy = y + h * 0.5
    hw = w * 0.5;     hh = h * 0.5
    if rotation == 0.0:
        return ((x, y), (x+w, y), (x+w, y+h), (x, y+h))
    a = rotation * 0.017453292519943295
    cs = math.cos(a); sn = math.sin(a)
    out = []
    for lx, ly in ((-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)):
        out.append((cx + lx*cs - ly*sn, cy + lx*sn + ly*cs))
    return tuple(out)


# --------------------------------------------------------------------------- #
# DISPATCHER — usa DRAW.Collide* quando presente, altrimenti fallback puri
# --------------------------------------------------------------------------- #
def _use_draw(draw, method, *args):
    """Chiama draw.<method>(*args) se esiste; altrimenti ritorna None (sentinel)."""
    if draw is not None:
        fn = getattr(draw, method, None)
        if fn is not None:
            return fn(*args)
    return None


def _rrect_to_rect_and_circles(rr):
    """Un RoundedRect e' un Rect centrale + 4 cerchi negli angoli.
    Per collisione approssimiamo con un solo Rect esteso e 4 cerchi."""
    r = rr.radius
    inner = Rect(rr.x + r, rr.y, rr.w - 2*r, rr.h) if rr.w > 2*r else None
    tall  = Rect(rr.x, rr.y + r, rr.w, rr.h - 2*r) if rr.h > 2*r else None
    corners = (
        Circle(rr.x + r,           rr.y + r,           r),
        Circle(rr.x + rr.w - r,    rr.y + r,           r),
        Circle(rr.x + r,           rr.y + rr.h - r,    r),
        Circle(rr.x + rr.w - r,    rr.y + rr.h - r,    r),
    )
    return inner, tall, corners


def _dispatch(a, b, draw):
    """Ritorna True/False. Non normalizza gli ordini: chiama la funzione giusta
    guardando entrambi i tipi (tabella statica)."""
    ta, tb = a._t, b._t

    # RoundedRect: scomponiamo in rect+cerchi e ricorriamo
    if ta == _T_RRECT:
        inner, tall, corners = _rrect_to_rect_and_circles(a)
        if inner and _dispatch(inner, b, draw): return True
        if tall  and _dispatch(tall,  b, draw): return True
        for c in corners:
            if _dispatch(c, b, draw): return True
        return False
    if tb == _T_RRECT:
        return _dispatch(b, a, draw)

    # TextureCollider: delega al proprio metodo
    if ta == _T_TEXCOL:
        return a._collides_shape(b, draw)
    if tb == _T_TEXCOL:
        return b._collides_shape(a, draw)

    # Polygon: gestito puro Python + SAT
    if ta == _T_POLYGON or tb == _T_POLYGON:
        return _dispatch_polygon(a, b, draw)

    key = (ta, tb)
    fn = _TABLE.get(key)
    if fn is None:
        fn = _TABLE.get((tb, ta))
        if fn is None:
            raise TypeError(
                f"check_collision: coppia non supportata "
                f"({a.__class__.__name__} vs {b.__class__.__name__})")
        return fn(b, a, draw)
    return fn(a, b, draw)


# --- Adattatori: ogni funzione firma (a, b, draw) -> bool ------------------
def _pt_pt(a, b, draw):
    return abs(a.x - b.x) < _EPS and abs(a.y - b.y) < _EPS

def _pt_rect(a, b, draw):
    r = _use_draw(draw, "CollidePointRect", a.x, a.y, b.x, b.y, b.w, b.h)
    if r is not None: return bool(r)
    return b.x <= a.x <= b.x + b.w and b.y <= a.y <= b.y + b.h

def _pt_rotrect(a, b, draw):
    r = _use_draw(draw, "CollidePointRotatedRect", a.x, a.y, b.x, b.y, b.w, b.h, b.rotation)
    if r is not None: return bool(r)
    return _point_in_polygon(a.x, a.y, _rect_corners(b.x, b.y, b.w, b.h, b.rotation))

def _pt_circle(a, b, draw):
    r = _use_draw(draw, "CollidePointCircle", a.x, a.y, b.cx, b.cy, b.r)
    if r is not None: return bool(r)
    dx = a.x - b.cx; dy = a.y - b.cy
    return dx*dx + dy*dy <= b.r * b.r

def _pt_ellipse(a, b, draw):
    r = _use_draw(draw, "CollidePointEllipse", a.x, a.y, b.cx, b.cy, b.rx, b.ry, b.rotation)
    if r is not None: return bool(r)
    if b.rx <= 0 or b.ry <= 0: return False
    if b.rotation != 0.0:
        ang = -b.rotation * 0.017453292519943295
        cs = math.cos(ang); sn = math.sin(ang)
        dx = a.x - b.cx; dy = a.y - b.cy
        px = dx*cs - dy*sn; py = dx*sn + dy*cs
    else:
        px = a.x - b.cx; py = a.y - b.cy
    return (px*px)/(b.rx*b.rx) + (py*py)/(b.ry*b.ry) <= 1.0

def _pt_line(a, b, draw):
    r = _use_draw(draw, "PointInLine", a.x, a.y, b.x1, b.y1, b.x2, b.y2, b.thickness)
    if r is not None: return bool(r)
    dx = b.x2 - b.x1; dy = b.y2 - b.y1
    if dx == 0 and dy == 0:
        return math.hypot(a.x - b.x1, a.y - b.y1) <= b.thickness * 0.5
    t = ((a.x - b.x1)*dx + (a.y - b.y1)*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(a.x - (b.x1 + t*dx), a.y - (b.y1 + t*dy)) <= b.thickness * 0.5

def _pt_tri(a, b, draw):
    r = _use_draw(draw, "CollidePointTriangle", a.x, a.y, b.x1, b.y1, b.x2, b.y2, b.x3, b.y3)
    if r is not None: return bool(r)
    d1 = (a.x-b.x2)*(b.y1-b.y2) - (b.x1-b.x2)*(a.y-b.y2)
    d2 = (a.x-b.x3)*(b.y2-b.y3) - (b.x2-b.x3)*(a.y-b.y3)
    d3 = (a.x-b.x1)*(b.y3-b.y1) - (b.x3-b.x1)*(a.y-b.y1)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)

def _rect_rect(a, b, draw):
    r = _use_draw(draw, "CollideRectRect", a.x, a.y, a.w, a.h, b.x, b.y, b.w, b.h)
    if r is not None: return bool(r)
    return not (a.x + a.w < b.x or a.x > b.x + b.w or
                a.y + a.h < b.y or a.y > b.y + b.h)

def _rect_rotrect(a, b, draw):
    r = _use_draw(draw, "CollideRectRotatedRect", a.x, a.y, a.w, a.h,
                  b.x, b.y, b.w, b.h, b.rotation)
    if r is not None: return bool(r)
    return _sat_convex(_rect_corners(a.x, a.y, a.w, a.h, 0.0),
                       _rect_corners(b.x, b.y, b.w, b.h, b.rotation))

def _rotrect_rotrect(a, b, draw):
    r = _use_draw(draw, "CollideRotatedRectRotatedRect",
                  a.x, a.y, a.w, a.h, a.rotation,
                  b.x, b.y, b.w, b.h, b.rotation)
    if r is not None: return bool(r)
    return _sat_convex(_rect_corners(a.x, a.y, a.w, a.h, a.rotation),
                       _rect_corners(b.x, b.y, b.w, b.h, b.rotation))

def _rect_circle(a, b, draw):
    r = _use_draw(draw, "CollideRectCircle", a.x, a.y, a.w, a.h, b.cx, b.cy, b.r)
    if r is not None: return bool(r)
    nx = max(a.x, min(b.cx, a.x + a.w))
    ny = max(a.y, min(b.cy, a.y + a.h))
    dx = nx - b.cx; dy = ny - b.cy
    return dx*dx + dy*dy <= b.r * b.r

def _rotrect_circle(a, b, draw):
    r = _use_draw(draw, "CollideRotatedRectCircle",
                  a.x, a.y, a.w, a.h, a.rotation, b.cx, b.cy, b.r)
    if r is not None: return bool(r)
    # rotazione inversa del centro cerchio in spazio locale rect
    cx = a.x + a.w * 0.5; cy = a.y + a.h * 0.5
    ang = -a.rotation * 0.017453292519943295
    cs = math.cos(ang); sn = math.sin(ang)
    dx = b.cx - cx; dy = b.cy - cy
    lx = dx*cs - dy*sn; ly = dx*sn + dy*cs
    nx = max(-a.w*0.5, min(lx, a.w*0.5))
    ny = max(-a.h*0.5, min(ly, a.h*0.5))
    ex = lx - nx; ey = ly - ny
    return ex*ex + ey*ey <= b.r * b.r

def _rect_ellipse(a, b, draw):
    r = _use_draw(draw, "CollideRectEllipse", a.x, a.y, a.w, a.h,
                  b.cx, b.cy, b.rx, b.ry, b.rotation)
    if r is not None: return bool(r)
    # Fallback completo: centro/vertici + intersezione esatta lato-ellisse.
    if _pt_rect(Point(b.cx, b.cy), a, draw): return True
    for (px, py) in ((a.x,a.y),(a.x+a.w,a.y),(a.x+a.w,a.y+a.h),(a.x,a.y+a.h)):
        if _pt_ellipse(Point(px,py), b, draw): return True
    for (x1, y1, x2, y2) in ((a.x, a.y, a.x+a.w, a.y),
                             (a.x+a.w, a.y, a.x+a.w, a.y+a.h),
                             (a.x+a.w, a.y+a.h, a.x, a.y+a.h),
                             (a.x, a.y+a.h, a.x, a.y)):
        if _segment_intersects_ellipse_pure(x1, y1, x2, y2, b.cx, b.cy, b.rx, b.ry, b.rotation):
            return True
    return False

def _rect_tri(a, b, draw):
    r = _use_draw(draw, "CollideRectTriangle", a.x, a.y, a.w, a.h,
                  b.x1, b.y1, b.x2, b.y2, b.x3, b.y3)
    if r is not None: return bool(r)
    tri_pts = ((b.x1,b.y1),(b.x2,b.y2),(b.x3,b.y3))
    rect_pts = _rect_corners(a.x, a.y, a.w, a.h, 0.0)
    return _sat_convex(rect_pts, tri_pts)

def _rect_line(a, b, draw):
    r = _use_draw(draw, "CollideLineRect", b.x1, b.y1, b.x2, b.y2,
                  a.x, a.y, a.w, a.h)
    if r is not None: return bool(r)
    if _pt_rect(Point(b.x1, b.y1), a, draw): return True
    if _pt_rect(Point(b.x2, b.y2), a, draw): return True
    corners = ((a.x,a.y),(a.x+a.w,a.y),(a.x+a.w,a.y+a.h),(a.x,a.y+a.h))
    for i in range(4):
        if _seg_seg_intersect(b.x1, b.y1, b.x2, b.y2,
                              corners[i][0], corners[i][1],
                              corners[(i+1)%4][0], corners[(i+1)%4][1]):
            return True
    return False

def _rotrect_line(a, b, draw):
    r = _use_draw(draw, "CollideLineRotatedRect", b.x1, b.y1, b.x2, b.y2,
                  a.x, a.y, a.w, a.h, a.rotation)
    if r is not None: return bool(r)
    if _pt_rotrect(Point(b.x1, b.y1), a, draw): return True
    if _pt_rotrect(Point(b.x2, b.y2), a, draw): return True
    corners = _rect_corners(a.x, a.y, a.w, a.h, a.rotation)
    for i in range(4):
        if _seg_seg_intersect(b.x1, b.y1, b.x2, b.y2,
                              corners[i][0], corners[i][1],
                              corners[(i+1)%4][0], corners[(i+1)%4][1]):
            return True
    return False

def _rotrect_ellipse(a, b, draw):
    r = _use_draw(draw, "CollideEllipseRotatedRect",
                  b.cx, b.cy, b.rx, b.ry, b.rotation,
                  a.x, a.y, a.w, a.h, a.rotation)
    if r is not None: return bool(r)
    return _dispatch_polygon(Polygon(_rect_corners(a.x, a.y, a.w, a.h, a.rotation)), b, draw)

def _rotrect_tri(a, b, draw):
    r = _use_draw(draw, "CollideTriangleRotatedRect",
                  b.x1, b.y1, b.x2, b.y2, b.x3, b.y3,
                  a.x, a.y, a.w, a.h, a.rotation)
    if r is not None: return bool(r)
    return _sat_convex(_rect_corners(a.x, a.y, a.w, a.h, a.rotation),
                       ((b.x1,b.y1),(b.x2,b.y2),(b.x3,b.y3)))

def _circle_circle(a, b, draw):
    r = _use_draw(draw, "CollideCircleCircle", a.cx, a.cy, a.r, b.cx, b.cy, b.r)
    if r is not None: return bool(r)
    dx = a.cx - b.cx; dy = a.cy - b.cy
    rr = a.r + b.r
    return dx*dx + dy*dy <= rr * rr

def _circle_ellipse(a, b, draw):
    r = _use_draw(draw, "CollideCircleEllipse", a.cx, a.cy, a.r,
                  b.cx, b.cy, b.rx, b.ry, b.rotation)
    if r is not None: return bool(r)
    return _pt_ellipse(Point(a.cx, a.cy),
                       Ellipse(b.cx, b.cy, b.rx + a.r, b.ry + a.r, b.rotation), draw)

def _circle_tri(a, b, draw):
    r = _use_draw(draw, "CollideCircleTriangle", a.cx, a.cy, a.r,
                  b.x1, b.y1, b.x2, b.y2, b.x3, b.y3)
    if r is not None: return bool(r)
    if _pt_tri(Point(a.cx, a.cy), b, draw): return True
    for (x1,y1),(x2,y2) in (((b.x1,b.y1),(b.x2,b.y2)),
                            ((b.x2,b.y2),(b.x3,b.y3)),
                            ((b.x3,b.y3),(b.x1,b.y1))):
        if _pt_line(Point(a.cx, a.cy), Line(x1,y1,x2,y2, a.r*2), draw):
            return True
    return False

def _circle_line(a, b, draw):
    r = _use_draw(draw, "CollideLineCircle", b.x1, b.y1, b.x2, b.y2, a.cx, a.cy, a.r)
    if r is not None: return bool(r)
    return _pt_line(Point(a.cx, a.cy),
                    Line(b.x1, b.y1, b.x2, b.y2, a.r*2 + b.thickness), draw)

def _ellipse_ellipse(a, b, draw):
    # BUG FIX: CollideEllipseEllipse ha una firma con rot1/rot2 SEPARATI dai
    # rispettivi centro/raggi (a differenza di quasi tutte le altre Collide*,
    # che incorporano la rotazione subito dopo i parametri della propria
    # forma). Prima questa chiamata non passava affatto a.rotation/b.rotation,
    # quindi qualunque Ellipse ruotata veniva silenziosamente testata come se
    # rotation=0 -- falsi negativi/positivi per ogni ellisse ruotata.
    r = _use_draw(draw, "CollideEllipseEllipse",
                  a.cx, a.cy, a.rx, a.ry, b.cx, b.cy, b.rx, b.ry,
                  a.rotation, b.rotation)
    if r is not None: return bool(r)
    # Fallback puro Python: se una delle due e' ruotata, la somma diretta dei
    # raggi (valida solo per cerchi/ellissi assi-allineate) non e' corretta.
    if a.rotation == 0.0 and b.rotation == 0.0:
        dx = a.cx - b.cx; dy = a.cy - b.cy
        rx = a.rx + b.rx; ry = a.ry + b.ry
        if rx <= 0 or ry <= 0: return False
        return (dx*dx)/(rx*rx) + (dy*dy)/(ry*ry) <= 1.0
    # Caso generale ruotato: contenimento del centro + campionamento del
    # perimetro di entrambe le ellissi (stessa strategia robusta usata da
    # CollideEllipseEllipse in PE_DRAW), cosi' il fallback resta corretto
    # anche quando `draw` non e' disponibile.
    if _pt_ellipse(Point(a.cx, a.cy), b, draw): return True
    if _pt_ellipse(Point(b.cx, b.cy), a, draw): return True
    samples = max(24, int(max(a.rx, a.ry, b.rx, b.ry) * 0.5))
    step = 2.0 * math.pi / samples
    ang_a = a.rotation * 0.017453292519943295
    cs_a = math.cos(ang_a); sn_a = math.sin(ang_a)
    ang_b = b.rotation * 0.017453292519943295
    cs_b = math.cos(ang_b); sn_b = math.sin(ang_b)
    for i in range(samples):
        t = i * step
        lx = a.rx * math.cos(t); ly = a.ry * math.sin(t)
        px = a.cx + lx*cs_a - ly*sn_a; py = a.cy + lx*sn_a + ly*cs_a
        if _pt_ellipse(Point(px, py), b, draw): return True
        lx = b.rx * math.cos(t); ly = b.ry * math.sin(t)
        px = b.cx + lx*cs_b - ly*sn_b; py = b.cy + lx*sn_b + ly*cs_b
        if _pt_ellipse(Point(px, py), a, draw): return True
    return False

def _ellipse_tri(a, b, draw):
    r = _use_draw(draw, "CollideEllipseTriangle",
                  a.cx, a.cy, a.rx, a.ry, b.x1, b.y1, b.x2, b.y2, b.x3, b.y3,
                  a.rotation)
    if r is not None: return bool(r)
    if _pt_ellipse(Point(b.x1, b.y1), a, draw): return True
    if _pt_ellipse(Point(b.x2, b.y2), a, draw): return True
    if _pt_ellipse(Point(b.x3, b.y3), a, draw): return True
    if _pt_tri(Point(a.cx, a.cy), b, draw): return True
    return False

def _ellipse_line(a, b, draw):
    r = _use_draw(draw, "CollideLineEllipse", b.x1, b.y1, b.x2, b.y2,
                  a.cx, a.cy, a.rx, a.ry, a.rotation)
    if r is not None: return bool(r)
    if _pt_ellipse(Point(b.x1, b.y1), a, draw): return True
    if _pt_ellipse(Point(b.x2, b.y2), a, draw): return True
    return _segment_intersects_ellipse_pure(b.x1, b.y1, b.x2, b.y2,
                                            a.cx, a.cy, a.rx, a.ry, a.rotation)

def _tri_tri(a, b, draw):
    r = _use_draw(draw, "CollideTriangleTriangle",
                  a.x1, a.y1, a.x2, a.y2, a.x3, a.y3,
                  b.x1, b.y1, b.x2, b.y2, b.x3, b.y3)
    if r is not None: return bool(r)
    return _sat_convex(((a.x1,a.y1),(a.x2,a.y2),(a.x3,a.y3)),
                       ((b.x1,b.y1),(b.x2,b.y2),(b.x3,b.y3)))

def _tri_line(a, b, draw):
    r = _use_draw(draw, "CollideLineTriangle", b.x1, b.y1, b.x2, b.y2,
                  a.x1, a.y1, a.x2, a.y2, a.x3, a.y3)
    if r is not None: return bool(r)
    if _pt_tri(Point(b.x1, b.y1), a, draw): return True
    if _pt_tri(Point(b.x2, b.y2), a, draw): return True
    for (x1,y1),(x2,y2) in (((a.x1,a.y1),(a.x2,a.y2)),
                            ((a.x2,a.y2),(a.x3,a.y3)),
                            ((a.x3,a.y3),(a.x1,a.y1))):
        if _seg_seg_intersect(b.x1, b.y1, b.x2, b.y2, x1, y1, x2, y2):
            return True
    return False

def _line_line(a, b, draw):
    r = _use_draw(draw, "CollideLineLine", a.x1, a.y1, a.x2, a.y2,
                  b.x1, b.y1, b.x2, b.y2)
    if r is not None: return bool(r)
    return _seg_seg_intersect(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2)


# Tabella dispatch: chiave (type_a, type_b). Se manca, il dispatcher prova
# a scambiare gli argomenti (la funzione riceve sempre a,b nell'ordine registrato).
_TABLE = {
    (_T_POINT,    _T_POINT):    _pt_pt,
    (_T_POINT,    _T_RECT):     _pt_rect,
    (_T_POINT,    _T_ROTRECT):  _pt_rotrect,
    (_T_POINT,    _T_CIRCLE):   _pt_circle,
    (_T_POINT,    _T_ELLIPSE):  _pt_ellipse,
    (_T_POINT,    _T_LINE):     _pt_line,
    (_T_POINT,    _T_TRIANGLE): _pt_tri,

    (_T_RECT,     _T_RECT):     _rect_rect,
    (_T_RECT,     _T_ROTRECT):  _rect_rotrect,
    (_T_ROTRECT,  _T_ROTRECT):  _rotrect_rotrect,
    (_T_RECT,     _T_CIRCLE):   _rect_circle,
    (_T_ROTRECT,  _T_CIRCLE):   _rotrect_circle,
    (_T_RECT,     _T_ELLIPSE):  _rect_ellipse,
    (_T_ROTRECT,  _T_ELLIPSE):  _rotrect_ellipse,
    (_T_RECT,     _T_TRIANGLE): _rect_tri,
    (_T_ROTRECT,  _T_TRIANGLE): _rotrect_tri,
    (_T_RECT,     _T_LINE):     _rect_line,
    (_T_ROTRECT,  _T_LINE):     _rotrect_line,

    (_T_CIRCLE,   _T_CIRCLE):   _circle_circle,
    (_T_CIRCLE,   _T_ELLIPSE):  _circle_ellipse,
    (_T_CIRCLE,   _T_TRIANGLE): _circle_tri,
    (_T_CIRCLE,   _T_LINE):     _circle_line,

    (_T_ELLIPSE,  _T_ELLIPSE):  _ellipse_ellipse,
    (_T_ELLIPSE,  _T_TRIANGLE): _ellipse_tri,
    (_T_ELLIPSE,  _T_LINE):     _ellipse_line,

    (_T_TRIANGLE, _T_TRIANGLE): _tri_tri,
    (_T_TRIANGLE, _T_LINE):     _tri_line,

    (_T_LINE,     _T_LINE):     _line_line,
}


# --- Polygon dispatch -------------------------------------------------------
def _dispatch_polygon(a, b, draw):
    # Assicura A = Polygon
    if a._t != _T_POLYGON:
        a, b = b, a
    # Polygon vs Polygon
    if b._t == _T_POLYGON:
        return _polygon_vs_polygon(a, b)
    # Polygon vs primitiva: converti primitiva in polygon quando possibile,
    # oppure test punto/lati.
    if b._t == _T_POINT:
        return _point_in_polygon(b.x, b.y, a.points)
    if b._t == _T_RECT:
        return _polygon_vs_polygon(a, Polygon(_rect_corners(b.x,b.y,b.w,b.h,0.0)))
    if b._t == _T_ROTRECT:
        return _polygon_vs_polygon(a,
            Polygon(_rect_corners(b.x, b.y, b.w, b.h, b.rotation)))
    if b._t == _T_TRIANGLE:
        return _polygon_vs_polygon(a,
            Polygon(((b.x1,b.y1),(b.x2,b.y2),(b.x3,b.y3))))
    if b._t == _T_LINE:
        # linea come "polygon degenere": test vertici + segmento vs lati poly
        if _point_in_polygon(b.x1, b.y1, a.points): return True
        if _point_in_polygon(b.x2, b.y2, a.points): return True
        pts = a.points; n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]; x2, y2 = pts[(i+1) % n]
            if _seg_seg_intersect(b.x1, b.y1, b.x2, b.y2, x1, y1, x2, y2):
                return True
        return False
    if b._t == _T_CIRCLE:
        # centro dentro poly?
        if _point_in_polygon(b.cx, b.cy, a.points): return True
        pts = a.points; n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]; x2, y2 = pts[(i+1) % n]
            if _pt_line(Point(b.cx, b.cy), Line(x1, y1, x2, y2, b.r*2), draw):
                return True
        return False
    if b._t == _T_ELLIPSE:
        # Centro dentro il poligono?
        if _point_in_polygon(b.cx, b.cy, a.points):
            return True
        # Vertici del poligono dentro l'ellisse?
        for p in a.points:
            if _pt_ellipse(Point(p[0], p[1]), b, draw):
                return True
        # **FIX: intersezione tra ogni lato del poligono e l'ellisse**
        pts = a.points
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if draw is not None:
                hit = draw._segment_intersects_ellipse(
                    x1, y1, x2, y2,
                    b.cx, b.cy, b.rx, b.ry, b.rotation
                )
            else:
                hit = _segment_intersects_ellipse_pure(
                    x1, y1, x2, y2,
                    b.cx, b.cy, b.rx, b.ry, b.rotation
                )
            if hit:
                return True
        return False
    raise TypeError(f"Polygon vs {b.__class__.__name__} non supportato")


# --------------------------------------------------------------------------- #
# TEXTURE COLLIDER — collider personalizzabili per immagini
# --------------------------------------------------------------------------- #
class TextureCollider(_Shape):
    """Collider per una texture (immagine) con forma personalizzabile.

    Le texture non hanno una forma di default: usa questo collider per dirgli
    esattamente come deve essere trattata la collisione. Modalita' disponibili:

        mode="rect"    : AABB della texture (dimensioni tx,ty,tw,th; supporta rotation).
        mode="rotrect" : OBB (usa rotation).
        mode="circle"  : cerchio inscritto o custom (radius=...).
        mode="ellipse" : ellisse.
        mode="polygon" : poligono arbitrario (points=[(x,y),...] RELATIVI al top-left).
        mode="pixel"   : maschera pixel-perfect (soglia alpha). Richiede PIL+numpy.
                         Usa `image_path=...` oppure `alpha_mask=<np.ndarray bool>`.

    Esempi
    ------
        # cerchio manuale
        col = TextureCollider(tx=100, ty=100, tw=64, th=64,
                              mode="circle", radius=30)

        # poligono che approssima una spada
        sword = TextureCollider(tx=100, ty=100, tw=32, th=128,
                                mode="polygon",
                                points=[(14,0),(18,0),(18,128),(14,128)])

        # pixel-perfect (dopo aver caricato l'immagine)
        col = TextureCollider(tx=0, ty=0, tw=64, th=64,
                              mode="pixel", image_path="assets/hero.png",
                              alpha_threshold=8)
    """
    __slots__ = ("tx", "ty", "tw", "th", "rotation", "mode",
                 "_shape_cache", "_mask", "_mask_w", "_mask_h",
                 "_alpha_threshold", "_points_rel", "_radius",
                 "_erx", "_ery")

    def __init__(self, tx, ty, tw, th, mode="rect", rotation=0.0,
                 radius=None, rx=None, ry=None,
                 points=None, image_path=None, alpha_mask=None,
                 alpha_threshold=1):
        self._t = _T_TEXCOL
        self.tx = float(tx); self.ty = float(ty)
        self.tw = float(tw); self.th = float(th)
        self.rotation = float(rotation)
        self.mode = mode
        self._shape_cache = None
        self._mask = None
        self._mask_w = self._mask_h = 0
        self._alpha_threshold = int(alpha_threshold)
        self._points_rel = None
        self._radius = radius
        self._erx = rx; self._ery = ry

        if mode == "polygon":
            if not points:
                raise ValueError("TextureCollider mode='polygon' richiede points=[...]")
            self._points_rel = tuple((float(x), float(y)) for x, y in points)
        elif mode == "pixel":
            if alpha_mask is not None:
                if not _HAS_NUMPY:
                    raise RuntimeError("Pixel-mask richiede numpy")
                arr = np.asarray(alpha_mask, dtype=bool)
                if arr.ndim != 2:
                    raise ValueError("alpha_mask deve essere 2D (H,W)")
                self._mask = arr
                self._mask_h, self._mask_w = arr.shape
            elif image_path is not None:
                self._load_mask(image_path)
            else:
                raise ValueError("TextureCollider mode='pixel' richiede image_path=... o alpha_mask=...")

    # -- setup ----------------------------------------------------------------
    def _load_mask(self, path):
        if not (_HAS_PIL and _HAS_NUMPY):
            raise RuntimeError("Pixel-mask richiede PIL e numpy")
        im = Image.open(path).convert("RGBA")
        arr = np.array(im)  # (H, W, 4)
        self._mask = arr[..., 3] >= self._alpha_threshold
        self._mask_h, self._mask_w = self._mask.shape

    # -- posizione dinamica ---------------------------------------------------
    def move_to(self, tx, ty):
        self.tx = float(tx); self.ty = float(ty)
        self._shape_cache = None
        return self

    def rotate(self, rotation):
        self.rotation = float(rotation)
        self._shape_cache = None
        return self

    # -- shape derivata (per collision & debug draw) --------------------------
    def _derived_shape(self):
        if self._shape_cache is not None:
            return self._shape_cache
        m = self.mode
        if m == "rect":
            if self.rotation == 0.0:
                s = Rect(self.tx, self.ty, self.tw, self.th)
            else:
                s = RotRect(self.tx, self.ty, self.tw, self.th, self.rotation)
        elif m == "rotrect":
            s = RotRect(self.tx, self.ty, self.tw, self.th, self.rotation)
        elif m == "circle":
            r = self._radius if self._radius is not None else min(self.tw, self.th) * 0.5
            s = Circle(self.tx + self.tw*0.5, self.ty + self.th*0.5, r)
        elif m == "ellipse":
            rx = self._erx if self._erx is not None else self.tw * 0.5
            ry = self._ery if self._ery is not None else self.th * 0.5
            s = Ellipse(self.tx + self.tw*0.5, self.ty + self.th*0.5, rx, ry, self.rotation)
        elif m == "polygon":
            # applica rotazione attorno al centro texture
            if self.rotation == 0.0:
                pts = tuple((self.tx + x, self.ty + y) for (x, y) in self._points_rel)
            else:
                cx = self.tx + self.tw * 0.5; cy = self.ty + self.th * 0.5
                a = self.rotation * 0.017453292519943295
                cs = math.cos(a); sn = math.sin(a)
                pts = []
                for (x, y) in self._points_rel:
                    wx = self.tx + x; wy = self.ty + y
                    dx = wx - cx; dy = wy - cy
                    pts.append((cx + dx*cs - dy*sn, cy + dx*sn + dy*cs))
                pts = tuple(pts)
            s = Polygon(pts)
        elif m == "pixel":
            # rappresentazione geometrica per debug: rect esterno
            if self.rotation == 0.0:
                s = Rect(self.tx, self.ty, self.tw, self.th)
            else:
                s = RotRect(self.tx, self.ty, self.tw, self.th, self.rotation)
        else:
            raise ValueError(f"TextureCollider mode sconosciuto: {m!r}")
        self._shape_cache = s
        return s

    # -- collisione con altra forma ------------------------------------------
    def _collides_shape(self, other, draw):
        if self.mode == "pixel":
            return self._collides_pixel(other, draw)
        return _dispatch(self._derived_shape(), other, draw)

    # -- pixel-mask fast path ------------------------------------------------
    def _collides_pixel(self, other, draw):
        # Bounding AABB del texture in mondo
        if self.rotation != 0.0:
            # per rotazioni usiamo l'AABB del OBB (piu' semplice, ok come pre-check)
            corners = _rect_corners(self.tx, self.ty, self.tw, self.th, self.rotation)
            xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
            aabb = Rect(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
        else:
            aabb = Rect(self.tx, self.ty, self.tw, self.th)

        # Fast-reject AABB
        if not _dispatch(aabb, other, draw):
            return False

        if not _HAS_NUMPY:
            return True  # senza numpy ci fermiamo al AABB (comportamento sicuro)

        # Ricava AABB intersezione in coordinate immagine (0..mask_w, 0..mask_h)
        mw = self._mask_w; mh = self._mask_h
        sx = mw / self.tw; sy = mh / self.th

        # AABB "other" in mondo -> in coord texture (senza rotazione)
        # Per semplicita': se rotation!=0, ruotiamo il punto/AABB della forma "other"
        # nel sistema locale del texture.
        if self.rotation != 0.0:
            cx = self.tx + self.tw * 0.5; cy = self.ty + self.th * 0.5
            ang = -self.rotation * 0.017453292519943295
            cs = math.cos(ang); sn = math.sin(ang)
            def to_local(px, py):
                dx = px - cx; dy = py - cy
                lx = dx*cs - dy*sn; ly = dx*sn + dy*cs
                return lx + self.tw*0.5, ly + self.th*0.5
        else:
            def to_local(px, py):
                return px - self.tx, py - self.ty

        # Ottieni AABB della "other" in coord locali texture
        oaabb = _shape_aabb(other)
        if oaabb is None:
            return True  # non riusciamo a calcolare AABB -> conservativo
        ox, oy, ow, oh = oaabb
        # 4 vertici -> locale -> bounding box locale
        pts_local = [to_local(ox, oy), to_local(ox+ow, oy),
                     to_local(ox+ow, oy+oh), to_local(ox, oy+oh)]
        lxs = [p[0] for p in pts_local]; lys = [p[1] for p in pts_local]
        lx0 = max(0.0, min(lxs)); ly0 = max(0.0, min(lys))
        lx1 = min(self.tw, max(lxs)); ly1 = min(self.th, max(lys))
        if lx0 > lx1 or ly0 > ly1:
            return False

        # Coordinate immagine — garantisce ALMENO un pixel (Point / AABB
        # zero-area campionerebbero altrimenti un range vuoto).
        ix0 = int(math.floor(lx0 * sx)); iy0 = int(math.floor(ly0 * sy))
        ix1 = int(math.ceil (lx1 * sx)); iy1 = int(math.ceil (ly1 * sy))
        if ix1 == ix0: ix1 = ix0 + 1
        if iy1 == iy0: iy1 = iy0 + 1
        ix0 = max(0, ix0); iy0 = max(0, iy0)
        ix1 = min(mw, ix1); iy1 = min(mh, iy1)
        if ix0 >= ix1 or iy0 >= iy1:
            return False

        # Cropped mask: c'e' almeno un pixel opaco?
        return bool(self._mask[iy0:iy1, ix0:ix1].any())


def _shape_aabb(s):
    """AABB (x,y,w,h) di una forma qualsiasi. None se non calcolabile."""
    t = s._t
    if t == _T_POINT:    return (s.x, s.y, 0.0, 0.0)
    if t == _T_RECT:     return (s.x, s.y, s.w, s.h)
    if t == _T_ROTRECT:
        c = _rect_corners(s.x, s.y, s.w, s.h, s.rotation)
        xs = [p[0] for p in c]; ys = [p[1] for p in c]
        return (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
    if t == _T_CIRCLE:   return (s.cx - s.r, s.cy - s.r, s.r*2, s.r*2)
    if t == _T_ELLIPSE:
        if s.rotation == 0.0:
            return (s.cx - s.rx, s.cy - s.ry, s.rx*2, s.ry*2)
        a = s.rotation * 0.017453292519943295
        cs = math.cos(a); sn = math.sin(a)
        hw = math.hypot(s.rx*cs, s.ry*sn); hh = math.hypot(s.rx*sn, s.ry*cs)
        return (s.cx - hw, s.cy - hh, hw*2, hh*2)
    if t == _T_LINE:
        x0 = min(s.x1, s.x2); y0 = min(s.y1, s.y2)
        return (x0, y0, abs(s.x2 - s.x1), abs(s.y2 - s.y1))
    if t == _T_TRIANGLE:
        x0 = min(s.x1, s.x2, s.x3); y0 = min(s.y1, s.y2, s.y3)
        return (x0, y0, max(s.x1,s.x2,s.x3)-x0, max(s.y1,s.y2,s.y3)-y0)
    if t == _T_POLYGON:  return s._aabb
    if t == _T_RRECT:    return (s.x, s.y, s.w, s.h)
    if t == _T_TEXCOL:   return (s.tx, s.ty, s.tw, s.th)
    return None


# --------------------------------------------------------------------------- #
# DEBUG DRAW
# --------------------------------------------------------------------------- #
def _draw_shape_outline(draw, shape, color, thickness=2.0):
    """Disegna il contorno della forma sul draw (istanza DRAW). Silente se draw=None."""
    if draw is None:
        return
    t = shape._t
    if t == _T_TEXCOL:
        _draw_shape_outline(draw, shape._derived_shape(), color, thickness)
        return
    if t == _T_POINT:
        # piccolo cerchio pieno per visualizzarlo
        if hasattr(draw, "DrawCircleOutline"):
            draw.DrawCircleOutline(shape.x, shape.y, 3.0, thickness=thickness, color=color)
        return
    if t == _T_RECT:
        if hasattr(draw, "DrawRectOutline"):
            draw.DrawRectOutline(shape.x, shape.y, shape.w, shape.h,
                                 thickness=thickness, color=color)
        return
    if t == _T_ROTRECT:
        if hasattr(draw, "DrawRectOutline"):
            draw.DrawRectOutline(shape.x, shape.y, shape.w, shape.h,
                                 thickness=thickness, color=color, rotation=shape.rotation)
        return
    if t == _T_CIRCLE:
        if hasattr(draw, "DrawCircleOutline"):
            draw.DrawCircleOutline(shape.cx, shape.cy, shape.r,
                                   thickness=thickness, color=color)
        return
    if t == _T_ELLIPSE:
        if hasattr(draw, "DrawEllipseOutline"):
            draw.DrawEllipseOutline(shape.cx, shape.cy, shape.rx, shape.ry,
                                    thickness=thickness, color=color, rotation=shape.rotation)
        return
    if t == _T_LINE:
        if hasattr(draw, "DrawLine"):
            draw.DrawLine(shape.x1, shape.y1, shape.x2, shape.y2,
                          thickness=max(thickness, shape.thickness), color=color)
        return
    if t == _T_TRIANGLE:
        if hasattr(draw, "DrawTriangleOutline"):
            draw.DrawTriangleOutline(shape.x1, shape.y1, shape.x2, shape.y2,
                                     shape.x3, shape.y3, thickness=thickness, color=color)
        return
    if t == _T_RRECT:
        if hasattr(draw, "DrawRoundedRectOutline"):
            draw.DrawRoundedRectOutline(shape.x, shape.y, shape.w, shape.h,
                                        shape.radius, thickness=thickness, color=color)
        return
    if t == _T_POLYGON:
        # disegna lati come linee
        if hasattr(draw, "DrawLine"):
            pts = shape.points; n = len(pts)
            for i in range(n):
                x1, y1 = pts[i]; x2, y2 = pts[(i+1) % n]
                draw.DrawLine(x1, y1, x2, y2, thickness=thickness, color=color)
        return



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

    # ------------------------------------------------------------------ #
    # EARLY-Z / DEPTH TEST — opt-in, retro-compatibile
    # ------------------------------------------------------------------ #
    # Uso tipico (nel loop di draw):
    #
    #   window.enable_early_z(True)                # una volta sola all'avvio
    #   # ---- pass 1: opachi front-to-back ----
    #   window.begin_opaque_pass()
    #   for spr in DRAW.sort_front_to_back(opaque_sprites, key=lambda s: s.z):
    #       spr.draw()                             # depth write ON, alpha OFF
    #   # ---- pass 2: semitrasparenti back-to-front ----
    #   window.begin_transparent_pass()
    #   for spr in DRAW.sort_back_to_front(alpha_sprites, key=lambda s: s.z):
    #       spr.draw()                             # depth write OFF, alpha ON
    #   window.end_depth_passes()                  # ripristina stato "normale"
    #
    # NOTA: il framebuffer del contesto deve avere un depth attachment.
    # Se stai usando il default screen framebuffer di moderngl, va già bene.
    # Se usi CameraGPU (FBO custom) e vuoi Early-Z anche dentro cam.begin(),
    # devi ricreare l'FBO con un depth renderbuffer — vedi CameraGPU per la
    # personalizzazione. Se non lo fai, il DEPTH_TEST resta attivo ma è un
    # no-op sul FBO senza depth (nessuna regressione visiva).
    def enable_early_z(self, enabled: bool = True) -> None:
        """
        Attiva/disattiva il Depth Test globale sul contesto moderngl.
        Chiamalo una volta sola (es. dopo la creazione della WINDOW).
        Retro-compatibile: se non lo chiami mai, il comportamento del motore
        è identico a prima.
        """
        if not hasattr(self, "ctx") or self.ctx is None:
            return
        if enabled:
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)
        self._early_z_enabled = bool(enabled)

    def begin_opaque_pass(self) -> None:
        """
        Prepara il contesto per disegnare gli oggetti opachi (pass 1).
        Depth write ON, depth func LESS: l'hardware scarta via Early-Z i pixel
        già coperti da un oggetto più vicino (ordina front-to-back per max effetto).
        """
        ctx = getattr(self, "ctx", None)
        if ctx is None:
            return
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"          # LESS
        # moderngl non ha un flag depth_mask separato; l'assenza di
        # DEPTH_TEST disattiva anche il write. Con DEPTH_TEST attivo,
        # il write è ON di default. Nulla da fare qui.

    def begin_transparent_pass(self) -> None:
        """
        Prepara il contesto per disegnare gli oggetti semitrasparenti (pass 2).
        Il depth test resta ON (per non disegnare "dietro" al mondo opaco), ma
        la scrittura del depth buffer va spenta per non impedire il corretto
        alpha-blending tra oggetti trasparenti sovrapposti. Ordina back-to-front.

        NOTA moderngl: non esiste un `depth_mask` diretto sul contesto; il
        pattern portabile è usare `depth_func = "<="` (o `"always"` se vuoi
        ignorare completamente l'occlusione) mantenendo il test attivo. Per
        controllo fine chiama manualmente `glDepthMask(False)` via
        `ctx.extra` se disponibile.
        """
        ctx = getattr(self, "ctx", None)
        if ctx is None:
            return
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<="

    def end_depth_passes(self) -> None:
        """
        Ripristina lo stato di rendering "classico" (nessun depth test).
        Chiamalo a fine frame se vuoi disegnare UI in ordine di draw call.
        """
        ctx = getattr(self, "ctx", None)
        if ctx is None:
            return
        ctx.depth_func = "<"
        if not getattr(self, "_early_z_enabled", False):
            ctx.disable(moderngl.DEPTH_TEST)

    # Helper statici di ordinamento — funzionano su qualunque iterable.
    @staticmethod
    def sort_front_to_back(items, key=None):
        """
        Ordina gli oggetti dal più vicino al più lontano (z crescente se
        z=0 è "davanti"). Usa il risultato per il pass opaco.
        """
        if key is None:
            key = lambda o: getattr(o, "z", 0.0)
        return sorted(items, key=key)

    @staticmethod
    def sort_back_to_front(items, key=None):
        """
        Ordina gli oggetti dal più lontano al più vicino. Usa il risultato
        per il pass alpha-blended.
        """
        if key is None:
            key = lambda o: getattr(o, "z", 0.0)
        return sorted(items, key=key, reverse=True)



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
                in uint i_color;      // RGBA8 packed (R=bits0-7 G=8-15 B=16-23 A=24-31)     // 0..255 (RGBA) — tint moltiplicativo
                uniform vec2 u_resolution;
                out vec2 v_uv;
                out vec4 v_color;

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
                    // FIX: passiamo un vec4 RGBA in [0,255] (coerente con
                    // gli altri instance program). Il fragment moltiplica per
                    // la texture del glifo (bianca) => tint colore + alpha.
                    v_color = vec4(float(i_color & 0xFFu), float((i_color >> 8u) & 0xFFu), float((i_color >> 16u) & 0xFFu), float((i_color >> 24u) & 0xFFu)) / 255.0;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_tex;
                in vec2 v_uv;
                in vec4 v_color;
                out vec4 f_color;
                void main() {
                    vec4 tex_color = texture(u_tex, v_uv);
                    f_color = vec4(tex_color.rgb * v_color.rgb,
                                   tex_color.a   * v_color.a);
                }
            """
        )

        quad = np.array([-1,-1, 1,-1, 1,1, -1,1], dtype='f4')
        indices = np.array([0,1,2, 0,2,3], dtype='i4')
        self.sprite_inst_vbo = self.ctx.buffer(quad.tobytes())
        self.sprite_inst_ibo = self.ctx.buffer(indices.tobytes())

        # 11 float-slot/istanza: pos(2)+size(2)+dir(2)+uv(4)+color(1, uint32
        # packed RGBA8 — vedi _pack_rgba_u32_as_f4).
        self.sprite_inst_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 11 * 4,
            dynamic=True
        )

        self.sprite_inst_vao = self.ctx.vertex_array(
            self.sprite_inst_prog,
            [
                (self.sprite_inst_vbo, "2f", "in_corner"),
                (self.sprite_inst_instance_vbo, "2f 2f 2f 4f 1u/i",
                "i_pos", "i_size", "i_dir", "i_uv", "i_color"),
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
                in uint i_color;      // RGBA8 packed (R=bits0-7 G=8-15 B=16-23 A=24-31)       // 0..255

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
                    v_color = vec4(float(i_color & 0xFFu), float((i_color >> 8u) & 0xFFu), float((i_color >> 16u) & 0xFFu), float((i_color >> 24u) & 0xFFu)) / 255.0;
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
            reserve=self.max_rects * 7 * 4,
            dynamic=True
        )

        self.ellipse_vao = self.ctx.vertex_array(
            self.ellipse_prog,
            [
                (self.ellipse_vbo, "2f", "in_corner"),
                (self.ellipse_instance_vbo, "2f 2f 2f 1u/i",
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
                in uint i_color;      // RGBA8 packed (R=bits0-7 G=8-15 B=16-23 A=24-31)

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
                    v_color = vec4(float(i_color & 0xFFu), float((i_color >> 8u) & 0xFFu), float((i_color >> 16u) & 0xFFu), float((i_color >> 24u) & 0xFFu)) / 255.0;
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
            reserve=self.max_rects * 8 * 4,
            dynamic=True
        )

        self.ellipse_outline_vao = self.ctx.vertex_array(
            self.ellipse_outline_prog,
            [
                (self.ellipse_outline_vbo, "2f", "in_corner"),
                (self.ellipse_outline_instance_vbo, "2f 2f 1f 2f 1u/i",
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
                in uint i_color;      // RGBA8 packed (R=bits0-7 G=8-15 B=16-23 A=24-31)
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
                    v_color = vec4(float(i_color & 0xFFu), float((i_color >> 8u) & 0xFFu), float((i_color >> 16u) & 0xFFu), float((i_color >> 24u) & 0xFFu)) / 255.0;
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

        # Buffer dinamico standard — ora 7 float-slot/istanza: pos(2)+size(2)+
        # dir(2)+color(1, uint32 packed RGBA8 — vedi _pack_rgba_u32_as_f4).
        self.rect_inst_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 7 * 4,
            dynamic=True
        )

        self.rect_inst_vao = self.ctx.vertex_array(
            self.rect_inst_prog,
            [
                (self.rect_inst_vbo, "2f", "in_corner"),
                (self.rect_inst_instance_vbo, "2f 2f 2f 1u/i",
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
        # Numba compilato, nessun overhead Python per rettangolo).
        self.rect_outline_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_corner;
                in vec2 i_pos;
                in vec2 i_size;
                in float i_thickness;
                in vec2 i_dir;
                in uint i_color;      // RGBA8 packed (R=bits0-7 G=8-15 B=16-23 A=24-31)
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
                    v_color = vec4(float(i_color & 0xFFu), float((i_color >> 8u) & 0xFFu), float((i_color >> 16u) & 0xFFu), float((i_color >> 24u) & 0xFFu)) / 255.0;
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

        # 8 float-slot/istanza: pos(2)+size(2)+thickness(1)+dir(2)+color(1, uint32 packed)
        self.rect_outline_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 8 * 4,
            dynamic=True
        )

        self.rect_outline_vao = self.ctx.vertex_array(
            self.rect_outline_prog,
            [
                (self.rect_outline_vbo, "2f", "in_corner"),
                (self.rect_outline_instance_vbo, "2f 2f 1f 2f 1u/i",
                "i_pos", "i_size", "i_thickness", "i_dir", "i_color"),
            ],
            index_buffer=self.rect_outline_ibo
        )

        if hasattr(self, "size"):
            self.rect_outline_prog["u_resolution"].value = (self.size[0], self.size[1])



    def _init_rounded_rect_gpu(self):
        """Pipeline instanced per DrawRoundedRectsBatch. Stessa architettura
        di _init_rect_gpu (SDF nel fragment, quad unitario condiviso,
        packing via kernel Numba compilato): 1 draw call instanced per chunk."""
        self.rrect_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_corner;
                in vec2 i_pos;
                in vec2 i_size;
                in float i_radius;
                in vec2 i_dir;
                in uint i_color;      // RGBA8 packed (R=bits0-7 G=8-15 B=16-23 A=24-31)
                uniform vec2 u_resolution;

                out vec2 v_local;
                out vec2 v_half_size;
                out float v_radius;
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
                    v_radius = i_radius;
                    v_color = vec4(float(i_color & 0xFFu), float((i_color >> 8u) & 0xFFu), float((i_color >> 16u) & 0xFFu), float((i_color >> 24u) & 0xFFu)) / 255.0;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_local;
                in vec2 v_half_size;
                in float v_radius;
                in vec4 v_color;
                out vec4 f_color;

                float sdRoundedBox(vec2 p, vec2 b, float r) {
                    vec2 q = abs(p) - b + vec2(r);
                    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r;
                }

                void main() {
                    // Clamp del raggio alla mezza-dimensione minore per non
                    // uscire dal quad (identico allo shrink CPU per il triangolo).
                    float r = min(v_radius, min(v_half_size.x, v_half_size.y));
                    float d = sdRoundedBox(v_local, v_half_size, r);
                    // AA indipendente da zoom/rotazione (stessa formula di rect_prog).
                    float aa = max(fwidth(d), 0.5);
                    float alpha = smoothstep(aa, -aa, d);
                    f_color = vec4(v_color.rgb, v_color.a * alpha);
                }
            """
        )

        quad = np.array([-1,-1, 1,-1, 1,1, -1,1], dtype='f4')
        indices = np.array([0,1,2, 0,2,3], dtype='i4')
        self.rrect_vbo = self.ctx.buffer(quad.tobytes())
        self.rrect_ibo = self.ctx.buffer(indices.tobytes())

        # 8 float-slot/istanza: pos(2)+size(2)+radius(1)+dir(2)+color(1, uint32 packed)
        self.rrect_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 8 * 4,
            dynamic=True,
        )
        self.rrect_vao = self.ctx.vertex_array(
            self.rrect_prog,
            [
                (self.rrect_vbo, "2f", "in_corner"),
                (self.rrect_instance_vbo, "2f 2f 1f 2f 1u/i",
                 "i_pos", "i_size", "i_radius", "i_dir", "i_color"),
            ],
            index_buffer=self.rrect_ibo,
        )
        if hasattr(self, "size"):
            self.rrect_prog["u_resolution"].value = (self.size[0], self.size[1])

    def _init_rounded_rect_outline_gpu(self):
        """Pipeline instanced per DrawRoundedRectsOutlineBatch. Speculare a
        _init_rounded_rect_gpu con thickness aggiuntivo."""
        self.rrect_outline_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_corner;
                in vec2 i_pos;
                in vec2 i_size;
                in float i_radius;
                in float i_thickness;
                in vec2 i_dir;
                in uint i_color;      // RGBA8 packed (R=bits0-7 G=8-15 B=16-23 A=24-31)
                uniform vec2 u_resolution;

                out vec2 v_local;
                out vec2 v_half_size;
                out float v_radius;
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
                    v_radius = i_radius;
                    v_thickness = i_thickness;
                    v_color = vec4(float(i_color & 0xFFu), float((i_color >> 8u) & 0xFFu), float((i_color >> 16u) & 0xFFu), float((i_color >> 24u) & 0xFFu)) / 255.0;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_local;
                in vec2 v_half_size;
                in float v_radius;
                in float v_thickness;
                in vec4 v_color;
                out vec4 f_color;

                float sdRoundedBox(vec2 p, vec2 b, float r) {
                    vec2 q = abs(p) - b + vec2(r);
                    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r;
                }

                void main() {
                    float r = min(v_radius, min(v_half_size.x, v_half_size.y));
                    float d = sdRoundedBox(v_local, v_half_size, r);
                    // dist > 0 dentro il rettangolo arrotondato (convenzione
                    // identica a rect_outline_prog per bordo coerente).
                    float dist = -d;
                    float aa = max(fwidth(dist), 0.5);
                    float outerA = smoothstep(-aa, aa, dist);
                    float innerA = smoothstep(v_thickness - aa,
                                              v_thickness + aa, dist);
                    float alpha = outerA * (1.0 - innerA);
                    f_color = vec4(v_color.rgb, v_color.a * alpha);
                }
            """
        )

        quad = np.array([-1,-1, 1,-1, 1,1, -1,1], dtype='f4')
        indices = np.array([0,1,2, 0,2,3], dtype='i4')
        self.rrect_outline_vbo = self.ctx.buffer(quad.tobytes())
        self.rrect_outline_ibo = self.ctx.buffer(indices.tobytes())

        # 9 float-slot/istanza: pos(2)+size(2)+radius(1)+thickness(1)+dir(2)+color(1, uint32 packed)
        self.rrect_outline_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 9 * 4,
            dynamic=True,
        )
        self.rrect_outline_vao = self.ctx.vertex_array(
            self.rrect_outline_prog,
            [
                (self.rrect_outline_vbo, "2f", "in_corner"),
                (self.rrect_outline_instance_vbo, "2f 2f 1f 1f 2f 1u/i",
                 "i_pos", "i_size", "i_radius", "i_thickness",
                 "i_dir", "i_color"),
            ],
            index_buffer=self.rrect_outline_ibo,
        )
        if hasattr(self, "size"):
            self.rrect_outline_prog["u_resolution"].value = (self.size[0], self.size[1])

    def _init_rounded_triangle_gpu(self):
        """Pipeline instanced per DrawRoundedTrianglesBatch. Il quad unitario
        e' rimappato per istanza sull'AABB del triangolo originale (passato
        come 2 vec2). Il fragment shader calcola la SDF del triangolo
        'shrunk' (i 3 vertici sono gia' rientrati di r_eff sulla CPU via
        kernel Numba) e sottrae r_eff per gli angoli arrotondati."""
        self.rtri_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_corner;         // (-1,-1)..(1,1) — quad AABB unitario
                in vec2 i_v0;
                in vec2 i_v1;
                in vec2 i_v2;
                in float i_radius;         // r_eff (gia' clampato su CPU)
                in vec2 i_aabb_min;
                in vec2 i_aabb_max;
                in uint i_color;      // RGBA8 packed (R=bits0-7 G=8-15 B=16-23 A=24-31)
                uniform vec2 u_resolution;

                out vec2 v_pos;
                out vec2 v_v0;
                out vec2 v_v1;
                out vec2 v_v2;
                out float v_radius;
                out vec4 v_color;

                void main() {
                    vec2 t = in_corner * 0.5 + 0.5;
                    vec2 world = mix(i_aabb_min, i_aabb_max, t);
                    vec2 norm = world / u_resolution * 2.0 - 1.0;
                    norm.y = -norm.y;
                    gl_Position = vec4(norm, 0.0, 1.0);
                    v_pos = world;
                    v_v0 = i_v0;
                    v_v1 = i_v1;
                    v_v2 = i_v2;
                    v_radius = i_radius;
                    v_color = vec4(float(i_color & 0xFFu), float((i_color >> 8u) & 0xFFu), float((i_color >> 16u) & 0xFFu), float((i_color >> 24u) & 0xFFu)) / 255.0;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_pos;
                in vec2 v_v0;
                in vec2 v_v1;
                in vec2 v_v2;
                in float v_radius;
                in vec4 v_color;
                out vec4 f_color;

                // SDF triangolo di Inigo Quilez — signed distance:
                // negativa dentro, positiva fuori.
                float sdTriangle(vec2 p, vec2 p0, vec2 p1, vec2 p2) {
                    vec2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
                    vec2 w0 = p - p0, w1 = p - p1, w2 = p - p2;
                    vec2 pq0 = w0 - e0 * clamp(dot(w0, e0) / dot(e0, e0), 0.0, 1.0);
                    vec2 pq1 = w1 - e1 * clamp(dot(w1, e1) / dot(e1, e1), 0.0, 1.0);
                    vec2 pq2 = w2 - e2 * clamp(dot(w2, e2) / dot(e2, e2), 0.0, 1.0);
                    float s = sign(e0.x * e2.y - e0.y * e2.x);
                    vec2 d = min(min(
                        vec2(dot(pq0, pq0), s * (w0.x * e0.y - w0.y * e0.x)),
                        vec2(dot(pq1, pq1), s * (w1.x * e1.y - w1.y * e1.x))),
                        vec2(dot(pq2, pq2), s * (w2.x * e2.y - w2.y * e2.x)));
                    return -sqrt(d.x) * sign(d.y);
                }

                void main() {
                    float d = sdTriangle(v_pos, v_v0, v_v1, v_v2) - v_radius;
                    float aa = max(fwidth(d), 0.5);
                    float alpha = smoothstep(aa, -aa, d);
                    f_color = vec4(v_color.rgb, v_color.a * alpha);
                }
            """
        )

        quad = np.array([-1,-1, 1,-1, 1,1, -1,1], dtype='f4')
        indices = np.array([0,1,2, 0,2,3], dtype='i4')
        self.rtri_vbo = self.ctx.buffer(quad.tobytes())
        self.rtri_ibo = self.ctx.buffer(indices.tobytes())

        # 12 float-slot/istanza: v0(2)+v1(2)+v2(2)+radius(1)+aabb_min(2)+aabb_max(2)+color(1, uint32 packed)
        self.rtri_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 12 * 4,
            dynamic=True,
        )
        self.rtri_vao = self.ctx.vertex_array(
            self.rtri_prog,
            [
                (self.rtri_vbo, "2f", "in_corner"),
                (self.rtri_instance_vbo, "2f 2f 2f 1f 2f 2f 1u/i",
                 "i_v0", "i_v1", "i_v2", "i_radius",
                 "i_aabb_min", "i_aabb_max", "i_color"),
            ],
            index_buffer=self.rtri_ibo,
        )
        if hasattr(self, "size"):
            self.rtri_prog["u_resolution"].value = (self.size[0], self.size[1])

    def _init_rounded_triangle_outline_gpu(self):
        """Pipeline instanced per DrawRoundedTrianglesOutlineBatch. Speculare
        a _init_rounded_triangle_gpu con thickness aggiuntivo. Il bordo usa
        la stessa formula outerA*(1-innerA) di rect_outline/rrect_outline
        (coerenza visiva)."""
        self.rtri_outline_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_corner;
                in vec2 i_v0;
                in vec2 i_v1;
                in vec2 i_v2;
                in float i_radius;
                in vec2 i_aabb_min;
                in vec2 i_aabb_max;
                in float i_thickness;
                in uint i_color;      // RGBA8 packed (R=bits0-7 G=8-15 B=16-23 A=24-31)
                uniform vec2 u_resolution;

                out vec2 v_pos;
                out vec2 v_v0;
                out vec2 v_v1;
                out vec2 v_v2;
                out float v_radius;
                out float v_thickness;
                out vec4 v_color;

                void main() {
                    vec2 t = in_corner * 0.5 + 0.5;
                    vec2 world = mix(i_aabb_min, i_aabb_max, t);
                    vec2 norm = world / u_resolution * 2.0 - 1.0;
                    norm.y = -norm.y;
                    gl_Position = vec4(norm, 0.0, 1.0);
                    v_pos = world;
                    v_v0 = i_v0;
                    v_v1 = i_v1;
                    v_v2 = i_v2;
                    v_radius = i_radius;
                    v_thickness = i_thickness;
                    v_color = vec4(float(i_color & 0xFFu), float((i_color >> 8u) & 0xFFu), float((i_color >> 16u) & 0xFFu), float((i_color >> 24u) & 0xFFu)) / 255.0;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_pos;
                in vec2 v_v0;
                in vec2 v_v1;
                in vec2 v_v2;
                in float v_radius;
                in float v_thickness;
                in vec4 v_color;
                out vec4 f_color;

                float sdTriangle(vec2 p, vec2 p0, vec2 p1, vec2 p2) {
                    vec2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
                    vec2 w0 = p - p0, w1 = p - p1, w2 = p - p2;
                    vec2 pq0 = w0 - e0 * clamp(dot(w0, e0) / dot(e0, e0), 0.0, 1.0);
                    vec2 pq1 = w1 - e1 * clamp(dot(w1, e1) / dot(e1, e1), 0.0, 1.0);
                    vec2 pq2 = w2 - e2 * clamp(dot(w2, e2) / dot(e2, e2), 0.0, 1.0);
                    float s = sign(e0.x * e2.y - e0.y * e2.x);
                    vec2 d = min(min(
                        vec2(dot(pq0, pq0), s * (w0.x * e0.y - w0.y * e0.x)),
                        vec2(dot(pq1, pq1), s * (w1.x * e1.y - w1.y * e1.x))),
                        vec2(dot(pq2, pq2), s * (w2.x * e2.y - w2.y * e2.x)));
                    return -sqrt(d.x) * sign(d.y);
                }

                void main() {
                    float d = sdTriangle(v_pos, v_v0, v_v1, v_v2) - v_radius;
                    // dist > 0 dentro la forma arrotondata (convenzione
                    // identica a rrect_outline_prog per bordo coerente).
                    float dist = -d;
                    float aa = max(fwidth(dist), 0.5);
                    float outerA = smoothstep(-aa, aa, dist);
                    float innerA = smoothstep(v_thickness - aa,
                                              v_thickness + aa, dist);
                    float alpha = outerA * (1.0 - innerA);
                    f_color = vec4(v_color.rgb, v_color.a * alpha);
                }
            """
        )

        quad = np.array([-1,-1, 1,-1, 1,1, -1,1], dtype='f4')
        indices = np.array([0,1,2, 0,2,3], dtype='i4')
        self.rtri_outline_vbo = self.ctx.buffer(quad.tobytes())
        self.rtri_outline_ibo = self.ctx.buffer(indices.tobytes())

        # 13 float-slot/istanza: v0(2)+v1(2)+v2(2)+radius(1)+aabb_min(2)+aabb_max(2)+thickness(1)+color(1, uint32 packed)
        self.rtri_outline_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 13 * 4,
            dynamic=True,
        )
        self.rtri_outline_vao = self.ctx.vertex_array(
            self.rtri_outline_prog,
            [
                (self.rtri_outline_vbo, "2f", "in_corner"),
                (self.rtri_outline_instance_vbo, "2f 2f 2f 1f 2f 2f 1f 1u/i",
                 "i_v0", "i_v1", "i_v2", "i_radius",
                 "i_aabb_min", "i_aabb_max",
                 "i_thickness", "i_color"),
            ],
            index_buffer=self.rtri_outline_ibo,
        )
        if hasattr(self, "size"):
            self.rtri_outline_prog["u_resolution"].value = (self.size[0], self.size[1])

    def _init_triangle_gpu(self):
        self.tri_inst_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 i_v1;
                in vec2 i_v2;
                in vec2 i_v3;
                in uint i_color;      // RGBA8 packed (R=bits0-7 G=8-15 B=16-23 A=24-31)
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
                    v_color = vec4(float(i_color & 0xFFu), float((i_color >> 8u) & 0xFFu), float((i_color >> 16u) & 0xFFu), float((i_color >> 24u) & 0xFFu)) / 255.0;
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

        # Buffer istanze persistente (7 float-slot per istanza: color packato in 1 uint32)
        self.tri_inst_instance_vbo = self.ctx.buffer(
            reserve=self.max_rects * 7 * 4,
            dynamic=True
        )

        self.tri_inst_vao = self.ctx.vertex_array(
            self.tri_inst_prog,
            [
                (self.tri_inst_instance_vbo, "2f 2f 2f 1u/i",
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
                in uint i_color;      // RGBA8 packed (R=bits0-7 G=8-15 B=16-23 A=24-31)
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
                    v_color = vec4(float(i_color & 0xFFu), float((i_color >> 8u) & 0xFFu), float((i_color >> 16u) & 0xFFu), float((i_color >> 24u) & 0xFFu)) / 255.0;
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
            reserve=self.max_rects * 6 * 4,
            dynamic=True
        )

        self.line_inst_vao = self.ctx.vertex_array(
            self.line_inst_prog,
            [
                (self.line_inst_vbo, "2f", "in_corner"),
                (self.line_inst_instance_vbo, "2f 2f 1f 1u/i",
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

        self.rect_inst_data = np.empty((self.max_rects, 7), dtype='f4')
        self.line_inst_data = np.empty((self.max_rects, 6), dtype='f4')
        self.tri_inst_data  = np.empty((self.max_rects, 7), dtype='f4')
        self.ellipse_instance_data = np.empty((self.max_rects, 7), dtype='f4')
        self.sprite_inst_data = np.empty((self.max_rects, 11), dtype='f4')
        # Buffer instance CPU per le versioni *Outline* Batch (stesso principio
        # zero-alloc-a-runtime dei buffer sopra).
        self.ellipse_outline_instance_data = np.empty((self.max_rects, 8), dtype='f4')
        self.rect_outline_instance_data = np.empty((self.max_rects, 8), dtype='f4')
        # Buffer instance CPU per le versioni Rounded* Batch.
        self.rrect_instance_data = np.empty((self.max_rects, 8), dtype='f4')
        self.rrect_outline_instance_data = np.empty((self.max_rects, 9), dtype='f4')
        self.rtri_instance_data = np.empty((self.max_rects, 12), dtype='f4')
        self.rtri_outline_instance_data = np.empty((self.max_rects, 13), dtype='f4')

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

        # Contatori CPU-accumulo per le forme arrotondate (rrect, rrect
        # outline, rtri, rtri outline): stessa tecnica zero-alloc-a-runtime
        # di self.rect_count / self._np_batch_buffer usata da DrawRect,
        # DrawTriangle ecc. Le DrawRoundedRect/DrawRoundedTriangle ecc.
        # 'immediate' scrivono qui invece di fare un Flush + upload + draw
        # call GPU per OGNI singola chiamata.
        self.rrect_count = 0
        self.rrect_outline_count = 0
        self.rtri_count = 0
        self.rtri_outline_count = 0
        # Traccia quale buffer CPU sta accumulando in questo momento
        # ('quad', 'rrect', 'rrect_outline', 'rtri', 'rtri_outline' o None).
        # Serve a preservare l'ordine di disegno (painter's algorithm):
        # quando una Draw* cambia 'tipo' rispetto all'ultima, scarichiamo
        # PRIMA tutti i buffer pendenti, cosi' le draw call GPU restano
        # nello stesso ordine delle chiamate Draw* dell'utente.
        self._active_kind = None

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
        self._init_rounded_rect_gpu()
        self._init_rounded_rect_outline_gpu()
        self._init_rounded_triangle_gpu()
        self._init_rounded_triangle_outline_gpu()
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
        self._flush_pending_draws()
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

        # L'alpha (colonna 9 nell'input) viene clampato a [0,255]. DrawSpritesBatch
        # non espone un tint RGB (le sprite sono texture piena-risoluzione), quindi
        # il colore packato è sempre bianco con alpha variabile.
        alpha_arr = np.clip(data[:, 9], 0.0, 255.0).astype('f4')
        white = np.empty((n, 4), dtype='f4')
        white[:, 0] = 255.0
        white[:, 1] = 255.0
        white[:, 2] = 255.0
        white[:, 3] = alpha_arr
        color_bits = _pack_rgba_u32_as_f4(white)

        # FIX 3: u_resolution gestito da SetResolution, non serve per-chiamata.
        self.atlas.tex.use(location=0)
        self.sprite_inst_prog["u_tex"].value = 0

        # Layout VBO: pos(2)+size(2)+dir(2)+uv(4)+color(1, uint32 packed) = 11 float-slot
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
            inst[:, 10]  = color_bits[sl]   # color (uint32 packed, letto come i_color)

            self.sprite_inst_instance_vbo.orphan(); self.sprite_inst_instance_vbo.write(memoryview(inst))
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

    def MeasureText(self, text, font="arial", size=24):
        """Ritorna (width, height) del bounding box del testo, usando lo
        stesso layout di DrawText/CollidePointText. Utile per centrare il
        testo anche in orizzontale (width), non solo in verticale.

        reuse=False: alloca buffer freschi invece di riusare quelli di
        DrawText, cosi' una misurazione non sporca lo stato interno usato
        da un'eventuale DrawText chiamata subito prima/dopo.
        """
        self._ensure_text_system()
        if not text:
            return 0.0, 0.0
        laid = self._layout_string(str(text), font, int(size), reuse=False)
        if laid is None:
            return 0.0, 0.0
        gx, gy, gw, gh, _guv = laid
        min_x = float(gx.min()); max_x = float((gx + gw).max())
        min_y = float(gy.min()); max_y = float((gy + gh).max())
        return max_x - min_x, max_y - min_y

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
        # FIX: prima solo l'alpha finiva nel buffer -> il parametro `color`
        # veniva ignorato e il testo appariva sempre bianco. Ora passiamo
        # RGBA completo (0..255) come tint moltiplicativo del glifo bianco.
        col_a = max(0.0, min(255.0, (a * alpha) * _INV_255))
        col_r = float(max(0.0, min(255.0, r)))
        col_g = float(max(0.0, min(255.0, g)))
        col_b = float(max(0.0, min(255.0, b)))

        cos_r, sin_r = self._cos_sin_deg(rotation)

        self._flush_pending_draws()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        self.atlas.tex.use(location=0)
        self.sprite_inst_prog["u_tex"].value = 0

        n = gx.shape[0]
        buf = self._text_batch_buf
        if buf.shape[0] < n:
            self._text_batch_buf = buf = np.empty((n, 11), dtype=np.float32)

        # BUG FIX (rotazione testo): stessa correzione di _numba_layout_glyphs.
        # Lo shader aggiunge un half_size NON ruotato per centrare ogni quad
        # (pensato per sprite/rettangoli che ruotano attorno al proprio
        # centro fisso). Se qui si ruota solo l'angolo top-left del glifo
        # (come faceva la vecchia riga "x + gx*cos_r - gy*sin_r"), quel
        # half_size non ruotato dello shader introduce un disallineamento
        # crescente con l'angolo -> lettere sovrapposte/sfalsate in verticale
        # a rotation!=0. Ruotiamo invece il CENTRO del glifo attorno
        # all'origine e pre-sottraiamo lo stesso half_size, cosi' si
        # cancella con quello riaggiunto dallo shader.
        half_w = gw * 0.5
        half_h = gh * 0.5
        cx = gx + half_w
        cy = gy + half_h
        buf[:n, 0] = x + (cx * cos_r - cy * sin_r) - half_w
        buf[:n, 1] = y + (cx * sin_r + cy * cos_r) - half_h
        buf[:n, 2] = gw
        buf[:n, 3] = gh
        buf[:n, 4] = cos_r
        buf[:n, 5] = sin_r
        buf[:n, 6:10] = guv
        buf[:n, 10] = _pack_rgba_u32_scalar_as_f4(col_r, col_g, col_b, col_a)

        self.sprite_inst_instance_vbo.orphan(); self.sprite_inst_instance_vbo.write(memoryview(buf[:n]))
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
            col_r = float(max(0.0, min(255.0, r)))
            col_g = float(max(0.0, min(255.0, g)))
            col_b = float(max(0.0, min(255.0, b)))
            # INCOERENZA FIX 9: usa _cos_sin_deg (fast-path per angoli
            # multipli di 360°) invece di calcolare cos/sin diretti come
            # faceva prima; ora DrawText e DrawTextBatch condividono la
            # stessa primitiva trigonometrica.
            cos_r, sin_r = self._cos_sin_deg(float(rotation))
            color_bits = np.float32(
                _pack_rgba_u32_scalar_as_f4(col_r, col_g, col_b, col_a)
            )
            chunks.append((gx, gy, gw, gh, guv,
                           float(x), float(y),
                           np.float32(cos_r),
                           np.float32(sin_r),
                           color_bits))

        if not chunks:
            return

        total = sum(c[0].shape[0] for c in chunks)
        if self._text_batch_buf.shape[0] < total:
            self._text_batch_buf = np.empty((total, 11), dtype=np.float32)
        out = self._text_batch_buf[:total]

        offset = 0
        for (gx, gy, gw, gh, guv, ox, oy, cos_r, sin_r, color_bits) in chunks:
            n = gx.shape[0]
            _numba_layout_glyphs(gx, gy, gw, gh, guv,
                                 ox, oy, cos_r, sin_r,
                                 color_bits,
                                 out[offset:offset + n])
            offset += n

        self._flush_pending_draws()
        if self.tex_rect_count > 0:
            self.RefreshTextures()

        self.atlas.tex.use(location=0)
        self.sprite_inst_prog["u_tex"].value = 0

        i = 0
        while i < total:
            chunk = min(self.max_rects, total - i)
            self.sprite_inst_instance_vbo.orphan(); self.sprite_inst_instance_vbo.write(memoryview(out[i:i + chunk]))
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
        arr = arr.reshape((h, w, 4))
        pixels = arr.tobytes()

        # PIXEL-PERFECT: teniamo una copia SOLO del canale alpha (w*h byte,
        # non w*h*4) lato CPU. E' l'unico dato che serve a CollidePointTexture
        # / CollidePointTextureBatch per il test pixel-esatto: niente readback
        # dalla GPU a runtime, niente ri-decodifica del file ad ogni hit-test.
        alpha_ch = np.ascontiguousarray(arr[:, :, 3])

        tex = self.ctx.texture((w, h), 4, pixels)
        if filter_mode.upper() == "NEAREST":
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        if name in self._texture_cache:
            self._texture_cache[name][0].release()

        self._texture_cache[name] = (tex, w, h, alpha_ch)

        sdl2.SDL_FreeFormat(format_ptr)
        sdl2.SDL_FreeSurface(conv_surf)
        sdl2.SDL_FreeSurface(surf_ptr)
        return True
    
    def UnloadTexture(self, name):
        if name in self._texture_cache:
            tex, _, _, _ = self._texture_cache.pop(name)
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
        # BUG FIX (painter's algorithm): usa lo stesso meccanismo _use_batch
        # delle altre primitive. Se prima si stava accumulando un'altra
        # famiglia (quad/rrect/rtri/...), _use_batch la scarica ORA; le
        # chiamate successive a texture accumulano in tex_rect_count e
        # verranno scaricate solo al prossimo cambio di famiglia o a
        # FlushAll(). Cosi' l'ordine delle draw call rispecchia esattamente
        # l'ordine delle chiamate Draw*, texture incluse.
        self._use_batch('tex')

        if name not in self._texture_cache:
            return

        tex, orig_w, orig_h, _ = self._texture_cache[name]

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

        self._use_batch('quad')
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

        self._use_batch('quad')
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

        self._use_batch('quad')
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

        self._use_batch('quad')
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

        self._use_batch('quad')
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
        Numba compilato (_numba_clip_rgba), che elimina 3-4 passaggi
        NumPy (clip x3, broadcast) su array grandi. Kernel sequenziale
        (non parallel=True): a parita' di N tipico per-frame, il costo
        fisso di dispatch dei thread supererebbe il lavoro da fare.
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
        self._flush_pending_draws()
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
        color_bits = _pack_rgba_u32_as_f4(rgba)
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
        # Packing dell'instance buffer via kernel Numba compilato.
        i = 0
        while i < n:
            chunk = min(self.max_rects, n - i)
            inst = self.rect_inst_data[:chunk]
            _numba_pack_rect_instances(
                pos[i:i+chunk], size[i:i+chunk],
                cos_arr[i:i+chunk], sin_arr[i:i+chunk],
                color_bits[i:i+chunk], inst
            )
            self.rect_inst_instance_vbo.orphan(); self.rect_inst_instance_vbo.write(memoryview(inst))
            self.rect_inst_vao.render(instances=chunk)
            i += chunk

    def DrawRectsOutlineBatch(self, positions, sizes, colors, thickness=1.0, alpha=255, rotation=0.0):
        """Versione 'solo bordo' di DrawRectsBatch. Stessa identica
        architettura GPU-instanced (packing vettoriale via kernel Numba
        parallelo + un'unica render(instances=chunk) per chunk): stessa
        fascia di prestazioni Batch di DrawRectsBatch."""
        self._flush_pending_draws()
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
        color_bits = _pack_rgba_u32_as_f4(rgba)
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
                color_bits[i:i+chunk], inst
            )
            self.rect_outline_instance_vbo.orphan(); self.rect_outline_instance_vbo.write(memoryview(inst))
            self.rect_outline_vao.render(instances=chunk)
            i += chunk



    def DrawRoundedRectsBatch(self, positions, sizes, radius,
                              colors, alpha=255, rotation=0.0):
        """Versione GPU-instanced per rettangoli arrotondati. Stessa
        architettura di DrawRectsBatch (packing Numba compilato +
        1 draw call instanced per chunk). `radius` puo' essere scalare
        oppure array di lunghezza n."""
        self._flush_pending_draws()
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
        color_bits = _pack_rgba_u32_as_f4(rgba)
        rot = np.asarray(rotation, dtype='f4').reshape(-1)
        if rot.size == 1:
            rot = np.full(n, rot[0], dtype='f4')
        elif rot.size != n:
            raise ValueError("rotation must be scalar or array of length n")
        self._check_finite_array(rot, "rotation")
        rot_rad = rot * self._DEG2RAD
        cos_arr = np.cos(rot_rad).astype('f4')
        sin_arr = np.sin(rot_rad).astype('f4')

        rad = np.asarray(radius, dtype='f4').reshape(-1)
        if rad.size == 1:
            rad = np.full(n, rad[0], dtype='f4')
        elif rad.size != n:
            raise ValueError("radius must be scalar or array of length n")
        self._check_finite_array(rad, "radius")
        if np.any(rad < 0):
            raise ValueError("radius must be >= 0")

        i = 0
        while i < n:
            chunk = min(self.max_rects, n - i)
            inst = self.rrect_instance_data[:chunk]
            _numba_pack_rrect_instances(
                pos[i:i+chunk], size[i:i+chunk], rad[i:i+chunk],
                cos_arr[i:i+chunk], sin_arr[i:i+chunk],
                color_bits[i:i+chunk], inst,
            )
            self.rrect_instance_vbo.orphan(); self.rrect_instance_vbo.write(memoryview(inst))
            self.rrect_vao.render(instances=chunk)
            i += chunk

    def DrawRoundedRect(self, x, y, w, h, radius,
                        color=(255, 255, 255, 255), rotation=0.0, alpha=255):
        """Versione 'immediate' di DrawRoundedRectsBatch. Accumula l'istanza
        in un buffer CPU pre-allocato (stessa tecnica zero-alloc di
        DrawRect/self._np_batch_buffer) invece di fare Flush + upload VBO +
        draw call GPU per OGNI singola chiamata: N chiamate consecutive
        diventano UNA sola draw call instanced quando il buffer viene
        scaricato (buffer pieno, cambio di primitiva o Flush/FlushAll).
        Stesso SDF/AA/shader della versione batch: bordo identico."""
        self._check_finite(x, y, w, h, radius, rotation,
                           names=("x", "y", "w", "h", "radius", "rotation"))
        if radius < 0:
            raise ValueError(f"radius must be >= 0 (got {radius})")

        self._use_batch('rrect')
        if self.rrect_count >= self.max_rects:
            self._flush_rrect()

        r, g, b, a = self._parse_color(color, alpha)
        cs, sn = self._cos_sin_deg(rotation)

        row = self.rrect_instance_data[self.rrect_count]
        row[0] = x;  row[1] = y
        row[2] = w;  row[3] = h
        row[4] = radius
        row[5] = cs; row[6] = sn
        row[7] = _pack_rgba_u32_scalar_as_f4(r, g, b, a)
        self.rrect_count += 1

    def DrawRoundedRectsOutlineBatch(self, positions, sizes, radius, colors,
                                     thickness=1.0, alpha=255, rotation=0.0):
        """Versione 'solo bordo' di DrawRoundedRectsBatch. Stessa architettura
        instanced (packing Numba compilato + 1 draw call per chunk). Il bordo
        usa la stessa formula outerA*(1-innerA) di DrawRectsOutlineBatch:
        garantisce che il bordo del rettangolo arrotondato sia visivamente
        identico al bordo di un rettangolo dritto quando radius=0."""
        self._flush_pending_draws()
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
        color_bits = _pack_rgba_u32_as_f4(rgba)
        rot = np.asarray(rotation, dtype='f4').reshape(-1)
        if rot.size == 1:
            rot = np.full(n, rot[0], dtype='f4')
        elif rot.size != n:
            raise ValueError("rotation must be scalar or array of length n")
        self._check_finite_array(rot, "rotation")
        rot_rad = rot * self._DEG2RAD
        cos_arr = np.cos(rot_rad).astype('f4')
        sin_arr = np.sin(rot_rad).astype('f4')

        rad = np.asarray(radius, dtype='f4').reshape(-1)
        if rad.size == 1:
            rad = np.full(n, rad[0], dtype='f4')
        elif rad.size != n:
            raise ValueError("radius must be scalar or array of length n")
        self._check_finite_array(rad, "radius")
        if np.any(rad < 0):
            raise ValueError("radius must be >= 0")

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
            inst = self.rrect_outline_instance_data[:chunk]
            _numba_pack_rrect_outline_instances(
                pos[i:i+chunk], size[i:i+chunk], rad[i:i+chunk],
                thick[i:i+chunk],
                cos_arr[i:i+chunk], sin_arr[i:i+chunk],
                color_bits[i:i+chunk], inst,
            )
            self.rrect_outline_instance_vbo.orphan(); self.rrect_outline_instance_vbo.write(memoryview(inst))
            self.rrect_outline_vao.render(instances=chunk)
            i += chunk

    def DrawRoundedRectOutline(self, x, y, w, h, radius, thickness=1.0,
                               color=(255, 255, 255, 255), rotation=0.0,
                               alpha=255):
        """Versione 'immediate' di DrawRoundedRectsOutlineBatch. Accumula in
        CPU come DrawRoundedRect: stessa fascia di prestazioni delle altre
        primitive non-Batch (DrawRectOutline ecc.), stesso shader/AA/bordo
        della versione batch."""
        self._check_finite(x, y, w, h, radius, thickness, rotation,
                           names=("x", "y", "w", "h", "radius",
                                  "thickness", "rotation"))
        if radius < 0:
            raise ValueError(f"radius must be >= 0 (got {radius})")
        if thickness <= 0:
            raise ValueError(f"thickness must be > 0 (got {thickness})")

        self._use_batch('rrect_outline')
        if self.rrect_outline_count >= self.max_rects:
            self._flush_rrect_outline()

        r, g, b, a = self._parse_color(color, alpha)
        cs, sn = self._cos_sin_deg(rotation)

        row = self.rrect_outline_instance_data[self.rrect_outline_count]
        row[0] = x;  row[1] = y
        row[2] = w;  row[3] = h
        row[4] = radius
        row[5] = thickness
        row[6] = cs; row[7] = sn
        row[8] = _pack_rgba_u32_scalar_as_f4(r, g, b, a)
        self.rrect_outline_count += 1

    def DrawLinesBatch(self, x1, y1, x2, y2, colors, thickness=1.0, alpha=255, rotation=0.0):
        self._flush_pending_draws()
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
        color_bits = _pack_rgba_u32_as_f4(rgba)

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
                thick[i:i+chunk], color_bits[i:i+chunk], inst
            )
            self.line_inst_instance_vbo.orphan(); self.line_inst_instance_vbo.write(memoryview(inst))
            self.line_inst_vao.render(instances=chunk)
            i += chunk


    def DrawTrianglesBatch(self, vertices, colors, alpha=255):
        self._flush_pending_draws()
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
        color_bits = _pack_rgba_u32_as_f4(rgba)

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
                color_bits[i:i+chunk], inst
            )
            self.tri_inst_instance_vbo.orphan(); self.tri_inst_instance_vbo.write(memoryview(inst))
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
        self._flush_pending_draws()
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
        color_bits = _pack_rgba_u32_as_f4(rgba)
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
            self.line_inst_instance_vbo.orphan(); self.line_inst_instance_vbo.write(memoryview(inst))
            self.line_inst_vao.render(instances=chunk)
            i += chunk


    def DrawRoundedTrianglesBatch(self, vertices, radius, colors, alpha=255):
        """Versione GPU-instanced per triangoli con angoli arrotondati.
        Stessa architettura di DrawTrianglesBatch (packing Numba compilato +
        1 draw call instanced per chunk). Ogni istanza:
          - CPU (Numba) calcola i 3 vertici 'shrunk' (spinta dei lati verso
            l'interno di r_eff) e l'AABB del triangolo originale;
          - GPU rasterizza un quad AABB e valuta SDF triangolo - r_eff nel
            fragment shader, con AA fwidth-based (identico alle altre
            primitive del motore).
        `radius` puo' essere scalare o array di lunghezza n."""
        self._flush_pending_draws()
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
        color_bits = _pack_rgba_u32_as_f4(rgba)

        rad = np.asarray(radius, dtype='f4').reshape(-1)
        if rad.size == 1:
            rad = np.full(n, rad[0], dtype='f4')
        elif rad.size != n:
            raise ValueError("radius must be scalar or array of length n")
        self._check_finite_array(rad, "radius")
        if np.any(rad < 0):
            raise ValueError("radius must be >= 0")

        v0 = np.ascontiguousarray(verts[:, 0, :])
        v1 = np.ascontiguousarray(verts[:, 1, :])
        v2 = np.ascontiguousarray(verts[:, 2, :])

        i = 0
        while i < n:
            chunk = min(self.max_rects, n - i)
            inst = self.rtri_instance_data[:chunk]
            _numba_pack_rtri_instances(
                v0[i:i+chunk], v1[i:i+chunk], v2[i:i+chunk],
                rad[i:i+chunk], color_bits[i:i+chunk], inst,
            )
            self.rtri_instance_vbo.orphan(); self.rtri_instance_vbo.write(memoryview(inst))
            self.rtri_vao.render(instances=chunk)
            i += chunk

    def DrawRoundedTriangle(self, x1, y1, x2, y2, x3, y3, radius,
                            color=(255, 255, 255, 255),
                            rotation=0.0, alpha=255):
        """Versione 'immediate' di DrawRoundedTrianglesBatch. La rotazione
        ruota i 3 vertici attorno al centroide sulla CPU (identico a
        DrawTriangle), poi accumula l'istanza in un buffer CPU pre-allocato
        (stessa tecnica zero-alloc di DrawTriangle/self._np_batch_buffer)
        invece di fare Flush + upload VBO + draw call GPU per OGNI singola
        chiamata. Riusa direttamente il kernel scalare Numba
        _rtri_shrink_and_aabb (lo stesso usato da DrawRoundedTrianglesBatch,
        solo senza il giro per gli array paralleli): bordo e AA identici
        alla versione batch."""
        self._check_finite(x1, y1, x2, y2, x3, y3, radius, rotation,
                           names=("x1", "y1", "x2", "y2", "x3", "y3",
                                  "radius", "rotation"))
        if radius < 0:
            raise ValueError(f"radius must be >= 0 (got {radius})")
        if rotation != 0.0:
            cx = (x1 + x2 + x3) / 3.0; cy = (y1 + y2 + y3) / 3.0
            cs, sn = self._cos_sin_deg(rotation)
            dx1 = x1 - cx; dy1 = y1 - cy
            dx2 = x2 - cx; dy2 = y2 - cy
            dx3 = x3 - cx; dy3 = y3 - cy
            x1 = cx + dx1*cs - dy1*sn; y1 = cy + dx1*sn + dy1*cs
            x2 = cx + dx2*cs - dy2*sn; y2 = cy + dx2*sn + dy2*cs
            x3 = cx + dx3*cs - dy3*sn; y3 = cy + dx3*sn + dy3*cs

        self._use_batch('rtri')
        if self.rtri_count >= self.max_rects:
            self._flush_rtri()

        r, g, b, a = self._parse_color(color, alpha)
        (sax, say, sbx, sby, scx, scy, r_eff,
         mnx, mny, mxx, mxy) = _rtri_shrink_and_aabb(x1, y1, x2, y2, x3, y3, radius)

        row = self.rtri_instance_data[self.rtri_count]
        row[0]  = sax; row[1]  = say
        row[2]  = sbx; row[3]  = sby
        row[4]  = scx; row[5]  = scy
        row[6]  = r_eff
        row[7]  = mnx; row[8]  = mny
        row[9]  = mxx; row[10] = mxy
        row[11] = _pack_rgba_u32_scalar_as_f4(r, g, b, a)
        self.rtri_count += 1

    def DrawRoundedTrianglesOutlineBatch(self, vertices, radius, colors,
                                         thickness=1.0, alpha=255):
        """Versione 'solo bordo' di DrawRoundedTrianglesBatch. Stessa
        architettura instanced. Il bordo usa la stessa formula
        outerA*(1-innerA) di DrawRectsOutlineBatch/DrawRoundedRectsOutlineBatch
        (coerenza visiva del bordo su tutte le primitive)."""
        self._flush_pending_draws()
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
        color_bits = _pack_rgba_u32_as_f4(rgba)

        rad = np.asarray(radius, dtype='f4').reshape(-1)
        if rad.size == 1:
            rad = np.full(n, rad[0], dtype='f4')
        elif rad.size != n:
            raise ValueError("radius must be scalar or array of length n")
        self._check_finite_array(rad, "radius")
        if np.any(rad < 0):
            raise ValueError("radius must be >= 0")

        thick = np.asarray(thickness, dtype='f4').reshape(-1)
        if thick.size == 1:
            thick = np.full(n, thick[0], dtype='f4')
        elif thick.size != n:
            raise ValueError("thickness must be scalar or array of length n")
        self._check_finite_array(thick, "thickness")
        if np.any(thick <= 0):
            raise ValueError("thickness must be > 0")

        v0 = np.ascontiguousarray(verts[:, 0, :])
        v1 = np.ascontiguousarray(verts[:, 1, :])
        v2 = np.ascontiguousarray(verts[:, 2, :])

        i = 0
        while i < n:
            chunk = min(self.max_rects, n - i)
            inst = self.rtri_outline_instance_data[:chunk]
            _numba_pack_rtri_outline_instances(
                v0[i:i+chunk], v1[i:i+chunk], v2[i:i+chunk],
                rad[i:i+chunk], thick[i:i+chunk],
                color_bits[i:i+chunk], inst,
            )
            self.rtri_outline_instance_vbo.orphan(); self.rtri_outline_instance_vbo.write(memoryview(inst))
            self.rtri_outline_vao.render(instances=chunk)
            i += chunk

    def DrawRoundedTriangleOutline(self, x1, y1, x2, y2, x3, y3, radius,
                                   thickness=1.0,
                                   color=(255, 255, 255, 255),
                                   rotation=0.0, alpha=255):
        """Versione 'immediate' di DrawRoundedTrianglesOutlineBatch. Accumula
        in CPU come DrawRoundedTriangle: stessa fascia di prestazioni delle
        altre primitive non-Batch. La rotazione ruota i 3 vertici attorno
        al centroide sulla CPU."""
        self._check_finite(x1, y1, x2, y2, x3, y3, radius, thickness, rotation,
                           names=("x1", "y1", "x2", "y2", "x3", "y3",
                                  "radius", "thickness", "rotation"))
        if radius < 0:
            raise ValueError(f"radius must be >= 0 (got {radius})")
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

        self._use_batch('rtri_outline')
        if self.rtri_outline_count >= self.max_rects:
            self._flush_rtri_outline()

        r, g, b, a = self._parse_color(color, alpha)
        (sax, say, sbx, sby, scx, scy, r_eff,
         mnx, mny, mxx, mxy) = _rtri_shrink_and_aabb(x1, y1, x2, y2, x3, y3, radius)

        row = self.rtri_outline_instance_data[self.rtri_outline_count]
        row[0]  = sax; row[1]  = say
        row[2]  = sbx; row[3]  = sby
        row[4]  = scx; row[5]  = scy
        row[6]  = r_eff
        row[7]  = mnx; row[8]  = mny
        row[9]  = mxx; row[10] = mxy
        row[11] = thickness
        row[12] = _pack_rgba_u32_scalar_as_f4(r, g, b, a)
        self.rtri_outline_count += 1

    def DrawEllipsesBatch(self, centers, radii, colors, alpha=255, rotation=0.0):
        """
        Path GPU instanced (SDF nel fragment shader): disegna un'ellisse
        matematicamente esatta, NON una mesh poligonale — a differenza di
        DrawEllipse/DrawCircle (path CPU) non esiste alcun concetto di
        `segments` qui, quindi il parametro non e' esposto (rimosso: prima
        veniva accettato solo per compatibilita' di firma e ignorato con un
        warning, il che era piu' fonte di confusione che di comodita').
        """
        self._flush_pending_draws()
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
        color_bits = _pack_rgba_u32_as_f4(rgba)

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
                color_bits[i:i+chunk], inst
            )
            self.ellipse_instance_vbo.orphan(); self.ellipse_instance_vbo.write(memoryview(inst))
            self.ellipse_vao.render(mode=moderngl.TRIANGLES, instances=chunk)
            i += chunk

    def DrawCirclesBatch(self, centers, radius, colors=(255,255,255,255),
                         alpha=255):
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
        self.DrawEllipsesBatch(centers, radii, colors, alpha=alpha, rotation=0.0)

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
        self._flush_pending_draws()
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
        color_bits = _pack_rgba_u32_as_f4(rgba)

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
                color_bits[i:i+chunk], inst
            )
            self.ellipse_outline_instance_vbo.orphan(); self.ellipse_outline_instance_vbo.write(memoryview(inst))
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
        self.rrect_count = 0
        self.rrect_outline_count = 0
        self.rtri_count = 0
        self.rtri_outline_count = 0
        self._active_kind = None


    def RefreshTextures(self):
        if self.tex_rect_count == 0 or self.current_texture is None:
            return

        self.current_texture.use(location=0)
        self.tex_prog["u_tex"].value = 0

        view = self._np_tex_buffer[:self.tex_rect_count]
        self.tex_vbo.orphan(); self.tex_vbo.write(memoryview(view))
        self.tex_vao.render(vertices=self.tex_rect_count * 6)
        self.tex_rect_count = 0

    def Flush(self):
        if self.rect_count == 0:
            return

        view = self._np_batch_buffer[:self.rect_count]
        self.rect_vbo.orphan(); self.rect_vbo.write(memoryview(view))
        self.rect_vao.render(vertices=self.rect_count * 6)
        self.rect_count = 0

    def _flush_rrect(self):
        """Scarica l'accumulo CPU di DrawRoundedRect verso la GPU: UNA sola
        draw call instanced per N rettangoli arrotondati accumulati, invece
        di una draw call per ognuno (stessa tecnica di Flush())."""
        if self.rrect_count == 0:
            return
        inst = self.rrect_instance_data[:self.rrect_count]
        self.rrect_instance_vbo.orphan(); self.rrect_instance_vbo.write(memoryview(inst))
        self.rrect_vao.render(instances=self.rrect_count)
        self.rrect_count = 0

    def _flush_rrect_outline(self):
        """Equivalente di _flush_rrect() per DrawRoundedRectOutline."""
        if self.rrect_outline_count == 0:
            return
        inst = self.rrect_outline_instance_data[:self.rrect_outline_count]
        self.rrect_outline_instance_vbo.orphan(); self.rrect_outline_instance_vbo.write(memoryview(inst))
        self.rrect_outline_vao.render(instances=self.rrect_outline_count)
        self.rrect_outline_count = 0

    def _flush_rtri(self):
        """Equivalente di _flush_rrect() per DrawRoundedTriangle."""
        if self.rtri_count == 0:
            return
        inst = self.rtri_instance_data[:self.rtri_count]
        self.rtri_instance_vbo.orphan(); self.rtri_instance_vbo.write(memoryview(inst))
        self.rtri_vao.render(instances=self.rtri_count)
        self.rtri_count = 0

    def _flush_rtri_outline(self):
        """Equivalente di _flush_rrect() per DrawRoundedTriangleOutline."""
        if self.rtri_outline_count == 0:
            return
        inst = self.rtri_outline_instance_data[:self.rtri_outline_count]
        self.rtri_outline_instance_vbo.orphan(); self.rtri_outline_instance_vbo.write(memoryview(inst))
        self.rtri_outline_vao.render(instances=self.rtri_outline_count)
        self.rtri_outline_count = 0

    def FlushRounded(self):
        """Scarica tutti e 4 gli accumulatori CPU delle forme arrotondate
        (rrect, rrect outline, rtri, rtri outline) verso la GPU. Puo' essere
        chiamato esplicitamente, ma normalmente ci pensano FlushAll() e il
        cambio di 'tipo' di disegno (_use_batch) a chiamarlo al momento
        giusto."""
        self._flush_rrect()
        self._flush_rrect_outline()
        self._flush_rtri()
        self._flush_rtri_outline()

    def _use_batch(self, kind):
        """Chiamato in testa a ogni Draw* 'immediate' che accumula in un
        buffer CPU (DrawRect/DrawLine/DrawTriangle/*Outline -> 'quad',
        DrawRoundedRect -> 'rrect', ecc.). Se il tipo di disegno cambia
        rispetto all'ultima primitiva accumulata, scarica PRIMA tutti i
        buffer pendenti sulla GPU: cosi' le draw call restano nello stesso
        ordine delle chiamate Draw* (painter's algorithm), anche se ogni
        forma arrotondata ora usa un VAO/shader separato dal quad piatto."""
        if self._active_kind is not None and self._active_kind != kind:
            self._flush_pending_draws()
        self._active_kind = kind

    def _flush_pending_draws(self):
        # BUG FIX (painter's algorithm): grazie al meccanismo _use_batch,
        # a ogni istante al piu' UNA di queste famiglie ha dati accumulati
        # (quella corrispondente a self._active_kind). L'ordine relativo
        # delle chiamate qui sotto non altera quindi il painter's algorithm:
        # gli altri contatori sono sempre 0. Le texture ora sono incluse.
        if self.tex_rect_count > 0:
            self.RefreshTextures()
        if self.rect_count > 0:
            self.Flush()
        self.FlushRounded()
        self._active_kind = None

    def FlushAll(self):
        """3. COMPITO: Disegna tutto quello che e' ancora accumulato.
        Grazie a _use_batch, a fine frame solo una famiglia ha dati
        pendenti: quale sia dipende dall'ultima chiamata Draw*, quindi
        l'ordine di questi flush non e' significativo."""
        if self.tex_rect_count > 0:
            self.RefreshTextures()
        if self.rect_count > 0:
            self.Flush()
        self.FlushRounded()
        self._active_kind = None

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
        if hasattr(self, "rrect_prog"):
            self.rrect_prog["u_resolution"].value = (width, height)
        if hasattr(self, "rrect_outline_prog"):
            self.rrect_outline_prog["u_resolution"].value = (width, height)
        if hasattr(self, "rtri_prog"):
            self.rtri_prog["u_resolution"].value = (width, height)
        if hasattr(self, "rtri_outline_prog"):
            self.rtri_outline_prog["u_resolution"].value = (width, height)

    def _release_draw(self):
        # Pulizia cache Python (questi hanno .clear())
        self._ellipse_cache.clear()
        self._bezier_cache.clear()

        if hasattr(self, "_texture_cache"):
            for tex_obj, _, _, _ in self._texture_cache.values():
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
        if a <= self._EPSILON:
            return False
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

    def CollidePointRotatedRectBatch(self, px_arr, py_arr, rx, ry, rw, rh, rotation=0.0):
        """Batch Point-vs-RotatedRect. `rotation` puo' essere UNO scalare
        (stessa rotazione per tutte le N rect, comportamento originale) OPPURE
        un array (N,) con una rotazione diversa per ogni rect.

        FIX COERENZA/PERF: la versione precedente faceva `float(rotation)`,
        quindi accettava SOLO uno scalare — passare un array (anche di soli
        zeri) sollevava TypeError ("only 0-dimensional arrays can be
        converted..."). Ora, come in CollidePointEllipseBatch, la
        trigonometria viene saltata del tutto se tutte le rotazioni sono 0
        (caso comunissimo: batch di rect assi-allineati), quindi nessuna
        regressione di prestazioni sul percorso già esistente."""
        px = np.asarray(px_arr, dtype='f4'); py = np.asarray(py_arr, dtype='f4')
        rx = np.asarray(rx, dtype='f4'); ry = np.asarray(ry, dtype='f4')
        rw = np.asarray(rw, dtype='f4'); rh = np.asarray(rh, dtype='f4')
        cx = rx + rw * 0.5; cy = ry + rh * 0.5
        dx = px - cx; dy = py - cy
        rot = np.asarray(rotation, dtype='f4')
        if np.any(rot != 0.0):
            ang = -rot * np.float32(_DEG2RAD_CONST)
            cs = np.cos(ang); sn = np.sin(ang)
            lx = dx * cs - dy * sn
            ly = dx * sn + dy * cs
        else:
            lx, ly = dx, dy
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

    def CollidePointTexture(self, px, py, name, tx, ty, tw=None, th=None,
                            rotation=0.0, flip_x=False, flip_y=False,
                            alpha_threshold=1):
        """Point-vs-Texture PIXEL-PERFECT.

        Prima uno scarto economico contro il rotated-rect di bounding
        (stesso costo di CollidePointRotatedRect: 1 mul/add per punto), poi
        SOLO se il punto e' dentro il rettangolo si fa un singolo lookup nel
        canale alpha tenuto in CPU (popolato da LoadTexture) — zero readback
        dalla GPU, zero riapertura del file su disco ad ogni hit-test.

        Parametri
        ---------
        name : str
            Nome della texture (quello passato a LoadTexture/DrawTexture).
        tx, ty, tw, th : float
            Posizione e dimensioni A SCHERMO con cui la texture e' stata
            disegnata (tw/th default = dimensione naturale, come DrawTexture).
        rotation : float
            Gradi, stessa convenzione di CollidePointRotatedRect/DrawTexture.
        alpha_threshold : int
            Valore alpha (0-255) sopra il quale il pixel conta come "pieno".
            Default 1 = qualunque pixel non totalmente trasparente.

        Ritorna False (nessuna eccezione) se `name` non e' in cache: comodo
        per chiamare la funzione senza dover controllare prima l'esistenza
        della texture.
        """
        entry = self._texture_cache.get(name)
        if entry is None:
            return False
        _tex, orig_w, orig_h, alpha_ch = entry

        w = orig_w if tw is None else tw
        h = orig_h if th is None else th
        if w <= 0.0 or h <= 0.0:
            return False

        cx = tx + w * 0.5
        cy = ty + h * 0.5
        ang = -float(rotation) * _DEG2RAD_CONST
        cs = math.cos(ang); sn = math.sin(ang)
        dx = px - cx; dy = py - cy
        local_x = cx + dx * cs - dy * sn
        local_y = cy + dx * sn + dy * cs

        # Early-out: identico a CollidePointRotatedRect, ma teniamo local_x/y
        # calcolati qui per non rifare cos/sin una seconda volta sotto.
        if not self.PointInRect(local_x, local_y, tx, ty, w, h):
            return False

        u = (local_x - tx) / w
        v = (local_y - ty) / h
        if flip_x: u = 1.0 - u
        if flip_y: v = 1.0 - v

        col = int(u * orig_w)
        row = int(v * orig_h)
        if col < 0 or col >= orig_w or row < 0 or row >= orig_h:
            return False

        return bool(alpha_ch[row, col] >= alpha_threshold)

    def CollidePointTextureBatch(self, px_arr, py_arr, name, x_arr, y_arr,
                                 w_arr=None, h_arr=None, rotation_arr=0.0,
                                 flip_x=False, flip_y=False,
                                 alpha_threshold=1):
        """Batch PIXEL-PERFECT: un punto (es. il mouse) contro N istanze
        della STESSA texture (tile picking, selezione unita' in un RTS,
        hit-test su uno sciame di sprite condiviso, ecc.).

        px_arr/py_arr possono essere scalari (es. mouse.x, mouse.y) o array
        della stessa lunghezza di x_arr/y_arr — broadcasting NumPy standard.

        Strategia in 2 stadi, stesso spirito di CollidePointTexture: lo
        stadio 1 (rotated-rect, vettoriale) scarta in un colpo solo la
        stragrande maggioranza delle istanze; lo stadio 2 (lookup alpha)
        gira SOLO sui sopravvissuti, non su tutti gli N.
        """
        entry = self._texture_cache.get(name)
        x_arr = np.asarray(x_arr, dtype='f4')
        n = x_arr.shape[0]
        if entry is None:
            return np.zeros(n, dtype=bool)
        _tex, orig_w, orig_h, alpha_ch = entry

        y_arr = np.asarray(y_arr, dtype='f4')
        w_arr = np.full(n, orig_w, dtype='f4') if w_arr is None else np.asarray(w_arr, dtype='f4')
        h_arr = np.full(n, orig_h, dtype='f4') if h_arr is None else np.asarray(h_arr, dtype='f4')
        rot_arr = np.broadcast_to(np.asarray(rotation_arr, dtype='f4'), (n,))

        # Stadio 1: early-out vettoriale (stesso costo di CollidePointRotatedRectBatch)
        hits = self.CollidePointRotatedRectBatch(px_arr, py_arr, x_arr, y_arr, w_arr, h_arr, rot_arr)
        out = np.zeros(n, dtype=bool)
        idx = np.nonzero(hits)[0]
        if idx.size == 0:
            return out

        px = np.broadcast_to(np.asarray(px_arr, dtype='f4'), (n,))[idx]
        py = np.broadcast_to(np.asarray(py_arr, dtype='f4'), (n,))[idx]
        xi = x_arr[idx]; yi = y_arr[idx]; wi = w_arr[idx]; hi = h_arr[idx]

        cx = xi + wi * 0.5; cy = yi + hi * 0.5
        ang = -rot_arr[idx] * np.float32(_DEG2RAD_CONST)
        cs = np.cos(ang); sn = np.sin(ang)
        ddx = px - cx; ddy = py - cy
        local_x = cx + ddx * cs - ddy * sn
        local_y = cy + ddx * sn + ddy * cs

        u = (local_x - xi) / wi
        v = (local_y - yi) / hi
        if flip_x: u = 1.0 - u
        if flip_y: v = 1.0 - v

        col = (u * orig_w).astype(np.int32)
        row = (v * orig_h).astype(np.int32)
        valid = (col >= 0) & (col < orig_w) & (row >= 0) & (row < orig_h)

        sub_idx = idx[valid]
        out[sub_idx] = alpha_ch[row[valid], col[valid]] >= alpha_threshold
        return out

    def CollidePointRoundedRect(self, px, py, x, y, w, h, radius, rotation=0.0):
        """Point-vs-RoundedRect diretto, senza creare oggetti temporanei."""
        return _point_in_rounded_rect_direct(px, py, x, y, w, h, radius, rotation)

    def CollidePointRoundedTriangle(self, px, py, x1, y1, x2, y2, x3, y3,
                                    radius, rotation=0.0):
        """Point-vs-RoundedTriangle coerente con DrawRoundedTriangle."""
        return _point_in_rounded_triangle_direct(px, py, x1, y1, x2, y2, x3, y3,
                                                 radius, rotation)

    def CollidePointPolygon(self, px, py, points):
        """Point-vs-Polygon diretto; funziona anche con poligoni concavi."""
        return _point_in_polygon(px, py, points)

    def CollidePointText(self, px, py, text, x, y, font="arial", size=24, rotation=0.0):
        """Point-vs-Text usando lo stesso layout di DrawText."""
        self._ensure_text_system()
        laid = self._layout_string(str(text), font, int(size), reuse=True)
        if laid is None:
            return False
        gx, gy, gw, gh, _guv = laid
        min_x = float(gx.min()); max_x = float((gx + gw).max())
        min_y = float(gy.min()); max_y = float((gy + gh).max())
        if rotation != 0.0:
            ang = -float(rotation) * _DEG2RAD_CONST
            cs = math.cos(ang); sn = math.sin(ang)
            dx = px - x; dy = py - y
            lx = dx * cs - dy * sn
            ly = dx * sn + dy * cs
        else:
            lx = px - x; ly = py - y
        return min_x <= lx <= max_x and min_y <= ly <= max_y

    def CollidePointRoundedRectBatch(self, px_arr, py_arr, x_arr, y_arr, w_arr, h_arr,
                                     radius_arr, rotation_arr=0.0):
        px = np.asarray(px_arr, dtype='f4'); py = np.asarray(py_arr, dtype='f4')
        x = np.asarray(x_arr, dtype='f4'); y = np.asarray(y_arr, dtype='f4')
        w = np.asarray(w_arr, dtype='f4'); h = np.asarray(h_arr, dtype='f4')
        r = np.asarray(radius_arr, dtype='f4')
        rot = np.asarray(rotation_arr, dtype='f4')
        cx = x + w * 0.5; cy = y + h * 0.5
        dx = px - cx; dy = py - cy
        if np.any(rot != 0.0):
            ang = -rot * np.float32(_DEG2RAD_CONST)
            cs = np.cos(ang); sn = np.sin(ang)
            lx = dx * cs - dy * sn
            ly = dx * sn + dy * cs
        else:
            lx, ly = dx, dy
        hw = w * 0.5; hh = h * 0.5
        rr = np.minimum(np.maximum(r, 0.0), np.minimum(hw, hh))
        qx = np.abs(lx) - hw + rr
        qy = np.abs(ly) - hh + rr
        d = np.minimum(np.maximum(qx, qy), 0.0) + np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) - rr
        return d <= 0.0

    def CollidePointRoundedTriangleBatch(self, px_arr, py_arr, vertices, radius):
        px = np.asarray(px_arr, dtype='f8')
        py = np.asarray(py_arr, dtype='f8')
        verts = np.asarray(vertices, dtype='f8')
        if verts.ndim == 2:
            n = verts.shape[0] // 3
            verts = verts.reshape(n, 3, 2)
        else:
            n = verts.shape[0]
        px = np.broadcast_to(px, (n,))
        py = np.broadcast_to(py, (n,))
        rad = np.asarray(radius, dtype='f8').reshape(-1)
        if rad.size == 1:
            rad = np.full(n, rad[0], dtype='f8')
        out = np.zeros(n, dtype=bool)
        for i in range(n):
            (x1, y1), (x2, y2), (x3, y3) = verts[i]
            out[i] = _point_in_rounded_triangle_direct(float(px[i]), float(py[i]),
                                                       float(x1), float(y1),
                                                       float(x2), float(y2),
                                                       float(x3), float(y3),
                                                       float(rad[i]), 0.0)
        return out
    # ================================================================
    # PE_COLLISION — API pubblica (integrata da PE_COLLISION.py)
    # ================================================================
    # Nel modulo originale queste erano funzioni sciolte con un singleton
    # globale (_RUNTIME) + bind_draw()/bind_window() per registrare quale
    # istanza DRAW usare. Qui `self' e' sempre il contesto draw/mouse: ogni
    # istanza DRAW/WINDOW tiene il proprio stato mouse, senza registrazioni
    # globali e senza rischio che piu' finestre si "rubino" lo stato.
    #
    # Le forme (Point, Rect, RotRect, Circle, Ellipse, Line, Triangle,
    # RoundedRect, Polygon, TextureCollider) sono agganciate come attributi
    # di classe subito dopo la definizione di DRAW (vedi fondo del file):
    # `draw.Rect(...)`, `draw.Circle(...)` ecc. funzionano su qualunque
    # istanza, oltre a restare importabili a livello di modulo.

    def _ensure_collision_state(self):
        """Inizializza lo stato mouse per-istanza al primo utilizzo (lazy,
        cosi' DRAW usato senza WINDOW/mouse non paga nulla)."""
        if getattr(self, "_col_ready", False):
            return
        self._mouse_x = 0.0; self._mouse_y = 0.0
        self._mouse_prev_x = 0.0; self._mouse_prev_y = 0.0
        self._mouse_buttons_down = set()
        self._mouse_buttons_pressed = set()
        self._mouse_buttons_released = set()
        self._mouse_wheel_x = 0.0; self._mouse_wheel_y = 0.0
        self._mouse_wheel_event = False
        self._col_ready = True

    def UpdateMouseState(self, mouse_x, mouse_y, events):
        """Da chiamare UNA VOLTA per frame (tipicamente in Loop(), PRIMA di
        update()/draw()) per alimentare MouseOver/MousePressed/
        MouseReleased/MouseClicked/MouseHeld/MouseDragging/MouseWheelOn.

        mouse_x, mouse_y : posizione corrente del cursore (es. da
                            SDL_GetMouseState), usata come fallback quando
                            nel frame non arrivano eventi mouse.
        events            : lista di PE_Event del frame corrente (PE_WINDOW
                            la costruisce gia' in Loop()).
        """
        self._ensure_collision_state()
        self._mouse_prev_x = self._mouse_x
        self._mouse_prev_y = self._mouse_y
        self._mouse_x = float(mouse_x)
        self._mouse_y = float(mouse_y)
        self._mouse_buttons_pressed = set()
        self._mouse_buttons_released = set()
        self._mouse_wheel_x = 0.0
        self._mouse_wheel_y = 0.0
        self._mouse_wheel_event = False

        for e in events:
            et = getattr(e, "type", None)
            if et == PE_MOUSEBUTTONDOWN:
                b = e.button
                self._mouse_buttons_down.add(b)
                self._mouse_buttons_pressed.add(b)
                self._mouse_x = float(e.x); self._mouse_y = float(e.y)
            elif et == PE_MOUSEBUTTONUP:
                b = e.button
                self._mouse_buttons_down.discard(b)
                self._mouse_buttons_released.add(b)
                self._mouse_x = float(e.x); self._mouse_y = float(e.y)
            elif et in (PE_MOUSEMOTION, PE_MOUSEDRAG):
                self._mouse_x = float(e.x); self._mouse_y = float(e.y)
            elif et == PE_MOUSEWHEEL:
                self._mouse_wheel_x = float(e.wheel_x)
                self._mouse_wheel_y = float(e.wheel_y)
                self._mouse_wheel_event = True
                self._mouse_x = float(e.x); self._mouse_y = float(e.y)

    def MousePosition(self):
        """Ritorna (x, y) posizione mouse corrente in coordinate finestra."""
        self._ensure_collision_state()
        return self._mouse_x, self._mouse_y

    def _mouse_point(self):
        self._ensure_collision_state()
        return Point(self._mouse_x, self._mouse_y)

    def _mouse_xy(self):
        self._ensure_collision_state()
        return self._mouse_x, self._mouse_y

    def _direct_shape_hit(self, px, py, *shape_args):
        """Hit-test punto->forma senza creare Point/Rect/Circle temporanei.

        Forme dirette supportate:
            MouseClicked(x, y, w, h)                         # rect default
            MouseClicked((x, y, w, h))                       # rect tuple/list
            MouseClicked("circle", cx, cy, r)
            MouseClicked("ellipse", cx, cy, rx, ry, rotation=0)
            MouseClicked("line", x1, y1, x2, y2, thickness=1, rotation=0)
            MouseClicked("triangle", x1, y1, x2, y2, x3, y3)
            MouseClicked("rounded_rect", x, y, w, h, radius, rotation=0)
            MouseClicked("rounded_triangle", x1, y1, x2, y2, x3, y3, radius, rotation=0)
            MouseClicked("polygon", [(x, y), ...])
            MouseClicked("texture", name, x, y, w=None, h=None, rotation=0, flip_x=False, flip_y=False, alpha_threshold=1)
            MouseClicked("text", text, x, y, font="arial", size=24, rotation=0)
        """
        if len(shape_args) == 1:
            spec = shape_args[0]
            if hasattr(spec, "_t"):
                return _dispatch(Point(px, py), spec, self)
            if isinstance(spec, (tuple, list)):
                if spec and isinstance(spec[0], str):
                    return self._direct_shape_hit(px, py, *spec)
                if len(spec) == 2:
                    sx, sy = spec
                    return abs(px - sx) <= _EPS and abs(py - sy) <= _EPS
                if len(spec) == 3:
                    cx, cy, r = spec
                    return self.CollidePointCircle(px, py, cx, cy, r)
                if len(spec) == 4:
                    x, y, w, h = spec
                    return self.CollidePointRect(px, py, x, y, w, h)
                if len(spec) == 5:
                    x, y, w, h, rotation = spec
                    return self.CollidePointRotatedRect(px, py, x, y, w, h, rotation)
                if len(spec) == 6:
                    x1, y1, x2, y2, x3, y3 = spec
                    return self.CollidePointTriangle(px, py, x1, y1, x2, y2, x3, y3)
            raise TypeError("Forma non valida: usa coordinate dirette, una tupla/lista, oppure ('tipo', ...).")

        if not shape_args:
            raise TypeError("Manca la forma da testare.")

        first = shape_args[0]
        if isinstance(first, str):
            kind = first.lower().replace("-", "_").replace(" ", "")
            args = shape_args[1:]
            if kind in ("rect", "rectangle", "rettangolo"):
                if len(args) == 4:
                    return self.CollidePointRect(px, py, *args)
                if len(args) == 5:
                    return self.CollidePointRotatedRect(px, py, *args)
            elif kind in ("rotrect", "rotatedrect", "rotated_rect"):
                if len(args) == 5:
                    return self.CollidePointRotatedRect(px, py, *args)
            elif kind in ("circle", "cerchio"):
                if len(args) == 3:
                    return self.CollidePointCircle(px, py, *args)
            elif kind in ("ellipse", "ellisse"):
                if len(args) == 4:
                    return self.CollidePointEllipse(px, py, *args)
                if len(args) == 5:
                    return self.CollidePointEllipse(px, py, *args)
            elif kind in ("line", "linea"):
                if len(args) == 4:
                    return self.PointInLine(px, py, *args)
                if len(args) == 5:
                    x1, y1, x2, y2, thickness = args
                    return self.PointInLine(px, py, x1, y1, x2, y2, thickness=thickness)
                if len(args) == 6:
                    x1, y1, x2, y2, thickness, rotation = args
                    return self.PointInLine(px, py, x1, y1, x2, y2, thickness=thickness, rotation=rotation)
            elif kind in ("triangle", "triangolo"):
                if len(args) == 6:
                    return self.CollidePointTriangle(px, py, *args)
            elif kind in ("roundedrect", "rounded_rect", "rrect", "rectrounded"):
                if len(args) == 5:
                    return self.CollidePointRoundedRect(px, py, *args)
                if len(args) == 6:
                    return self.CollidePointRoundedRect(px, py, *args)
            elif kind in ("roundedtriangle", "rounded_triangle", "rtri"):
                if len(args) == 7:
                    return self.CollidePointRoundedTriangle(px, py, *args)
                if len(args) == 8:
                    return self.CollidePointRoundedTriangle(px, py, *args)
            elif kind in ("polygon", "poly", "poligono"):
                if len(args) == 1:
                    return self.CollidePointPolygon(px, py, args[0])
            elif kind in ("texture", "sprite", "image"):
                if 3 <= len(args) <= 9:
                    return self.CollidePointTexture(px, py, *args)
            elif kind in ("text", "testo"):
                if 3 <= len(args) <= 6:
                    return self.CollidePointText(px, py, *args)
            raise TypeError(f"Parametri non validi per forma {first!r}.")

        if len(shape_args) == 2:
            sx, sy = shape_args
            return abs(px - sx) <= _EPS and abs(py - sy) <= _EPS
        if len(shape_args) == 3:
            return self.CollidePointCircle(px, py, *shape_args)
        if len(shape_args) == 4:
            return self.CollidePointRect(px, py, *shape_args)
        if len(shape_args) == 5:
            return self.CollidePointRotatedRect(px, py, *shape_args)
        if len(shape_args) == 6:
            return self.CollidePointTriangle(px, py, *shape_args)
        raise TypeError("Coordinate dirette non riconosciute: usa ('tipo', ...) per forme ambigue/avanzate.")

    def _outline_direct_shape(self, *shape_args, color=(0, 255, 0, 255), thickness=2.0):
        if len(shape_args) == 1 and hasattr(shape_args[0], "_t"):
            _draw_shape_outline(self, shape_args[0], color, thickness)
            return
        if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)):
            spec = shape_args[0]
            if spec and isinstance(spec[0], str):
                self._outline_direct_shape(*spec, color=color, thickness=thickness)
                return
            if len(spec) == 4:
                self.DrawRectOutline(*spec, thickness=thickness, color=color)
                return
        if shape_args and isinstance(shape_args[0], str):
            kind = shape_args[0].lower().replace("-", "_").replace(" ", "")
            args = shape_args[1:]
            if kind in ("rect", "rectangle", "rettangolo") and len(args) in (4, 5):
                if len(args) == 4:
                    self.DrawRectOutline(*args, thickness=thickness, color=color)
                else:
                    x, y, w, h, rotation = args
                    self.DrawRectOutline(x, y, w, h, thickness=thickness, color=color, rotation=rotation)
            elif kind in ("circle", "cerchio") and len(args) == 3:
                self.DrawCircleOutline(*args, thickness=thickness, color=color)
            elif kind in ("ellipse", "ellisse") and len(args) in (4, 5):
                if len(args) == 4:
                    self.DrawEllipseOutline(*args, thickness=thickness, color=color)
                else:
                    cx, cy, rx, ry, rotation = args
                    self.DrawEllipseOutline(cx, cy, rx, ry, thickness=thickness, color=color, rotation=rotation)
            elif kind in ("line", "linea") and len(args) >= 4:
                self.DrawLine(args[0], args[1], args[2], args[3], thickness=max(thickness, args[4] if len(args) >= 5 else thickness), color=color)
            elif kind in ("triangle", "triangolo") and len(args) == 6:
                self.DrawTriangleOutline(*args, thickness=thickness, color=color)
            elif kind in ("roundedrect", "rounded_rect", "rrect", "rectrounded") and len(args) in (5, 6):
                if len(args) == 5:
                    self.DrawRoundedRectOutline(*args, thickness=thickness, color=color)
                else:
                    x, y, w, h, radius, rotation = args
                    self.DrawRoundedRectOutline(x, y, w, h, radius, thickness=thickness, color=color, rotation=rotation)
            elif kind in ("roundedtriangle", "rounded_triangle", "rtri") and len(args) in (7, 8):
                if len(args) == 7:
                    self.DrawRoundedTriangleOutline(*args, thickness=thickness, color=color)
                else:
                    x1, y1, x2, y2, x3, y3, radius, rotation = args
                    self.DrawRoundedTriangleOutline(x1, y1, x2, y2, x3, y3, radius, thickness=thickness, color=color, rotation=rotation)
            return
        if len(shape_args) == 4:
            self.DrawRectOutline(*shape_args, thickness=thickness, color=color)
        elif len(shape_args) == 3:
            self.DrawCircleOutline(*shape_args, thickness=thickness, color=color)
        elif len(shape_args) == 6:
            self.DrawTriangleOutline(*shape_args, thickness=thickness, color=color)

    def _shape_from_spec(self, spec):
        if hasattr(spec, "_t"):
            return spec
        if not isinstance(spec, (tuple, list)):
            raise TypeError("CheckCollision richiede forme DRAW oppure tuple ('tipo', ...).")
        if spec and isinstance(spec[0], str):
            kind = spec[0].lower().replace("-", "_").replace(" ", "")
            args = spec[1:]
            if kind in ("rect", "rectangle", "rettangolo") and len(args) == 4:
                return Rect(*args)
            if kind in ("rotrect", "rotatedrect", "rotated_rect") and len(args) == 5:
                return RotRect(*args)
            if kind in ("circle", "cerchio") and len(args) == 3:
                return Circle(*args)
            if kind in ("ellipse", "ellisse") and len(args) in (4, 5):
                return Ellipse(*args)
            if kind in ("line", "linea") and len(args) in (4, 5):
                return Line(*args)
            if kind in ("triangle", "triangolo") and len(args) == 6:
                return Triangle(*args)
            if kind in ("roundedrect", "rounded_rect", "rrect", "rectrounded") and len(args) == 5:
                return RoundedRect(*args)
            if kind in ("polygon", "poly", "poligono") and len(args) == 1:
                return Polygon(args[0])
            if kind in ("texture", "sprite", "image") and 4 <= len(args) <= 12:
                return TextureCollider(*args)
            raise TypeError(f"Specifica forma non valida: {spec!r}")
        if len(spec) == 2:
            return Point(*spec)
        if len(spec) == 3:
            return Circle(*spec)
        if len(spec) == 4:
            return Rect(*spec)
        if len(spec) == 5:
            return RotRect(*spec)
        if len(spec) == 6:
            return Triangle(*spec)
        raise TypeError(f"Specifica forma non riconosciuta: {spec!r}")

    def _mouse_args(self, first, extra, default_button, show, color, thickness):
        button = default_button
        legacy_show = show
        legacy_color = color
        legacy_thickness = thickness
        shape_args = (first, *extra)
        if extra and (hasattr(first, "_t") or isinstance(first, (tuple, list))):
            rest = list(extra)
            if rest and isinstance(rest[0], int):
                button = rest.pop(0)
            if rest and isinstance(rest[0], bool):
                legacy_show = rest.pop(0)
            if rest and isinstance(rest[0], (tuple, list)):
                legacy_color = rest.pop(0)
            if rest and isinstance(rest[0], (int, float)):
                legacy_thickness = rest.pop(0)
            # Compatibilità con la vecchia firma:
            # MouseClicked(shape, button, show, color, thickness).
            # Quando la forma è già un oggetto/tupla, gli argomenti extra non
            # sono coordinate della forma ma vecchie opzioni posizionali.
            shape_args = (first,)
        return shape_args, button, legacy_show, legacy_color, legacy_thickness

    def _shape_only_args(self, first, extra, show, color, thickness):
        legacy_show = show
        legacy_color = color
        legacy_thickness = thickness
        shape_args = (first, *extra)
        if extra and (hasattr(first, "_t") or isinstance(first, (tuple, list))):
            rest = list(extra)
            if rest and isinstance(rest[0], bool):
                legacy_show = rest.pop(0)
            if rest and isinstance(rest[0], (tuple, list)):
                legacy_color = rest.pop(0)
            if rest and isinstance(rest[0], (int, float)):
                legacy_thickness = rest.pop(0)
            shape_args = (first,)
        return shape_args, legacy_show, legacy_color, legacy_thickness

    def CheckCollision(self, a, b, show=False, color=(0, 255, 0, 255),
                       thickness=2.0):
        """Rileva collisione fra due forme qualsiasi (Point, Rect, RotRect,
        Circle, Ellipse, Line, Triangle, RoundedRect, Polygon,
        TextureCollider). Se show=True disegna il contorno di entrambe.

        Esempio
        -------
            if draw.CheckCollision(player_rect, enemy_circle):
                player.hp -= 1
        """
        a = self._shape_from_spec(a)
        b = self._shape_from_spec(b)
        result = _dispatch(a, b, self)
        if show:
            _draw_shape_outline(self, a, color, thickness)
            _draw_shape_outline(self, b, color, thickness)
        return result

    def MouseOver(self, shape, *shape_args, show=False, color=(0, 255, 255, 255),
                  thickness=2.0):
        """True se il cursore del mouse e' attualmente SOPRA la forma."""
        px, py = self._mouse_xy()
        args, show, color, thickness = self._shape_only_args(shape, shape_args, show, color, thickness)
        hit = self._direct_shape_hit(px, py, *args)
        if show:
            self._outline_direct_shape(*args, color=color, thickness=thickness)
        return hit

    def MousePressed(self, shape, *shape_args, button=PE_MOUSE_LEFT, show=False,
                     color=(255, 255, 0, 255), thickness=2.0):
        """True per UN SOLO frame quando `button` viene premuto MENTRE il
        mouse e' sopra `shape`. Richiede che UpdateMouseState() sia stato
        chiamato in questo frame."""
        self._ensure_collision_state()
        args, button, show, color, thickness = self._mouse_args(shape, shape_args, button, show, color, thickness)
        px, py = self._mouse_xy()
        hit = (button in self._mouse_buttons_pressed) and self._direct_shape_hit(px, py, *args)
        if show:
            self._outline_direct_shape(*args, color=color, thickness=thickness)
        return hit

    def MouseReleased(self, shape, *shape_args, button=PE_MOUSE_LEFT, show=False,
                      color=(255, 128, 0, 255), thickness=2.0):
        """True per UN SOLO frame quando `button` viene rilasciato sopra
        `shape`."""
        self._ensure_collision_state()
        args, button, show, color, thickness = self._mouse_args(shape, shape_args, button, show, color, thickness)
        px, py = self._mouse_xy()
        hit = (button in self._mouse_buttons_released) and self._direct_shape_hit(px, py, *args)
        if show:
            self._outline_direct_shape(*args, color=color, thickness=thickness)
        return hit

    def MouseClicked(self, shape, *shape_args, button=PE_MOUSE_LEFT, show=False,
                     color=(0, 255, 0, 255), thickness=2.0):
        """Alias di MouseReleased — semanticamente 'click completato sulla
        forma'."""
        return self.MouseReleased(shape, *shape_args, button=button, show=show,
                                  color=color, thickness=thickness)

    def MouseHeld(self, shape, *shape_args, button=PE_MOUSE_LEFT, show=False,
                 color=(255, 0, 255, 255), thickness=2.0):
        """True FINCHE' `button` resta premuto E il mouse resta sopra
        `shape`."""
        self._ensure_collision_state()
        args, button, show, color, thickness = self._mouse_args(shape, shape_args, button, show, color, thickness)
        px, py = self._mouse_xy()
        hit = (button in self._mouse_buttons_down) and self._direct_shape_hit(px, py, *args)
        if show:
            self._outline_direct_shape(*args, color=color, thickness=thickness)
        return hit

    def MouseDragging(self, shape, *shape_args, button=PE_MOUSE_LEFT, show=False,
                      color=(0, 128, 255, 255), thickness=2.0):
        """True mentre l'utente sta trascinando (bottone premuto + mouse in
        movimento) e il cursore e' sopra `shape`."""
        self._ensure_collision_state()
        moved = (self._mouse_x != self._mouse_prev_x) or \
                (self._mouse_y != self._mouse_prev_y)
        args, button, show, color, thickness = self._mouse_args(shape, shape_args, button, show, color, thickness)
        px, py = self._mouse_xy()
        hit = moved and (button in self._mouse_buttons_down) and self._direct_shape_hit(px, py, *args)
        if show:
            self._outline_direct_shape(*args, color=color, thickness=thickness)
        return hit

    def MouseWheelOn(self, shape, *shape_args, show=False, color=(200, 200, 0, 255),
                     thickness=2.0):
        """Ritorna (dx, dy) della rotellina se avvenuta sopra `shape` in
        questo frame, altrimenti (0, 0). Uso: `dx, dy = draw.MouseWheelOn(btn)`."""
        self._ensure_collision_state()
        args, show, color, thickness = self._shape_only_args(shape, shape_args, show, color, thickness)
        px, py = self._mouse_xy()
        if self._mouse_wheel_event and self._direct_shape_hit(px, py, *args):
            if show:
                self._outline_direct_shape(*args, color=color, thickness=thickness)
            return self._mouse_wheel_x, self._mouse_wheel_y
        return 0.0, 0.0

    # ------------------------------------------------------------------ #
    # WRAPPER "FACILI" — versioni semplificate delle funzioni *Batch.
    #
    # Le funzioni *Batch (DrawRectsBatch, DrawEllipsesBatch, ecc.) vogliono
    # array NumPy "paralleli": una lista di posizioni, una lista di
    # dimensioni, una lista di colori... costruiti a mano dal chiamante.
    # Comodo per prestazioni massime quando i dati sono già in NumPy, ma
    # scomodo per scrivere in fretta.
    #
    # Questi wrapper fanno l'opposto: si passa UNA SOLA lista di tuple (una
    # tupla per istanza), con GLI STESSI parametri, nello stesso ordine,
    # delle funzioni "immediate" equivalenti (DrawRect, DrawLine,
    # DrawEllipse, ...). Non serve importare numpy né costruire array a
    # parte: il wrapper spacchetta le tuple in liste Python parallele e
    # richiama la funzione *Batch già esistente, che internamente le
    # converte in NumPy — stessa identica pipeline GPU-instanced, stessa
    # fascia di prestazioni Batch, zero lavoro extra per chi chiama.
    #
    # I parametri opzionali in coda a ogni tupla possono essere omessi
    # (vengono riempiti con gli stessi default delle funzioni immediate):
    #
    #     draw.DrawRects([
    #         (10, 10, 50, 50),                              # bianco, alpha 255
    #         (100, 10, 30, 30, (255, 0, 0, 255)),            # rosso
    #         (200, 10, 40, 40, (0, 255, 0, 255), 128, 45.0),  # verde, alpha 128, ruotato
    #     ])
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pad_item(item, n_required, defaults):
        """Completa una tupla `item` con i valori di default mancanti in
        coda. `n_required` = numero minimo di valori posizionali che
        l'utente DEVE fornire; `defaults` = default dei rimanenti parametri
        opzionali, nello stesso ordine in cui compaiono nella firma."""
        item = tuple(item)
        if len(item) < n_required:
            raise ValueError(
                f"ogni elemento deve avere almeno {n_required} valori "
                f"posizionali, ricevuti {len(item)}: {item}"
            )
        n_optional_given = len(item) - n_required
        if n_optional_given > len(defaults):
            raise ValueError(
                "troppi valori nell'elemento (attesi al massimo "
                f"{n_required + len(defaults)}, ricevuti {len(item)}): {item}"
            )
        return item + tuple(defaults[n_optional_given:])

    def DrawRects(self, items):
        """
        Versione facile di DrawRectsBatch.

        items: iterable di tuple
            (x, y, w, h, color=(255,255,255,255), alpha=255, rotation=0.0)
        """
        items = list(items)
        if not items:
            return
        positions, sizes, colors, alphas, rotations = [], [], [], [], []
        for raw in items:
            x, y, w, h, color, alpha, rotation = self._pad_item(
                raw, 4, ((255, 255, 255, 255), 255, 0.0)
            )
            positions.append((x, y))
            sizes.append((w, h))
            colors.append(color)
            alphas.append(alpha)
            rotations.append(rotation)
        self.DrawRectsBatch(positions, sizes, colors,
                            alpha=alphas, rotation=rotations)

    def DrawRectsOutline(self, items):
        """
        Versione facile di DrawRectsOutlineBatch.

        items: iterable di tuple
            (x, y, w, h, thickness=1.0, color=(255,255,255,255),
             rotation=0.0, alpha=255)
        """
        items = list(items)
        if not items:
            return
        positions, sizes, thicknesses, colors, rotations, alphas = \
            [], [], [], [], [], []
        for raw in items:
            x, y, w, h, thickness, color, rotation, alpha = self._pad_item(
                raw, 4, (1.0, (255, 255, 255, 255), 0.0, 255)
            )
            positions.append((x, y))
            sizes.append((w, h))
            thicknesses.append(thickness)
            colors.append(color)
            rotations.append(rotation)
            alphas.append(alpha)
        self.DrawRectsOutlineBatch(positions, sizes, colors,
                                   thickness=thicknesses, alpha=alphas,
                                   rotation=rotations)

    def DrawRoundedRects(self, items):
        """
        Versione facile di DrawRoundedRectsBatch.

        items: iterable di tuple
            (x, y, w, h, radius, color=(255,255,255,255),
             rotation=0.0, alpha=255)
        """
        items = list(items)
        if not items:
            return
        positions, sizes, radii, colors, rotations, alphas = \
            [], [], [], [], [], []
        for raw in items:
            x, y, w, h, radius, color, rotation, alpha = self._pad_item(
                raw, 5, ((255, 255, 255, 255), 0.0, 255)
            )
            positions.append((x, y))
            sizes.append((w, h))
            radii.append(radius)
            colors.append(color)
            rotations.append(rotation)
            alphas.append(alpha)
        self.DrawRoundedRectsBatch(positions, sizes, radii, colors,
                                   alpha=alphas, rotation=rotations)

    def DrawRoundedRectsOutline(self, items):
        """
        Versione facile di DrawRoundedRectsOutlineBatch.

        items: iterable di tuple
            (x, y, w, h, radius, thickness=1.0, color=(255,255,255,255),
             rotation=0.0, alpha=255)
        """
        items = list(items)
        if not items:
            return
        positions, sizes, radii, thicknesses, colors, rotations, alphas = \
            [], [], [], [], [], [], []
        for raw in items:
            x, y, w, h, radius, thickness, color, rotation, alpha = self._pad_item(
                raw, 5, (1.0, (255, 255, 255, 255), 0.0, 255)
            )
            positions.append((x, y))
            sizes.append((w, h))
            radii.append(radius)
            thicknesses.append(thickness)
            colors.append(color)
            rotations.append(rotation)
            alphas.append(alpha)
        self.DrawRoundedRectsOutlineBatch(positions, sizes, radii, colors,
                                          thickness=thicknesses, alpha=alphas,
                                          rotation=rotations)

    def DrawLines(self, items):
        """
        Versione facile di DrawLinesBatch.

        items: iterable di tuple
            (x1, y1, x2, y2, thickness=1.0, color=(255,255,255,255),
             rotation=0.0, alpha=255)
        """
        items = list(items)
        if not items:
            return
        x1s, y1s, x2s, y2s, thicknesses, colors, rotations, alphas = \
            [], [], [], [], [], [], [], []
        for raw in items:
            x1, y1, x2, y2, thickness, color, rotation, alpha = self._pad_item(
                raw, 4, (1.0, (255, 255, 255, 255), 0.0, 255)
            )
            x1s.append(x1); y1s.append(y1); x2s.append(x2); y2s.append(y2)
            thicknesses.append(thickness)
            colors.append(color)
            rotations.append(rotation)
            alphas.append(alpha)
        self.DrawLinesBatch(x1s, y1s, x2s, y2s, colors,
                            thickness=thicknesses, alpha=alphas,
                            rotation=rotations)

    def DrawTriangles(self, items):
        """
        Versione facile di DrawTrianglesBatch. NOTA: DrawTrianglesBatch non
        supporta la rotazione via GPU; se `rotation` != 0 i 3 vertici
        vengono ruotati sulla CPU attorno al centroide (stessa formula di
        DrawTriangle) prima di essere impacchettati nel batch.

        items: iterable di tuple
            (x1, y1, x2, y2, x3, y3, color=(255,255,255,255),
             rotation=0.0, alpha=255)
        """
        items = list(items)
        if not items:
            return
        vertices, colors, alphas = [], [], []
        for raw in items:
            x1, y1, x2, y2, x3, y3, color, rotation, alpha = self._pad_item(
                raw, 6, ((255, 255, 255, 255), 0.0, 255)
            )
            if rotation != 0.0:
                cx = (x1 + x2 + x3) / 3.0; cy = (y1 + y2 + y3) / 3.0
                cs, sn = self._cos_sin_deg(rotation)
                dx1, dy1 = x1 - cx, y1 - cy
                dx2, dy2 = x2 - cx, y2 - cy
                dx3, dy3 = x3 - cx, y3 - cy
                x1, y1 = cx + dx1*cs - dy1*sn, cy + dx1*sn + dy1*cs
                x2, y2 = cx + dx2*cs - dy2*sn, cy + dx2*sn + dy2*cs
                x3, y3 = cx + dx3*cs - dy3*sn, cy + dx3*sn + dy3*cs
            vertices.append(((x1, y1), (x2, y2), (x3, y3)))
            colors.append(color)
            alphas.append(alpha)
        self.DrawTrianglesBatch(vertices, colors, alpha=alphas)

    def DrawTrianglesOutline(self, items):
        """
        Versione facile di DrawTrianglesOutlineBatch. Stessa nota sulla
        rotazione di DrawTriangles (ruotata sulla CPU prima del batch).

        items: iterable di tuple
            (x1, y1, x2, y2, x3, y3, thickness=1.0,
             color=(255,255,255,255), rotation=0.0, alpha=255)
        """
        items = list(items)
        if not items:
            return
        vertices, thicknesses, colors, alphas = [], [], [], []
        for raw in items:
            x1, y1, x2, y2, x3, y3, thickness, color, rotation, alpha = \
                self._pad_item(raw, 6, (1.0, (255, 255, 255, 255), 0.0, 255))
            if rotation != 0.0:
                cx = (x1 + x2 + x3) / 3.0; cy = (y1 + y2 + y3) / 3.0
                cs, sn = self._cos_sin_deg(rotation)
                dx1, dy1 = x1 - cx, y1 - cy
                dx2, dy2 = x2 - cx, y2 - cy
                dx3, dy3 = x3 - cx, y3 - cy
                x1, y1 = cx + dx1*cs - dy1*sn, cy + dx1*sn + dy1*cs
                x2, y2 = cx + dx2*cs - dy2*sn, cy + dx2*sn + dy2*cs
                x3, y3 = cx + dx3*cs - dy3*sn, cy + dx3*sn + dy3*cs
            vertices.append(((x1, y1), (x2, y2), (x3, y3)))
            thicknesses.append(thickness)
            colors.append(color)
            alphas.append(alpha)
        self.DrawTrianglesOutlineBatch(vertices, colors,
                                       thickness=thicknesses, alpha=alphas)

    def DrawRoundedTriangles(self, items):
        """
        Versione facile di DrawRoundedTrianglesBatch. Stessa nota sulla
        rotazione di DrawTriangles (ruotata sulla CPU prima del batch).

        items: iterable di tuple
            (x1, y1, x2, y2, x3, y3, radius, color=(255,255,255,255),
             rotation=0.0, alpha=255)
        """
        items = list(items)
        if not items:
            return
        vertices, radii, colors, alphas = [], [], [], []
        for raw in items:
            x1, y1, x2, y2, x3, y3, radius, color, rotation, alpha = \
                self._pad_item(raw, 7, ((255, 255, 255, 255), 0.0, 255))
            if rotation != 0.0:
                cx = (x1 + x2 + x3) / 3.0; cy = (y1 + y2 + y3) / 3.0
                cs, sn = self._cos_sin_deg(rotation)
                dx1, dy1 = x1 - cx, y1 - cy
                dx2, dy2 = x2 - cx, y2 - cy
                dx3, dy3 = x3 - cx, y3 - cy
                x1, y1 = cx + dx1*cs - dy1*sn, cy + dx1*sn + dy1*cs
                x2, y2 = cx + dx2*cs - dy2*sn, cy + dx2*sn + dy2*cs
                x3, y3 = cx + dx3*cs - dy3*sn, cy + dx3*sn + dy3*cs
            vertices.append(((x1, y1), (x2, y2), (x3, y3)))
            radii.append(radius)
            colors.append(color)
            alphas.append(alpha)
        self.DrawRoundedTrianglesBatch(vertices, radii, colors, alpha=alphas)

    def DrawRoundedTrianglesOutline(self, items):
        """
        Versione facile di DrawRoundedTrianglesOutlineBatch. Stessa nota
        sulla rotazione di DrawTriangles (ruotata sulla CPU prima del
        batch).

        items: iterable di tuple
            (x1, y1, x2, y2, x3, y3, radius, thickness=1.0,
             color=(255,255,255,255), rotation=0.0, alpha=255)
        """
        items = list(items)
        if not items:
            return
        vertices, radii, thicknesses, colors, alphas = [], [], [], [], []
        for raw in items:
            (x1, y1, x2, y2, x3, y3, radius, thickness, color,
             rotation, alpha) = self._pad_item(
                raw, 7, (1.0, (255, 255, 255, 255), 0.0, 255)
            )
            if rotation != 0.0:
                cx = (x1 + x2 + x3) / 3.0; cy = (y1 + y2 + y3) / 3.0
                cs, sn = self._cos_sin_deg(rotation)
                dx1, dy1 = x1 - cx, y1 - cy
                dx2, dy2 = x2 - cx, y2 - cy
                dx3, dy3 = x3 - cx, y3 - cy
                x1, y1 = cx + dx1*cs - dy1*sn, cy + dx1*sn + dy1*cs
                x2, y2 = cx + dx2*cs - dy2*sn, cy + dx2*sn + dy2*cs
                x3, y3 = cx + dx3*cs - dy3*sn, cy + dx3*sn + dy3*cs
            vertices.append(((x1, y1), (x2, y2), (x3, y3)))
            radii.append(radius)
            thicknesses.append(thickness)
            colors.append(color)
            alphas.append(alpha)
        self.DrawRoundedTrianglesOutlineBatch(vertices, radii, colors,
                                              thickness=thicknesses,
                                              alpha=alphas)

    def DrawEllipses(self, items):
        """
        Versione facile di DrawEllipsesBatch. NOTA: come DrawEllipsesBatch,
        il path GPU disegna un'ellisse esatta via SDF — qui non esiste il
        parametro `segments` (ha senso solo per DrawEllipse/DrawCircle,
        path CPU poligonale).

        items: iterable di tuple
            (cx, cy, rx, ry, color=(255,255,255,255), rotation=0.0,
             alpha=255)
        """
        items = list(items)
        if not items:
            return
        centers, radii, colors, rotations, alphas = [], [], [], [], []
        for raw in items:
            cx, cy, rx, ry, color, rotation, alpha = self._pad_item(
                raw, 4, ((255, 255, 255, 255), 0.0, 255)
            )
            centers.append((cx, cy))
            radii.append((rx, ry))
            colors.append(color)
            rotations.append(rotation)
            alphas.append(alpha)
        self.DrawEllipsesBatch(centers, radii, colors,
                               alpha=alphas, rotation=rotations)

    def DrawCircles(self, items):
        """
        Versione facile di DrawCirclesBatch.

        items: iterable di tuple
            (cx, cy, r, color=(255,255,255,255), alpha=255)
        """
        items = list(items)
        if not items:
            return
        centers, radii, colors, alphas = [], [], [], []
        for raw in items:
            cx, cy, r, color, alpha = self._pad_item(
                raw, 3, ((255, 255, 255, 255), 255)
            )
            centers.append((cx, cy))
            radii.append(r)
            colors.append(color)
            alphas.append(alpha)
        self.DrawCirclesBatch(centers, radii, colors, alpha=alphas)

    def DrawEllipsesOutline(self, items):
        """
        Versione facile di DrawEllipsesOutlineBatch.

        items: iterable di tuple
            (cx, cy, rx, ry, thickness=1.0, color=(255,255,255,255),
             rotation=0.0, alpha=255)
        """
        items = list(items)
        if not items:
            return
        centers, radii, thicknesses, colors, rotations, alphas = \
            [], [], [], [], [], []
        for raw in items:
            cx, cy, rx, ry, thickness, color, rotation, alpha = self._pad_item(
                raw, 4, (1.0, (255, 255, 255, 255), 0.0, 255)
            )
            centers.append((cx, cy))
            radii.append((rx, ry))
            thicknesses.append(thickness)
            colors.append(color)
            rotations.append(rotation)
            alphas.append(alpha)
        self.DrawEllipsesOutlineBatch(centers, radii, thickness=thicknesses,
                                      colors=colors, alpha=alphas,
                                      rotation=rotations)

    def DrawCirclesOutline(self, items):
        """
        Versione facile di DrawCircleOutlineBatch.

        items: iterable di tuple
            (cx, cy, r, thickness=1.0, color=(255,255,255,255), alpha=255)
        """
        items = list(items)
        if not items:
            return
        centers, radii, thicknesses, colors, alphas = [], [], [], [], []
        for raw in items:
            cx, cy, r, thickness, color, alpha = self._pad_item(
                raw, 3, (1.0, (255, 255, 255, 255), 255)
            )
            centers.append((cx, cy))
            radii.append(r)
            thicknesses.append(thickness)
            colors.append(color)
            alphas.append(alpha)
        self.DrawCircleOutlineBatch(centers, radii, thickness=thicknesses,
                                    colors=colors, alpha=alphas)

    def DrawBezierCurves(self, items, segments=None, smooth=True, alpha=255):
        """
        Versione facile di DrawBezierCurvesBatch. NOTA: `segments`/`smooth`
        controllano la tassellazione e — come nella funzione Batch
        sottostante — si applicano a TUTTE le curve del batch, non sono
        quindi per-item; lo stesso vale per `alpha`, qui globale per
        l'intero batch.

        items: iterable di tuple
            (p0, p1, p2, thickness=2.0, color=(255,255,255,255),
             rotation=0.0)
        p0/p1/p2 sono coppie (x, y).
        """
        items = list(items)
        if not items:
            return
        p0s, p1s, p2s, thicknesses, colors = [], [], [], [], []
        for raw in items:
            p0, p1, p2, thickness, color, rotation = self._pad_item(
                raw, 3, (2.0, (255, 255, 255, 255), 0.0)
            )
            if rotation != 0.0:
                ocx = (p0[0] + p1[0] + p2[0]) / 3.0
                ocy = (p0[1] + p1[1] + p2[1]) / 3.0
                cs, sn = self._cos_sin_deg(rotation)

                def _rot(p, ocx=ocx, ocy=ocy, cs=cs, sn=sn):
                    dx, dy = p[0] - ocx, p[1] - ocy
                    return (ocx + dx * cs - dy * sn, ocy + dx * sn + dy * cs)

                p0, p1, p2 = _rot(p0), _rot(p1), _rot(p2)
            p0s.append(p0); p1s.append(p1); p2s.append(p2)
            thicknesses.append(thickness)
            colors.append(color)
        self.DrawBezierCurvesBatch(p0s, p1s, p2s, thickness=thicknesses,
                                   colors=colors, segments=segments,
                                   smooth=smooth, alpha=alpha)

    def DrawSprites(self, items):
        """
        Alias 'facile' — nome coerente con le altre wrapper (senza
        suffisso Batch) di DrawSpritesBatch, che già accetta una lista di
        tuple così com'è.

        items: iterable di tuple
            (x, y, w, h, rot, u0, v0, u1, v1, alpha)
        """
        self.DrawSpritesBatch(items)

    def DrawTexts(self, items):
        """
        Alias 'facile' — nome coerente con le altre wrapper di
        DrawTextBatch, che già accetta una lista di tuple così com'è.

        items: iterable di tuple
            (text, x, y, font, size, color, alpha, rotation)
        """
        self.DrawTextBatch(items)

#AGGANCIAMENTI
DRAW.Point = Point
DRAW.Rect = Rect
DRAW.RotRect = RotRect
DRAW.Circle = Circle
DRAW.Ellipse = Ellipse
DRAW.Line = Line
DRAW.Triangle = Triangle
DRAW.RoundedRect = RoundedRect
DRAW.Polygon = Polygon
DRAW.TextureCollider = TextureCollider