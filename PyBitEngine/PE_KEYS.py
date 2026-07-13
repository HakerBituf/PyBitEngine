# File: PE_KEYS.py
import sdl2 as _sdl2

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
