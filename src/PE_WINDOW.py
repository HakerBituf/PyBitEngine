import sdl2
import sdl2.sdlimage as img
import moderngl
import time
import os, ctypes, gc

from .PE_DRAW import DRAW
from .PE_KEYS import (PE_Event, PE_KEYDOWN, PE_KEYUP,
                      PE_MOUSEDRAG, PE_MOUSEMOTION,
                      PE_MOUSEBUTTONDOWN, PE_MOUSEBUTTONUP, PE_MOUSEWHEEL)


def _set_window_icon(window, path: str):
    ext     = os.path.splitext(path)[1].lower()
    surface = None

    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".ico"]:
        surface = img.IMG_Load(path.encode("utf-8"))
    else:
        print("Formato non supportato:", ext)
        return

    if not surface:
        print("Errore caricamento icona")
        return

    sdl2.SDL_SetWindowIcon(window, surface)
    sdl2.SDL_FreeSurface(surface)


class WINDOW(DRAW):
    def __init__(self, title="PyEngine",
                 geometry=("center", "center", 800, 600),
                 icon: str = "",
                 fullscreen: bool = False,
                 borderless: bool = False,
                 VSync: bool = False,
                 MSAA: bool = False,
                 MSS: int = 4,
                 max_fps: int = None,
                 gc_auto: bool = False,
                 gc_mode: str = "frames",
                 gc_interval: float = 600,
                 gc_thresholds: tuple = None,
                 gc_obj_number: int = 700,
                 max_draw_elements: int = 131072):

        # ------------------------------------------------------------------ #
        # GC — configurazione iniziale
        # ------------------------------------------------------------------ #
        self._gc_auto       = gc_auto
        self._gc_mode       = gc_mode        # "frames" | "time" | "smart"
        self._gc_interval   = max(1, gc_interval)
        self._gc_thresholds = gc_thresholds
        self._gc_counter    = 0
        self._gc_cycle_count = 0   # BUG B fix: contatore dedicato per l'escalation gen=2
        self._gc_last_time  = time.perf_counter()
        self._gc_stats      = {"count": 0, "total_time": 0.0}
        # Soglia gen0 per la modalità "smart".
        # Python di default usa 700; usiamo lo stesso valore per evitare
        # raccolte troppo frequenti.
        self._gc_smart_threshold = gc_obj_number

        if gc_mode not in ("frames", "time", "smart"):
            raise ValueError(
                f"gc_mode must be 'frames', 'time' or 'smart' (got {gc_mode!r})"
            )
        if gc_thresholds is not None and len(gc_thresholds) != 3:
            raise ValueError(
                "gc_thresholds must be a 3-tuple (gen0, gen1, gen2)"
            )

        # BUG A fix: questa logica era invertita rispetto a SetGCAuto() e alla
        # sua docstring. Con gc_auto=False (il default!) il codice originale
        # chiamava gc.disable() e MAI gc.enable(): risultato, con i parametri
        # di default la libreria disabilitava per sempre il garbage collector
        # ciclico di Python senza eseguire alcuna raccolta manuale al suo
        # posto (il blocco "GC manuale" nel Loop() è condizionato proprio su
        # self._gc_auto). Qualsiasi ciclo di riferimenti — incluso quello che
        # questa stessa classe crea con resize_watcher (self -> _resize_cb ->
        # closure -> self) — non sarebbe mai stato raccolto: memory leak
        # garantito già con i parametri predefiniti.
        #
        # Ora la logica è identica (e quindi sempre coerente) a quella di
        # SetGCAuto():
        #   gc_auto=True  -> controllo manuale: gc.disable() + ForceGC
        #                     periodico nel Loop().
        #   gc_auto=False -> GC automatico di Python: gc.enable() (+ soglie
        #                     personalizzate se fornite).
        if self._gc_auto:
            gc.disable()
            # NON chiamare gc.set_debug() — qualsiasi flag di debug (anche
            # DEBUG_STATS) stampa su stderr ad ogni raccolta e distrugge
            # le performance; vantaggi zero a runtime.
        else:
            gc.enable()
            if gc_thresholds is not None:
                gc.set_threshold(*gc_thresholds)

        # ------------------------------------------------------------------ #
        # Finestra e contesto OpenGL
        # ------------------------------------------------------------------ #
        self.title    = title
        self.position = geometry[0], geometry[1]
        self.size     = geometry[2], geometry[3]
        self.color    = (1.0, 1.0, 1.0)
        self.fps      = 0
        self._fps_timer  = 0.0
        self._fps_frames = 0

        # ------------------------------------------------------------------ #
        # BUG FIX: saturazione GPU inutile
        # ------------------------------------------------------------------ #
        # Loop() non aveva MAI alcun limite di frame rate. Con VSync=False
        # (il default!) il while principale gira a velocità massima: anche
        # una scena vuota arriva a centinaia/migliaia di FPS, saturando la
        # GPU al 100% per nessun motivo (consumo energetico, ventole,
        # calore, e su laptop scarica la batteria inutilmente).
        #
        # Con max_fps=None (default) applichiamo un comportamento sensato
        # automaticamente:
        #   - VSync=True  -> nessun cap: il driver sincronizza già lo swap
        #                     al refresh del monitor, SDL_GL_SwapWindow
        #                     blocca da solo.
        #   - VSync=False -> cap automatico a 240 FPS, così chi non sa di
        #                     dover limitare il frame rate non si ritrova
        #                     comunque a bruciare GPU per niente.
        # Un valore esplicito (anche 0 = nessun limite) sovrascrive sempre
        # questa scelta automatica.
        if max_fps is None:
            self._max_fps = 0 if VSync else 240
        else:
            self._max_fps = max(0, int(max_fps))
        self._frame_duration = (1.0 / self._max_fps) if self._max_fps > 0 else 0.0

        sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)
        img.IMG_Init(img.IMG_INIT_PNG | img.IMG_INIT_JPG)

        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 3)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 3)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_PROFILE_MASK,
                                  sdl2.SDL_GL_CONTEXT_PROFILE_CORE)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DOUBLEBUFFER, 1)
        if MSAA:
            sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_MULTISAMPLEBUFFERS, 1)
            sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_MULTISAMPLESAMPLES, MSS)

        posL = list(self.position)
        if self.position[0] == "center": posL[0] = sdl2.SDL_WINDOWPOS_CENTERED
        if self.position[1] == "center": posL[1] = sdl2.SDL_WINDOWPOS_CENTERED

        self.window = sdl2.SDL_CreateWindow(
            title.encode(),
            posL[0], posL[1],
            self.size[0], self.size[1],
            sdl2.SDL_WINDOW_OPENGL |
            sdl2.SDL_WINDOW_SHOWN  |
            sdl2.SDL_WINDOW_RESIZABLE
        )

        if icon != "":
            _set_window_icon(self.window, icon)

        if fullscreen and not borderless:
            sdl2.SDL_SetWindowFullscreen(self.window,
                                          sdl2.SDL_WINDOW_FULLSCREEN)
        elif borderless and not fullscreen:
            sdl2.SDL_SetWindowFullscreen(self.window,
                                          sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP)

        self._fluid_resize_enabled = False

        self.gl_context = sdl2.SDL_GL_CreateContext(self.window)
        sdl2.SDL_GL_MakeCurrent(self.window, self.gl_context)

        if VSync:
            sdl2.SDL_GL_SetSwapInterval(1)
        else:
            sdl2.SDL_GL_SetSwapInterval(0)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._init_draw(max_elements=max_draw_elements)
        self.ctx.viewport = (0, 0, self.size[0], self.size[1])
        self.SetResolution(self.size[0], self.size[1])
        self.SetBackground((255, 255, 255))
        self.running = True

        # PERF FIX 16: buffer ctypes riusati per SDL_GetMouseState nel loop
        # eventi. In precedenza ogni SDL_MOUSEWHEEL allocava due ctypes.c_int(0)
        # nuovi (oggetti Python) creando spazzatura ad ogni scroll.
        self._mouse_x_buf = ctypes.c_int(0)
        self._mouse_y_buf = ctypes.c_int(0)

        # --- FIX: event watcher per il resize fluido su Windows ---
        def resize_watcher(userdata, event_ptr):
            event = event_ptr.contents
            if event.type == sdl2.SDL_WINDOWEVENT:
                if event.window.event == sdl2.SDL_WINDOWEVENT_SIZE_CHANGED:
                    w, h = event.window.data1, event.window.data2
                    self._on_window_resize(w, h)
                    if self._fluid_resize_enabled:
                        self.ctx.clear(*self.color)
                        self.draw()
                        sdl2.SDL_GL_SwapWindow(self.window)
            return 0

        self._resize_cb = sdl2.SDL_EventFilter(resize_watcher)
        sdl2.SDL_AddEventWatch(self._resize_cb, None)
        # -----------------------------------------------------------

    # ------------------------------------------------------------------ #
    # Hook da sovrascrivere
    # ------------------------------------------------------------------ #

    def update(self, dt, events):
        pass

    def draw(self):
        pass

    def _on_window_resize(self, w, h):
        if w == 0 or h == 0:
            return
        self.size = (w, h)
        self.ctx.viewport = (0, 0, w, h)
        self.SetResolution(w, h)

    # ------------------------------------------------------------------ #
    # GC — API pubblica
    # ------------------------------------------------------------------ #

    def SetGCAuto(self, enabled: bool, mode: str = "frames",
                  interval: float = 600, thresholds: tuple = None):
        """
        Configura la gestione del garbage collector.

        - enabled : True  → controllo manuale (gc.disable + ForceGC periodico)
                    False → GC automatico di Python
        - mode    : "frames"  → ForceGC ogni N fotogrammi
                    "time"    → ForceGC ogni T secondi
                    "smart"   → ForceGC quando gen0 supera la soglia
        - interval: N fotogrammi o T secondi tra una raccolta e l'altra
        - thresholds: (gen0, gen1, gen2) — valido solo se enabled=False
        """
        self._gc_auto      = enabled
        if mode not in ("frames", "time", "smart"):
            raise ValueError(
                f"mode must be 'frames', 'time' or 'smart' (got {mode!r})"
            )
        if thresholds is not None and len(thresholds) != 3:
            raise ValueError(
                "thresholds must be a 3-tuple (gen0, gen1, gen2)"
            )
        self._gc_mode      = mode
        self._gc_interval  = max(1, interval)
        self._gc_counter   = 0
        self._gc_cycle_count = 0
        self._gc_last_time = time.perf_counter()

        if enabled:
            gc.disable()
            # IMPORTANTE: non chiamare MAI gc.set_debug() qui.
            # Anche DEBUG_STATS stampa su stderr ad ogni raccolta
            # e dimezza i frame rate su scene pesanti.
        else:
            # FIX: ri-abilitare il GC se era stato disabilitato in precedenza
            gc.enable()
            if thresholds is not None:
                gc.set_threshold(*thresholds)

    def SetGCInterval(self, interval: float):
        """Cambia l'intervallo di raccolta (fotogrammi o secondi)."""
        self._gc_interval = max(1, interval)

    def SetGCSmartThreshold(self, threshold: int):
        """Soglia di oggetti in gen0 per la modalità 'smart' (default: 700)."""
        self._gc_smart_threshold = max(100, threshold)

    def SetGCThresholds(self, gen0: int, gen1: int, gen2: int):
        """Imposta le soglie generazionali per il GC automatico di Python."""
        gc.set_threshold(gen0, gen1, gen2)

    def ForceGC(self, generation: int = 0):
        """
        Forza una raccolta GC immediata.

        generation:
          0 → solo generazione giovane (veloce, ~0.1 ms — usare di routine)
          1 → fino alla gen 1 (media)
          2 → raccolta completa   (lenta, ~1–5 ms — usare raramente)

        Restituisce (n_oggetti_raccolti, tempo_secondi).
        """
        start     = time.perf_counter()
        collected = gc.collect(generation)
        elapsed   = time.perf_counter() - start
        self._gc_stats["count"]      += 1
        self._gc_stats["total_time"] += elapsed
        return collected, elapsed

    def _periodic_force_gc(self):
        """
        BUG B fix: raccolta periodica chiamata dal Loop() per tutte le
        modalità ("frames"/"time"/"smart"). Usa un contatore DEDICATO
        (`_gc_cycle_count`) che non viene mai azzerato dal trigger stesso,
        a differenza del vecchio codice che resettava `_gc_counter` (usato
        anche per decidere QUANDO raccogliere) e quindi non poteva mai
        raggiungere il multiplo necessario per scatenare una gen=2: il
        "ogni 10 cicli, raccolta completa" non scattava MAI in pratica.
        Ogni 10 raccolte di routine (gen 0) ne viene quindi eseguita una
        completa (gen 2) per eliminare i riferimenti circolari accumulati,
        in modo identico per tutte le modalità.
        """
        self._gc_cycle_count += 1
        gen = 2 if (self._gc_cycle_count % 10 == 0) else 0
        return self.ForceGC(gen)

    def GetGCStats(self):
        """Restituisce statistiche leggere sulle raccolte manuali."""
        return self._gc_stats.copy()

    def IsGCEnabled(self):
        """True se il GC ciclico di Python è attivo (gc_auto=False)."""
        return gc.isenabled()

    def GetGCThresholds(self):
        """Restituisce le soglie generazionali correnti (gen0, gen1, gen2)."""
        return gc.get_threshold()
    
    def GetScreenResolution(self):
        """
        Restituisce la risoluzione nativa dello schermo (monitor) su cui 
        si trova attualmente la finestra, come tupla (larghezza, altezza).
        """
        # Ottiene l'indice del monitor associato alla finestra (ottimo per il multi-monitor)
        display_index = sdl2.SDL_GetWindowDisplayIndex(self.window)
        if display_index < 0:
            display_index = 0  # Fallback sul monitor principale se la finestra non è ancora pronta

        mode = sdl2.SDL_DisplayMode()
        if sdl2.SDL_GetDesktopDisplayMode(display_index, ctypes.byref(mode)) == 0:
            return mode.w, mode.h
        
        # Fallback nel caso in cui SDL dovesse fallire il recupero delle info
        return 0, 0

    # ------------------------------------------------------------------ #
    # Finestra — API pubblica
    # ------------------------------------------------------------------ #

    def SetCursorVisible(self, visible: bool):
        """Mostra o nasconde il cursore in base al valore booleano passato."""
        sdl2.SDL_ShowCursor(1 if visible else 0)

    def SetFluidResize(self, enabled: bool):
        self._fluid_resize_enabled = enabled

    def SetBackground(self, color: tuple = (255, 255, 255)):
        if len(color) == 3:
            r, g, b = color
            self.color = (r/255, g/255, b/255, 1.0)
        else:
            r, g, b, a = color
            self.color = (r/255, g/255, b/255, a/255)

    def SetTitle(self, title: str = "PyBitEngine"):
        sdl2.SDL_SetWindowTitle(self.window, title.encode())

    def SetIcon(self, icon: str = ""):
        if icon != "":
            _set_window_icon(self.window, icon)

    def GetFPS(self):
        return self.fps

    def SetMaxFPS(self, max_fps: int = 0):
        """
        Imposta il tetto massimo di fotogrammi al secondo.

        max_fps : int
            0  -> nessun limite (VSync, se attivo, resta l'unico freno)
            >0 -> il Loop() dorme il tempo necessario a non superare
                  questo valore, evitando di saturare la GPU per niente
                  quando la scena è leggera e VSync è disattivato.
        """
        self._max_fps = max(0, int(max_fps))
        self._frame_duration = (1.0 / self._max_fps) if self._max_fps > 0 else 0.0

    def GetMaxFPS(self):
        return self._max_fps

    def SetFullscreen(self, fullscreen: bool = None, mode: str = "borderless"):
        if mode == "borderless":
            sdl2.SDL_SetWindowFullscreen(
                self.window,
                sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP if fullscreen else 0)
        elif mode == "fullscreen":
            sdl2.SDL_SetWindowFullscreen(
                self.window,
                sdl2.SDL_WINDOW_FULLSCREEN if fullscreen else 0)

    # ------------------------------------------------------------------ #
    # Loop principale
    # ------------------------------------------------------------------ #

    def Loop(self):
        event      = sdl2.SDL_Event()
        last_time  = time.perf_counter()
        events_list = []

        # BUG C fix: il cleanup (rilascio buffer GPU, rimozione event
        # watcher, distruzione contesto/finestra SDL) era posto DOPO il
        # while, senza alcun try/finally. Se update()/draw() (forniti
        # dall'utente) sollevavano un'eccezione, l'intero blocco di
        # pulizia veniva saltato: handle SDL/OpenGL e cache Python
        # restavano vivi indefinitamente (leak di risorse non gestite
        # dal garbage collector di Python, perché SDL/moderngl non sono
        # tracciati da esso).
        try:
            while self.running:
                now = time.perf_counter()
                dt  = min(now - last_time, 0.1)
                last_time = now

                # Calcolo FPS
                self._fps_timer  += dt
                self._fps_frames += 1
                if self._fps_timer >= 1.0:
                    self.fps         = self._fps_frames
                    self._fps_frames = 0
                    self._fps_timer -= 1.0

                # --- Raccolta eventi ---
                events_list.clear()
                append = events_list.append

                while sdl2.SDL_PollEvent(event):
                    t = event.type

                    if t == sdl2.SDL_QUIT:
                        self.running = False

                    elif t == sdl2.SDL_WINDOWEVENT:
                        # BUG FIX 2: il resize e' gestito ESCLUSIVAMENTE dal
                        # watcher SDL_AddEventWatch (SIZE_CHANGED), che scatta
                        # anche durante il resize "live" su Windows. SDL emette
                        # in sequenza sia SIZE_CHANGED sia RESIZED per lo
                        # stesso resize: gestire anche qui il branch causava
                        # _on_window_resize (e _build_gpu_resources delle
                        # camere) invocato due volte per resize, con rischio
                        # di doppio release() degli stessi oggetti GPU.
                        # Se in futuro si volesse rimuovere il watcher,
                        # reintrodurre qui il branch:
                        #   if event.window.event in (
                        #       sdl2.SDL_WINDOWEVENT_RESIZED,
                        #       sdl2.SDL_WINDOWEVENT_SIZE_CHANGED,
                        #   ): self._on_window_resize(
                        #       event.window.data1, event.window.data2)
                        pass

                    elif t == sdl2.SDL_KEYDOWN:
                        append(PE_Event(type=PE_KEYDOWN,
                                        key=event.key.keysym.sym))
                    elif t == sdl2.SDL_KEYUP:
                        append(PE_Event(type=PE_KEYUP,
                                        key=event.key.keysym.sym))

                    elif t == sdl2.SDL_MOUSEMOTION:
                        if event.motion.state != 0:
                            append(PE_Event(type=PE_MOUSEDRAG,
                                            x=event.motion.x, y=event.motion.y,
                                            dx=event.motion.xrel,
                                            dy=event.motion.yrel))
                        else:
                            append(PE_Event(type=PE_MOUSEMOTION,
                                            x=event.motion.x, y=event.motion.y,
                                            dx=event.motion.xrel,
                                            dy=event.motion.yrel))

                    elif t == sdl2.SDL_MOUSEBUTTONDOWN:
                        append(PE_Event(type=PE_MOUSEBUTTONDOWN,
                                        button=event.button.button,
                                        x=event.button.x, y=event.button.y,
                                        clicks=event.button.clicks))

                    elif t == sdl2.SDL_MOUSEBUTTONUP:
                        append(PE_Event(type=PE_MOUSEBUTTONUP,
                                        button=event.button.button,
                                        x=event.button.x, y=event.button.y))

                    elif t == sdl2.SDL_MOUSEWHEEL:
                        # PERF FIX 16: riutilizza i buffer ctypes preallocati
                        # in __init__ invece di crearli ogni evento (scroll
                        # intenso => spazzatura Python significativa).
                        mx = self._mouse_x_buf
                        my = self._mouse_y_buf
                        sdl2.SDL_GetMouseState(ctypes.byref(mx), ctypes.byref(my))
                        append(PE_Event(type=PE_MOUSEWHEEL,
                                        x=mx.value, y=my.value,
                                        wheel_x=event.wheel.x,
                                        wheel_y=event.wheel.y))

                # --- Frame ---
                self.update(dt, events_list)
                self.ctx.clear(*self.color)
                self.draw()
                sdl2.SDL_GL_SwapWindow(self.window)

                # --- Frame limiter (BUG FIX: saturazione GPU inutile) ---
                # Senza questo blocco, con VSync=False il while sopra girava
                # a velocità massima: zero pause, zero freni, GPU al 100%
                # anche per disegnare un singolo rettangolo statico.
                # Dormiamo il tempo restante rispetto al budget di frame
                # (1/max_fps secondi), misurato da 'now' (inizio di questo
                # stesso frame). time.sleep() su molti sistemi (specialmente
                # Windows) ha una granularità di ~15ms, quindi per restare
                # precisi lasciamo un margine di sicurezza e affiniamo gli
                # ultimi ~1ms con un breve spin-wait (costa pochissima CPU,
                # zero GPU, ed evita di "sforare" il target dormendo troppo).
                if self._frame_duration > 0.0:
                    target_end = now + self._frame_duration
                    remaining = target_end - time.perf_counter()
                    if remaining > 0.001:
                        time.sleep(max(0.0, remaining - 0.001))
                    if remaining > 0.0:
                        while time.perf_counter() < target_end:
                            pass

                # --- GC manuale ---
                if self._gc_auto:
                    mode = self._gc_mode

                    if mode == "frames":
                        self._gc_counter += 1
                        if self._gc_counter >= self._gc_interval:
                            self._periodic_force_gc()
                            self._gc_counter = 0

                    elif mode == "time":
                        now_gc = time.perf_counter()
                        if now_gc - self._gc_last_time >= self._gc_interval:
                            self._periodic_force_gc()
                            self._gc_last_time = now_gc

                    elif mode == "smart":
                        # gc.get_count() è O(1) e non alloca — sicuro da chiamare
                        # ad ogni frame.
                        if gc.get_count()[0] > self._gc_smart_threshold:
                            self._periodic_force_gc()
        finally:
            # --- Cleanup --- (eseguito SEMPRE: uscita normale o eccezione)
            self._release_draw()

            if hasattr(self, "_resize_cb"):
                sdl2.SDL_DelEventWatch(self._resize_cb, None)

            img.IMG_Quit()
            sdl2.SDL_GL_DeleteContext(self.gl_context)
            sdl2.SDL_DestroyWindow(self.window)
            sdl2.SDL_Quit()
