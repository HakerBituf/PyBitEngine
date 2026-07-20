"""PE_TIME — timer e scheduling non bloccanti per PyBitEngine.

Due famiglie di strumenti, per due casi d'uso diversi:

1) SCHEDULER A FRAME (Scheduler, After, Every, Countdown, Cooldown,
   Stopwatch): NON usano thread OS, costo quasi nullo, vanno aggiornati
   a mano dentro il tuo game loop passando il dt (Update(dt) /
   countdown.Update(dt)). Da usare per QUALSIASI cosa che tocca stato di
   gioco, rendering o comunque il contesto OpenGL/SDL2 — cioe' il 99%
   dei casi in un gioco.

2) TIMER SU THREAD REALE (AsyncTimer / AsyncAfter): basati sul modulo
   standard `threading`, utili SOLO per attese che non devono toccare
   OpenGL/SDL2 direttamente (es. aspettare un file, un download, un
   calcolo pesante in background) mentre il gioco continua a girare.
   Il loro callback gira su un thread separato: se deve toccare lo
   stato del gioco o chiamare funzioni DRAW/WINDOW, NON farlo dentro il
   callback -- usa RunOnMainThread(fn, *args) per accodarlo in sicurezza
   e chiama PumpMainThread() una volta a frame dal thread principale.
"""

from __future__ import annotations

import queue as _queue
import threading
import time
from typing import Callable


# ============================================================
# 1) SCHEDULER A FRAME (non bloccante, zero-thread, zero-alloc a regime)
# ============================================================

class _TimerHandle:
    """Handle interno di un timer registrato su uno Scheduler."""

    __slots__ = ("remaining", "interval", "callback", "args", "kwargs",
                 "repeat", "repeats_left", "alive")

    def __init__(self, delay, callback, args, kwargs, interval, repeats_left):
        self.remaining = delay
        self.interval = interval
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self.repeat = interval is not None
        self.repeats_left = repeats_left  # None = infinito
        self.alive = True

    def Cancel(self):
        """Interrompe il timer. Non bloccante: ha effetto dal prossimo
        Update() dello scheduler che lo possiede."""
        self.alive = False


class Scheduler:
    """Gestisce N timer a frame senza usare thread OS. Va aggiornato
    manualmente con Update(dt) dal thread principale (dentro il tuo
    update(dt, events)). Rimozione via swap-pop: zero shift, zero
    allocazioni dopo il warmup iniziale."""

    __slots__ = ("_timers",)

    def __init__(self):
        self._timers: list[_TimerHandle] = []

    def After(self, seconds: float, callback: Callable, *args, **kwargs) -> _TimerHandle:
        """Esegue callback(*args, **kwargs) una sola volta dopo `seconds`
        secondi di gioco. Non bloccante: ritorna subito un handle. Chiama
        handle.Cancel() (o scheduler.Cancel(handle)) per interromperlo in
        qualsiasi momento prima che scatti."""
        handle = _TimerHandle(float(seconds), callback, args, kwargs, None, None)
        self._timers.append(handle)
        return handle

    def Every(self, seconds: float, callback: Callable, *args,
              times: int | None = None, immediate: bool = False, **kwargs) -> _TimerHandle:
        """Esegue callback ogni `seconds` secondi.
        times=None    -> ripete all'infinito finche' non viene cancellato.
        times=N        -> si ferma da solo dopo N esecuzioni.
        immediate=True -> la prima esecuzione scatta al prossimo Update()
                          invece che dopo `seconds`."""
        first_delay = 0.0 if immediate else float(seconds)
        handle = _TimerHandle(first_delay, callback, args, kwargs, float(seconds), times)
        self._timers.append(handle)
        return handle

    def Cancel(self, handle: _TimerHandle):
        """Interrompe un timer restituito da After/Every."""
        handle.alive = False

    def CancelAll(self):
        """Interrompe tutti i timer attivi in un colpo solo (utile su
        cambio scena/livello/game-over)."""
        for t in self._timers:
            t.alive = False

    def Update(self, dt: float):
        """Da chiamare una volta a frame. Idioma standard di
        reverse-iteration + swap-pop: nessuno shift di lista, nessuna
        allocazione, sicuro anche se un callback cancella se' stesso o
        un altro timer durante l'esecuzione.

        BUG FIX: iteriamo SOLO sui timer esistenti all'inizio del frame
        (n = len(timers) all'ingresso) e riferiamo la swap-pop al confine
        `end` locale, non a `timers[-1]`. Cosi' un callback che chiama
        After/Every appende in coda senza rischiare che la swap-pop del
        timer corrente scarti il nuovo handle appena creato (che sara'
        invece visto al prossimo Update)."""
        timers = self._timers
        end = len(timers)     # numero di timer da processare in questo frame
        i = end - 1
        while i >= 0:
            t = timers[i]
            if t.alive:
                t.remaining -= dt
                if t.remaining <= 0.0:
                    t.callback(*t.args, **t.kwargs)
                    if t.repeat and t.alive and (t.repeats_left is None or t.repeats_left > 1):
                        if t.repeats_left is not None:
                            t.repeats_left -= 1
                        # += invece di = : mantiene la precisione media
                        # anche se il frame precedente ha "sforato" leggermente.
                        t.remaining += t.interval
                    else:
                        t.alive = False
            if not t.alive:
                # Swap con l'ultimo timer della finestra "originale" (end-1),
                # NON con timers[-1]: cosi' i nuovi handle appesi dai callback
                # (posizioni >= end) restano intatti in coda.
                last = end - 1
                if i != last:
                    timers[i] = timers[last]
                # Rimuovi lo slot last spostando eventuali nuovi handle
                # (in end..len-1) di una posizione a sinistra. In pratica:
                # pop(last) — O(n_new) nel caso limite, ma n_new e' quasi
                # sempre 0 e comunque <= al numero di timer creati nel frame.
                timers.pop(last)
                end -= 1
            i -= 1

    def Count(self) -> int:
        """Numero di timer attivi in questo momento (utile per debug/HUD)."""
        return len(self._timers)


# Scheduler globale di comodo: per la maggior parte dei giochi basta
# questo, senza doverne istanziare uno a mano.
_default_scheduler = Scheduler()


def After(seconds, callback, *args, **kwargs) -> _TimerHandle:
    return _default_scheduler.After(seconds, callback, *args, **kwargs)


def Every(seconds, callback, *args, times=None, immediate=False, **kwargs) -> _TimerHandle:
    return _default_scheduler.Every(seconds, callback, *args, times=times, immediate=immediate, **kwargs)


def Cancel(handle: _TimerHandle):
    _default_scheduler.Cancel(handle)


def CancelAll():
    _default_scheduler.CancelAll()


def Update(dt: float):
    """Aggiorna lo scheduler globale. Chiamalo una volta a frame, es.
    all'inizio del tuo update(dt, events): PE_TIME.Update(dt)."""
    _default_scheduler.Update(dt)


def Count() -> int:
    return _default_scheduler.Count()


# ============================================================
# 2) COUNTDOWN / COOLDOWN / STOPWATCH — oggetti leggeri per entita'
# ============================================================

class Countdown:
    """Timer a uso singolo, senza scheduler: pensato per essere tenuto
    come attributo su un oggetto di gioco (proiettile, nemico, powerup,
    animazione...) e aggiornato a mano ogni frame. Zero allocazioni dopo
    la creazione, __slots__ per stare leggero anche con migliaia di
    istanze vive contemporaneamente."""

    __slots__ = ("duration", "remaining", "running")

    def __init__(self, duration: float, start: bool = True):
        self.duration = float(duration)
        self.remaining = self.duration if start else 0.0
        self.running = start

    def Start(self, duration: float | None = None):
        """Avvia (o riavvia) il countdown. Se duration e' None riusa
        l'ultima durata impostata."""
        if duration is not None:
            self.duration = float(duration)
        self.remaining = self.duration
        self.running = True

    def Stop(self):
        """Interrompe il countdown senza farlo scattare."""
        self.running = False

    def Update(self, dt: float) -> bool:
        """Aggiorna il countdown. Ritorna True esattamente nel frame in
        cui scade — comodo per `if countdown.Update(dt): spawn()`."""
        if not self.running:
            return False
        self.remaining -= dt
        if self.remaining <= 0.0:
            self.remaining = 0.0
            self.running = False
            return True
        return False

    @property
    def expired(self) -> bool:
        return not self.running and self.remaining <= 0.0

    @property
    def progress(self) -> float:
        """Da 0.0 (appena partito) a 1.0 (scaduto). Comodo per barre di
        caricamento, fade, animazioni a tempo."""
        if self.duration <= 0.0:
            return 1.0
        return 1.0 - max(0.0, min(1.0, self.remaining / self.duration))


class Cooldown(Countdown):
    """Variante semantica di Countdown per attacchi/abilita': parte
    'pronta' e si consuma con Trigger()."""

    __slots__ = ()

    def __init__(self, duration: float):
        super().__init__(duration, start=False)

    @property
    def ready(self) -> bool:
        return not self.running

    def Trigger(self) -> bool:
        """Se pronto, consuma il cooldown e ritorna True; altrimenti non
        fa nulla e ritorna False. Pattern tipico:
        if arma.cooldown.Trigger(): spara()"""
        if self.ready:
            self.Start()
            return True
        return False


class Stopwatch:
    """Cronometro FPS-indipendente, con pausa/ripresa. Non serve
    aggiornarlo ogni frame: legge time.perf_counter() on-demand quando
    interroghi `elapsed`."""

    __slots__ = ("_start", "_accumulated", "_running")

    def __init__(self, start: bool = True):
        self._start = time.perf_counter() if start else None
        self._accumulated = 0.0
        self._running = start

    def Start(self):
        if not self._running:
            self._start = time.perf_counter()
            self._running = True

    def Pause(self):
        if self._running:
            self._accumulated += time.perf_counter() - self._start
            self._running = False

    def Reset(self):
        self._accumulated = 0.0
        self._start = time.perf_counter() if self._running else None

    @property
    def elapsed(self) -> float:
        if self._running:
            return self._accumulated + (time.perf_counter() - self._start)
        return self._accumulated


# ============================================================
# 3) TIMER SU THREAD REALE — solo per lavoro che NON tocca OpenGL/SDL2
# ============================================================

_main_thread_queue: "_queue.Queue" = _queue.Queue()


def RunOnMainThread(callback: Callable, *args, **kwargs):
    """Accoda una funzione da eseguire sul thread principale al prossimo
    PumpMainThread(). Thread-safe: chiamabile da qualsiasi thread.
    OBBLIGATORIO se il callback di un AsyncTimer/AsyncAfter deve toccare
    lo stato di gioco o chiamare funzioni DRAW/WINDOW (OpenGL non e'
    thread-safe)."""
    _main_thread_queue.put((callback, args, kwargs))


def PumpMainThread(max_items: int | None = None):
    """Esegue sul thread chiamante (deve essere il thread principale)
    le funzioni accodate con RunOnMainThread. Chiamala una volta a
    frame, es. dentro update(dt, events): PE_TIME.PumpMainThread().
    Non blocca mai: se la coda e' vuota ritorna subito."""
    q = _main_thread_queue
    n = 0
    while max_items is None or n < max_items:
        try:
            callback, args, kwargs = q.get_nowait()
        except _queue.Empty:
            break
        callback(*args, **kwargs)
        n += 1


class AsyncTimer:
    """Wrapper leggero su threading.Timer: attesa non bloccante su un
    thread OS reale, interrompibile con Cancel().

    ATTENZIONE: il callback gira su un thread separato dal loop
    OpenGL/SDL2. Non chiamare da qui funzioni DRAW/WINDOW e non toccare
    stato di gioco condiviso — usa RunOnMainThread(fn, ...) dentro il
    callback per rimandare l'esecuzione al thread principale in
    sicurezza.
    """

    __slots__ = ("_timer", "_callback", "_args", "_kwargs", "_alive")

    def __init__(self, seconds: float, callback: Callable, *args, **kwargs):
        self._callback = callback
        self._args = args
        self._kwargs = kwargs
        self._alive = True
        self._timer = threading.Timer(float(seconds), self._run)
        self._timer.daemon = True  # non tiene vivo il processo alla chiusura del gioco
        self._timer.start()

    def _run(self):
        if self._alive:
            self._callback(*self._args, **self._kwargs)

    def Cancel(self):
        """Interrompe il timer se non e' ancora scattato. Se e' gia'
        scattato (o sta scattando in questo istante) non ha effetto:
        e' un limite intrinseco di threading.Timer, non e' possibile
        interrompere un callback a meta' esecuzione."""
        self._alive = False
        self._timer.cancel()

    @property
    def alive(self) -> bool:
        return self._alive and self._timer.is_alive()


def AsyncAfter(seconds: float, callback: Callable, *args, **kwargs) -> AsyncTimer:
    """Esegue callback su un thread separato dopo `seconds`, senza
    bloccare il thread principale/il game loop. Vedi AsyncTimer per gli
    avvertimenti su OpenGL/thread-safety."""
    return AsyncTimer(seconds, callback, *args, **kwargs)


__all__ = [
    "Scheduler", "After", "Every", "Cancel", "CancelAll", "Update", "Count",
    "Countdown", "Cooldown", "Stopwatch",
    "AsyncTimer", "AsyncAfter", "RunOnMainThread", "PumpMainThread",
]
