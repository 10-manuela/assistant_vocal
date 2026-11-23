"""Lecture en file d'attente : la phrase N+1 se prépare pendant que la phrase N est jouée."""
from __future__ import annotations

import logging
import queue
import threading

from assistant_vocal.audio_io import play as _play
from assistant_vocal.audio_io import stop as _stop

log = logging.getLogger("speaker")


class Speaker:
    def __init__(self, device=None, play_fn=None, stop_fn=None, max_pending: int = 2):
        self._play = play_fn or (lambda audio, rate: _play(audio, rate, device))
        self._stop = stop_fn or _stop
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self.on_start = None  # appelé avant chaque phrase : on_start(turn, speech)
        threading.Thread(target=self._loop, daemon=True, name="speaker").start()

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                turn, speech = item
                if turn.cancel.is_set():
                    continue
                if self.on_start:
                    self.on_start(turn, speech)
                self._play(speech.audio, speech.sample_rate)
                if not turn.cancel.is_set():  # phrase jouée en entier
                    turn.played.append(speech.text)
            except Exception:
                log.exception("Erreur de lecture audio")
            finally:
                self._queue.task_done()

    def say(self, turn, speech) -> bool:
        """Met une phrase en file. Bloque si la file est pleine. False si le tour est annulé."""
        while not turn.cancel.is_set():
            try:
                self._queue.put((turn, speech), timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def wait(self) -> None:
        """Attend que tout ce qui est en file ait été joué (ou abandonné après stop)."""
        self._queue.join()

    def stop(self) -> None:
        """Coupe le son en cours et vide la file."""
        self._stop()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()

    def close(self) -> None:
        self.stop()
        self._queue.put(None)