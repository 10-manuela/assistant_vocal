"""Capture micro en flux continu, par blocs de taille fixe."""
from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd


class MicCapture:
    """Flux micro. Chaque bloc = block_ms millisecondes, float32 mono."""

    def __init__(self, sample_rate=16000, block_ms=32, device=None):
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_ms / 1000)  # 512 à 16 kHz
        self.device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status):
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # on préfère perdre un bloc que bloquer le micro

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float = 1.0) -> np.ndarray:
        return self._queue.get(timeout=timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()