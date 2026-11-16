"""Lecture audio dans les haut-parleurs."""
from __future__ import annotations

import numpy as np
import sounddevice as sd


def play(audio: np.ndarray, sample_rate: int, device=None, blocking: bool = True) -> None:
    sd.play(audio.astype(np.float32), samplerate=sample_rate, device=device)
    if blocking:
        sd.wait()


def stop() -> None:
    sd.stop()


def beep(freq=440.0, seconds=0.4, sample_rate=16000, device=None) -> None:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    play(0.3 * np.sin(2 * np.pi * freq * t), sample_rate, device)