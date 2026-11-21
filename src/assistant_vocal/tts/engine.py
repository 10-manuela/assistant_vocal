"""Moteur de synthèse vocale : interface commune et découpage en phrases."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol

import numpy as np

from assistant_vocal.config import ROOT_DIR, AppConfig
from assistant_vocal.llm.sentences import sentences


class TTSError(RuntimeError):
    """Erreur de synthèse vocale."""


@dataclass
class Speech:
    text: str
    audio: np.ndarray  # float32, mono
    sample_rate: int
    elapsed: float  # temps de calcul en secondes

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0

    @property
    def rtf(self) -> float:
        """Facteur temps réel : < 1 signifie plus rapide que la parole."""
        return self.elapsed / self.duration if self.duration else 0.0


class Backend(Protocol):
    def load(self) -> None: ...

    def synthesize(self, text: str, emotion: str | None = None) -> tuple[np.ndarray, int]:
        """Retourne (audio float32 mono, fréquence d'échantillonnage)."""
        ...


class TTS:
    def __init__(self, backend: Backend):
        self.backend = backend

    def warmup(self) -> None:
        self.backend.load()
        self.synthesize("Bonjour.")

    def synthesize(self, text: str, emotion: str | None = None) -> Speech:
        start = time.perf_counter()
        audio, rate = self.backend.synthesize(text, emotion)
        elapsed = time.perf_counter() - start
        return Speech(text, np.asarray(audio, dtype=np.float32), rate, elapsed)

    def stream(self, tokens: Iterable[str], emotion: str | None = None) -> Iterator[Speech]:
        """Transforme un flux de texte (LLM) en audio, phrase par phrase."""
        for sentence in sentences(tokens):
            if sentence.strip():
                yield self.synthesize(sentence, emotion)


def build_tts(cfg: AppConfig) -> TTS:
    models_dir = ROOT_DIR / cfg.tts.models_dir
    if cfg.tts.backend == "kokoro":
        from assistant_vocal.tts.kokoro_backend import KokoroBackend

        return TTS(
            KokoroBackend(
                models_dir,
                cfg.tts.kokoro_model,
                cfg.tts.kokoro_voices,
                cfg.tts.voice,
                speed=cfg.tts.speed,
            )
        )
    if cfg.tts.backend == "piper":
        from assistant_vocal.tts.piper_backend import PiperBackend

        return TTS(PiperBackend(cfg.tts.piper_voice, models_dir))
    raise TTSError(f"Moteur TTS inconnu : '{cfg.tts.backend}'")