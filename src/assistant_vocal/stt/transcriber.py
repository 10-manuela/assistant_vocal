"""Transcription voix vers texte avec faster-whisper."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from assistant_vocal.config import AppConfig

SAMPLE_RATE = 16000


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float
    audio_seconds: float
    elapsed: float  # temps de calcul en secondes

    @property
    def rtf(self) -> float:
        """Facteur temps réel : < 1 signifie plus rapide que la parole."""
        return self.elapsed / self.audio_seconds if self.audio_seconds else 0.0


class Transcriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "fr",
        beam_size: int = 1,
        initial_prompt: str | None = None,
        model=None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self._model = model  # injectable pour les tests
        self.initial_prompt = initial_prompt

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "Transcriber":
        return cls(
            model_size=cfg.stt.model,
            device=cfg.stt.device,
            compute_type=cfg.stt.compute_type,
            language=cfg.language,
            beam_size=cfg.stt.beam_size,
            initial_prompt=cfg.stt.initial_prompt or None,
        )

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def warmup(self) -> None:
        """Charge le modèle et fait un premier passage à vide."""
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Transcript:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"{SAMPLE_RATE} Hz attendu, reçu {sample_rate} Hz")
        audio = np.asarray(audio, dtype=np.float32)
        model = self._load()

        start = time.perf_counter()
        segments, info = model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,  # évite les phrases inventées sur du silence
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        elapsed = time.perf_counter() - start

        return Transcript(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            audio_seconds=len(audio) / SAMPLE_RATE,
            elapsed=elapsed,
        )