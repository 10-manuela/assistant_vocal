"""Moteur Piper : voix légère sur CPU."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from assistant_vocal.tts.engine import TTSError


class PiperBackend:
    def __init__(self, voice: str, models_dir: Path):
        self.voice_name = voice
        self.model_path = Path(models_dir) / f"{voice}.onnx"
        self._voice = None

    def load(self) -> None:
        if self._voice is not None:
            return
        if not self.model_path.exists():
            raise TTSError(
                f"Voix introuvable : {self.model_path}\n"
                f"Télécharge-la : python -m piper.download_voices {self.voice_name} "
                f"--download-dir {self.model_path.parent}"
            )
        from piper import PiperVoice

        self._voice = PiperVoice.load(str(self.model_path))

    def synthesize(self, text: str, emotion: str | None = None) -> tuple[np.ndarray, int]:
        self.load()
        parts, rate = [], self._voice.config.sample_rate
        for chunk in self._voice.synthesize(text):
            parts.append(chunk.audio_float_array)
            rate = chunk.sample_rate
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return audio, rate  # Piper ignore `emotion` : il n'est pas expressif