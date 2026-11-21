"""Moteur Kokoro (ONNX) : voix plus naturelle que Piper, sur CPU."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from assistant_vocal.tts.engine import TTSError


class KokoroBackend:
    def __init__(
        self,
        models_dir: Path,
        model_file: str = "kokoro-v1.0.onnx",
        voices_file: str = "voices-v1.0.bin",
        voice: str = "ff_siwis",
        lang: str = "fr-fr",
        speed: float = 1.0,
    ):
        self.model_path = Path(models_dir) / model_file
        self.voices_path = Path(models_dir) / voices_file
        self.voice = voice
        self.lang = lang
        self.speed = speed
        self._kokoro = None

    def load(self) -> None:
        if self._kokoro is not None:
            return
        for path in (self.model_path, self.voices_path):
            if not path.exists():
                raise TTSError(
                    f"Fichier Kokoro introuvable : {path}\n"
                    "Télécharge-le depuis la release 'model-files-v1.0' du dépôt "
                    "thewh1teagle/kokoro-onnx (voir les commandes curl du module)."
                )
        from kokoro_onnx import Kokoro

        try:
            self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
        except Exception as exc:  # fichier tronqué, protobuf invalide...
            raise TTSError(f"Chargement de Kokoro impossible (fichier incomplet ?) : {exc}") from exc

    def synthesize(self, text: str, emotion: str | None = None) -> tuple[np.ndarray, int]:
        self.load()
        audio, rate = self._kokoro.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang
        )
        return np.asarray(audio, dtype=np.float32), int(rate)
        # Kokoro n'a pas de contrôle d'émotion : `emotion` est ignoré pour l'instant.