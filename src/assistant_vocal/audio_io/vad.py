"""Détection de parole (Silero VAD) avec début et fin d'énoncé."""
from __future__ import annotations

import numpy as np
import torch
from silero_vad import load_silero_vad


class SpeechDetector:
    """Traite des blocs de 512 échantillons (16 kHz) et signale début/fin de parole."""

    def __init__(self, sample_rate=16000, threshold=0.5, min_silence_ms=700, block_ms=32):
        assert sample_rate == 16000, "Silero VAD : 16 kHz attendu dans ce module"
        self.threshold = threshold
        self.silence_blocks_needed = max(1, int(min_silence_ms / block_ms))
        self.model = load_silero_vad()
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self.in_speech = False
        self._silence_count = 0

    def probability(self, block: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.model(torch.from_numpy(block), 16000).item())

    def process(self, block: np.ndarray) -> str | None:
        """Retourne 'start', 'end' ou None."""
        is_speech = self.probability(block) >= self.threshold
        if is_speech:
            self._silence_count = 0
            if not self.in_speech:
                self.in_speech = True
                return "start"
        elif self.in_speech:
            self._silence_count += 1
            if self._silence_count >= self.silence_blocks_needed:
                self.in_speech = False
                self._silence_count = 0
                return "end"
        return None