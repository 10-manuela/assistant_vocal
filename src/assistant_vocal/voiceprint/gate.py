"""Filtre de locuteur : ne laisse passer que les voix enregistrées (empreinte vocale).

L'empreinte reste en mémoire pour la durée de la session : rien n'est écrit sur le disque.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OWNER = "owner"
SHORT_POLICIES = ("continuity", "accept", "reject")
MAX_SECONDS = 30  # au-delà, l'empreinte n'est pas plus précise


@dataclass
class GateResult:
    accepted: bool
    score: float | None  # None quand aucune comparaison n'a eu lieu
    speaker: str | None
    reason: str  # no_voice | short | match | mismatch


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


class SpeakerGate:
    def __init__(
        self,
        embedder,
        threshold: float = 0.45,
        sample_rate: int = 16000,
        min_check_seconds: float = 1.0,
        min_enroll_seconds: float = 4.0,
        short_policy: str = "continuity",
    ):
        if short_policy not in SHORT_POLICIES:
            raise ValueError(f"short_policy doit valoir l'une de {SHORT_POLICIES}")
        if embedder.sample_rate != sample_rate:
            raise ValueError(
                f"Le modèle attend {embedder.sample_rate} Hz, le micro fournit {sample_rate} Hz"
            )
        self.embedder = embedder
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_check_samples = int(min_check_seconds * sample_rate)
        self.min_enroll_samples = int(min_enroll_seconds * sample_rate)
        self.short_policy = short_policy
        self.voices: dict[str, np.ndarray] = {}
        self._last_accepted = True

    @property
    def active(self) -> bool:
        """Le filtre ne s'applique qu'une fois au moins une voix enregistrée."""
        return bool(self.voices)

    def _embed(self, audio) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)[-MAX_SECONDS * self.sample_rate:]
        return _unit(self.embedder.embed(audio))

    # --- voix enregistrées -----------------------------------------------

    def enroll(self, name: str, audio) -> None:
        if len(audio) < self.min_enroll_samples:
            raise ValueError(
                f"Enregistrement trop court : {len(audio) / self.sample_rate:.1f} s, "
                f"{self.min_enroll_samples / self.sample_rate:.1f} s minimum"
            )
        self.voices[name] = self._embed(audio)
        self._last_accepted = True

    def remove(self, name: str) -> bool:
        return self.voices.pop(name, None) is not None

    def guests(self) -> list[str]:
        return [n for n in self.voices if n != OWNER]

    def new_guest_name(self) -> str:
        i = 1
        while f"guest{i}" in self.voices:
            i += 1
        return f"guest{i}"

    def clear_guests(self) -> int:
        names = self.guests()
        for n in names:
            del self.voices[n]
        return len(names)

    # --- vérification ----------------------------------------------------

    def check(self, audio, remember: bool = True) -> GateResult:
        """Compare un énoncé aux voix enregistrées.

        remember=False : la décision ne sert pas de référence pour les énoncés courts
        (utilisé pour la vérification d'une interruption).
        """
        if not self.voices:
            return GateResult(True, None, None, "no_voice")

        if len(audio) < self.min_check_samples:
            accepted = {
                "accept": True,
                "reject": False,
                "continuity": self._last_accepted,
            }[self.short_policy]
            return GateResult(accepted, None, None, "short")

        emb = self._embed(audio)
        scores = {name: float(np.dot(ref, emb)) for name, ref in self.voices.items()}
        name, best = max(scores.items(), key=lambda kv: kv[1])
        accepted = best >= self.threshold
        if remember:
            self._last_accepted = accepted
        return GateResult(accepted, best, name if accepted else None,
                          "match" if accepted else "mismatch")


def build_gate(cfg) -> SpeakerGate | None:
    """Construit le filtre depuis la config (None si désactivé).

    Le premier appel télécharge le modèle de locuteur depuis Hugging Face.
    """
    v = cfg.voiceprint
    if not v.enabled:
        return None
    from speakeronnx import SpeakerEmbedder

    return SpeakerGate(
        SpeakerEmbedder(model=v.model),
        threshold=v.threshold,
        sample_rate=cfg.audio.sample_rate,
        min_check_seconds=v.min_check_ms / 1000,
        min_enroll_seconds=v.min_enroll_ms / 1000,
        short_policy=v.short_policy,
    )