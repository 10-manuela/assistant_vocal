"""Liste et sélection des périphériques audio."""
from __future__ import annotations

import sounddevice as sd


def list_devices() -> list[dict]:
    """Retourne tous les périphériques avec leurs capacités."""
    return [
        {
            "index": i,
            "name": d["name"],
            "inputs": d["max_input_channels"],
            "outputs": d["max_output_channels"],
            "default_rate": d["default_samplerate"],
        }
        for i, d in enumerate(sd.query_devices())
    ]


def default_input() -> int | None:
    idx = sd.default.device[0]
    return idx if idx is not None and idx >= 0 else None


def default_output() -> int | None:
    idx = sd.default.device[1]
    return idx if idx is not None and idx >= 0 else None