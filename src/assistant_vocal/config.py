"""Chargement de la configuration du projet."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 32
    input_device: int | str | None = None
    output_device: int | str | None = None
    vad_threshold: float = 0.5
    min_silence_ms: int = 700


@dataclass
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    temperature: float = 0.7
    num_ctx: int = 4096
    keep_alive: str = "30m"
    timeout: float = 120.0


@dataclass
class StorageConfig:
    db_path: str = "data/assistant.db"


@dataclass
class STTConfig:
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1


@dataclass
class TTSConfig:
    backend: str = "piper"
    voice: str = "fr_FR-siwis-medium"
    models_dir: str = "models/tts"


@dataclass
class AppConfig:
    language: str = "fr"
    log_level: str = "INFO"
    audio: AudioConfig = field(default_factory=AudioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


def _build(cls, data: dict):
    """Construit une dataclass en ignorant les clés inconnues."""
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


_SECTIONS = {
    "audio": AudioConfig,
    "llm": LLMConfig,
    "storage": StorageConfig,
    "stt": STTConfig,
    "tts": TTSConfig,
}


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    base = _build(AppConfig, {k: v for k, v in raw.items() if k not in _SECTIONS})
    for name, cls in _SECTIONS.items():
        setattr(base, name, _build(cls, raw.get(name) or {}))
    return base
