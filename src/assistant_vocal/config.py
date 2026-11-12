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


@dataclass
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    temperature: float = 0.7


@dataclass
class AppConfig:
    language: str = "fr"
    log_level: str = "INFO"
    audio: AudioConfig = field(default_factory=AudioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def _build(cls, data: dict):
    """Construit une dataclass en ignorant les clés inconnues."""
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    audio = _build(AudioConfig, raw.pop("audio", {}) or {})
    llm = _build(LLMConfig, raw.pop("llm", {}) or {})
    base = _build(AppConfig, {k: v for k, v in raw.items() if k not in ("audio", "llm")})
    base.audio, base.llm = audio, llm
    return base