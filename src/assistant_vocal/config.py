"""Chargement de la configuration du projet."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"

DEFAULT_SYSTEM_PROMPT = (
    "Tu es un assistant vocal pour un entretien professionnel. "
    "Tu parles à l'oral : réponds en français, en deux ou trois phrases courtes maximum, "
    "sans liste, sans numérotation, sans markdown. "
    "Si le sujet est vaste, donne l'idée principale et propose de continuer."
)


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
    host: str = "http://127.0.0.1:11434"
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
    beam_size: int = 5
    initial_prompt: str | None = None


@dataclass
class TTSConfig:
    backend: str = "piper"
    voice: str = "ff_siwis"
    models_dir: str = "models/tts"
    kokoro_model: str = "kokoro-v1.0.onnx"
    kokoro_voices: str = "voices-v1.0.bin"
    speed: float = 1.0
    piper_voice: str = "fr_FR-siwis-medium"


@dataclass
class VoiceprintConfig:
    enabled: bool = True
    model: str = "wespeaker-resnet34"
    threshold: float = 0.45
    min_check_ms: int = 1000
    min_enroll_ms: int = 4000
    short_policy: str = "continuity"  # continuity | accept | reject
    guest_timeout_s: float = 60.0


@dataclass
class OrchestratorConfig:
    barge_in_ms: int = 300
    verified_barge_in_ms: int = 1000
    min_utterance_ms: int = 300
    preroll_ms: int = 320
    history_limit: int = 12
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass
class AppConfig:
    language: str = "fr"
    log_level: str = "INFO"
    audio: AudioConfig = field(default_factory=AudioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    voiceprint: VoiceprintConfig = field(default_factory=VoiceprintConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)


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
    "voiceprint": VoiceprintConfig,
    "orchestrator": OrchestratorConfig,
}


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    base = _build(AppConfig, {k: v for k, v in raw.items() if k not in _SECTIONS})
    for name, cls in _SECTIONS.items():
        setattr(base, name, _build(cls, raw.get(name) or {}))
    return base