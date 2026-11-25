"""Étapes que le chef de dialogue demande à l'orchestrateur d'exécuter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Say:
    """Phrase préparée, jouée telle quelle (sans passer par le LLM)."""
    text: str
    record: bool = False  # True : ajoutée à la conversation enregistrée


@dataclass
class Generate:
    """Le LLM répond en streaming, avec ce rôle système."""
    system: str
    instruction: str | None = None  # consigne cachée, non enregistrée


@dataclass
class Session:
    """Ouvre une nouvelle conversation enregistrée."""
    title: str


@dataclass
class RecordUser:
    """Enregistre la phrase de l'utilisateur dans la conversation."""


@dataclass
class Compute:
    """Étapes calculées au moment de l'exécution (ex. : attendre l'analyse)."""
    fn: Callable[[], list]


@dataclass
class Quit:
    """Termine la discussion."""
