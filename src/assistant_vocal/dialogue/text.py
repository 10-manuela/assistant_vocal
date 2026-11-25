"""Utilitaires de texte."""
from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Minuscules, sans accents ni ponctuation (l'apostrophe devient une espace)."""
    text = unicodedata.normalize("NFD", (text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(normalize(text).split())


def join_french(items: list[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b et c'."""
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " et " + items[-1]
