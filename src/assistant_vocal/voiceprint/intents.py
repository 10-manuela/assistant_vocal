"""Commandes vocales liées aux voix (provisoire : sera repris par le module `commands`)."""
from __future__ import annotations

import re
import unicodedata

ADD_GUEST = "add_guest"
REMOVE_GUEST = "remove_guest"

_REMOVE = [
    r"\b(retire|retirer|enleve|enlever|supprime|supprimer)\b.*\b(invite|invitee|personne|participant|collegue)\b",
    r"\b(il|elle|l invite|l invitee|mon collegue|ma collegue|la personne)\b.*\b(est parti|est partie|s en va|nous quitte)\b",
]
_ADD = [
    r"\b(ajoute|ajouter|ajoutez|invite|inviter|faire venir|fais venir|faire entrer)\b.*\b(quelqu un|personne|invite|invitee|participant|collegue)\b",
    r"\b(quelqu un|une personne|une autre personne|un invite|mon collegue|ma collegue)\b.*\b(participer|rejoindre|rejoint|se joindre|se joint)\b",
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)  # l'apostrophe devient une espace
    return re.sub(r"\s+", " ", text).strip()


def detect_intent(text: str) -> str | None:
    """Retourne 'add_guest', 'remove_guest' ou None."""
    t = _normalize(text)
    if any(re.search(p, t) for p in _REMOVE):
        return REMOVE_GUEST
    if any(re.search(p, t) for p in _ADD):
        return ADD_GUEST
    return None