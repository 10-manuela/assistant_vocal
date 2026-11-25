"""Reconnaissance des commandes vocales, des oui/non, du prénom, du métier et du poste choisi."""
from __future__ import annotations

import re

from assistant_vocal.dialogue.text import normalize

QUIT, MENU, INTERVIEW, DISCOVERY, REPEAT = "quit", "menu", "interview", "discovery", "repeat"

# Variantes que le STT peut écrire pour « Mervi »
_NAME = re.compile(r"\bmer(?:vi|vy|vie|vis|veille|vil|wi|vey)\b")

_PATTERNS = [
    (QUIT, [
        r"\bau revoir\b",
        r"\b(quitte|quitter|termine|terminer|arrete|arreter|stoppe|stopper)\b.*\b(discussion|conversation|session|entretien|application)\b",
        r"^(mer\w+ )?(quitte|quitter)$",
    ]),
    (MENU, [
        r"\bmenu\b",
        r"\b(changer|change|changez) de mode\b",
        r"\bautre mode\b",
    ]),
    (INTERVIEW, [r"\bentretien\b"]),
    (DISCOVERY, [r"\b(decouverte|decouvrir|metier|metiers|discussion)\b"]),
]

# « Répète » : seulement sous forme courte et exacte, pour ne pas confondre avec une réponse
_REPEAT = re.compile(
    r"^(mer\w+ )?(pouvez vous |peux tu |pourriez vous |pourrais tu )?"
    r"(repeter|repetez|repete)( la question| s il vous plait| s il te plait)*$"
)

_YES = re.compile(r"^(oui|ouais|yes|bien sur|d accord|tout a fait|absolument|exactement|je veux bien|ok|affirmatif)\b")
_NO = re.compile(r"^(non|pas du tout|surtout pas|negatif)\b")


def has_name(text: str) -> bool:
    return _NAME.search(normalize(text)) is not None


def detect_command(text: str, strict: bool = False, max_words: int = 6) -> str | None:
    """Retourne 'quit', 'menu', 'interview', 'discovery', 'repeat' ou None.

    strict=True (pendant l'entretien ou la discussion) : il faut dire « Mervi » ;
    sinon une phrase de réponse pourrait être prise pour une commande.
    Sans le prénom, seules les phrases courtes (<= max_words) sont considérées.
    """
    t = normalize(text)
    if _REPEAT.match(t):
        return REPEAT
    named = _NAME.search(t) is not None
    if not named and (strict or len(t.split()) > max_words):
        return None
    for command, patterns in _PATTERNS:
        if any(re.search(p, t) for p in patterns):
            return command
    return None


def is_yes(text: str, max_words: int = 4) -> bool:
    t = normalize(text)
    return len(t.split()) <= max_words and _YES.match(t) is not None


def is_no(text: str, max_words: int = 4) -> bool:
    t = normalize(text)
    return len(t.split()) <= max_words and _NO.match(t) is not None


_NAME_PATTERNS = re.compile(
    r"(?:je m['’ ]?appelle|moi c['’ ]?est|mon pr[ée]nom est|mon nom est)\s+([A-Za-zÀ-ÿ'’-]{2,20})",
    re.IGNORECASE,
)
_NOT_NAMES = {"un", "une", "le", "la", "les", "etudiant", "etudiante", "ici", "pas"}


def extract_name(text: str) -> str | None:
    """Prénom annoncé dans la présentation (« je m'appelle Manuela »)."""
    m = _NAME_PATTERNS.search(text or "")
    if not m:
        return None
    name = m.group(1).strip("'’-")
    if normalize(name) in _NOT_NAMES:
        return None
    return name[:1].upper() + name[1:].lower()


_JOB_AFTER_NOUN = re.compile(r"\b(?:m[ée]tiers?|postes?|professions?)\s+(?:de |d['’ ]|du |des |en )?(.+)$")
_JOB_AFTER_VERB = re.compile(
    r"\b(?:parler|parlons|discuter|discutons|d[ée]couvrir|conna[iî]tre|savoir|int[ée]ress[ée]e? par|int[ée]ress[ée]e? [àa])\s+"
    r"(?:(?:de|du|des|sur|le|la|les|un|une)\s+|(?:d|l)['’]\s*)*(.+)$"
)
_POLITE = re.compile(r"\s*(s['’ ]il (?:te|vous) pla[iî]t|merci)\s*$")


def extract_job(text: str, max_words: int = 8) -> str | None:
    """Métier cité dans la phrase (« je voudrais découvrir le métier de médiateur culturel »)."""
    t = (text or "").lower().strip().strip(".!?,;:")
    t = _POLITE.sub("", t).strip(" ,.")
    m = _JOB_AFTER_NOUN.search(t) or _JOB_AFTER_VERB.search(t)
    job = (m.group(1) if m else t).strip(" ,.")
    job = re.sub(r"^(?:le |la |les |l['’])", "", job).strip()
    if not job or len(job.split()) > max_words:
        return None
    return job


_ORDINALS = {"premier": 0, "premiere": 0, "deuxieme": 1, "second": 1, "seconde": 1,
             "troisieme": 2, "quatrieme": 3, "dernier": -1, "derniere": -1}


def match_profile(text: str, offered: list[dict]) -> dict | None:
    """Poste choisi parmi ceux proposés : mots-clés, ou « le premier », « le deuxième »..."""
    t = f" {normalize(text)} "
    scores = []
    for p in offered:
        s = sum(1 for kw in p["mots_cles"] if f" {normalize(kw)} " in t)
        scores.append(s)
    best = max(scores, default=0)
    if best > 0 and scores.count(best) == 1:
        return offered[scores.index(best)]
    if best == 0:
        for word in t.split():
            if word in _ORDINALS:
                idx = _ORDINALS[word]
                if -len(offered) <= idx < len(offered):
                    return offered[idx]
    return None
