"""Découpe un flux de morceaux de texte en phrases complètes."""
from __future__ import annotations

import re
from typing import Iterable, Iterator

_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


def sentences(tokens: Iterable[str], min_chars: int = 15) -> Iterator[str]:
    """Émet chaque phrase dès qu'elle est terminée.

    Les phrases trop courtes (« Oui. ») sont regroupées avec la suivante pour
    que la voix ne soit pas hachée.
    """
    buf = ""
    for token in tokens:
        buf += token
        while True:
            cut = next((m for m in _BOUNDARY.finditer(buf) if m.start() >= min_chars), None)
            if cut is None:
                break
            yield buf[: cut.start()].strip()
            buf = buf[cut.end():]
    if buf.strip():
        yield buf.strip()