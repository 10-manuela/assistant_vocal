"""Juge : le LLM vérifie des critères précis ; la note et la décision sont calculées par le code."""
from __future__ import annotations

import json
import logging
import re

from assistant_vocal.dialogue.interview import Evaluation, Question
from assistant_vocal.dialogue.text import word_count

log = logging.getLogger("judge")

SYSTEM = (
    "Tu es un recruteur expérimenté, exigeant et impartial. Tu évalues la réponse orale d'un candidat, "
    "transcrite automatiquement : la ponctuation peut être imparfaite, ne la pénalise pas. "
    "Tu juges uniquement le contenu de la réponse, jamais la personne. "
    "Tu réponds uniquement par un objet JSON."
)


def build_prompt(profile_name: str, question: Question, answer: str) -> str:
    criteria = "\n".join(f"{i}. {c}" for i, c in enumerate(question.criteria, 1))
    example = json.dumps({"criteres": [i % 2 == 0 for i in range(len(question.criteria))],
                          "exemple": False, "structure": True, "langage": 1})
    return (
        f"Poste visé : {profile_name}\n"
        f"Question posée : {question.text}\n\n"
        f"Critères attendus :\n{criteria}\n\n"
        f"Réponse du candidat :\n\"\"\"{answer}\"\"\"\n\n"
        "Pour chaque critère, indique true seulement s'il est clairement présent dans la réponse, sinon false. "
        "Sois exigeant : une réponse vague ou hors sujet vaut false.\n"
        "Indique aussi :\n"
        "- \"exemple\" : true si la réponse contient un exemple concret ou un fait précis ;\n"
        "- \"structure\" : true si la réponse est organisée et va à l'essentiel ;\n"
        "- \"langage\" : 0 (confus ou trop familier), 1 (correct) ou 2 (clair et professionnel).\n\n"
        f"Réponds exactement sous cette forme : {example}"
    )


def _bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "vrai", "oui", "1")
    return bool(v)


def parse_evaluation(raw: str, n_criteria: int) -> Evaluation | None:
    """Lit la réponse JSON du LLM. Retourne None si elle est inutilisable."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        met = data["criteres"]
        if not isinstance(met, list) or len(met) != n_criteria:
            return None
        language = int(data.get("langage", 1))
        return Evaluation(
            criteria_met=[_bool(x) for x in met],
            example=_bool(data.get("exemple", False)),
            structured=_bool(data.get("structure", False)),
            language=min(2, max(0, language)),
        )
    except (ValueError, KeyError, TypeError):
        return None


class LLMJudge:
    def __init__(self, llm, min_words: int = 6, attempts: int = 2):
        self.llm = llm
        self.min_words = min_words
        self.attempts = attempts

    def evaluate(self, profile_name: str, question: Question, answer: str) -> Evaluation | None:
        if word_count(answer) < self.min_words:  # trop court : inutile d'interroger le LLM
            return Evaluation([False] * len(question.criteria), language=0, skipped=True)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(profile_name, question, answer)},
        ]
        for attempt in range(self.attempts):
            try:
                raw = self.llm.chat(messages, fmt="json",
                                    options={"temperature": 0.0 if attempt == 0 else 0.3, "num_predict": 200})
            except Exception as exc:  # Ollama injoignable, délai dépassé...
                log.warning("Évaluation impossible : %s", exc)
                return None
            evaluation = parse_evaluation(raw, len(question.criteria))
            if evaluation is not None:
                return evaluation
        log.warning("Réponse du juge inutilisable pour la question %s", question.id)
        return None
