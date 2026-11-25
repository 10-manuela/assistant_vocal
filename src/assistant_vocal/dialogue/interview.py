"""Entretien : contenu, tirage aléatoire des questions, notation et décision."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from assistant_vocal.dialogue.text import join_french

CONTENT_PATH = Path(__file__).parent / "content" / "interview.yaml"
ORDINALS = ["Première", "Deuxième", "Troisième", "Quatrième", "Cinquième",
            "Sixième", "Septième", "Huitième", "Neuvième", "Dixième"]


@dataclass
class Question:
    id: str
    text: str
    criteria: list[str]
    category: str  # parcours | motivation | comportemental | situation | projet | profil

    @property
    def technical(self) -> bool:
        return self.category == "profil"


@dataclass
class Evaluation:
    criteria_met: list[bool]
    example: bool = False
    structured: bool = False
    language: int = 1  # 0 confus ou familier, 1 correct, 2 clair et professionnel
    skipped: bool = False  # réponse trop courte : non soumise au LLM

    def score(self, w_content: float = 0.6, w_form: float = 0.2, w_language: float = 0.2) -> float:
        content = sum(self.criteria_met) / len(self.criteria_met) if self.criteria_met else 0.0
        form = (int(self.example) + int(self.structured)) / 2
        return w_content * content + w_form * form + w_language * (self.language / 2)


@dataclass
class Scenario:
    company: dict
    offered: list[dict]


@dataclass
class InterviewSession:
    scenario: Scenario
    profile: dict
    questions: list[Question]
    index: int = 0
    answers: list[str] = field(default_factory=list)
    futures: list = field(default_factory=list)
    partial: str = ""
    follow_up_done: bool = False

    @property
    def current(self) -> Question:
        return self.questions[self.index]


@dataclass
class Result:
    retained: bool | None  # None : trop peu de réponses évaluées pour décider
    score: float  # 0-100
    technical: float | None  # 0-100
    threshold: float  # 0-100
    evaluated: int
    total: int
    knockout: bool  # niveau technique minimal non atteint
    strengths: list[str]
    gaps: list[str]
    language: float | None  # 0-1


# --- contenu -----------------------------------------------------------------

def load_content(path: Path | str | None = None) -> dict:
    content = yaml.safe_load(Path(path or CONTENT_PATH).read_text(encoding="utf-8"))
    validate_content(content)
    return content


def validate_content(content: dict) -> None:
    """Détecte les erreurs de saisie dans le fichier YAML avant l'entretien."""
    profiles = content.get("profiles") or []
    if not any(p.get("obligatoire") for p in profiles):
        raise ValueError("Aucun profil marqué « obligatoire: true »")
    needed = content.get("nb_profils", 4)
    if len(profiles) < needed:
        raise ValueError(f"Il faut au moins {needed} profils")
    composition = content["composition"]
    for p in profiles:
        for key in ("id", "nom", "court", "mots_cles", "criteres", "seuil", "eliminatoire", "questions"):
            if key not in p:
                raise ValueError(f"Profil {p.get('id', '?')} : champ « {key} » manquant")
        if len(p["questions"]) < composition.count("profil"):
            raise ValueError(f"Profil {p['id']} : pas assez de questions")
    for category in set(composition) - {"profil"}:
        if len(content["generic"].get(category, [])) < composition.count(category):
            raise ValueError(f"Catégorie « {category} » : pas assez de questions")
    for q in [q for p in profiles for q in p["questions"]] + [q for v in content["generic"].values() for q in v]:
        if not q.get("criteres"):
            raise ValueError(f"Question {q.get('id', '?')} : aucun critère")


# --- tirage ------------------------------------------------------------------

def build_scenario(content: dict, rng: random.Random) -> Scenario:
    """Entreprise au hasard, et 4 profils recherchés dont le profil obligatoire."""
    company = rng.choice(content["companies"])
    profiles = content["profiles"]
    required = [p for p in profiles if p.get("obligatoire")]
    others = [p for p in profiles if not p.get("obligatoire")]
    offered = required + rng.sample(others, content.get("nb_profils", 4) - len(required))
    rng.shuffle(offered)
    return Scenario(company, offered)


def build_session(content: dict, scenario: Scenario, profile: dict, rng: random.Random) -> InterviewSession:
    """Tire les questions selon la composition de l'entretien (sans doublon)."""
    pools: dict[str, list[dict]] = {
        cat: rng.sample(qs, len(qs)) for cat, qs in content["generic"].items()
    }
    pools["profil"] = rng.sample(profile["questions"], len(profile["questions"]))
    questions = []
    for category in content["composition"]:
        q = pools[category].pop()
        questions.append(Question(q["id"], q["text"], list(q["criteres"]), category))
    return InterviewSession(scenario, profile, questions)


# --- notation ----------------------------------------------------------------

def compute_result(session: InterviewSession, evaluations: list[Evaluation | None], *,
                   w_content=0.6, w_form=0.2, w_language=0.2, w_technical=2.0,
                   min_evaluated_ratio=0.7) -> Result:
    """Score pondéré + niveau technique minimal : la décision est calculée, pas improvisée."""
    scored = []  # (question, évaluation, score 0-1)
    for q, ev in zip(session.questions, evaluations):
        if ev is not None:
            scored.append((q, ev, ev.score(w_content, w_form, w_language)))

    total = len(session.questions)
    threshold = session.profile["seuil"] * 100
    if not scored:
        return Result(None, 0.0, None, threshold, 0, total, False, [], [], None)

    weights = [w_technical if q.technical else 1.0 for q, _, _ in scored]
    score = 100 * sum(w * s for w, (_, _, s) in zip(weights, scored)) / sum(weights)
    tech = [s for q, _, s in scored if q.technical]
    technical = 100 * sum(tech) / len(tech) if tech else None
    knockout = technical is not None and technical < session.profile["eliminatoire"] * 100

    # Points forts : critères couverts dans les meilleures réponses ; à travailler : les manquants des pires
    strengths: list[str] = []
    for q, ev, _ in sorted(scored, key=lambda x: -x[2]):
        strengths += [c for c, ok in zip(q.criteria, ev.criteria_met) if ok and c not in strengths]
    gaps: list[str] = []
    for q, ev, _ in sorted(scored, key=lambda x: x[2]):
        gaps += [c for c, ok in zip(q.criteria, ev.criteria_met) if not ok and c not in gaps]

    if len(scored) / total < min_evaluated_ratio:
        retained = None
    else:
        retained = score >= threshold and not knockout
    return Result(retained, score, technical, threshold, len(scored), total, knockout,
                  strengths[:2], gaps[:2], sum(ev.language for _, ev, _ in scored) / (2 * len(scored)))


def format_verdict(result: Result, session: InterviewSession, name: str | None = None) -> str:
    """Bilan parlé, construit à partir des chiffres et des critères du poste."""
    poste = session.profile["nom"]
    company = session.scenario.company["nom"]
    hello = f"{name}, " if name else ""

    if result.retained is None:
        return (f"{hello}je n'ai pu analyser que {result.evaluated} réponses sur {result.total}. "
                "C'est trop peu pour rendre une décision fiable, donc je préfère ne pas me prononcer. "
                "Vous pouvez refaire l'entretien quand vous voulez.")

    if result.retained:
        parts = [f"{hello}après analyse de vos réponses, j'ai le plaisir de vous annoncer que votre candidature "
                 f"est retenue pour le poste de {poste}, chez {company}."]
    else:
        parts = [f"{hello}après analyse de vos réponses, je ne peux pas retenir votre candidature pour le poste "
                 f"de {poste}, chez {company}, pour le moment."]
        if result.knockout:
            parts.append("Vos réponses techniques n'ont pas atteint le niveau minimal que je demande pour ce poste.")
    parts.append(f"Votre score global est de {round(result.score)} sur 100, le seuil étant de {round(result.threshold)}.")
    if result.strengths:
        parts.append("Vos points forts : " + join_french(result.strengths) + ".")
    if result.gaps:
        parts.append("Les points à travailler : " + join_french(result.gaps) + ".")
    if result.language is not None:
        if result.language >= 0.75:
            parts.append("Votre expression est claire et professionnelle.")
        elif result.language < 0.4:
            parts.append("Soignez la clarté de votre expression et votre vocabulaire professionnel.")
        else:
            parts.append("Votre expression est correcte, avec une marge de progression en précision.")
    parts.append("Rappelez-vous qu'il s'agit d'une simulation d'entraînement.")
    return " ".join(parts)
