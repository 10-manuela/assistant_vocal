"""Mervi : chef de dialogue. Décide ce qu'elle dit selon le mode (menu, entretien, découverte)."""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from assistant_vocal.config import DialogueConfig
from assistant_vocal.dialogue import commands as cmd
from assistant_vocal.dialogue.interview import (
    ORDINALS, build_scenario, build_session, compute_result, format_verdict,
)
from assistant_vocal.dialogue.steps import Compute, Generate, Quit, RecordUser, Say, Session
from assistant_vocal.dialogue.text import join_french

ACKS = ["Très bien, merci.", "D'accord, je note.", "Merci pour cette réponse.",
        "Je vois, merci.", "Bien, passons à la suite."]
FOLLOW_UPS = ["Pouvez-vous développer, avec un exemple concret ?",
              "Pouvez-vous préciser votre réponse, avec un exemple ?"]

DISCOVERY_SYSTEM = (
    "Tu es Mervi, une assistante vocale chaleureuse. Tu discutes avec ton interlocuteur du métier suivant : {job}. "
    "Tu le tutoies. Tu parles à l'oral : deux ou trois phrases courtes, sans liste, sans numérotation, sans markdown. "
    "Tu poses une seule question à la fois pour approfondir : missions, compétences, formations, quotidien, "
    "difficultés, évolutions, débouchés. Tu développes ses réponses avec lui et tu t'adaptes à ce qui l'intéresse. "
    "Si tu n'es pas sûre d'une information, notamment des chiffres comme les salaires, dis-le clairement plutôt que d'inventer."
)
DISCOVERY_KICKOFF = ("Commence la discussion : présente ce métier en une phrase, "
                     "puis pose-moi une première question pour savoir ce qui m'intéresse.")

CONFIRM_ACTIONS = {
    cmd.QUIT: "quitter la discussion",
    cmd.MENU: "revenir au menu",
    cmd.INTERVIEW: "recommencer un nouvel entretien",
    cmd.DISCOVERY: "passer à la découverte d'un métier",
}


class Mode(str, Enum):
    MENU = "menu"
    INTERVIEW_SETUP = "interview_setup"
    INTERVIEW = "interview"
    VERDICT = "verdict"  # questions terminées, bilan pas encore prononcé
    DISCOVERY_SETUP = "discovery_setup"
    DISCOVERY = "discovery"


class Mervi:
    def __init__(self, content: dict, judge, settings: DialogueConfig | None = None,
                 rng: random.Random | None = None, executor=None):
        self.content = content
        self.judge = judge
        self.s = settings or DialogueConfig()
        self.rng = rng or random.Random(self.s.seed)
        self.executor = executor or ThreadPoolExecutor(max_workers=1)
        self.mode = Mode.MENU
        self.name: str | None = None
        self.scenario = None
        self.session = None
        self.job: str | None = None
        self._confirm: str | None = None

    # --- accueil et menu -------------------------------------------------

    def greeting_steps(self, ask_voice: bool = True) -> list:
        text = ("Bonjour, je suis Mervi, ton assistante vocale. Je suis là pour échanger avec toi sur un sujet précis, "
                "te faire passer un entretien d'embauche, ou t'aider à découvrir un métier.")
        if ask_voice:
            text += " Avant de commencer, présente-toi en quelques phrases, pour que je reconnaisse ta voix."
        return [Say(text)]

    def set_user_intro(self, text: str) -> None:
        self.name = cmd.extract_name(text) or self.name

    def on_enrolled(self) -> list:
        thanks = f"Merci {self.name}." if self.name else "Merci."
        return [Say(f"{thanks} J'ai bien enregistré ta voix.")] + self.menu_steps()

    def menu_steps(self, polite: bool = False) -> list:
        self.mode = Mode.MENU
        if polite:  # juste après un entretien
            return [Say("Souhaitez-vous refaire un entretien, découvrir un métier, ou quitter ? "
                        "Dites : Mervi, mode entretien ; Mervi, mode découverte ; ou Mervi, au revoir.")]
        return [Say(
            "Voici ce que je peux faire avec toi. "
            "Premier mode : l'entretien professionnel. Je joue le rôle d'une recruteuse, je te pose dix questions, "
            "puis je te dis si ta candidature est retenue. Pour le lancer, dis : Mervi, mode entretien. "
            "Deuxième mode : la découverte d'un métier. Nous discutons ensemble du métier de ton choix. "
            "Pour le lancer, dis : Mervi, mode découverte. "
            "À tout moment, dis : Mervi, menu, pour changer de mode, ou : Mervi, au revoir, pour quitter. "
            "Quel mode choisis-tu ?")]

    # --- point d'entrée --------------------------------------------------

    def handle(self, text: str) -> list:
        text = (text or "").strip()
        if self.mode == Mode.VERDICT:  # bilan interrompu : on le reprend
            return [Compute(self._deliver_verdict)]

        if self._confirm is not None:
            steps = self._resolve_confirm(text)
            if steps is not None:
                return steps

        strict = self.mode in (Mode.INTERVIEW, Mode.DISCOVERY)
        command = cmd.detect_command(text, strict=strict)
        if command is not None and not self._ignored_here(command):
            steps = self._on_command(command)
            if steps is not None:
                return steps

        if self.mode == Mode.MENU:
            return [Say("Je n'ai pas compris. Dis : Mervi, mode entretien, ou : Mervi, mode découverte.")]
        if self.mode == Mode.INTERVIEW_SETUP:
            return self._choose_profile(text)
        if self.mode == Mode.INTERVIEW:
            return self._answer(text)
        if self.mode == Mode.DISCOVERY_SETUP:
            return self._choose_job(text)
        return [RecordUser(), Generate(DISCOVERY_SYSTEM.format(job=self.job))]

    # --- commandes -------------------------------------------------------

    def _ignored_here(self, command: str) -> bool:
        return ((command == cmd.INTERVIEW and self.mode == Mode.INTERVIEW_SETUP)
                or (command == cmd.DISCOVERY and self.mode == Mode.DISCOVERY_SETUP))

    def _on_command(self, command: str) -> list | None:
        if command == cmd.REPEAT:
            if self.mode == Mode.INTERVIEW:
                return [Say(self._question_text(self.session.index))]
            if self.mode == Mode.MENU:
                return self.menu_steps()
            return None  # en discussion : traité comme une phrase normale
        # En entretien, ces commandes feraient perdre la progression : on confirme à voix haute
        if self.mode == Mode.INTERVIEW and command in CONFIRM_ACTIONS:
            self._confirm = command
            return [Say(f"Voulez-vous vraiment abandonner l'entretien en cours pour {CONFIRM_ACTIONS[command]} ? "
                        "Répondez par oui ou par non.")]
        return self._perform(command)

    def _resolve_confirm(self, text: str) -> list | None:
        command, self._confirm = self._confirm, None
        if cmd.is_yes(text):
            return self._perform(command)
        if cmd.is_no(text):
            steps = [Say("D'accord, nous continuons.")]
            if self.mode == Mode.INTERVIEW:
                steps.append(Say(self._question_text(self.session.index)))
            return steps
        return None  # ni oui ni non : la phrase est traitée normalement

    def _perform(self, command: str) -> list:
        if command == cmd.QUIT:
            return [Say("Merci pour cet échange, et bonne continuation ! À bientôt."), Quit()]
        if command == cmd.MENU:
            self.session = self.scenario = None
            return self.menu_steps()
        if command == cmd.INTERVIEW:
            return self._start_interview_setup()
        return self._start_discovery_setup()

    # --- mode entretien --------------------------------------------------

    def _hello(self) -> str:
        return f"Enchantée {self.name}. " if self.name else ""

    def _start_interview_setup(self) -> list:
        self.scenario = build_scenario(self.content, self.rng)
        self.session = None
        self.mode = Mode.INTERVIEW_SETUP
        company = self.scenario.company
        names = join_french([p["nom"] for p in self.scenario.offered])
        return [
            Session(f"Entretien chez {company['nom']}"),
            Say("Très bien, passons à l'entretien. Nous allons nous vouvoyer, comme en situation réelle. "
                f"{self._hello()}Je suis recruteuse chez {company['nom']}. {company['pitch']} "
                f"Nous recherchons actuellement quatre profils : {names}. "
                "Pour quel poste souhaitez-vous candidater ?", record=True),
        ]

    def _choose_profile(self, text: str) -> list:
        profile = cmd.match_profile(text, self.scenario.offered)
        if profile is None:
            names = join_french([p["court"] for p in self.scenario.offered])
            return [Say(f"Je n'ai pas bien saisi. Quel poste vous intéresse : {names} ?")]
        self.session = build_session(self.content, self.scenario, profile, self.rng)
        self.mode = Mode.INTERVIEW
        criteria = join_french(profile["criteres"])
        return [
            RecordUser(),
            Say(f"Très bien, vous candidatez au poste de {profile['nom']}. Pour ce poste, j'évaluerai surtout : "
                f"{criteria}. Je vais vous poser {len(self.session.questions)} questions, et je vous donnerai ma "
                "décision à la fin. Répondez comme dans un vrai entretien.", record=True),
            Say(self._question_text(0), record=True),
        ]

    def _question_text(self, index: int) -> str:
        s = self.session
        last = index == len(s.questions) - 1
        label = "Dernière question" if last else (
            f"{ORDINALS[index]} question" if index < len(ORDINALS) else f"Question {index + 1}")
        return f"{label}. {s.questions[index].text}"

    def _answer(self, text: str) -> list:
        s = self.session
        if len(text.split()) < self.s.min_answer_words and not s.follow_up_done:
            s.follow_up_done = True
            s.partial = text
            return [RecordUser(), Say(self.rng.choice(FOLLOW_UPS), record=True)]

        answer = f"{s.partial} {text}".strip()
        s.partial, s.follow_up_done = "", False
        s.answers.append(answer)
        s.futures.append(self.executor.submit(self.judge.evaluate, s.profile["nom"], s.current, answer))

        steps: list = [RecordUser()]
        if s.index + 1 < len(s.questions):
            s.index += 1
            steps.append(Say(f"{self.rng.choice(ACKS)} {self._question_text(s.index)}", record=True))
        else:
            self.mode = Mode.VERDICT
            steps += [
                Say("Merci, c'était la dernière question. Je vais maintenant analyser vos réponses, "
                    "un instant s'il vous plaît.", record=True),
                Compute(self._deliver_verdict),
            ]
        return steps

    def _deliver_verdict(self) -> list:
        s = self.session
        evaluations = []
        deadline = time.monotonic() + self.s.judge_timeout_s
        for future in s.futures:
            try:
                evaluations.append(future.result(timeout=max(0.0, deadline - time.monotonic())))
            except Exception:
                evaluations.append(None)
        result = compute_result(
            s, evaluations,
            w_content=self.s.weight_content, w_form=self.s.weight_form, w_language=self.s.weight_language,
            w_technical=self.s.weight_profile_questions, min_evaluated_ratio=self.s.min_evaluated_ratio,
        )
        verdict = format_verdict(result, s, self.name)
        self.last_result = result
        self.session = self.scenario = None
        return [Say(verdict, record=True)] + self.menu_steps(polite=True)

    # --- mode découverte -------------------------------------------------

    def _start_discovery_setup(self) -> list:
        self.mode = Mode.DISCOVERY_SETUP
        self.session = self.scenario = None
        return [Say("D'accord, parlons métiers. Quel métier souhaites-tu découvrir ?")]

    def _choose_job(self, text: str) -> list:
        job = cmd.extract_job(text)
        if job is None:
            return [Say("Je n'ai pas bien compris le métier. Peux-tu me le redire ?")]
        self.job = job
        self.mode = Mode.DISCOVERY
        return [Session(f"Découverte du métier : {job}"), RecordUser(),
                Generate(DISCOVERY_SYSTEM.format(job=job), DISCOVERY_KICKOFF)]

    def close(self) -> None:
        shutdown = getattr(self.executor, "shutdown", None)
        if shutdown:
            shutdown(wait=False)


def build_mervi(cfg, llm_factory=None) -> Mervi:
    """Construit Mervi depuis la config (le juge a son propre client Ollama)."""
    from assistant_vocal.dialogue.interview import load_content
    from assistant_vocal.dialogue.judge import LLMJudge
    from assistant_vocal.llm import LLMClient

    judge_llm = (llm_factory or LLMClient.from_config)(cfg)
    return Mervi(load_content(), LLMJudge(judge_llm, cfg.dialogue.min_answer_words), cfg.dialogue)
