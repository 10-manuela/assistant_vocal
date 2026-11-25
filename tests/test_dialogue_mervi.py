import random
from concurrent.futures import Future

import pytest

from assistant_vocal.config import DialogueConfig
from assistant_vocal.dialogue import Compute, Generate, Mervi, Mode, Quit, RecordUser, Say, Session
from assistant_vocal.dialogue.interview import Evaluation, load_content

LONG = "Dans mon dernier projet, j'ai dû organiser le travail de quatre personnes et livrer à temps."


class SyncExecutor:
    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


class FakeJudge:
    def __init__(self, good=True):
        self.good = good
        self.calls = []

    def evaluate(self, profile_name, question, answer):
        self.calls.append((profile_name, question.id, answer))
        n = len(question.criteria)
        if self.good:
            return Evaluation([True] * n, True, True, 2)
        return Evaluation([False] * n, False, False, 0)


def make(good=True, seed=3):
    judge = FakeJudge(good)
    mervi = Mervi(load_content(), judge, DialogueConfig(), random.Random(seed), SyncExecutor())
    return mervi, judge


def says(steps):
    return " ".join(s.text for s in steps if isinstance(s, Say))


def kinds(steps):
    return [type(s).__name__ for s in steps]


def run_compute(steps):
    """Exécute les étapes Compute comme le fait l'orchestrateur."""
    out = []
    for s in steps:
        out += s.fn() if isinstance(s, Compute) else [s]
    return out


def start_interview(mervi, choice="développeuse IA"):
    mervi.handle("Mervi, mode entretien")
    return mervi.handle(choice)


def finish_interview(mervi):
    steps = None
    for _ in range(10):
        steps = mervi.handle(LONG)
    return steps


# --- accueil et menu --------------------------------------------------------

def test_greeting_introduces_mervi_and_asks_for_the_voice():
    mervi, _ = make()
    text = says(mervi.greeting_steps(ask_voice=True))
    assert "Mervi" in text and "assistante vocale" in text
    assert "entretien d'embauche" in text and "découvrir un métier" in text
    assert "présente-toi" in text
    assert "présente-toi" not in says(mervi.greeting_steps(ask_voice=False))


def test_menu_after_enrollment_tells_what_to_say():
    mervi, _ = make()
    mervi.set_user_intro("Bonjour, je m'appelle Manuela et je suis étudiante")
    text = says(mervi.on_enrolled())
    assert "Manuela" in text
    for phrase in ("Mervi, mode entretien", "Mervi, mode découverte", "Mervi, menu", "Mervi, au revoir"):
        assert phrase in text
    assert mervi.mode == Mode.MENU


def test_unclear_request_in_the_menu_is_repeated_helpfully():
    mervi, _ = make()
    assert "mode entretien" in says(mervi.handle("euh bonjour"))
    assert mervi.mode == Mode.MENU


# --- entretien --------------------------------------------------------------

def test_interview_setup_presents_company_and_four_profiles():
    mervi, _ = make()
    steps = mervi.handle("Mervi, mode entretien")
    assert isinstance(steps[0], Session) and steps[1].record
    text = says(steps)
    assert "vouvoyer" in text and "recruteuse chez" in text
    assert "développeur ou développeuse en intelligence artificielle" in text  # toujours proposé
    assert mervi.mode == Mode.INTERVIEW_SETUP
    assert len(mervi.scenario.offered) == 4


def test_unrecognised_position_is_asked_again():
    mervi, _ = make()
    mervi.handle("mode entretien")
    steps = mervi.handle("euh je ne sais pas")
    assert "Quel poste" in says(steps) and mervi.mode == Mode.INTERVIEW_SETUP


def test_choosing_the_ai_profile_announces_criteria_and_first_question():
    mervi, _ = make()
    steps = start_interview(mervi)
    text = says(steps)
    assert mervi.mode == Mode.INTERVIEW
    assert "j'évaluerai surtout" in text and "10 questions" in text
    assert "Première question" in text
    assert mervi.session.profile["id"] == "ia"
    assert kinds(steps)[0] == "RecordUser"


def test_each_answer_is_judged_and_followed_by_the_next_question():
    mervi, judge = make()
    start_interview(mervi)
    steps = mervi.handle(LONG)
    assert len(judge.calls) == 1 and judge.calls[0][2] == LONG
    assert "Deuxième question" in says(steps)
    assert kinds(steps) == ["RecordUser", "Say"]


def test_short_answer_triggers_one_follow_up_then_is_combined():
    mervi, judge = make()
    start_interview(mervi)
    steps = mervi.handle("oui")
    assert "développer" in says(steps) or "préciser" in says(steps)
    assert mervi.session.index == 0 and judge.calls == []
    mervi.handle("non")  # toujours court : on passe, sans redemander
    assert mervi.session.index == 1
    assert judge.calls[0][2] == "oui non"


def test_full_interview_ends_with_a_verdict_and_back_to_the_menu():
    mervi, judge = make(good=True)
    start_interview(mervi)
    steps = finish_interview(mervi)
    assert mervi.mode == Mode.VERDICT and "analyser" in says(steps)
    assert isinstance(steps[-1], Compute)
    final = run_compute(steps)
    assert len(judge.calls) == 10
    text = says(final)
    assert "est retenue" in text and "sur 100" in text
    assert "Mervi, mode entretien" in text  # propose la suite
    assert mervi.mode == Mode.MENU and mervi.session is None
    assert mervi.last_result.retained is True


def test_weak_interview_is_rejected():
    mervi, _ = make(good=False)
    start_interview(mervi)
    text = says(run_compute(finish_interview(mervi)))
    assert "ne peux pas retenir" in text


def test_unjudged_answers_give_no_decision_instead_of_an_invented_one():
    mervi, judge = make()
    judge.evaluate = lambda *a: None  # le juge est tombé
    start_interview(mervi)
    text = says(run_compute(finish_interview(mervi)))
    assert "ne pas me prononcer" in text and mervi.last_result.retained is None


def test_interrupted_verdict_is_delivered_on_the_next_utterance():
    mervi, _ = make()
    start_interview(mervi)
    finish_interview(mervi)  # le Compute n'est jamais exécuté : l'utilisateur a coupé la parole
    assert mervi.mode == Mode.VERDICT
    steps = mervi.handle("euh pardon")
    assert isinstance(steps[0], Compute)
    assert "sur 100" in says(run_compute(steps))


def test_repeat_the_current_question():
    mervi, _ = make()
    start_interview(mervi)
    mervi.handle(LONG)
    steps = mervi.handle("Pouvez-vous répéter la question ?")
    assert says(steps) == "Deuxième question. " + mervi.session.current.text.split("Deuxième question. ")[-1]
    assert mervi.session.index == 1


def test_different_seeds_give_different_interviews():
    firsts = set()
    for seed in range(30):
        mervi, _ = make(seed=seed)
        firsts.add(says(mervi.handle("mode entretien"))[:400])
    assert len(firsts) > 3


# --- changement de mode et sortie ------------------------------------------

def test_commands_need_the_name_during_the_interview():
    mervi, _ = make()
    start_interview(mervi)
    steps = mervi.handle("mode découverte")  # sans « Mervi » : c'est une (courte) réponse
    assert mervi.mode == Mode.INTERVIEW and mervi._confirm is None
    assert "abandonner" not in says(steps)
    assert mervi.session.partial == "mode découverte"  # traitée comme une réponse, Mervi demande de développer


def test_switching_mode_during_the_interview_asks_for_confirmation():
    mervi, _ = make()
    start_interview(mervi)
    steps = mervi.handle("Mervi, mode découverte")
    assert "abandonner l'entretien" in says(steps) and mervi.mode == Mode.INTERVIEW
    steps = mervi.handle("oui")
    assert mervi.mode == Mode.DISCOVERY_SETUP and "Quel métier" in says(steps)


def test_refusing_the_switch_resumes_the_interview_at_the_same_question():
    mervi, _ = make()
    start_interview(mervi)
    steps = mervi.handle("Mervi, au revoir")
    assert "abandonner l'entretien" in says(steps)
    steps = mervi.handle("non")
    assert "nous continuons" in says(steps) and "Première question" in says(steps)
    assert mervi.mode == Mode.INTERVIEW and not any(isinstance(s, Quit) for s in steps)


def test_quitting_the_interview_after_confirmation():
    mervi, _ = make()
    start_interview(mervi)
    mervi.handle("Mervi, au revoir")
    assert any(isinstance(s, Quit) for s in mervi.handle("oui"))


def test_answering_something_else_cancels_the_confirmation():
    mervi, _ = make()
    start_interview(mervi)
    mervi.handle("Mervi, au revoir")
    steps = mervi.handle(LONG)  # ni oui ni non : c'était une réponse
    assert mervi.mode == Mode.INTERVIEW and mervi.session.index == 1 and not any(isinstance(s, Quit) for s in steps)


def test_menu_command_returns_to_the_menu():
    mervi, _ = make()
    start_interview(mervi)
    mervi.handle("Mervi, menu")
    assert "Mervi, mode entretien" in says(mervi.handle("oui"))
    assert mervi.mode == Mode.MENU and mervi.session is None


def test_quit_from_the_menu_is_immediate():
    mervi, _ = make()
    steps = mervi.handle("au revoir")
    assert any(isinstance(s, Quit) for s in steps) and "bonne continuation" in says(steps)


# --- découverte d'un métier -------------------------------------------------

def test_discovery_mode_asks_for_the_job_then_hands_over_to_the_llm():
    mervi, _ = make()
    steps = mervi.handle("Mervi, mode découverte")
    assert "Quel métier" in says(steps) and mervi.mode == Mode.DISCOVERY_SETUP

    steps = mervi.handle("le métier de médiateur culturel")
    assert kinds(steps) == ["Session", "RecordUser", "Generate"]
    assert steps[0].title == "Découverte du métier : médiateur culturel"
    gen = steps[2]
    assert "médiateur culturel" in gen.system and "tutoies" in gen.system
    assert gen.instruction and "première question" in gen.instruction
    assert mervi.mode == Mode.DISCOVERY


def test_discovery_conversation_continues_with_the_llm():
    mervi, _ = make()
    mervi.handle("mode découverte")
    mervi.handle("développeur web")
    steps = mervi.handle("Qu'est-ce qu'on fait au quotidien dans ce métier ?")
    assert kinds(steps) == ["RecordUser", "Generate"] and steps[1].instruction is None


def test_discovery_job_can_contain_the_word_discovery_in_setup():
    mervi, _ = make()
    mervi.handle("mode découverte")
    steps = mervi.handle("je voudrais découvrir le métier de designer")
    assert mervi.mode == Mode.DISCOVERY and "designer" in steps[2].system


def test_discovery_needs_the_name_to_leave():
    mervi, _ = make()
    mervi.handle("mode découverte")
    mervi.handle("architecte")
    assert kinds(mervi.handle("au revoir")) == ["RecordUser", "Generate"]  # sans « Mervi » : simple phrase
    assert any(isinstance(s, Quit) for s in mervi.handle("Mervi, au revoir"))


def test_switching_from_discovery_to_interview_needs_no_confirmation():
    mervi, _ = make()
    mervi.handle("mode découverte")
    mervi.handle("architecte")
    steps = mervi.handle("Mervi, mode entretien")
    assert mervi.mode == Mode.INTERVIEW_SETUP and isinstance(steps[0], Session)
