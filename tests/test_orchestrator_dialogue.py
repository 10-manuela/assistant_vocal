"""Intégration orchestrateur + Mervi.

Fichier autonome : il ne dépend d'aucun autre fichier de test (pas d'import relatif).
"""
import queue
import random
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np

from assistant_vocal.config import DialogueConfig
from assistant_vocal.dialogue import Mervi
from assistant_vocal.dialogue.interview import Evaluation, load_content
from assistant_vocal.orchestrator import Orchestrator, Speaker, State
from assistant_vocal.storage import ConversationStore
from assistant_vocal.tts import TTS
from assistant_vocal.voiceprint import OWNER, SpeakerGate

# --- faux modules (copiés ici pour que le fichier se suffise à lui-même) ---------

BLOCK = 512


class FakeVAD:
    """Même logique que SpeechDetector, pilotée par la valeur du bloc."""
    threshold = 0.5

    def __init__(self, silence_needed=2):
        self.silence_needed = silence_needed
        self.in_speech = False
        self.silence = 0
        self.last_probability = 0.0

    def process(self, block):
        self.last_probability = float(block[0])
        if self.last_probability >= self.threshold:
            self.silence = 0
            if not self.in_speech:
                self.in_speech = True
                return "start"
        elif self.in_speech:
            self.silence += 1
            if self.silence >= self.silence_needed:
                self.in_speech = False
                return "end"
        return None


class FakeLLM:
    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = []

    def stream(self, messages):
        self.calls.append(messages)
        yield from self.tokens


class FakeBackend:
    def load(self):
        pass

    def synthesize(self, text, emotion=None):
        return np.zeros(1600, dtype=np.float32), 16000


class FakePlayer:
    def __init__(self, block=False):
        self.block = block
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def play(self, audio, rate):
        self.calls += 1
        self.started.set()
        if self.block:
            self.release.wait(5)

    def stop(self):
        self.release.set()


SR = 16000


class FakeEmbedder:
    sample_rate = SR

    def embed(self, audio):
        v = np.zeros(8, dtype=np.float32)
        v[int(round(float(np.median(audio)) * 10))] = 1.0
        return v


class SeqSTT:
    """Renvoie les textes dans l'ordre (le dernier est répété)."""

    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio, sample_rate=16000):
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return SimpleNamespace(text=text, elapsed=0.01)


def vblk(p, label):
    """Bloc dont [0] pilote le VAD factice et dont le niveau identifie la voix."""
    b = np.full(BLOCK, label / 10, dtype=np.float32)
    b[0] = p
    return b


def speak(orch, label, voiced=12, silence=3):
    for _ in range(voiced):
        orch.handle_block(vblk(0.9, label))
    for _ in range(silence):
        orch.handle_block(np.zeros(BLOCK, dtype=np.float32))


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


# --- tests ---------------------------------------------------------------------

def make(texts, with_gate=False, enrolled=False, good=True, events=None):
    store = ConversationStore(":memory:")
    player = FakePlayer()
    speaker = Speaker(play_fn=player.play, stop_fn=player.stop)
    llm = FakeLLM(["Ce métier est passionnant.", " Qu'est-ce qui t'attire ?"])
    mervi = Mervi(load_content(), FakeJudge(good), DialogueConfig(), random.Random(3), SyncExecutor())
    gate = None
    if with_gate:
        gate = SpeakerGate(FakeEmbedder(), threshold=0.45, sample_rate=SR,
                           min_check_seconds=0.2, min_enroll_seconds=0.4)
        if enrolled:
            gate.enroll(OWNER, np.full(8000, 0.1, dtype=np.float32))
    orch = Orchestrator(
        SeqSTT(texts), llm, TTS(FakeBackend()), store, FakeVAD(), speaker,
        gate=gate, dialogue=mervi, block_ms=32, barge_in_ms=96, verified_barge_in_ms=224,
        min_utterance_ms=96, preroll_ms=64,
        on_event=(lambda k, d: events.append((k, d))) if events is not None else None,
    )
    return orch, mervi, store, llm, player


def turn(orch, label=1):
    speak(orch, label)
    assert orch.wait_idle(10)


def contents(store, orch, role=None):
    return [m.content for m in store.get_messages(orch.conversation_id) if role in (None, m.role)]


def test_start_with_voice_filter_greets_then_enrolls_then_presents_the_modes():
    events = []
    orch, mervi, store, llm, player = make(["Bonjour, je m'appelle Manuela"], with_gate=True, events=events)
    orch.start()
    greeting_calls = player.calls
    assert greeting_calls >= 1 and orch._pending == OWNER

    turn(orch)  # l'utilisateur se présente
    assert orch.gate.active and mervi.name == "Manuela"
    assert player.calls > greeting_calls  # le menu est dit
    assert ("enrolled", OWNER) in events
    assert contents(store, orch) == []  # accueil et menu ne sont pas enregistrés
    assert llm.calls == []


def test_start_without_voice_filter_says_greeting_and_menu_at_once():
    orch, mervi, store, llm, player = make(["x"])
    orch.start()
    assert player.calls >= 3 and orch._pending is None


def test_a_full_interview_is_saved_as_its_own_conversation_and_ends_with_a_verdict():
    orch, mervi, store, llm, player = make(["Mervi, mode entretien", "développeuse IA", LONG])
    orch.start()
    turn(orch)
    assert store.get_conversation(orch.conversation_id).title.startswith("Entretien chez")
    assert len(store.list_conversations()) == 1  # la conversation vide du départ a été supprimée
    assert len(contents(store, orch, "assistant")) == 1  # présentation de l'entreprise

    turn(orch)  # choix du poste
    assert "développeuse IA" in contents(store, orch, "user")
    assert any("Première question" in c for c in contents(store, orch, "assistant"))

    for _ in range(10):
        turn(orch)
    assert llm.calls == []  # l'entretien n'utilise pas le LLM pour parler
    assert len(contents(store, orch, "user")) == 11
    assert "est retenue" in contents(store, orch, "assistant")[-1]
    assert orch.state == State.LISTENING and mervi.mode.value == "menu"


def test_weak_candidate_gets_a_rejection_through_the_whole_chain():
    orch, mervi, store, llm, player = make(["mode entretien", "IA", LONG], good=False)
    orch.start()
    for _ in range(12):
        turn(orch)
    assert "ne peux pas retenir" in contents(store, orch, "assistant")[-1]


def test_goodbye_stops_the_main_loop():
    orch, mervi, store, llm, player = make(["Mervi, au revoir"])
    orch.start()
    turn(orch)
    assert orch.finished

    class SilentMic:
        def read(self, timeout=0.5):
            raise queue.Empty

    orch.run(SilentMic())  # rend la main tout de suite au lieu de boucler


def test_discovery_mode_talks_with_the_llm_about_the_chosen_job():
    orch, mervi, store, llm, player = make(
        ["Mervi, mode découverte", "le métier de médiateur culturel",
         "Qu'est-ce qu'on fait au quotidien ?"])
    orch.start()
    turn(orch)
    assert llm.calls == []  # le choix du mode est un texte préparé

    turn(orch)  # le métier : le LLM prend le relais
    assert store.get_conversation(orch.conversation_id).title == "Découverte du métier : médiateur culturel"
    first = llm.calls[0]
    assert first[0]["role"] == "system" and "médiateur culturel" in first[0]["content"]
    assert first[-1]["role"] == "user" and "première question" in first[-1]["content"]  # consigne cachée
    assert contents(store, orch, "assistant") == ["Ce métier est passionnant. Qu'est-ce qui t'attire ?"]
    assert all("première question" not in c for c in contents(store, orch))  # non enregistrée

    turn(orch)  # la discussion continue, avec la mémoire de l'échange
    second = llm.calls[1]
    assert second[-1]["content"] == "Qu'est-ce qu'on fait au quotidien ?"
    assert any(m["role"] == "assistant" and "passionnant" in m["content"] for m in second)


def test_a_stranger_is_ignored_by_the_dialogue_too():
    events = []
    orch, mervi, store, llm, player = make(["Mervi, mode entretien"], with_gate=True, enrolled=True, events=events)
    turn(orch, label=2)  # une autre voix
    assert any(k == "ignored" for k, _ in events)
    assert mervi.mode.value == "menu" and contents(store, orch) == [] and player.calls == 0
    turn(orch, label=1)  # le propriétaire
    assert mervi.mode.value == "interview_setup"
