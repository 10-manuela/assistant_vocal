import threading
from types import SimpleNamespace

import numpy as np

from assistant_vocal.orchestrator import Orchestrator, Speaker, State
from assistant_vocal.storage import ConversationStore
from assistant_vocal.tts import TTS

BLOCK = 512


def blk(p: float) -> np.ndarray:
    return np.full(BLOCK, p, dtype=np.float32)


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


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio, sample_rate=16000):
        return SimpleNamespace(text=self.text, elapsed=0.05)


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


def make(stt_text="Bonjour Manuela", block=False, conversation_id=None, store=None):
    store = store or ConversationStore(":memory:")
    player = FakePlayer(block)
    speaker = Speaker(play_fn=player.play, stop_fn=player.stop)
    llm = FakeLLM(["Bonjour, je suis ravie.", " Comment allez-vous ?"])
    orch = Orchestrator(
        FakeSTT(stt_text), llm, TTS(FakeBackend()), store, FakeVAD(), speaker,
        block_ms=32, barge_in_ms=96, min_utterance_ms=96, preroll_ms=64,
        conversation_id=conversation_id,
    )
    return orch, store, llm, player


def say_something(orch, voiced=12, silence=3):
    for _ in range(voiced):
        orch.handle_block(blk(0.9))
    for _ in range(silence):
        orch.handle_block(blk(0.0))


def test_full_turn_is_answered_and_saved():
    orch, store, llm, player = make()
    say_something(orch)
    assert orch.wait_idle(5)
    msgs = store.get_messages(orch.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].content == "Bonjour Manuela"
    assert msgs[1].content == "Bonjour, je suis ravie. Comment allez-vous ?"
    assert player.calls == 2  # une lecture par phrase
    sent = llm.calls[0]
    assert sent[0]["role"] == "system"
    assert sent[-1]["content"] == "Bonjour Manuela"


def test_empty_transcript_is_ignored():
    orch, store, llm, player = make(stt_text="")
    say_something(orch)
    assert orch.wait_idle(5)
    assert store.get_messages(orch.conversation_id) == []
    assert player.calls == 0


def test_barge_in_stops_the_assistant():
    orch, store, llm, player = make(block=True)
    say_something(orch)
    assert player.started.wait(5)
    assert orch.state == State.SPEAKING

    for _ in range(4):  # l'utilisateur parle par-dessus
        orch.handle_block(blk(0.9))
    assert orch.state == State.LISTENING
    assert orch.wait_idle(5)

    roles = [m.role for m in store.get_messages(orch.conversation_id)]
    assert roles == ["user"]  # phrase interrompue : non enregistrée
    assert player.calls == 1  # la 2e phrase n'a jamais été jouée


def test_short_noise_does_not_interrupt():
    orch, store, llm, player = make(block=True)
    say_something(orch)
    assert player.started.wait(5)

    orch.handle_block(blk(0.9))  # une toux : 2 blocs seulement
    orch.handle_block(blk(0.9))
    for _ in range(3):
        orch.handle_block(blk(0.0))
    assert orch.state == State.SPEAKING

    player.release.set()
    assert orch.wait_idle(5)
    roles = [m.role for m in store.get_messages(orch.conversation_id)]
    assert roles == ["user", "assistant"]


def test_resume_loads_previous_messages():
    store = ConversationStore(":memory:")
    conv = store.create_conversation()
    store.add_message(conv.id, "user", "salut")
    store.add_message(conv.id, "assistant", "bonjour")
    orch, *_ = make(conversation_id=conv.id, store=store)
    assert [m["content"] for m in orch.history] == ["salut", "bonjour"]