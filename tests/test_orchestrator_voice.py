from types import SimpleNamespace

import numpy as np

from assistant_vocal.orchestrator import Orchestrator, Speaker, State
from assistant_vocal.storage import ConversationStore
from assistant_vocal.tts import TTS
from assistant_vocal.voiceprint import OWNER, SpeakerGate

from test_orchestrator import BLOCK, FakeBackend, FakeLLM, FakePlayer, FakeVAD

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


def make_gated(texts=("Bonjour Manuela",), block=False, enrolled=True, events=None):
    store = ConversationStore(":memory:")
    player = FakePlayer(block)
    speaker = Speaker(play_fn=player.play, stop_fn=player.stop)
    llm = FakeLLM(["Bonjour, je suis ravie.", " Comment allez-vous ?"])
    gate = SpeakerGate(FakeEmbedder(), threshold=0.45, sample_rate=SR,
                       min_check_seconds=0.2, min_enroll_seconds=0.4)
    if enrolled:
        gate.enroll(OWNER, np.full(8000, 0.1, dtype=np.float32))
    orch = Orchestrator(
        SeqSTT(texts), llm, TTS(FakeBackend()), store, FakeVAD(), speaker,
        gate=gate, block_ms=32, barge_in_ms=96, verified_barge_in_ms=224,
        min_utterance_ms=96, preroll_ms=64,
        on_event=(lambda k, d: events.append((k, d))) if events is not None else None,
    )
    return orch, store, llm, player, gate


def roles(store, orch):
    return [m.role for m in store.get_messages(orch.conversation_id)]


def test_owner_is_answered():
    orch, store, llm, player, gate = make_gated()
    speak(orch, 1)
    assert orch.wait_idle(5)
    assert roles(store, orch) == ["user", "assistant"]


def test_stranger_is_ignored():
    events = []
    orch, store, llm, player, gate = make_gated(events=events)
    speak(orch, 2)
    assert orch.wait_idle(5)
    assert roles(store, orch) == []
    assert llm.calls == [] and player.calls == 0
    assert orch.state == State.LISTENING
    assert any(k == "ignored" for k, _ in events)


def test_owner_enrollment_then_conversation():
    orch, store, llm, player, gate = make_gated(enrolled=False)
    orch.begin_enrollment(OWNER)
    speak(orch, 1)
    assert orch.wait_idle(5)
    assert gate.active and OWNER in gate.voices
    assert roles(store, orch) == []  # l'enrôlement n'entre pas dans la conversation
    assert llm.calls == []
    assert player.calls == 1  # phrase de confirmation

    speak(orch, 1)
    assert orch.wait_idle(5)
    assert roles(store, orch) == ["user", "assistant"]
    speak(orch, 2)  # un inconnu
    assert orch.wait_idle(5)
    assert roles(store, orch) == ["user", "assistant"]


def test_short_enrollments_accumulate():
    orch, store, llm, player, gate = make_gated(enrolled=False)
    orch.begin_enrollment(OWNER)
    speak(orch, 1, voiced=5)
    assert orch.wait_idle(5)
    assert not gate.active  # pas assez de voix : on redemande
    speak(orch, 1, voiced=5)
    assert orch.wait_idle(5)
    assert gate.active


def test_guest_can_be_added_and_removed_by_voice():
    texts = ["Je voudrais ajouter quelqu'un à l'entretien", "Bonjour tout le monde",
             "L'invité est parti"]
    orch, store, llm, player, gate = make_gated(texts=texts)

    speak(orch, 1)  # le propriétaire demande d'ajouter quelqu'un
    assert orch.wait_idle(5)
    assert llm.calls == [] and roles(store, orch) == []  # commande : pas de LLM
    assert orch._pending == "guest1"

    speak(orch, 2)  # l'invité se présente : sa voix est enregistrée
    assert orch.wait_idle(5)
    assert "guest1" in gate.voices and orch._pending is None

    speak(orch, 2)  # l'invité est maintenant écouté
    assert orch.wait_idle(5)
    assert roles(store, orch) == ["user", "assistant"]

    speak(orch, 1)  # le propriétaire retire l'invité
    assert orch.wait_idle(5)
    assert gate.guests() == []
    speak(orch, 2)  # l'invité est de nouveau ignoré
    assert orch.wait_idle(5)
    assert roles(store, orch) == ["user", "assistant"]


def test_stranger_cannot_interrupt():
    orch, store, llm, player, gate = make_gated(block=True)
    speak(orch, 1)
    assert player.started.wait(5)
    assert orch.state == State.SPEAKING

    speak(orch, 2, voiced=10)  # un inconnu parle par-dessus
    assert orch.state == State.SPEAKING

    for _ in range(10):  # le propriétaire parle par-dessus
        orch.handle_block(vblk(0.9, 1))
    assert orch.state == State.LISTENING
    assert orch.wait_idle(5)


def test_brief_noise_does_not_interrupt_when_gate_is_active():
    orch, store, llm, player, gate = make_gated(block=True)
    speak(orch, 1)
    assert player.started.wait(5)
    for _ in range(4):  # 128 ms de voix du propriétaire : sous le délai de vérification
        orch.handle_block(vblk(0.9, 1))
    assert orch.state == State.SPEAKING
    player.release.set()
    assert orch.wait_idle(5)