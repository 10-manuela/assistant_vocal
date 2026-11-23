import numpy as np
import pytest

from assistant_vocal.voiceprint import OWNER, SpeakerGate, detect_intent

SR = 16000


class FakeEmbedder:
    """Une « voix » = un niveau constant : 0.1 -> empreinte e1, 0.2 -> e2, etc."""
    sample_rate = SR

    def embed(self, audio):
        v = np.zeros(8, dtype=np.float32)
        v[int(round(float(np.median(audio)) * 10))] = 1.0
        return v


def voice(label, seconds):
    return np.full(int(SR * seconds), label / 10, dtype=np.float32)


def make_gate(**kw):
    kw.setdefault("min_check_seconds", 1.0)
    kw.setdefault("min_enroll_seconds", 4.0)
    return SpeakerGate(FakeEmbedder(), threshold=0.45, sample_rate=SR, **kw)


def test_inactive_gate_accepts_everything():
    gate = make_gate()
    r = gate.check(voice(2, 2))
    assert r.accepted and r.reason == "no_voice" and not gate.active


def test_owner_accepted_stranger_rejected():
    gate = make_gate()
    gate.enroll(OWNER, voice(1, 5))
    assert gate.active
    ok = gate.check(voice(1, 2))
    assert ok.accepted and ok.speaker == OWNER and ok.score == pytest.approx(1.0)
    ko = gate.check(voice(2, 2))
    assert not ko.accepted and ko.reason == "mismatch" and ko.speaker is None


def test_enrollment_too_short_is_refused():
    with pytest.raises(ValueError):
        make_gate().enroll(OWNER, voice(1, 2))


def test_guest_is_accepted_then_removed():
    gate = make_gate()
    gate.enroll(OWNER, voice(1, 5))
    name = gate.new_guest_name()
    assert name == "guest1"
    gate.enroll(name, voice(2, 5))
    assert gate.check(voice(2, 2)).accepted
    assert gate.clear_guests() == 1
    assert not gate.check(voice(2, 2)).accepted
    assert gate.check(voice(1, 2)).accepted  # le propriétaire reste


def test_short_utterance_continuity_follows_last_decision():
    gate = make_gate(short_policy="continuity")
    gate.enroll(OWNER, voice(1, 5))
    gate.check(voice(1, 2))
    assert gate.check(voice(1, 0.3)).accepted  # trop court, mais le dernier énoncé était valide
    gate.check(voice(2, 2))  # un inconnu parle
    r = gate.check(voice(1, 0.3))
    assert not r.accepted and r.reason == "short"


@pytest.mark.parametrize("policy,expected", [("accept", True), ("reject", False)])
def test_short_utterance_fixed_policies(policy, expected):
    gate = make_gate(short_policy=policy)
    gate.enroll(OWNER, voice(1, 5))
    assert gate.check(voice(2, 0.3)).accepted is expected


def test_remember_false_does_not_change_continuity():
    gate = make_gate()
    gate.enroll(OWNER, voice(1, 5))
    gate.check(voice(2, 2), remember=False)  # vérification d'interruption : sans effet
    assert gate.check(voice(1, 0.3)).accepted


def test_sample_rate_mismatch_is_refused():
    class Bad:
        sample_rate = 8000

    with pytest.raises(ValueError):
        SpeakerGate(Bad(), sample_rate=SR)


@pytest.mark.parametrize("text", [
    "Je voudrais ajouter quelqu'un à l'entretien",
    "Peux-tu inviter une personne ?",
    "Mon collègue va nous rejoindre",
    "Fais venir un invité s'il te plaît",
])
def test_add_guest_phrases(text):
    assert detect_intent(text) == "add_guest"


@pytest.mark.parametrize("text", [
    "L'invité est parti",
    "Tu peux enlever l'invité",
    "Mon collègue nous quitte",
])
def test_remove_guest_phrases(text):
    assert detect_intent(text) == "remove_guest"


@pytest.mark.parametrize("text", [
    "Je voudrais ajouter une ligne à mon CV",
    "Je suis une personne motivée qui aime apprendre",
    "Bonjour, je m'appelle Manuela",
])
def test_normal_sentences_are_not_commands(text):
    assert detect_intent(text) is None