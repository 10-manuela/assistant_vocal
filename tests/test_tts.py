import numpy as np
import pytest

from assistant_vocal.tts import TTS, TTSError


class FakeBackend:
    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def synthesize(self, text, emotion=None):
        self.calls.append((text, emotion))
        return np.zeros(16000, dtype=np.float32), 16000  # 1 s de silence


def test_synthesize_returns_float32_audio():
    s = TTS(FakeBackend()).synthesize("Bonjour")
    assert s.audio.dtype == np.float32
    assert s.sample_rate == 16000


def test_duration_and_rtf():
    s = TTS(FakeBackend()).synthesize("Bonjour")
    assert s.duration == pytest.approx(1.0)
    assert s.rtf >= 0


def test_stream_synthesizes_one_audio_per_sentence():
    backend = FakeBackend()
    tokens = ["Bonjour,", " je suis", " ravie.", " Comment", " allez-vous", " ?"]
    out = list(TTS(backend).stream(tokens, emotion="joie"))
    assert [s.text for s in out] == ["Bonjour, je suis ravie.", "Comment allez-vous ?"]
    assert all(e == "joie" for _, e in backend.calls)


def test_stream_skips_empty_text():
    assert list(TTS(FakeBackend()).stream(["   "])) == []


def test_missing_voice_gives_clear_error(tmp_path):
    from assistant_vocal.tts.piper_backend import PiperBackend

    with pytest.raises(TTSError, match="download_voices"):
        PiperBackend("fr_FR-inconnue", tmp_path).load()
        