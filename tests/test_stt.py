from types import SimpleNamespace

import numpy as np
import pytest

from assistant_vocal.stt import Transcriber


class FakeModel:
    def __init__(self, texts):
        self.texts = texts
        self.received = None

    def transcribe(self, audio, **kwargs):
        self.received = (audio, kwargs)
        segments = (SimpleNamespace(text=t) for t in self.texts)
        return segments, SimpleNamespace(language="fr", language_probability=0.97)


def test_segments_are_joined_and_stripped():
    t = Transcriber(model=FakeModel(["  Bonjour,", " je m'appelle Manuela. "]))
    result = t.transcribe(np.zeros(16000))
    assert result.text == "Bonjour, je m'appelle Manuela."
    assert result.language == "fr"


def test_audio_is_converted_to_float32_and_options_passed():
    fake = FakeModel(["ok"])
    t = Transcriber(language="fr", beam_size=1, model=fake)
    t.transcribe(np.zeros(8000, dtype=np.float64))
    audio, kwargs = fake.received
    assert audio.dtype == np.float32
    assert kwargs["language"] == "fr"
    assert kwargs["vad_filter"] is True


def test_duration_and_rtf():
    t = Transcriber(model=FakeModel(["ok"]))
    result = t.transcribe(np.zeros(32000))
    assert result.audio_seconds == pytest.approx(2.0)
    assert result.rtf >= 0


def test_wrong_sample_rate_rejected():
    t = Transcriber(model=FakeModel([]))
    with pytest.raises(ValueError):
        t.transcribe(np.zeros(100), sample_rate=44100)