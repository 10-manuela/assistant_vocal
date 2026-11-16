import numpy as np
import pytest

from assistant_vocal.audio_io.vad import SpeechDetector

SR = 16000
BLOCK = 512


def _blocks(signal: np.ndarray):
    for i in range(0, len(signal) - BLOCK + 1, BLOCK):
        yield signal[i : i + BLOCK].astype(np.float32)


@pytest.fixture(scope="module")
def detector():
    return SpeechDetector(sample_rate=SR)


def test_silence_is_not_speech(detector):
    detector.reset()
    events = [detector.process(b) for b in _blocks(np.zeros(SR * 2))]
    assert "start" not in events


def test_noise_is_not_speech(detector):
    detector.reset()
    rng = np.random.default_rng(0)
    noise = 0.01 * rng.standard_normal(SR * 2)
    events = [detector.process(b) for b in _blocks(noise)]
    assert "start" not in events


def test_blocks_have_expected_size():
    from assistant_vocal.audio_io.capture import MicCapture

    assert MicCapture(sample_rate=16000, block_ms=32).block_size == 512