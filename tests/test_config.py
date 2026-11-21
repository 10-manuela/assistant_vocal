from assistant_vocal.config import load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.language == "fr"
    assert cfg.audio.sample_rate == 16000
    assert cfg.llm.model == "llama3.2:3b"
    assert cfg.tts.backend in ("piper", "kokoro")