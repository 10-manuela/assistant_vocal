from assistant_vocal.config import load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.language == "fr"
    assert cfg.audio.sample_rate == 16000
    assert cfg.llm.model == "qwen2.5:3b"