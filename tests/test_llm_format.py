import json

from assistant_vocal.llm.client import LLMClient, make_message


class FakeResponse:
    def __init__(self, objs):
        self.lines = [json.dumps(o).encode() + b"\n" for o in objs]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_format_and_options_are_sent_to_ollama():
    sent = []

    def opener(req, timeout=None):
        sent.append(json.loads(req.data))
        return FakeResponse([{"message": {"role": "assistant", "content": "{}"}, "done": True}])

    llm = LLMClient(temperature=0.7, num_ctx=4096, opener=opener)
    llm.chat([make_message("user", "x")], fmt="json", options={"temperature": 0.0, "num_predict": 200})
    assert sent[0]["format"] == "json"
    assert sent[0]["options"]["temperature"] == 0.0
    assert sent[0]["options"]["num_predict"] == 200
    assert sent[0]["options"]["num_ctx"] == 4096  # les autres réglages restent


def test_default_call_is_unchanged():
    sent = []

    def opener(req, timeout=None):
        sent.append(json.loads(req.data))
        return FakeResponse([{"message": {"role": "assistant", "content": "ok"}, "done": True}])

    LLMClient(opener=opener).chat([make_message("user", "x")])
    assert "format" not in sent[0]
    assert sent[0]["options"]["temperature"] == 0.7
