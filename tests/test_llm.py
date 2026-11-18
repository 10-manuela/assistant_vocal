import base64
import json
import urllib.error

import pytest

from assistant_vocal.llm import LLMClient, LLMError, make_message, sentences


class FakeResponse:
    def __init__(self, objs):
        self.lines = [json.dumps(o).encode() + b"\n" for o in objs]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.lines)


def make_opener(objs, captured=None):
    def opener(req, timeout=None):
        if captured is not None:
            captured.append(json.loads(req.data))
        return FakeResponse(objs)

    return opener


def chunk(text, done=False, **extra):
    return {"message": {"role": "assistant", "content": text}, "done": done, **extra}


def test_stream_yields_tokens_and_sends_payload():
    sent = []
    objs = [chunk("Bon"), chunk("jour"), chunk("", done=True)]
    llm = LLMClient(model="m", temperature=0.3, opener=make_opener(objs, sent))
    out = list(llm.stream([make_message("user", "salut")]))
    assert out == ["Bon", "jour"]
    assert sent[0]["model"] == "m"
    assert sent[0]["stream"] is True
    assert sent[0]["options"]["temperature"] == 0.3


def test_images_are_base64_encoded():
    msg = make_message("user", "décris", images=[b"abc"])
    assert msg["images"] == [base64.b64encode(b"abc").decode()]


def test_stats_from_final_chunk():
    objs = [chunk("a"), chunk("", done=True, eval_count=20, eval_duration=1_000_000_000)]
    llm = LLMClient(opener=make_opener(objs))
    llm.chat([make_message("user", "x")])
    assert llm.last_stats.tokens == 20
    assert llm.last_stats.tokens_per_s == pytest.approx(20.0)
    assert llm.last_stats.first_token_s is not None


def test_tool_calls_are_collected():
    call = {"function": {"name": "nouvelle_conversation", "arguments": {}}}
    objs = [
        {"message": {"role": "assistant", "content": "", "tool_calls": [call]}, "done": False},
        chunk("", done=True),
    ]
    llm = LLMClient(opener=make_opener(objs))
    llm.chat([make_message("user", "x")])
    assert llm.last_tool_calls == [call]


def test_connection_error_becomes_llmerror():
    def opener(req, timeout=None):
        raise urllib.error.URLError("refusé")

    llm = LLMClient(opener=opener)
    with pytest.raises(LLMError):
        llm.chat([make_message("user", "x")])


def test_sentences_split_as_they_complete():
    tokens = ["Bonjour,", " je suis", " ravie.", " Comment", " allez-vous", " ?", " Bien."]
    assert list(sentences(tokens)) == ["Bonjour, je suis ravie.", "Comment allez-vous ?", "Bien."]


def test_sentences_merge_short_ones():
    assert list(sentences(["Oui. Très bien merci."])) == ["Oui. Très bien merci."]


def test_sentences_flush_unfinished_text():
    assert list(sentences(["Pas de point final"])) == ["Pas de point final"]