import json

from assistant_vocal.dialogue.interview import Question
from assistant_vocal.dialogue.judge import LLMJudge, build_prompt, parse_evaluation

Q = Question("ia-1", "Expliquez le surapprentissage.", ["la définition", "un exemple", "une parade"], "profil")
LONG = "Le surapprentissage arrive quand un modèle apprend par cœur ses données d'entraînement et généralise mal."


def test_parse_clean_json():
    ev = parse_evaluation('{"criteres": [true, false, true], "exemple": true, "structure": false, "langage": 2}', 3)
    assert ev.criteria_met == [True, False, True]
    assert ev.example and not ev.structured and ev.language == 2


def test_parse_tolerates_text_around_json_and_string_booleans():
    raw = 'Voici : ```json\n{"criteres": ["true", "false", "oui"], "exemple": "false", "structure": true, "langage": 5}\n```'
    ev = parse_evaluation(raw, 3)
    assert ev.criteria_met == [True, False, True]
    assert ev.language == 2  # borné à 0-2


def test_parse_rejects_unusable_answers():
    assert parse_evaluation("pas de json", 3) is None
    assert parse_evaluation('{"criteres": [true]}', 3) is None  # mauvaise longueur
    assert parse_evaluation('{"autre": 1}', 3) is None
    assert parse_evaluation('{"criteres": [true, true, true], "langage": "x"}', 3) is None
    assert parse_evaluation("", 3) is None


def test_prompt_lists_criteria_and_candidate_answer():
    prompt = build_prompt("développeur IA", Q, LONG)
    assert "1. la définition" in prompt and "3. une parade" in prompt
    assert LONG in prompt and "développeur IA" in prompt


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, fmt=None, options=None):
        self.calls.append((messages, fmt, options))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


GOOD = json.dumps({"criteres": [True, True, False], "exemple": True, "structure": True, "langage": 1})


def test_short_answer_is_scored_without_calling_the_llm():
    llm = FakeLLM()
    ev = LLMJudge(llm, min_words=6).evaluate("ia", Q, "je ne sais pas")
    assert ev.skipped and not any(ev.criteria_met) and ev.language == 0
    assert llm.calls == []


def test_judge_asks_for_json_with_low_temperature():
    llm = FakeLLM(GOOD)
    ev = LLMJudge(llm).evaluate("ia", Q, LONG)
    assert ev.criteria_met == [True, True, False]
    _, fmt, options = llm.calls[0]
    assert fmt == "json" and options["temperature"] == 0.0


def test_judge_retries_once_when_the_answer_is_unusable():
    llm = FakeLLM("n'importe quoi", GOOD)
    assert LLMJudge(llm).evaluate("ia", Q, LONG) is not None
    assert len(llm.calls) == 2


def test_judge_gives_up_after_two_bad_answers():
    llm = FakeLLM("non", "toujours non")
    assert LLMJudge(llm).evaluate("ia", Q, LONG) is None


def test_judge_survives_an_ollama_error():
    assert LLMJudge(FakeLLM(RuntimeError("Ollama injoignable"))).evaluate("ia", Q, LONG) is None
