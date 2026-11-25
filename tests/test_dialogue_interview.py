import random

import pytest

from assistant_vocal.dialogue.interview import (
    Evaluation, InterviewSession, Question, Scenario, build_scenario, build_session,
    compute_result, format_verdict, load_content, validate_content,
)

CONTENT = load_content()


def perfect(q):
    return Evaluation([True] * len(q.criteria), example=True, structured=True, language=2)


def zero(q):
    return Evaluation([False] * len(q.criteria), language=0)


def make_session(seed=1, profile_id="ia"):
    rng = random.Random(seed)
    scenario = build_scenario(CONTENT, rng)
    profile = next(p for p in CONTENT["profiles"] if p["id"] == profile_id)
    return build_session(CONTENT, scenario, profile, rng)


def test_content_is_valid_and_complete():
    validate_content(CONTENT)
    assert len(CONTENT["profiles"]) >= 5
    assert sum(1 for p in CONTENT["profiles"] if p.get("obligatoire")) == 1


def test_invalid_content_is_detected():
    broken = {**CONTENT, "profiles": [{**p, "obligatoire": False} for p in CONTENT["profiles"]]}
    with pytest.raises(ValueError, match="obligatoire"):
        validate_content(broken)


def test_every_interview_searches_four_profiles_including_the_ai_one():
    for seed in range(200):
        scenario = build_scenario(CONTENT, random.Random(seed))
        assert len(scenario.offered) == 4
        assert len({p["id"] for p in scenario.offered}) == 4
        assert "ia" in {p["id"] for p in scenario.offered}


def test_other_profiles_vary_between_interviews():
    seen = {tuple(sorted(p["id"] for p in build_scenario(CONTENT, random.Random(s)).offered))
            for s in range(100)}
    assert len(seen) > 3


def test_ten_questions_without_duplicates_following_the_composition():
    for seed in range(100):
        session = make_session(seed)
        assert len(session.questions) == 10
        assert len({q.id for q in session.questions}) == 10
        assert [("profil" if q.technical else q.category) for q in session.questions] == CONTENT["composition"]
        own = {q["id"] for q in session.profile["questions"]}
        assert sum(q.id in own for q in session.questions) == 4


def test_questions_change_from_one_interview_to_the_next():
    drawn = {tuple(q.id for q in make_session(s).questions) for s in range(50)}
    assert len(drawn) > 25


def test_perfect_candidate_is_retained():
    session = make_session()
    result = compute_result(session, [perfect(q) for q in session.questions])
    assert result.retained is True and result.score == pytest.approx(100)
    assert result.knockout is False


def test_weak_candidate_is_rejected():
    session = make_session()
    result = compute_result(session, [zero(q) for q in session.questions])
    assert result.retained is False and result.score == 0


def test_weak_technical_level_is_a_knockout_even_with_a_good_score():
    session = make_session()
    session.profile = {**session.profile, "seuil": 0.3, "eliminatoire": 0.45}
    evals = [zero(q) if q.technical else perfect(q) for q in session.questions]
    result = compute_result(session, evals)
    assert result.score >= 30 and result.knockout is True and result.retained is False
    assert "niveau minimal" in format_verdict(result, session)


def test_too_few_evaluated_answers_means_no_decision():
    session = make_session()
    evals = [perfect(q) if i < 5 else None for i, q in enumerate(session.questions)]
    result = compute_result(session, evals)
    assert result.retained is None and result.evaluated == 5
    assert "ne pas me prononcer" in format_verdict(result, session)


def test_technical_questions_weigh_more():
    session = make_session()
    technical_ok = [perfect(q) if q.technical else zero(q) for q in session.questions]
    general_ok = [zero(q) if q.technical else perfect(q) for q in session.questions]
    assert compute_result(session, technical_ok).score > compute_result(session, general_ok).score


def test_verdict_text_names_the_job_company_score_and_gaps():
    session = make_session()
    evals = [perfect(q) if i % 2 == 0 else zero(q) for i, q in enumerate(session.questions)]
    result = compute_result(session, evals)
    text = format_verdict(result, session, name="Manuela")
    assert "Manuela" in text and session.profile["nom"] in text
    assert session.scenario.company["nom"] in text
    assert "Les points à travailler" in text and "Vos points forts" in text
    assert "simulation d'entraînement" in text
    assert ("retenue" in text) or ("ne peux pas retenir" in text)


def test_language_remark_depends_on_the_language_score():
    session = make_session()
    clear = compute_result(session, [perfect(q) for q in session.questions])
    messy = compute_result(session, [zero(q) for q in session.questions])
    assert "claire et professionnelle" in format_verdict(clear, session)
    assert "Soignez la clarté" in format_verdict(messy, session)


def test_evaluation_score_range():
    q = Question("x", "?", ["a", "b"], "profil")
    assert perfect(q).score() == pytest.approx(1.0)
    assert zero(q).score() == 0.0
