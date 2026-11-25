import pytest

from assistant_vocal.dialogue import commands as cmd


@pytest.mark.parametrize("text,strict,expected", [
    ("Mervi, mode entretien", False, "interview"),
    ("mode entretien professionnel", False, "interview"),
    ("je voudrais passer un entretien", False, "interview"),
    ("Mervi, mode découverte", False, "discovery"),
    ("je veux découvrir un métier", False, "discovery"),
    ("menu", False, "menu"),
    ("Mervi, change de mode", False, "menu"),
    ("au revoir", False, "quit"),
    ("Mervi, au revoir", True, "quit"),
    ("Merveille, au revoir", True, "quit"),  # variante d'écriture du STT
    ("Mervi, quitte la discussion", True, "quit"),
    ("Mervi, arrête l'entretien", True, "quit"),
    # pendant l'entretien ou la discussion, il faut dire « Mervi »
    ("mode entretien", True, None),
    ("au revoir", True, None),
    ("menu", True, None),
    # une réponse ordinaire n'est jamais une commande
    ("j'ai terminé mon projet", False, None),
    ("J'ai passé un entretien chez une grande entreprise de la région", False, None),
    ("Je travaille sur un projet de découverte du patrimoine avec des musées de la région", False, None),
])
def test_detect_command(text, strict, expected):
    assert cmd.detect_command(text, strict=strict) == expected


@pytest.mark.parametrize("text", [
    "répète", "Pouvez-vous répéter la question ?", "Mervi, répète", "répétez s'il vous plaît",
])
def test_repeat_works_without_the_name(text):
    assert cmd.detect_command(text, strict=True) == "repeat"


def test_repeat_is_not_triggered_inside_an_answer():
    assert cmd.detect_command("je répète que j'aime coder", strict=True) is None


def test_yes_no():
    assert cmd.is_yes("Oui, bien sûr") and cmd.is_yes("d'accord") and not cmd.is_yes("non")
    assert cmd.is_no("Non merci") and cmd.is_no("pas du tout") and not cmd.is_no("oui")
    assert not cmd.is_yes("oui mais je voulais d'abord parler de mon expérience passée")


@pytest.mark.parametrize("text,expected", [
    ("Bonjour, je m'appelle Manuela et je suis étudiante", "Manuela"),
    ("moi c'est léa", "Léa"),
    ("Mon prénom est Jean-Paul", "Jean-paul"),
    ("je suis étudiante en art", None),
    ("je m'appelle un", None),
    ("", None),
])
def test_extract_name(text, expected):
    assert cmd.extract_name(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("le métier de médiateur culturel", "médiateur culturel"),
    ("Je voudrais découvrir le métier de data analyst s'il te plaît", "data analyst"),
    ("développeur web", "développeur web"),
    ("parlons de design", "design"),
    ("je m'intéresse au métier d'architecte", "architecte"),
    ("", None),
    ("alors en fait je ne sais pas trop quoi dire sur ce sujet précis aujourd'hui", None),
])
def test_extract_job(text, expected):
    assert cmd.extract_job(text) == expected


OFFERED = [
    {"id": "ia", "mots_cles": ["ia", "intelligence artificielle"]},
    {"id": "data", "mots_cles": ["data analyst", "data"]},
    {"id": "ux", "mots_cles": ["ux", "designer"]},
    {"id": "projet", "mots_cles": ["chef de projet"]},
]


@pytest.mark.parametrize("text,expected", [
    ("Je postule comme développeuse IA", "ia"),
    ("l'intelligence artificielle", "ia"),
    ("designer UX", "ux"),
    ("chef de projet", "projet"),
    ("le deuxième", "data"),
    ("le dernier", "projet"),
    ("développeur", None),   # ambigu ou inconnu
    ("data et ia", None),    # deux postes à égalité
])
def test_match_profile(text, expected):
    found = cmd.match_profile(text, OFFERED)
    assert (found["id"] if found else None) == expected
