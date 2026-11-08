"""The parts of a masked value that are masked wherever they stand: a surname, a profile's handle.

Such parts stand alone in a profile written as JSON, a pasted CV or a list of repositories.
"""

from polanon import Anonymizer, Guard, Span, default_guards


def mask(text, mapping=None):
    return Anonymizer(default_guards(with_ner=False)).anonymize_sync(text, mapping)


class FixedGuard(Guard):
    """Reports one value where it stands, as a guard a caller writes for values it knows does."""

    name = "fixed"

    def __init__(self, value, entity_type, score=1.0, declares=frozenset()):
        self.value, self.type, self.score = value, entity_type, score
        self.entity_types = declares

    def detect(self, text, enabled):
        start = text.find(self.value)
        if start < 0 or self.type not in enabled:
            return []
        return [Span(start, start + len(self.value), self.type, self.score, self.name)]


def test_a_surname_alone_is_masked_after_the_full_name():
    """A lone surname went out in the clear after the full name was masked."""
    text = "Jan Kowalski prowadził projekt. Kowalski odpowiadał też za testy."
    result = mask(text)
    assert result.anonymized == "<OSOBA_1> prowadził projekt. <OSOBA_2> odpowiadał też za testy."
    assert result.mapping == {"<OSOBA_1>": "Jan Kowalski", "<OSOBA_2>": "Kowalski"}
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


def test_each_part_of_a_double_surname_is_masked():
    result = mask("Anna Nowak-Kępa prowadzi zajęcia. Pani Kępa je zaplanowała.")
    assert "Kępa" not in result.anonymized


def test_a_lone_surname_in_lowercase_stays_an_ordinary_word():
    """Kot is a surname and "kot" a cat, as for any known one-word name."""
    result = mask("Anna Kot przyszła. Mój kot został w domu.")
    assert result.anonymized == "<OSOBA_1> przyszła. Mój kot został w domu."


def test_a_surname_known_from_an_earlier_turn_is_masked_on_its_own():
    first = mask("Jan Kowalski")
    second = mask("Kowalski dzwonił.", first.mapping)
    assert second.anonymized == "<OSOBA_2> dzwonił."
    assert second.mapping["<OSOBA_2>"] == "Kowalski"


def test_a_name_a_language_model_guessed_gives_no_surname_without_a_known_first_name():
    """A model's "Google Ads" as a person must not mask "Ads" in every sentence after it."""
    model = FixedGuard("Google Ads", "PERSON", score=0.85)
    result = Anonymizer([model]).anonymize_sync("Kampania w Google Ads. Ads to reklamy.")
    assert result.anonymized == "Kampania w <OSOBA_1>. Ads to reklamy."


def test_the_handle_of_a_profile_link_is_masked_wherever_it_stands():
    """A login stands on its own in a JSON field, after "@" and in the names of repositories."""
    text = (
        '{"login": "jkowalski", "blog": "https://jkowalski.github.io"}\n'
        "Nick @jkowalski, repozytorium jkowalski-portfolio."
    )
    result = mask(text)
    assert result.anonymized == (
        '{"login": "<LINK_1>", "blog": "<LINK_2>"}\n'
        "Nick @<LINK_1>, repozytorium <LINK_1>-portfolio."
    )
    assert result.mapping == {"<LINK_1>": "jkowalski", "<LINK_2>": "https://jkowalski.github.io"}
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


def test_a_repository_link_keeps_the_project_name_readable():
    result = mask("Projekt: https://github.com/jkowalski/weather-app")
    assert result.anonymized == "Projekt: <LINK_1>/weather-app"


def test_the_handle_of_a_well_known_brand_stays_readable():
    """ "github.com/microsoft/vscode" links to a project; "Microsoft Excel" names nobody."""
    result = mask("Fork https://github.com/microsoft/vscode. Znam Microsoft Excel.")
    assert result.anonymized == "Fork <LINK_1>/vscode. Znam Microsoft Excel."


def test_a_link_the_callers_own_guard_reports_gives_its_handle_too():
    """A caller's guard that reports links as LINK, a type of its own until now, keeps working."""
    profile = FixedGuard("https://github.com/jkowalski", "LINK", declares=frozenset({"LINK"}))
    result = Anonymizer([profile]).anonymize_sync(
        'GitHub: https://github.com/jkowalski\n"login": "jkowalski"'
    )
    assert result.anonymized == 'GitHub: <LINK_1>\n"login": "<LINK_2>"'
    assert result.mapping == {"<LINK_1>": "https://github.com/jkowalski", "<LINK_2>": "jkowalski"}


def test_a_name_from_the_callers_own_guard_gives_its_surname_even_with_an_unknown_first_name():
    """The caller knows the name, so a first name no list holds does not stop the surname."""
    profile = FixedGuard("Bożydar Gąsiorek", "PERSON")
    result = Anonymizer([profile]).anonymize_sync("Bożydar Gąsiorek. Gąsiorek ma prawo jazdy.")
    assert result.anonymized == "<OSOBA_1>. <OSOBA_2> ma prawo jazdy."
