"""The parts of a masked value that are masked wherever they stand: a surname, a profile's handle.

Such parts stand alone in a profile written as JSON, a pasted CV or a list of repositories.
"""

import re

import pytest

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


def test_a_title_is_not_masked_on_its_own_after_a_name_on_a_foreign_letter():
    """After "z. Hd. Frau Wiśniewska" the receptionist, "die Frau am Empfang", was masked too."""
    text = "z. Hd. Frau Wiśniewska\nDie Frau am Empfang hat angerufen."
    assert mask(text).anonymized == "z. Hd. Frau <OSOBA_1>\nDie Frau am Empfang hat angerufen."


def test_a_second_title_is_not_masked_as_the_person_either():
    text = "Attn: Frau Prof. Anna Schmidt\nProf. Lis und Prof Nowak."
    masked = mask(text).anonymized
    assert masked.count("Prof") == 3
    assert "Frau Prof." in masked


def test_a_name_after_a_title_and_without_it_is_one_person():
    text = "Attn: Mr Oliver Bennett\n\nKind regards,\nOliver Bennett"
    assert mask(text).anonymized == "Attn: Mr <OSOBA_1>\n\nKind regards,\n<OSOBA_1>"


def test_a_foreign_title_is_no_part_of_a_name():
    from polanon._known_parts import known_parts

    assert known_parts("Frau Wiśniewska", "PERSON", trusted=True) == [
        "Wiśniewska",
        "Wiśniewskiej",
        "Wiśniewską",
    ]
    assert known_parts("Mrs Ottoline Carter", "PERSON", trusted=True) == ["Ottoline", "Carter"]


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


def test_a_feminine_surname_in_another_case_is_masked_in_the_next_turn():
    """The dative "Bielskiej" does not begin with "Bielska", so it went out after the full name."""
    first = mask("Pracownica Natalia Bielska prosi o korektę.")
    text = "Dopisz, że Bielskiej potrzebne jest zaświadczenie."
    second = mask(text, first.mapping)
    assert second.anonymized == "Dopisz, że <OSOBA_2> potrzebne jest zaświadczenie."
    assert Anonymizer.restore(second.anonymized, second.mapping) == text


@pytest.mark.parametrize(
    ("name", "later"),
    [
        ("Natalia Bielska", "Dopisz, że Bielskiej potrzebne jest zaświadczenie."),
        ("Natalia Bielska", "Rozmawiałem z Bielską wczoraj."),
        ("Ewa Nowacka", "Przekaż Nowackiej dokumenty."),
        ("Ewa Nowacka", "Spotkanie z Nowacką w piątek."),
        ("Maria Zawadzka", "Zadzwoń do Zawadzkiej."),
        ("Maria Zawadzka", "Umówiłem się z Zawadzką."),
        ("Anna Nowak-Bielska", "Zadzwoń do Nowak-Bielskiej."),
    ],
)
def test_a_feminine_surname_in_another_case_is_masked_in_the_same_text(name, later):
    text = f"Pracownica {name} prosi o korektę. {later}"
    result = mask(text)
    surname = name.split()[-1]
    assert surname[:-1] not in result.anonymized
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("Anna Polska", "Pracuje w polskiej firmie."),
        ("Natalia Bielska", "Pracuje w bielskiej firmie."),
    ],
)
def test_a_lowercase_adjective_spelled_like_a_known_surname_stays(name, text):
    mapping = {"<OSOBA_1>": name}
    assert mask(text, mapping).anonymized == text


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


def test_a_login_made_of_a_known_surname_is_the_person_too():
    """A backup's settings an agent read held "user.name=Tomasz Malinowski" and the directory
    /home/tmalinowski: the surname was masked, the login made of it went out. A word in lowercase
    that is the surname with an initial, a dot, an underscore or digits around it is the person
    too; the surname alone in lowercase is not, since Zima or Kot are ordinary words as well."""
    text = (
        "user.name=Tomasz Malinowski\n"
        "RESTIC_CACHE_DIR=/home/tmalinowski/.cache\n"
        "ssh t.malinowski@nas, konto malinowski85 i malinowski_t, a malinowski sam nie."
    )
    masked = mask(text).anonymized
    for login in ("tmalinowski", "t.malinowski", "malinowski85", "malinowski_t"):
        assert login not in masked, login
    assert " malinowski sam " in masked


def test_a_surname_with_letters_glued_on_is_no_login():
    """A surname declines as an adjective does, and a lowercase word with an ending glued on,
    "malinowskie", may be an ordinary one, so letters after a surname make a login only after a
    dot, "_" or "-"."""
    text = "Pisze Tomasz Malinowski, a odmiany malinowskie są w katalogu; login malinowski.t."
    masked = mask(text).anonymized
    assert "odmiany malinowskie" in masked
    assert "malinowski.t" not in masked


def test_a_login_names_its_person_wherever_the_surname_stands():
    """A login the name rule reads, "t.malinowski", gives the surname as a known value, so the
    surname written as a name and the other logins made of it are masked as well."""
    text = "Login t.malinowski. Potem Malinowski zalogował się jako tmalinowski."
    masked = mask(text).anonymized
    assert "malinowski" not in masked.lower(), masked


def test_an_e_mail_made_of_a_name_gives_its_login_and_surname():
    """The address was masked whole, but its local part, a first name and a surname, was no known
    value: the login and the surname went out further down the text, next to the person's
    token."""
    text = (
        "Pisała anna.kowalczyk@example.com. Konto akowalczyk ma dostęp, a Kowalczyk prosi o hasło."
    )
    result = mask(text)
    assert "kowalczyk" not in result.anonymized.lower(), result.anonymized
    assert "Kowalczyk" in result.mapping.values()


def test_a_five_letter_surname_gives_logins_written_with_a_separator():
    """A login of a five-letter surname with a dot, "_" or "-" is the person too: "nowak" was
    below the six letters the rule asked for. Glued straight on, a short surname stays a word,
    since "nowakowie" or "kotek" are ordinary ones."""
    text = "Pisze Jan Nowak; konta nowak_j i nowak.j, a nowakowie przyjdą."
    masked = mask(text).anonymized
    assert "nowak_j" not in masked, masked
    assert "nowak.j" not in masked, masked
    assert "nowakowie" in masked


def test_a_short_surname_in_another_case_is_masked_in_a_later_text():
    """After "Katarzyna Lis" and "Tomasz Kot" were masked, "Lisowi" and "Kotowi" went out in the
    clear and "Nowakowi" did not: three letters are too few to be masked inside a longer word,
    and a case ending glued on made one."""
    turn = Anonymizer(default_guards(with_ner=False)).turn()
    turn.mask("Lista: Katarzyna Lis, Tomasz Kot, Jan Nowak.")
    later = [
        "Przekaż Lisowi notatkę.",
        "Przekaż Kotowi i Nowakowi.",
        "Od Lisa nic nie przyszło, z Kotem i Lisem rozmawiałem, u Kotów nikt nie odbiera.",
    ]
    for text in later:
        masked = turn.mask(text)
        assert re.search("Lis|Kot|Nowak", masked) is None, masked
        assert turn.restore(masked) == text


@pytest.mark.parametrize(
    "text",
    [
        "Lista zadań na listopad.",
        "Kotara w oknie, kotlet na talerzu.",
        "Lisy i koty mieszkają w lesie.",
        "Daj lisowi spokój, a kotu mleka.",
        "Lisowski i Kotarski dzwonili.",
    ],
)
def test_a_short_surname_leaves_the_words_that_begin_with_it_alone(text):
    """Only a case ending makes the word the surname: "Lista", "Kotara" or "Lisowski" are no
    "Lis" and no "Kot", and a case in lowercase is the animal."""
    mapping = {"<OSOBA_1>": "Katarzyna Lis", "<OSOBA_2>": "Tomasz Kot"}
    assert mask(text, mapping).anonymized == text


@pytest.mark.parametrize(
    ("name", "later", "forms"),
    [
        ("Adam Kozieł", "Przekaż Koziołowi umowę, a Kozła zaproś.", ["Koziołowi", "Kozła"]),
        ("Adam Kozieł", "Rozmawiałem o Koziele z Kozłem.", ["Koziele", "Kozłem"]),
        ("Paweł Wróbel", "Nie ma Wróbla, przekaż Wróblowi notatkę.", ["Wróbla", "Wróblowi"]),
        ("Jan Bednarek", "Bednarka nie było, Bednarkowi przekaż.", ["Bednarka", "Bednarkowi"]),
        ("Jan Krawiec", "Krawca nie było, byliśmy u Krawców.", ["Krawca", "Krawców"]),
        ("Jan Mróz", "Mroza nie było.", ["Mroza"]),
        ("Tomasz Kot", "Rozmawialiśmy o Kocie.", ["Kocie"]),
    ],
)
def test_a_surname_whose_stem_changes_is_masked_in_its_other_cases(name, later, forms):
    """ "Koziołowi" and "Kozła" do not begin with "Kozieł", nor "Wróbla" with "Wróbel": the
    surname alone was masked after the full name, its cases went out in the clear."""
    turn = Anonymizer(default_guards(with_ner=False)).turn()
    turn.mask(f"Umowę podpisał {name}.")
    masked = turn.mask(later)
    for form in forms:
        assert form not in masked, masked
    assert turn.restore(masked) == later


@pytest.mark.parametrize(
    "text",
    [
        "Na łące stał kozioł, a wróbel siedział na płocie.",
        "Bez kozła i bez wróbla, o kocie ani słowa.",
        "Kozioł skoczył przez płot.",
    ],
)
def test_an_animal_spelled_like_a_known_surname_stays(text):
    mapping = {"<OSOBA_1>": "Adam Kozieł", "<OSOBA_2>": "Paweł Wróbel", "<OSOBA_3>": "Jan Kot"}
    assert mask(text, mapping).anonymized == text


@pytest.mark.parametrize(
    ("first", "later", "forms"),
    [
        (
            "Umowę zawarto z Grzegorzem Lisem.",
            "Czy Lisowi wysłać ją poleconym? Lis nie odbiera.",
            ["Lis"],
        ),
        ("Pismo Jana Kowalskiego.", "Kowalski dzwonił, Kowalskiemu oddzwonię.", ["Kowalski"]),
    ],
)
def test_a_name_first_masked_in_another_case_gives_its_surname_in_the_nominative(
    first, later, forms
):
    """ "Grzegorzem Lisem" gave the part "Lisem", which "Lis" and "Lisowi" do not hold, so both
    went out in the clear in the next message."""
    turn = Anonymizer(default_guards(with_ner=False)).turn()
    turn.mask(first)
    masked = turn.mask(later)
    for form in forms:
        assert form not in masked, masked
    assert turn.restore(masked) == later


def test_the_parts_of_a_call_s_names_are_forgotten_with_its_values():
    """The parts are worked out once per value and kept until `forget`, which the gateway calls
    after each call, as it does for the keys of the values."""
    from polanon import _known_parts, _matching
    from polanon.guards import names_pl

    mask("Umowę podpisał Adam Kozieł.")
    assert _known_parts._known_parts.cache_info().currsize > 0
    _matching.forget()
    assert _known_parts._known_parts.cache_info().currsize == 0
    assert names_pl._noun_surname_forms.cache_info().currsize == 0
    assert names_pl._nominatives.cache_info().currsize == 0


@pytest.mark.parametrize(
    ("name", "later", "forms"),
    [
        ("Anna Duda", "Dudy nie było, przekaż Dudzie notatkę.", ["Dudy", "Dudzie"]),
        ("Jan Sowa", "Z Sową rozmawiałem, Sowie wysłałem umowę.", ["Sową", "Sowie"]),
        ("Ewa Mucha", "Musze przekaż, że Muchy szukał kurier.", ["Musze", "Muchy"]),
    ],
)
def test_a_surname_in_a_is_masked_in_its_other_cases(name, later, forms):
    """No case of "Duda" but the nominative holds "Duda", so the surname alone was masked after
    the full name and "Dudy" and "Dudzie" went out."""
    turn = Anonymizer(default_guards(with_ner=False)).turn()
    turn.mask(f"Umowę podpisał(a) {name}.")
    masked = turn.mask(later)
    for form in forms:
        assert form not in masked, masked
    assert turn.restore(masked) == later


def test_a_bird_spelled_like_a_known_surname_in_a_stays():
    mapping = {"<OSOBA_1>": "Jan Sowa", "<OSOBA_2>": "Anna Wrona"}
    text = "Na drzewie siedziała sowa, a obok wrony; o sowie i wronie ani słowa."
    assert mask(text, mapping).anonymized == text


def test_a_name_first_masked_in_a_case_the_rules_cannot_read_gives_its_nominatives():
    """ "Adama Kozieła" gave the part "Kozieła" alone, so "Kozieł" and "Koziołowi" went out in the
    next message. "Adama" is only a genitive, so the surname is one too, of "Kozieł" or "Kozioł"."""
    text = "Kozieł dzwonił, przekaż Koziołowi notatkę."
    masked = mask(text, {"<OSOBA_1>": "Adama Kozieła"})
    assert "Koz" not in masked.anonymized, masked.anonymized
    assert Anonymizer.restore(masked.anonymized, masked.mapping) == text


def test_a_first_name_that_may_be_a_nominative_gives_no_guess_at_the_surname():
    """ "Kamila Kozieła" may be a woman of that name, whose surname is "Kozieła" itself."""
    text = "Kozieł dzwonił."
    assert mask(text, {"<OSOBA_1>": "Kamila Kozieła"}).anonymized == text
