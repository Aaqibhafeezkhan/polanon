"""The sweep flags invented tokens and personal data the model produced on its own."""

import pytest

from polanon import PersonNameGuard, RegexGuard, sweep
from polanon.policy import ENTITY_TYPES

ALL = set(ENTITY_TYPES)
GUARDS = [RegexGuard()]
NAME_GUARDS = [RegexGuard(), PersonNameGuard()]


def test_clean_answer_reports_nothing():
    report = sweep("Wszystko w porządku.", {"<EMAIL_1>": "a@b.example"}, GUARDS, ALL)
    assert report.is_clean


def test_invented_token_is_reported():
    report = sweep("Napisz do <OSOBA_7>.", {"<OSOBA_1>": "Jan"}, GUARDS, ALL)
    assert report.unknown_tokens == ["<OSOBA_7>"]


def test_restored_known_value_is_not_reported_as_new():
    report = sweep("Kontakt: a@b.example", {"<EMAIL_1>": "a@b.example"}, GUARDS, ALL)
    assert report.new_spans == []


def test_hallucinated_personal_data_is_reported():
    report = sweep("Przykładowy PESEL to 50810112354.", {}, GUARDS, ALL)
    assert [s.entity_type for s in report.new_spans] == ["PESEL"]
    assert not report.is_clean


def test_known_values_read_differently_by_a_guard_are_not_reported_as_new():
    """A postcode rule reads "80-470" inside the known account "80-470-3"; two known values read
    together as one name; neither is anything the conversation did not already know."""
    from polanon import PersonNameGuard

    guards = [RegexGuard(), PersonNameGuard()]
    assert sweep("Konto: 80-470-3", {"<KONTO_1>": "80-470-3"}, guards, ALL).is_clean
    mapping = {"<OSOBA_1>": "Anna", "<OSOBA_2>": "Nowak"}
    assert sweep("Pani Anna Nowak odpisała.", mapping, guards, ALL).is_clean


def test_a_new_name_next_to_known_ones_is_still_reported():
    from polanon import PersonNameGuard

    mapping = {"<OSOBA_1>": "Anna Nowak"}
    report = sweep("Anna Nowak i Jan Kowalski", mapping, [PersonNameGuard()], ALL)
    assert [s.entity_type for s in report.new_spans] == ["PERSON"]


def test_known_value_repeated_in_the_answer_is_not_reported_as_new():
    report = sweep(
        "Napisz na a@b.example, a kopię na a@b.example.", {"<EMAIL_1>": "a@b.example"}, GUARDS, ALL
    )
    assert report.is_clean


def test_new_identifier_hidden_by_an_invisible_character_is_reported():
    """anonymize() reads through a zero-width space, so the sweep has to as well."""
    answer = f"Nowy PESEL 4405{chr(0x200B)}1401359."
    report = sweep(answer, {"<OSOBA_1>": "Jan Kowalski"}, GUARDS, ALL)
    assert [answer[span.start : span.end] for span in report.new_spans] == [
        f"4405{chr(0x200B)}1401359"
    ]


@pytest.mark.parametrize("name", [f"Jan{chr(0x00A0)}Kowalski", "JAN KOWALSKI"])
def test_known_name_spelled_differently_is_not_reported_as_new(name):
    report = sweep(f"Odpowiedz {name}.", {"<OSOBA_1>": "Jan Kowalski"}, NAME_GUARDS, ALL)
    assert report.is_clean


def test_known_name_with_a_glued_ending_is_not_reported_as_new():
    """A restored token takes the Polish ending the model put after it: the ending is the same
    person, not personal data the model invented."""
    report = sweep(
        "Odpowiedź dla Jan Kowalskia. Przekażę Jan Kowalskiowi.",
        {"<OSOBA_1>": "Jan Kowalski"},
        NAME_GUARDS,
        ALL,
    )
    assert report.new_spans == []


def test_a_different_name_beside_an_inflected_known_one_is_still_reported():
    text = "Odpowiedź dla Jan Kowalskia. Pozdrawia Anna Nowak."
    report = sweep(text, {"<OSOBA_1>": "Jan Kowalski"}, NAME_GUARDS, ALL)
    assert [text[span.start : span.end] for span in report.new_spans] == ["Anna Nowak"]


def test_a_known_name_grown_into_a_longer_one_is_still_reported():
    """Letters glued on the right that do not spell a case ending are new data: a longer surname
    built on the same stem as a known one is not the same person."""
    for known, answer in [
        ("Jan Nowak", "Panie Jan Nowakowski, dzieki."),
        ("Jan Kowal", "Panie Jan Kowalczyk, dzieki."),
        ("Ewa Lis", "Pani Ewa Lisowska, dzieki."),
    ]:
        report = sweep(answer, {"<OSOBA_1>": known}, NAME_GUARDS, ALL)
        assert report.new_spans, f"{known} inside {answer!r} went unreported"


def test_letters_glued_on_the_left_of_a_known_name_are_still_reported():
    report = sweep("Pani Joanna Nowak odpisała.", {"<OSOBA_1>": "Anna Nowak"}, NAME_GUARDS, ALL)
    assert report.new_spans


def test_a_known_phone_number_inside_a_new_pesel_is_still_reported():
    """The extension never crosses digits: the PESEL keeps its first and last digit uncovered."""
    report = sweep("PESEL 50810112354.", {"<TELEFON_1>": "810112354"}, GUARDS, ALL)
    assert [span.entity_type for span in report.new_spans] == ["PESEL"]


@pytest.mark.parametrize(
    ("token", "known", "answer"),
    [
        ("<ADRES_1>", "ul. Długa 5", "Wysyłka na ul. Długa 5a."),
        ("<REJESTRACJA_1>", "WPI 1234", "Auto o tablicach WPI 1234E."),
        ("<IP_1>", "fe80::1", "Serwer to fe80::1a na lanie."),
        ("<EMAIL_1>", "jan@firma.example", "Pisz na jan@firma.examplea."),
    ],
)
def test_a_letter_glued_onto_a_non_name_known_value_is_still_reported(token, known, answer):
    """Only names decline: a letter glued onto any other kind of known value — an address, a plate,
    an IP address, an e-mail — changes the value, so the detection stays reported."""
    report = sweep(answer, {token: known}, GUARDS, ALL)
    assert report.new_spans, f"{known} with a glued letter went unreported in {answer!r}"


def test_overlapping_known_values_stay_clean():
    mapping = {"<OSOBA_1>": "Jan Kowalski", "<FIRMA_1>": "Kowalski Sp. z o.o."}
    report = sweep("Zgloszenie przyjala firma Jan Kowalski Sp. z o.o.", mapping, NAME_GUARDS, ALL)
    assert report.is_clean


def test_a_short_known_name_with_a_glued_ending_is_not_reported_as_new():
    """An ending glued right onto a short known value covers the word: "Jan" restored as "Jana"."""
    report = sweep("Wiadomość od Jana.", {"<OSOBA_1>": "Jan"}, NAME_GUARDS, ALL)
    assert report.new_spans == []
