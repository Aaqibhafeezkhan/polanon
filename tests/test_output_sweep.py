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
