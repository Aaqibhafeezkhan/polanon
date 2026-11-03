"""The admin denylist: literal terms masked wherever and however they are written."""

import pytest

from polanon import Anonymizer, DenylistGuard, default_guards

ORG = {"ORG"}


def _found(terms: list[str], text: str, enabled=ORG) -> list[str]:
    return [text[s.start : s.end] for s in DenylistGuard(terms).detect(text, enabled)]


@pytest.mark.parametrize(
    "text",
    [
        "Budżet na Projekt Żuraw rośnie.",
        "Budżet na Projekt\xa0Żuraw rośnie.",
        "Budżet na Projekt  Żuraw rośnie.",
        "Budżet na Projekt\nŻuraw rośnie.",
        "Budżet na projekt zuraw rośnie.",
        "Budżet na PROJEKT ŻURAW rośnie.",
    ],
)
def test_a_term_is_found_whatever_the_spacing_case_or_polish_letters(text):
    written = text.removeprefix("Budżet na ").removesuffix(" rośnie.")
    assert _found(["Projekt Żuraw"], text) == [written]


def test_a_term_typed_without_polish_letters_finds_them_in_the_text():
    text = "Rezerwacja w Hotel Pod Różą potwierdzona."
    assert _found(["hotel pod roza"], text) == ["Hotel Pod Różą"]


def test_overlapping_terms_are_masked_as_one_span():
    text = "Kierownikiem jest Projekt Atlas Wiśniewski."
    assert _found(["Projekt Atlas", "Atlas Wiśniewski"], text) == ["Projekt Atlas Wiśniewski"]


def test_of_two_terms_starting_together_the_longer_is_masked():
    assert _found(["Atlas", "Atlas Wiśniewski"], "To Atlas Wiśniewski.") == ["Atlas Wiśniewski"]


@pytest.mark.parametrize("text", ["Atlasowy projekt", "Projekt Atlasu", "Kontratlas"])
def test_a_term_inside_a_longer_word_is_left_alone(text):
    assert _found(["Atlas"], text) == []


def test_nothing_is_returned_without_terms_or_when_org_masking_is_off():
    assert _found([], "Projekt Atlas") == []
    assert _found(["  ", ""], "Projekt Atlas") == []
    assert _found(["Projekt Atlas"], "Projekt Atlas", enabled={"PESEL"}) == []


async def test_overlapping_terms_reach_the_provider_as_one_token():
    terms = ["Projekt Atlas", "Atlas Wiśniewski"]
    anonymizer = Anonymizer(default_guards(with_ner=False, denylist=terms))
    result = await anonymizer.anonymize("Kierownikiem jest Projekt Atlas Wiśniewski.")
    assert result.anonymized == "Kierownikiem jest <FIRMA_1>."
    assert result.mapping == {"<FIRMA_1>": "Projekt Atlas Wiśniewski"}


@pytest.mark.parametrize(
    ("term", "text"),
    [
        pytest.param("Nowak\u2013Kowal", "Umowa z Nowak\u2013Kowal sp.", id="en dash"),
        pytest.param("Projekt Z\u0307uraw", "Raport: Projekt \u017buraw gotowy", id="decomposed"),
        pytest.param("Pro\u00adjekt Atlas", "Pro\u00adjekt Atlas startuje", id="soft hyphen"),
    ],
)
def test_a_term_copied_from_word_a_pdf_or_macos_matches_the_text(term, text):
    """The text is read normalised, the terms were not, and they matched nothing."""
    result = Anonymizer(default_guards(with_ner=False, denylist=[term])).anonymize_sync(text)
    assert list(result.mapping) == ["<FIRMA_1>"]
