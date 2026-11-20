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


def test_a_company_listed_with_its_legal_form_is_found_without_it():
    """PolSales added the short name of every client by hand: an e-mail says "Siłownia Testowa",
    the register says "Siłownia Testowa Sp. z o.o.", and the short name went out."""
    terms = ["Siłownia Testowa Sp. z o.o.", "Klub Biegowy Orlik S.A."]
    text = "Siłownia Testowa i KLUB BIEGOWY ORLIK pytają, a Siłownia Testowa Sp. z o.o. płaci."
    assert _found(terms, text) == [
        "Siłownia Testowa",
        "KLUB BIEGOWY ORLIK",
        "Siłownia Testowa Sp. z o.o.",
    ]


@pytest.mark.parametrize(
    "term", ["Budex sp. z o.o.", "Partner Sp. z o.o.", "Kot Sp. j.", "Polska S.A.", "AB, sp. k."]
)
def test_one_word_before_a_legal_form_is_not_masked_alone(term):
    """One word is in too many sentences that do not name the company: "Nasz partner"."""
    text = f"Budex, nasz partner, Partner, mój kot, Polska i AB. Umowa z {term}"
    assert _found([term], text) == [term]


def test_a_shortened_name_keeps_its_letter_case():
    """ "To moje biuro rachunkowe" names no company; "Biuro Rachunkowe" or in capitals does."""
    terms = ["Biuro Rachunkowe Sp. z o.o."]
    assert _found(terms, "To moje biuro rachunkowe.") == []
    assert _found(terms, "Biuro Rachunkowe i BIURO RACHUNKOWE.") == [
        "Biuro Rachunkowe",
        "BIURO RACHUNKOWE",
    ]


@pytest.mark.parametrize("term", ["Kowalski Sp. j.", "Jan Kowalski Sp. j."])
def test_a_person_named_by_a_partnership_stays_one_person(term):
    """ "Kowalski Sp. j." shortened to "Kowalski" cut "Jan Kowalski" into a person "Jan" and a
    company "Kowalski", and the person came back as "Jana Kowalski"."""
    anonymizer = Anonymizer(default_guards(with_ner=False, denylist=[term]))
    result = anonymizer.anonymize_sync("Jan Kowalski przyszedł do Jana Kowalskiego.")
    assert result.anonymized == "<OSOBA_1> przyszedł do <OSOBA_1:dopełniacz>."
    assert result.mapping == {"<OSOBA_1>": "Jan Kowalski"}


@pytest.mark.parametrize(
    "term", ["1234567890", "123-456-78-90", "123-45-67-890", "123 456 78 90", "123.456.78.90"]
)
def test_a_number_is_found_however_its_digits_are_grouped(term):
    text = (
        "NIP 123-456-78-90, 123-45-67-890, 123 456 78 90, 1234567890, 123  456\t78\n90, "
        "123 - 456 - 78 - 90; obcy 12345678901."
    )
    assert _found([term], text) == [
        "123-456-78-90",
        "123-45-67-890",
        "123 456 78 90",
        "1234567890",
        "123  456\t78\n90",
        "123 - 456 - 78 - 90",
    ]


def test_a_listed_number_written_with_more_whitespace_is_still_found():
    """Main matched any whitespace between the parts of a term; a number must as well."""
    text = "Umowa nr 4471  2290  1183, 4471\t2290\n1183."
    assert _found(["4471 2290 1183"], text) == ["4471  2290  1183", "4471\t2290\n1183"]


@pytest.mark.parametrize("term", ["12345", "100200", "12032024"])
def test_a_number_of_fewer_than_nine_digits_keeps_its_grouping(term):
    """Grouped otherwise, six or eight digits are a date, an amount or a postcode."""
    text = f"kod 12-345, data 10.02.00 i 12.03.2024, kwota 100.200 zł; numer {term}."
    assert _found([term], text) == [term]


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
