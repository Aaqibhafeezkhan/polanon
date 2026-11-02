"""Company names found by their legal form, without a model."""

import base64
import time

import pytest

from polanon import Anonymizer, CompanyGuard, RegexGuard, default_guards, merge_spans
from polanon.guards.company_pl import COMPANY_SCORE

ORG = {"ORG"}


def _found(text: str, enabled=ORG) -> list[str]:
    return [text[s.start : s.end] for s in CompanyGuard().detect(text, enabled)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Faktura od Nowak Transport Sp. z o.o. z Krakowa", "Nowak Transport Sp. z o.o."),
        ("zakupy w Malinowski Handel S.A., Katowice", "Malinowski Handel S.A."),
        ("Bud-Mar sp. z o.o. sp. k. wystawiła fakturę", "Bud-Mar sp. z o.o. sp. k."),
        ("Umowa z Kowalski i Synowie sp. j. na rok", "Kowalski i Synowie sp. j."),
        ("Obsługę prowadzi Fundacja Dobra Wola od maja", "Fundacja Dobra Wola"),
        ("Zamówienie u P.P.H.U. Kowalski wysłane", "P.P.H.U. Kowalski"),
        ("Zakłady Mięsne Kowalczyk przysłały ofertę", "Zakłady Mięsne Kowalczyk"),
    ],
)
def test_a_legal_form_anchors_the_whole_company_name(text, expected):
    assert _found(text) == [expected]


@pytest.mark.parametrize(
    "form",
    [
        "S.A.",
        "SA",
        "sp. z o.o",
        "sp z oo",
        "SP. Z O.O.",
        "spółka jawna",
        "Spółka Komandytowa",
        "spółka komandytowo-akcyjna",
        "S.K.A.",
        "spółka cywilna",
        "spółka partnerska",
    ],
)
def test_a_legal_form_anchors_the_name_in_every_common_spelling(form):
    text = f"Faktura od Kowalczyk Budownictwo {form} z Krakowa"
    assert _found(text) == [f"Kowalczyk Budownictwo {form}"]


@pytest.mark.parametrize("prefix", ["PPHU", "P.H.U.", "F.H.U.", "FHU", "PUH"])
def test_a_trade_abbreviation_anchors_the_name_of_a_sole_trader(prefix):
    assert _found(f"Faktura od {prefix} Kowalczyk z Krakowa") == [f"{prefix} Kowalczyk"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Umowa z „Kowalczyk Budownictwo” Sp. z o.o. z siedzibą w Krakowie",
            "„Kowalczyk Budownictwo” Sp. z o.o.",
        ),
        (
            'Umowa z "Kowalczyk Budownictwo" Sp. z o.o. z siedzibą w Krakowie',
            '"Kowalczyk Budownictwo" Sp. z o.o.',
        ),
        ("Umowa z Kowalczyk Budownictwo 2000 Sp. z o.o.", "Kowalczyk Budownictwo 2000 Sp. z o.o."),
        ("Umowa z Kowalczyk (Polska) Sp. z o.o. z Krakowa", "Kowalczyk (Polska) Sp. z o.o."),
        ("Umowa z Kowalczyk.pl Sp. z o.o. z Krakowa", "Kowalczyk.pl Sp. z o.o."),
        ("Umowa z eBudex Sp. z o.o. z Krakowa", "eBudex Sp. z o.o."),
        ("Wsparła nas Fundacja „Pomoc Dzieciom” w Warszawie", "Fundacja „Pomoc Dzieciom”"),
        (
            "Przedsiębiorstwo Handlowe „ABC” Sp. z o.o.",
            "Przedsiębiorstwo Handlowe „ABC” Sp. z o.o.",
        ),
    ],
)
def test_a_name_in_quotes_or_with_a_number_or_brackets_is_taken_whole(text, expected):
    assert _found(text) == [expected]


def test_a_long_name_is_taken_from_its_first_word():
    """The founder's surname usually comes first, so a long name must not lose its head."""
    text = "Umowa z Kowalczyk Biuro Rachunkowe Doradztwo Podatkowe Kadry Sp. z o.o. z Radomia"
    assert _found(text) == ["Kowalczyk Biuro Rachunkowe Doradztwo Podatkowe Kadry Sp. z o.o."]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Z poważaniem\nJan Kowalski\nKierownik Sprzedaży\nBudex Sp. z o.o.\n\nPS. Jan wraca.",
            ["Budex Sp. z o.o."],
        ),
        (
            "Faktura VAT Budex Sp. z o.o. nr 5/2024. Prosimy zapłacić Budex Sp. z o.o. do piątku.",
            ["Budex Sp. z o.o.", "Budex Sp. z o.o."],
        ),
        ("Spotkanie będzie w Krakowie. Budex S.A. przyśle dwie osoby.", ["Budex S.A."]),
        (
            "Darowiznę przekazała Fundacja Dobra Wola. Jan Kowalski podziękował.",
            ["Fundacja Dobra Wola"],
        ),
        (
            "Pełnomocnikiem jest Kancelaria Adwokacka Nowak. Pani Ewa Zielińska odbierze akta.",
            ["Kancelaria Adwokacka Nowak"],
        ),
        ("PROSZĘ O FAKTURĘ DLA BUDEX SP. Z O.O.", ["BUDEX SP. Z O.O."]),
        ("Zapłacono 5 Budex Sp. z o.o.", ["Budex Sp. z o.o."]),
    ],
)
def test_a_name_stops_at_a_line_break_a_sentence_end_or_a_heading(text, expected):
    assert _found(text) == expected


def test_a_name_wrapped_onto_the_next_line_is_taken_whole():
    """A long line was wrapped by an editor or a PDF; a signature's short lines are not."""
    text = (
        "Niniejszym potwierdzamy, że umowę zawarto z firmą Kowalczyk\n"
        "Budownictwo Sp. z o.o. w dniu 5 maja."
    )
    assert _found(text) == ["Kowalczyk\nBudownictwo Sp. z o.o."]
    assert _found("Kowalczyk Budownictwo\nSp. z o.o.") == ["Kowalczyk Budownictwo\nSp. z o.o."]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Muster GmbH i Kowalski Handel Sp. z o.o.", ["Kowalski Handel Sp. z o.o."]),
        ("Alfa S.A. i Beta Sp. z o.o.", ["Alfa S.A.", "Beta Sp. z o.o."]),
        ("Kowalski Spółka Jawna i Nowak Sp. z o.o.", ["Kowalski Spółka Jawna", "Nowak Sp. z o.o."]),
    ],
)
def test_a_name_stops_at_another_companys_legal_form(text, expected):
    assert _found(text) == expected


async def test_a_foreign_and_a_polish_company_get_a_token_each():
    text = "Muster GmbH i Kowalski Handel Sp. z o.o. podpisały umowę"
    result = await Anonymizer(default_guards(with_ner=False)).anonymize(text)
    assert result.anonymized == "<FIRMA_1> i <FIRMA_2> podpisały umowę"
    assert result.mapping == {
        "<FIRMA_1>": "Muster GmbH",
        "<FIRMA_2>": "Kowalski Handel Sp. z o.o.",
    }


def test_a_name_split_by_a_connector_stays_one_span():
    """Two patterns describe the same organisation; the guard must not hand over two spans."""
    text = "Sprawę prowadzi Kancelaria Adwokacka Nowak i Wspólnicy s.c. z Poznania"
    assert _found(text) == ["Kancelaria Adwokacka Nowak i Wspólnicy s.c."]


@pytest.mark.parametrize(
    "text",
    [
        "Byliśmy w USA. Adres to ul. Długa 5",
        "spółka z ograniczoną odpowiedzialnością bez nazwy",
        "Dzisiaj rano poszedłem do sklepu i kupiłem chleb.",
        "Warszawa, Kraków i Gdańsk to duże miasta.",
        "Zakład pracy zapewnia szkolenia.",
        "Wyrok SA w Krakowie z 12.03.2019",
        "Prosimy o kontakt. Sp. z o.o. jest formą prawną.",
        "SAMSUNG Galaxy kosztuje 3000 zł",
        "Faktura nr FV/12/2024 z dnia 5 maja 2024",
    ],
)
def test_ordinary_prose_is_left_alone(text):
    assert _found(text) == []


@pytest.mark.parametrize(
    "blob",
    [
        base64.urlsafe_b64encode(bytes(range(256)) * 150).decode(),
        bytes(range(256)).hex().upper() * 100,
    ],
    ids=["base64url", "hex"],
)
def test_a_long_run_of_word_characters_is_scanned_in_linear_time(blob):
    """A pasted base64url blob or hex dump of 50,000 characters once took most of a minute."""
    started = time.perf_counter()
    CompanyGuard().detect(blob, ORG)
    assert time.perf_counter() - started < 1.0


def test_nothing_is_returned_when_org_masking_is_off():
    assert _found("Nowak Transport Sp. z o.o.", enabled={"PESEL"}) == []


def test_the_company_span_outranks_a_name_guess_but_not_a_validated_identifier():
    """merge_spans has to keep the NIP and drop a statistical guess inside the company name."""
    text = "Nowak Transport Sp. z o.o., NIP 8992611767"
    spans = CompanyGuard().detect(text, ORG) + RegexGuard().detect(text, {"NIP"})
    merged = merge_spans(spans)
    covered = sorted(text[s.start : s.end] for s in merged)
    assert covered == ["8992611767", "Nowak Transport Sp. z o.o."]
    assert COMPANY_SCORE < 1.0
