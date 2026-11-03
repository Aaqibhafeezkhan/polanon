"""Checksum validators and the regex guard. These run without a language model."""

import time

import pytest

from polanon import Anonymizer
from polanon.guards.international import InternationalGuard
from polanon.guards.recognizers_pl import (
    RegexGuard,
    is_valid_iban,
    is_valid_ipv4,
    is_valid_ipv6,
    is_valid_luhn,
    is_valid_nip,
    is_valid_pesel,
    is_valid_pl_id,
    is_valid_plate,
    is_valid_regon,
    is_valid_vin,
)
from polanon.policy import ENTITY_TYPES
from polanon.types import merge_spans

ALL = set(ENTITY_TYPES)


@pytest.mark.parametrize(
    ("value", "ok"),
    [("50810112354", True), ("02412900121", True), ("50810112350", False), ("12345678901", False)],
)
def test_pesel(value, ok):
    assert is_valid_pesel(value) is ok


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("5260000005", True),
        ("526-000-00-05", True),
        ("PL8992611767", True),
        ("PL 899-261-17-67", True),
        ("5260000006", False),
        ("1234567890", False),
        ("PL1234567890", False),
    ],
)
def test_nip(value, ok):
    assert is_valid_nip(value) is ok


@pytest.mark.parametrize(
    ("value", "ok"), [("123456785", True), ("123456786", False), ("000000000", True)]
)
def test_regon(value, ok):
    assert is_valid_regon(value) is ok


@pytest.mark.parametrize(
    ("value", "ok"), [("ABC412345", True), ("ABC512345", False), ("CAB500000", True)]
)
def test_pl_id(value, ok):
    assert is_valid_pl_id(value) is ok


def test_iban_and_luhn():
    assert is_valid_iban("PL61109010140000071219812874")
    assert not is_valid_iban("PL61109010140000071219812875")
    assert is_valid_luhn("4111 1111 1111 1111")
    assert not is_valid_luhn("4111 1111 1111 1112")


def test_regex_guard_finds_valid_and_skips_invalid():
    text = (
        "Jan, PESEL 50810112354, NIP 526-000-00-05, tel. 600 100 200, "
        "mail jan@example.com, zły PESEL 50810112350."
    )
    spans = RegexGuard().detect(text, ALL)
    found = {(s.entity_type, text[s.start : s.end]) for s in spans}
    assert ("PESEL", "50810112354") in found
    assert ("NIP", "526-000-00-05") in found
    assert ("EMAIL", "jan@example.com") in found
    assert ("PHONE", "600 100 200") in found
    assert ("PESEL", "50810112350") not in found


def test_regex_guard_respects_enabled_set():
    assert RegexGuard().detect("PESEL 50810112354", {"NIP"}) == []


# --- address, contact and device identifiers added after the first field test ---------------

ADDRESS_CASES = [
    ("mieszkam przy al. Jana Pawła II 43A m. 12.", "al. Jana Pawła II 43A m. 12"),
    ("ul. Krótka 5", "ul. Krótka 5"),
    ("ul. 3 Maja 12/4", "ul. 3 Maja 12/4"),
    ("Adres: os. Tysiąclecia 15", "os. Tysiąclecia 15"),
    ("pl. Zbawiciela 2", "pl. Zbawiciela 2"),
    ("ul. Jana III Sobieskiego 22", "ul. Jana III Sobieskiego 22"),
    # A patron's title, a flat after "lok." and no space after the prefix's dot.
    ("na adres ul. gen. Andersa 15, 00-159 Warszawa.", "ul. gen. Andersa 15"),
    ("na adres ul. ks. Skorupki 5, 00-546 Warszawa.", "ul. ks. Skorupki 5"),
    ("na adres ul. św. Filipa 17, 31-150 Kraków.", "ul. św. Filipa 17"),
    ("na adres ul. prof. Szafera 3, 31-543 Kraków.", "ul. prof. Szafera 3"),
    ("na adres al. Kard. Wyszyńskiego 10, 70-200 Szczecin.", "al. Kard. Wyszyńskiego 10"),
    ("na adres ul. Długa 5 lok. 12, 00-238 Warszawa.", "ul. Długa 5 lok. 12"),
    ("na adres ul.Długa 5, 00-238 Warszawa.", "ul.Długa 5"),
]


@pytest.mark.parametrize(("text", "expected"), ADDRESS_CASES)
def test_street_address_is_masked_as_one_unit(text, expected):
    spans = [s for s in RegexGuard().detect(text, {"ADDRESS"}) if s.entity_type == "ADDRESS"]
    assert [text[s.start : s.end] for s in spans] == [expected]


def test_address_match_fails_fast_when_a_street_prefix_precedes_a_run_of_capital_i():
    """OCR reads vertical table rules as "I"; the failed match used to try 2^n readings of them."""
    started = time.perf_counter()
    RegexGuard().detect("Adres: ul. " + "I " * 40 + "| podpis", {"ADDRESS"})
    assert time.perf_counter() - started < 1.0


def test_email_match_stays_linear_when_a_long_run_has_no_at_sign():
    """A pasted hex dump or base64 blob was rescanned from every one of its positions."""
    started = time.perf_counter()
    RegexGuard().detect("a" * 200_000, {"EMAIL"})
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("e-mail: zofia.żak@example.com", "zofia.żak@example.com"),
        ("kontakt@przykład.example", "kontakt@przykład.example"),
    ],
)
def test_email_with_polish_letters_is_masked_as_one_unit(text, expected):
    """ASCII-only EMAIL left diacritics outside the token or skipped the address (#7)."""
    spans = [s for s in RegexGuard().detect(text, {"EMAIL"}) if s.entity_type == "EMAIL"]
    assert [text[s.start : s.end] for s in spans] == [expected]


def test_address_does_not_swallow_the_postcode_that_follows():
    text = "ul. Krótka 5, 00-950 Warszawa"
    found = {(s.entity_type, text[s.start : s.end]) for s in RegexGuard().detect(text, ALL)}
    assert ("ADDRESS", "ul. Krótka 5") in found
    assert ("POSTAL_CODE", "00-950") in found


@pytest.mark.parametrize(
    "text",
    ["fax (22) 123 45 68", "tel. 22 5551234", "+48 22 123 45 67", "785-112-334", "800 100 200"],
)
def test_phone_formats_used_in_poland_are_detected(text):
    assert any(s.entity_type == "PHONE" for s in RegexGuard().detect(text, {"PHONE"}))


def test_mac_address_is_detected():
    text = "MAC 00:1B:44:11:3A:B7"
    spans = RegexGuard().detect(text, {"MAC_ADDRESS"})
    assert [text[s.start : s.end] for s in spans] == ["00:1B:44:11:3A:B7"]


def test_vin_is_detected_but_a_plain_number_run_is_not():
    assert RegexGuard().detect("VIN WVWZZZ1JZ3W386752", {"VIN"})
    assert RegexGuard().detect("kod 12345678901234567", {"VIN"}) == []


@pytest.mark.parametrize(
    "text", ["Nr rejestracyjny WA 12345", "nr rej. KR 8842X", "tablice: GD 1234A"]
)
def test_plate_is_detected_when_labelled(text):
    assert any(s.entity_type == "PLATE" for s in RegexGuard().detect(text, {"PLATE"}))


@pytest.mark.parametrize(
    "text",
    [
        "Norma ISO 27001 i RODO 2016/679",
        "Kod produktu XYZ1234 oraz SKU AB123",
        "Zamówienie ORD 2024 w systemie SAP",
    ],
)
def test_unlabelled_codes_are_not_mistaken_for_plates(text):
    assert RegexGuard().detect(text, {"PLATE"}) == []


@pytest.mark.parametrize(
    ("text", "plate"),
    [
        ("tablica rejestracyjna: PO 5RX12", "PO 5RX12"),
        ("Tablice rejestracyjne: PO 5RX12", "PO 5RX12"),
        ("pojazd o numerze rejestracyjnym KR 8842X", "KR 8842X"),
        ("nr rejestracyjny pojazdu: KR 8842X", "KR 8842X"),
        ("numery rejestracyjne: KR 4567C", "KR 4567C"),
        ("z numerem rejestracyjnym SK 98765", "SK 98765"),
        ("numeru rejestracyjnego DW 5544P", "DW 5544P"),
        ("nr rej. kr 8842x", "kr 8842x"),
    ],
)
def test_plate_is_detected_when_the_label_is_inflected(text, plate):
    spans = RegexGuard().detect(text, {"PLATE"})
    assert [text[s.start : s.end] for s in spans] == [plate]


@pytest.mark.parametrize(
    "text",
    [
        "Tablica na 2024 rok jest gotowa.",
        "Rejestracja do 12345 osób jest otwarta.",
        "Tablice ze 1500 zł kosztują więcej.",
        "Rejestracja od 2025 roku obowiązkowa.",
    ],
)
def test_plate_is_not_detected_when_a_preposition_and_a_number_follow_the_label(text):
    assert RegexGuard().detect(text, {"PLATE"}) == []


def test_plate_keeps_its_type_when_it_also_reads_as_a_us_zip_code():
    """To the international rules "WA 12345" is a Washington ZIP code; the label says plate."""
    text = "Samochód o numerze rejestracyjnym WA 12345 wjechał na parking."
    detections = RegexGuard().detect(text, ALL) + InternationalGuard().detect(text, ALL)
    spans = merge_spans(detections, text)
    assert [(s.entity_type, text[s.start : s.end]) for s in spans] == [("PLATE", "WA 12345")]


def test_only_the_identifier_is_masked_not_its_label():
    text = "Spółka wpisana do KRS: 0000987654"
    spans = RegexGuard().detect(text, {"KRS"})
    assert [text[s.start : s.end] for s in spans] == ["0000987654"]


@pytest.mark.parametrize(
    ("text", "krs"),
    [
        ("Spółka wpisana do KRS nr 0000987654, kapitał 5000 zł.", "0000987654"),
        ("Sąd Rejonowy, KRS 987654, NIP 5260000005.", "987654"),
    ],
)
def test_krs_is_detected_when_written_with_nr_or_without_leading_zeros(text, krs):
    """An unpadded "KRS 987654" passes the ID card check digit and was masked, label and all."""
    spans = merge_spans(RegexGuard().detect(text, ALL), text)
    found = [(s.entity_type, text[s.start : s.end]) for s in spans]
    assert ("KRS", krs) in found
    assert all(entity_type != "PL_ID" for entity_type, _ in found)


def test_mac_address_is_detected_in_cisco_dotted_notation():
    text = "Switch: show mac address-table -> 001b.4411.3ab7 na porcie Gi0/1"
    spans = RegexGuard().detect(text, {"MAC_ADDRESS"})
    assert [text[s.start : s.end] for s in spans] == ["001b.4411.3ab7"]


def test_mac_address_is_not_detected_when_a_dotted_number_has_only_digits():
    assert RegexGuard().detect("Wersja 1.2.3, build 2024.1015.0001.", {"MAC_ADDRESS"}) == []


def test_regon_is_masked_whole_when_the_branch_code_is_written_apart():
    text = "REGON 123456785 10003 (oddział)"
    spans = merge_spans(RegexGuard().detect(text, {"REGON"}), text)
    assert [text[s.start : s.end] for s in spans] == ["123456785 10003"]


def test_regon_is_still_found_when_some_other_number_follows_it():
    text = "REGON 123456785 12345 (numer sprawy)"
    spans = merge_spans(RegexGuard().detect(text, {"REGON"}), text)
    assert [text[s.start : s.end] for s in spans] == ["123456785"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Data urodzenia: 12.05.1985", "12.05.1985"),
        ("urodzony 5 marca 1990 roku", "5 marca 1990"),
        ("data 2024-03-15", "2024-03-15"),
    ],
)
def test_dates_are_detected_when_the_admin_enables_them(text, expected):
    spans = RegexGuard().detect(text, {"DATE"})
    assert [text[s.start : s.end] for s in spans] == [expected]


@pytest.mark.parametrize(
    ("validator", "value"),
    [
        (is_valid_pesel, "5081011235"),  # ten digits
        (is_valid_pesel, "50810112355"),  # checksum off by one
        (is_valid_pesel, "90993112341"),  # month 99 exists in no calendar
        (is_valid_pesel, "90013212349"),  # day 32
        (is_valid_nip, "526000000"),  # nine digits
        (is_valid_nip, "5260000006"),  # checksum off by one
        (is_valid_regon, "14170791"),  # eight digits
        (is_valid_regon, "141707918"),  # checksum off by one
        (is_valid_regon, "1417079190000"),  # thirteen digits
        (is_valid_regon, "14170791900001"),  # fourteen digits, bad second checksum
        (is_valid_pl_id, "AB412345"),  # two letters
        (is_valid_pl_id, "ABC412346"),  # check digit off by one
        (is_valid_iban, "PL6110901014000007121981287"),  # one digit short
        (is_valid_iban, "PL61109010140000071219812875"),  # mod-97 fails
        (is_valid_luhn, "12345"),  # too short to be a card
        (is_valid_luhn, "4111111111111112"),  # checksum off by one
        (is_valid_vin, "WVWZZZ1KZAW12345"),  # sixteen characters
        (is_valid_vin, "WVWIZZ1KZAW123456"),  # I is never used in a VIN
        (is_valid_vin, "12345678901234567"),  # seventeen digits are some other number
        (is_valid_plate, "K8492J"),  # one-letter area code
        (is_valid_plate, "KRAKOWS"),  # no digit in the tail
    ],
)
def test_a_malformed_identifier_is_rejected(validator, value):
    assert validator(value) is False


@pytest.mark.parametrize(
    ("validator", "value"),
    [
        (is_valid_regon, "14170791900001"[:9]),
        (is_valid_vin, "WVWZZZ1KZAW123456"),
        (is_valid_plate, "KR 8492J"),
        (is_valid_luhn, "4111111111111111"),
        (is_valid_pesel, "508101 12354"),  # birth date and serial number apart
    ],
)
def test_a_well_formed_identifier_is_accepted(validator, value):
    assert validator(value) is True


def test_a_fourteen_digit_regon_with_both_checksums_is_accepted():
    """The long REGON carries a second checksum over the first nine digits plus the branch code."""
    base = "141707919"
    weights = (2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8)
    for branch in range(10000, 10100):
        candidate = f"{base}{branch}"
        total = sum(int(c) * w for c, w in zip(candidate[:13], weights, strict=True)) % 11
        if (0 if total == 10 else total) == int(candidate[13]):
            assert is_valid_regon(candidate)
            return
    pytest.fail("no valid 14-digit REGON found in the search range")


def test_a_polish_account_number_without_the_pl_prefix_is_still_an_iban():
    text = "Przelew na konto 61 1090 1014 0000 0712 1981 2874 do piątku."
    spans = RegexGuard().detect(text, {"IBAN"})
    assert [text[s.start : s.end] for s in spans] == ["61 1090 1014 0000 0712 1981 2874"]


PL_IBAN = "PL61 1090 1014 0000 0712 1981 2874"


@pytest.mark.parametrize(
    ("text", "iban"),
    [
        (f"Konto: {PL_IBAN} PLN", PL_IBAN),
        (f"Konto {PL_IBAN} BIC WBKPPLPP", PL_IBAN),
        (f"Nr konta: {PL_IBAN} PKO BP, termin 14 dni.", PL_IBAN),
        (f"IBAN {PL_IBAN} SWIFT WBKPPLPP", PL_IBAN),
        (f"Konto {PL_IBAN} 2024 r.", PL_IBAN),
        (f"Przelew na {PL_IBAN} PLN 1200", PL_IBAN),
        (f"Przelew na {PL_IBAN} 100 zł", PL_IBAN),
        ("Rachunek PL61109010140000071219812874 EUR", "PL61109010140000071219812874"),
        ("Konto PL61109010140000071219812874 PKO BP do piątku.", "PL61109010140000071219812874"),
        ("IBAN: SE45 5000 0000 0583 9825 7466 BIC: ESSESESS", "SE45 5000 0000 0583 9825 7466"),
        ("IBAN: AT61 1904 3002 3457 3201 EUR", "AT61 1904 3002 3457 3201"),
        ("IBAN: BE68 5390 0754 7034 BIC GEBABEBB", "BE68 5390 0754 7034"),
    ],
)
def test_iban_is_cut_at_its_registered_length_when_a_short_word_follows(text, iban):
    """The pattern took "PLN" or "BIC" for one more group, so the checksum failed on the lot."""
    spans = RegexGuard().detect(text, {"IBAN"})
    assert [text[s.start : s.end] for s in spans] == [iban]


@pytest.mark.parametrize(
    ("text", "entity_type", "value"),
    [
        ("moje konto: pl61 1090 1014 0000 0712 1981 2874", "IBAN", PL_IBAN.lower()),
        (
            "przelej na pl61109010140000071219812874 do piątku",
            "IBAN",
            "pl61109010140000071219812874",
        ),
        ("IBAN gb82 west 1234 5698 7654 32", "IBAN", "gb82 west 1234 5698 7654 32"),
        ("IBAN de89 3704 0044 0532 0130 00", "IBAN", "de89 3704 0044 0532 0130 00"),
        ("IBAN ch93 0076 2011 6238 5295 7", "IBAN", "ch93 0076 2011 6238 5295 7"),
        ("mój dowód to abc412345, dzięki", "PL_ID", "abc412345"),
        ("vin auta: wvwzzz1jz3w386752", "VIN", "wvwzzz1jz3w386752"),
    ],
)
def test_identifier_is_detected_when_typed_in_lowercase(text, entity_type, value):
    spans = merge_spans(RegexGuard().detect(text, ALL), text)
    assert [(s.entity_type, text[s.start : s.end]) for s in spans] == [(entity_type, value)]


@pytest.mark.parametrize(
    "text",
    [
        "kod ab12 abcd efgh ijkl w systemie",  # no such country, so not an IBAN in lowercase
        "dla 120000 osób, ich 250000 głosów",  # a short word and a number, not an ID card
    ],
)
def test_lowercase_text_is_not_mistaken_for_an_iban_or_an_id_card(text):
    assert RegexGuard().detect(text, {"IBAN", "PL_ID"}) == []


@pytest.mark.parametrize(
    ("text", "card"),
    [
        ("Karta 4111 1111 1111 1111 wygasa 12/27.", "4111 1111 1111 1111"),
        ("Dane karty: 4111 1111 1111 1111 12/27, CVV na odwrocie.", "4111 1111 1111 1111"),
        ("Pozycja 3 4111 1111 1111 1111 obciążona dwukrotnie.", "4111 1111 1111 1111"),
        ("tel. 600 700 800 4111 1111 1111 1111", "4111 1111 1111 1111"),
        ("Karta 4111-1111-1111-1111.", "4111-1111-1111-1111"),
        ("Amex 3782 822463 10005 do zwrotu.", "3782 822463 10005"),
        ("Numer 4111111111111111, ważna do 12/27.", "4111111111111111"),
    ],
)
def test_card_number_is_masked_exactly_when_other_numbers_stand_next_to_it(text, card):
    """The whole run of numbers used to be tested, failed the checksum and left the card in clear;
    and the span ended on the separator after the card, gluing the token to the next word."""
    spans = RegexGuard().detect(text, {"CREDIT_CARD"})
    assert [text[s.start : s.end] for s in spans] == [card]


@pytest.mark.parametrize("value", ["0048 600 700 800", "0048 512 345 678", "0048512345678"])
def test_phone_is_not_typed_as_a_card_when_dialled_with_0048(value):
    """These pass the Luhn check, but no card number starts with 0, so the phone keeps its type."""
    assert is_valid_luhn(value)
    text = f"Proszę dzwonić: {value}"
    spans = merge_spans(RegexGuard().detect(text, ALL), text)
    assert [(s.entity_type, text[s.start : s.end]) for s in spans] == [("PHONE", value)]


@pytest.mark.parametrize(
    "text",
    [
        "Proszę o korektę faktury FV 2024123 z dnia 5 marca, kwota się nie zgadza.",
        "Zamówienie ZK 1000457 nie dotarło, a WZ 2024001 wystawiono wczoraj.",
        "Numer faktury: FV2024001",
        "Rozchód RW 7654321 zaksięgowany.",
        "Nowy numer zamówienia: zk1234567.",
    ],
)
def test_document_number_is_not_mistaken_for_a_passport(text):
    assert RegexGuard().detect(text, {"PL_PASSPORT"}) == []


@pytest.mark.parametrize(
    ("text", "passport"),
    [
        ("Paszport nr AB1234567", "AB1234567"),
        ("nr paszportu: EA 1234567", "EA 1234567"),
        ("passport no. EA1234567", "EA1234567"),
        ("mój paszport to ea1234567", "ea1234567"),
        ("nr paszportu: FV 1234567", "FV 1234567"),  # the label outweighs the document symbol
        ("Dane: Jan Kowalski, EA 1234567, ważny do 2030 r.", "EA 1234567"),
    ],
)
def test_passport_is_detected_when_labelled_or_not_a_document_symbol(text, passport):
    spans = RegexGuard().detect(text, {"PL_PASSPORT"})
    assert [text[s.start : s.end] for s in spans] == [passport]


@pytest.mark.parametrize(
    ("text", "amount"),
    [
        ("Wartość umowy: PLN 100000 netto, płatne w dwóch ratach.", "PLN 100000"),
        ("Kwota kredytu EUR 124000, oprocentowanie 5%.", "EUR 124000"),
        ("Budżet projektu: USD 120000, z czego połowa w Q1.", "USD 120000"),
        ("Wynagrodzenie roczne PLN 155000 brutto.", "PLN 155000"),
    ],
)
def test_amount_is_not_an_id_card_when_it_opens_with_a_currency_code(text, amount):
    """Each of these amounts passes the ID card's mod-10 check, as one in ten of them does."""
    assert is_valid_pl_id(amount)
    assert RegexGuard().detect(text, {"PL_ID"}) == []


def test_id_card_with_a_currency_series_is_detected_after_an_id_card_label():
    text = "Seria i numer dowodu: CHF912345"
    spans = RegexGuard().detect(text, {"PL_ID"})
    assert [text[s.start : s.end] for s in spans] == ["CHF912345"]


@pytest.mark.parametrize(
    ("text", "entity_type", "value"),
    [
        ("Numer PESEL: 5 0 8 1 0 1 1 2 3 5 4", "PESEL", "5 0 8 1 0 1 1 2 3 5 4"),
        ("PESEL 508101 12354 (data urodzenia i numer)", "PESEL", "508101 12354"),
        ("Identyfikator podatkowy NIP: 5 2 6 0 0 0 0 0 0 5", "NIP", "5 2 6 0 0 0 0 0 0 5"),
        (
            "Numer rachunku: 6 1 1 0 9 0 1 0 1 4 0 0 0 0 0 7 1 2 1 9 8 1 2 8 7 4",
            "IBAN",
            "6 1 1 0 9 0 1 0 1 4 0 0 0 0 0 7 1 2 1 9 8 1 2 8 7 4",
        ),
    ],
)
def test_identifier_is_detected_when_its_digits_are_spaced(text, entity_type, value):
    """Forms print one box per digit, and text extraction keeps the gaps between them."""
    spans = merge_spans(RegexGuard().detect(text, ALL), text)
    assert [(s.entity_type, text[s.start : s.end]) for s in spans] == [(entity_type, value)]


@pytest.mark.parametrize(
    "text", ["Zamówienie 123456 12345 w drodze.", "Pozycje: 1 2 3 4 5 6 7 8 9 0 1 2"]
)
def test_spaced_digits_are_not_a_pesel_when_the_checksum_fails(text):
    assert RegexGuard().detect(text, {"PESEL", "NIP"}) == []


@pytest.mark.parametrize(
    "value",
    [
        "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
        "2001:db8::ff00:42:8329",
        "::1",
        "fe80::1ff:fe23:4567:890a",
        "::ffff:192.168.0.1",
    ],
)
def test_an_ipv6_address_is_detected_in_any_of_its_written_forms(value):
    """Merged, because the IPv4-mapped form matches both rules and merge_spans is what decides."""
    text = f"Nowy adres to {value}, port 8443."
    spans = merge_spans(RegexGuard().detect(text, {"IP_ADDRESS"}))
    assert [text[s.start : s.end] for s in spans] == [value]


@pytest.mark.parametrize(
    "text",
    [
        "MAC 3C:22:FB:1A:0B:9E nie jest adresem IP.",
        "Godzina 12:30, cena 10:1, stosunek 3:4.",
        "Zobacz https://example.test:8443/panel",
    ],
)
def test_something_that_only_looks_like_ipv6_is_left_alone(text):
    assert RegexGuard().detect(text, {"IP_ADDRESS"}) == []


def test_a_mac_address_is_still_a_mac_address():
    text = "Interfejs ma MAC 3C:22:FB:1A:0B:9E."
    spans = RegexGuard().detect(text, {"MAC_ADDRESS"})
    assert [text[s.start : s.end] for s in spans] == ["3C:22:FB:1A:0B:9E"]


@pytest.mark.parametrize(
    "value", ["0048601234567", "0048 601 234 567", "+48601234567", "601 234 567"]
)
def test_a_polish_number_is_found_however_it_was_dialled(value):
    text = f"Zadzwoń {value} po siedemnastej."
    spans = RegexGuard().detect(text, {"PHONE"})
    assert [text[s.start : s.end] for s in spans] == [value]


@pytest.mark.parametrize(
    ("validator", "value"),
    [
        (is_valid_ipv4, "999.1.1.1"),
        (is_valid_ipv4, "10.0.0"),
        (is_valid_ipv6, "2001:db8::ff00::1"),  # two "::" abbreviations
        (is_valid_ipv6, "3C:22:FB:1A:0B:9E"),  # a MAC address, not an address
        (is_valid_ipv6, "12:30"),
    ],
)
def test_something_shaped_like_an_address_but_invalid_is_rejected(validator, value):
    assert validator(value) is False


@pytest.mark.parametrize(
    "text",
    [
        "ITEM # DESCRIPTION QTY UNIT PRICE LINE TOTAL M0061 Catalyst 1 liter",
        "Total VAT     214.57",
        "Portal Klienta 3 dostępny",
    ],
)
def test_a_word_that_merely_ends_in_a_street_prefix_is_not_an_address(text):
    """A total line read as "al. M0061 Catalyst 1" and put a token inside the word TOTAL."""
    assert RegexGuard().detect(text, {"ADDRESS"}) == []


@pytest.mark.parametrize("text", ["al. Jerozolimskie 44", "Al. Niepodległości 18", "ul. Długa 5"])
def test_a_real_street_prefix_still_starts_an_address(text):
    spans = RegexGuard().detect(text, {"ADDRESS"})
    assert [text[s.start : s.end] for s in spans] == [text]


def masked(text: str, enabled=ALL) -> str:
    return Anonymizer([RegexGuard()], enabled=set(enabled)).anonymize_sync(text).anonymized


def test_pesel_with_check_digit_zero_is_masked():
    """One PESEL in ten ends in 0: its weighted sum is a multiple of ten."""
    assert is_valid_pesel("50810110048")
    assert masked("PESEL 50810110048") == "PESEL <PESEL_1>"


def test_pesel_of_someone_born_on_the_31st_is_masked():
    assert is_valid_pesel("50813110007")
    assert masked("PESEL 50813110007") == "PESEL <PESEL_1>"


@pytest.mark.parametrize("pesel", ["50920110004", "45921510007"])
def test_pesel_of_someone_born_in_december_is_masked(pesel):
    assert is_valid_pesel(pesel)
    assert masked(f"PESEL {pesel}") == "PESEL <PESEL_1>"


def test_nine_digit_regon_with_check_digit_zero_is_masked():
    assert is_valid_regon("123456160")
    assert masked("REGON 123456160", {"REGON"}) == "REGON <REGON_1>"


@pytest.mark.parametrize(
    "label",
    ["tel.", "tel. kom.", "kom.", "nr tel.", "telefon", "komórka", "fax", "phone"],
)
def test_nine_digit_number_after_phone_label_is_a_phone_not_regon(label):
    """A checksum-valid nine-digit phone after a phone label was labelled REGON (#8)."""
    text = f"{label} 123456785"
    spans = merge_spans(RegexGuard().detect(text, {"REGON", "PHONE"}), text)
    assert [(s.entity_type, text[s.start : s.end]) for s in spans] == [("PHONE", "123456785")]


def test_nine_digit_number_after_regon_label_stays_regon():
    text = "REGON 123456785"
    spans = merge_spans(RegexGuard().detect(text, {"REGON", "PHONE"}), text)
    assert [(s.entity_type, text[s.start : s.end]) for s in spans] == [("REGON", "123456785")]


@pytest.mark.parametrize("text", ["numer 123456785", "hotel 123456785", "Pastel 123456785"])
def test_unlabelled_checksum_valid_nine_digits_stay_regon(text):
    """A phone label starts a word: the "tel" at the end of "hotel" is none."""
    spans = merge_spans(RegexGuard().detect(text, {"REGON", "PHONE"}), text)
    assert [(s.entity_type, text[s.start : s.end]) for s in spans] == [("REGON", "123456785")]


def test_fourteen_digit_number_with_an_invalid_company_part_is_not_a_regon():
    assert not is_valid_regon("12345678000005")
    assert masked("Zamówienie 12345678000005", {"REGON"}) == "Zamówienie 12345678000005"


def test_fourteen_digit_regon_with_check_digit_zero_is_masked():
    assert is_valid_regon("12345678500010")
    assert masked("REGON 12345678500010", {"REGON"}) == "REGON <REGON_1>"


def test_shortest_registered_iban_is_masked():
    assert is_valid_iban("NO9386011117947")
    assert masked("Konto NO93 8601 1117 947 w banku.", {"IBAN"}) == "Konto <IBAN_1> w banku."


def test_iban_with_wrong_check_digits_is_not_masked():
    """PL61... is valid; PL60... leaves remainder 0, which no real IBAN has."""
    assert not is_valid_iban("PL60109010140000071219812874")
    text = "Konto PL60 1090 1014 0000 0712 1981 2874 w banku."
    assert masked(text, {"IBAN"}) == text


@pytest.mark.parametrize("card", ["4000123412341235005", "4000 1234 1234 1235 005"])
def test_nineteen_digit_card_is_masked(card):
    assert masked(f"Karta {card} ważna.", {"CREDIT_CARD"}) == "Karta <KARTA_1> ważna."


def test_hyphenated_iban_followed_by_a_currency_is_cut_at_its_length():
    text = "Konto: PL61-1090-1014-0000-0712-1981-2874-PLN"
    assert masked(text, {"IBAN"}) == "Konto: <IBAN_1>-PLN"


def test_lowercase_identifier_passing_mod_97_is_not_an_iban():
    """A session id or hash passes mod 97 one time in 97; only capitals get the fallback."""
    text = "Sesja zz64deadbeefcafe1234 wygasła."
    assert is_valid_iban("zz64deadbeefcafe1234")
    assert masked(text, {"IBAN"}) == text


def test_bare_26_digit_number_with_a_wrong_checksum_is_not_an_iban():
    text = "Nr zamówienia 12 3456 7890 1234 5678 9012 3456 w systemie."
    assert masked(text, {"IBAN"}) == text


def test_card_is_not_read_out_of_the_middle_of_a_longer_number():
    """4000 1234 1234 0004 passes Luhn, but here it is a slice of a 24-digit reference."""
    text = "Numer referencyjny 1111 4000 1234 1234 0004 9999 w systemie."
    assert masked(text, {"CREDIT_CARD"}) == text


@pytest.mark.parametrize("card", ["4000 1234 1234 4", "4000123412344"])
def test_thirteen_digit_card_is_masked(card):
    assert masked(f"Karta {card} ważna.", {"CREDIT_CARD"}) == "Karta <KARTA_1> ważna."


def test_three_digit_group_ends_a_card_candidate():
    """ "4000 1234 123 4004" passes Luhn as a whole, but a card never has a short inner group."""
    text = "Kod 4000 1234 123 4004 na fakturze."
    assert masked(text, {"CREDIT_CARD"}) == text


def test_longest_valid_card_is_taken_when_a_prefix_also_passes_luhn():
    """Taking the 16-digit prefix sends the last three digits of the card in the clear."""
    text = "Karta 4000 1234 1234 1234 008 ważna."
    assert masked(text, {"CREDIT_CARD"}) == "Karta <KARTA_1> ważna."


def test_split_number_with_a_pesel_checksum_but_no_label_is_not_a_pesel():
    """A date-prefixed order number has the same shape and passes the checksum one time in ten."""
    text = "Zamówienie 508101-12354 wysłane."
    assert masked(text, {"PESEL"}) == text


def test_lowercase_passport_after_a_longer_label_is_masked():
    assert masked("Numer paszportu numer: ab1234567", {"PL_PASSPORT"}) == (
        "Numer paszportu numer: <PASZPORT_1>"
    )


def test_lowercase_invoice_symbol_after_a_registration_label_is_not_a_plate():
    """ "data rejestracji fv 12345" is the day an invoice was booked, whatever the letter case."""
    text = "Data rejestracji fv 12345 w systemie."
    assert masked(text, {"PLATE"}) == text


@pytest.mark.parametrize(
    "text",
    [
        "konto PL61 1090 1014\n0000 0712 1981 2874 koniec",
        "konto PL61 1090 1014 0000\r\n0712 1981 2874 koniec",
    ],
)
def test_iban_wrapped_onto_the_next_line_is_masked(text):
    """A narrow column in a PDF or an e-mail breaks an IBAN between its groups."""
    assert masked(text, {"IBAN"}) == "konto <IBAN_1> koniec"


def test_iban_at_the_end_of_a_line_does_not_take_the_next_line_with_it():
    text = "PL61 1090 1014 0000 0712 1981 2874\n2024 Faktura"
    assert masked(text, {"IBAN"}) == "<IBAN_1>\n2024 Faktura"


SECOND_IBAN = "PL07 1090 1014 0000 0000 0000 0001"  # made-up digits with valid check digits


@pytest.mark.parametrize(
    "joint",
    [
        pytest.param("\n", id="one per line"),
        pytest.param(" oraz ", id="joined by a word"),
        pytest.param(", ", id="comma"),
    ],
)
def test_the_second_of_two_grouped_ibans_is_masked_too(joint):
    """The first candidate ran on into the second IBAN, and the scan resumed after it."""
    text = f"Konta: {PL_IBAN}{joint}{SECOND_IBAN}"
    ibans = [
        text[span.start : span.end]
        for span in RegexGuard().detect(text, ALL)
        if span.entity_type == "IBAN"
    ]
    assert ibans == [PL_IBAN, SECOND_IBAN]


def _found(text, entity_type):
    return [
        text[span.start : span.end]
        for span in RegexGuard().detect(text, ALL)
        if span.entity_type == entity_type
    ]


@pytest.mark.parametrize(
    "text",
    ["ur. 15.03.1985r. w Krakowie", "urodzony 15 marca 1985r.", "data urodzenia: 1985-03-15r"],
)
def test_a_birth_date_with_the_year_abbreviation_glued_on_is_masked(text):
    """A word boundary after the year let "1985r." go out whole."""
    assert len(_found(text, "BIRTH_DATE")) == 1
    assert not _found(text, "BIRTH_DATE")[0].endswith("r")


@pytest.mark.parametrize(
    "text",
    [
        "wpisana do KRS pod numerem 0000123456",
        "KRS pod nr 0000123456",
        "KRS No. 0000123456",
        "KRS: 0000123456",
    ],
)
def test_krs_is_found_in_the_usual_phrasings(text):
    assert _found(text, "KRS") == ["0000123456"]


@pytest.mark.parametrize(
    ("text", "address"),
    [
        ("Mieszka przy ulicy Długiej 5 od lat.", "ulicy Długiej 5"),
        ("Przeprowadził się na ulicę Polną 12/4.", "ulicę Polną 12/4"),
        ("Biuro na placu Wolności 2.", "placu Wolności 2"),
        ("Dom w alei Róż 3.", "alei Róż 3"),
        ("Mieszkanie na osiedlu Słonecznym 14.", "osiedlu Słonecznym 14"),
    ],
)
def test_an_address_after_an_inflected_street_word_is_masked(text, address):
    """Only the nominative "ulica" was a street word, and prose uses the other cases."""
    assert _found(text, "ADDRESS") == [address]


@pytest.mark.parametrize(
    "text",
    [
        "15.03.2024 4111 1111 1111 1111 Zakup 100,00 PLN",
        "15.03.2024 4111111111111111 Zakup 100,00 PLN",
        "Karta ważna do 2027 4111 1111 1111 1111",
        "2024-03-15 4111 1111 1111 1111 zakup",
    ],
)
def test_a_card_right_after_a_four_digit_number_is_masked(text):
    """After the year of a date, every following group was skipped and the card went out."""
    assert [found.replace(" ", "") for found in _found(text, "CREDIT_CARD")] == ["4111111111111111"]


def test_a_long_chain_of_four_digit_groups_is_still_not_sliced_into_a_card():
    """Five groups or more are an account or reference number, not a year and a card."""
    assert _found("Ref. 1234 4111 1111 1111 1111 9999", "CREDIT_CARD") == []
