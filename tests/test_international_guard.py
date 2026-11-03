"""Foreign invoice formats: what the guard finds, and the invoice lines it must leave alone."""

import time

import pytest

from polanon import Anonymizer, InternationalGuard, default_guards
from polanon.guards.international import is_valid_aba_routing, is_valid_uk_postcode_inward
from polanon.policy import ENTITY_TYPES

ALL = set(ENTITY_TYPES)
GUARD = InternationalGuard()


def found(text: str, enabled: set[str] = ALL) -> set[tuple[str, str]]:
    return {(span.entity_type, text[span.start : span.end]) for span in GUARD.detect(text, enabled)}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("VAT No: GB999 9999 73", ("TAX_ID", "GB999 9999 73")),
        ("VAT REG No: GB 123 4567 89", ("TAX_ID", "GB 123 4567 89")),
        ("VAT Number EU123456789", ("TAX_ID", "EU123456789")),
        ("MwSt/VAT ID: 12345678", ("TAX_ID", "12345678")),
        ("UID: CHE-116.281.710 MWST", ("TAX_ID", "CHE-116.281.710 MWST")),
        ("Leverantör SE556677889901", ("TAX_ID", "SE556677889901")),
        ("USt-IdNr. DE123456789", ("TAX_ID", "DE123456789")),
        ("Org nr: 5561947986", ("COMPANY_ID", "5561947986")),
        ("Company Registered in England and Wales No. 123456", ("COMPANY_ID", "123456")),
        ("Bankgiro: 5833-3543", ("BANK_ACCOUNT", "5833-3543")),
        ("Plusgiro: 12 34 56-7", ("BANK_ACCOUNT", "12 34 56-7")),
        ("BACS: Sort Code 12-34-56", ("BANK_ACCOUNT", "12-34-56")),
        ("Beneficiary Account Number: 14169892", ("BANK_ACCOUNT", "14169892")),
        ("Routing number: 021000021", ("BANK_ACCOUNT", "021000021")),
        ("Postkonto: 80-470-3", ("BANK_ACCOUNT", "80-470-3")),
        ("phone: (+41) 12345678", ("PHONE", "(+41) 12345678")),
        ("Tfn: +46 8 123 456 78", ("PHONE", "+46 8 123 456 78")),
        ("Phone: 496 0123", ("PHONE", "496 0123")),
        ("please call 0117 496 0588.", ("PHONE", "0117 496 0588")),
        ("1776 Liberty Way", ("ADDRESS", "1776 Liberty Way")),
        ("44 Shore St", ("ADDRESS", "44 Shore St")),
        ("Bahnhofstrasse 21", ("ADDRESS", "Bahnhofstrasse 21")),
        ("12 rue de la Paix", ("ADDRESS", "12 rue de la Paix")),
        ("Via Roma 5", ("ADDRESS", "Via Roma 5")),
        ("PO Box 45276", ("ADDRESS", "PO Box 45276")),
        ("Macduff AB4 1TX", ("POSTAL_CODE", "AB4 1TX")),
        ("Kansas City, KS 66102", ("POSTAL_CODE", "KS 66102")),
        ("Whitehorse Y1A 2C6", ("POSTAL_CODE", "Y1A 2C6")),
        ("CH-8001 Zürich", ("POSTAL_CODE", "CH-8001")),
        ("Postfach\n8021 Zürich", ("POSTAL_CODE", "8021")),
        ("111 22 Stockholm", ("POSTAL_CODE", "111 22")),
        ("ACME Inc", ("ORG", "ACME Inc")),
        ("Bank: BARCLAYS BANK PLC", ("ORG", "BARCLAYS BANK PLC")),
        ("Alpenblick Treuhand GmbH", ("ORG", "Alpenblick Treuhand GmbH")),
        ("Nordvik Konsult AB", ("ORG", "Nordvik Konsult AB")),
        ("Dear Ms Carter,", ("PERSON", "Carter")),
        ("z. Hd. Frau Wiśniewska", ("PERSON", "Wiśniewska")),
        ("CEO: John Doe", ("PERSON", "John Doe")),
        ("TfL Contact:\nJohn Stairs", ("PERSON", "John Stairs")),
        ("please contact F.Coggle Ext 12345", ("PERSON", "F.Coggle")),
        ("Invoiced To  \nHenry Rearden", ("PERSON", "Henry Rearden")),
        ("Vår referens: Karin Lindqvist", ("PERSON", "Karin Lindqvist")),
        ("Kind regards,\nOliver Bennett", ("PERSON", "Oliver Bennett")),
        ("Mit freundlichen Grüßen\nLukas Meier", ("PERSON", "Lukas Meier")),
    ],
)
def test_a_foreign_value_is_found_with_its_type(text, expected):
    assert expected in found(text)


@pytest.mark.parametrize(
    "text",
    [
        "VAT rate 20.0%",
        "Total VAT     214.57",
        "VAT 0% (reverse charge)",
        "Subtotal  1072.83",
        "INVOICE NUMBER 952384",
        "ORDER NUMBER 000037",
        "M0061 Catalyst 1 liter 10 Ea. 59.99 9.99 500.00",
        "10 M9005 Plastic Bottle \u2013 15.2 FL Oz (450 ml) 1.49 14.90",
        "Please send payment within 30 days of receiving this invoice.",
        "Buchhaltung und Lohnabrechnung, August 2026        CHF 2 400.00",
        "Tax point date 01/09/2017",
        "please contact Customer Service",
        "Your Ref PROD",
        "size A4 1CM",
        "SAMPLE STANDARD UK GBP INVOICE",
        "TERMS AND CONDITIONS",
    ],
)
def test_an_ordinary_invoice_line_is_left_alone(text):
    assert found(text) == set()


def test_only_enabled_types_are_returned():
    text = "VAT No: GB999 9999 73\nACME Inc\nPhone: 496 0123"
    assert {entity_type for entity_type, _ in found(text, {"ORG"})} == {"ORG"}
    assert found(text, set()) == set()


def test_the_aba_checksum_separates_a_routing_number_from_any_nine_digits():
    assert is_valid_aba_routing("021000021")
    assert not is_valid_aba_routing("021000022")
    assert not is_valid_aba_routing("12345")


def test_a_uk_inward_code_never_uses_the_letters_royal_mail_leaves_out():
    assert is_valid_uk_postcode_inward("AB4 1TX")
    assert not is_valid_uk_postcode_inward("A4 1CM")


async def test_a_foreign_invoice_reaches_the_provider_without_its_identifiers():
    text = (
        "Harbour Lane Supplies Ltd\n27 Kingfisher Road\nBristol BS1 4DJ\n"
        "Tel: +44 117 496 0321\nVAT Reg No: GB 123 4567 82\n"
        "Sort Code: 20-45-77  Account No: 41365870\n"
        "Bill To:\nNowak Transport Sp. z o.o.\nVAT: PL8992611767\n"
        "Kind regards,\nOliver Bennett\n"
    )
    result = await Anonymizer(default_guards(with_ner=False)).anonymize(text)

    for value in (
        "Harbour Lane Supplies Ltd",
        "27 Kingfisher Road",
        "BS1 4DJ",
        "+44 117 496 0321",
        "GB 123 4567 82",
        "20-45-77",
        "41365870",
        "Nowak Transport",
        "8992611767",
        "Oliver Bennett",
    ):
        assert value not in result.anonymized, value
    assert "<NIP_1>" in result.anonymized
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


# --- Polish text the foreign rules must leave alone ---


@pytest.mark.parametrize(
    "text",
    [
        "Sterownik PLC Siemens S7-1200 trzeba przeprogramować.",
        "Klasa AB budynku",
        "Masa Netto KG",
        "Raport SAS z analizy danych.",
        "Zestaw SE dla klienta",
        "1. Sterownik PLC Siemens do wymiany.",
        "Wyrok SA w Krakowie jest prawomocny.",
        "Zegarek Casio Limited Edition w promocji.",
    ],
)
def test_a_polish_phrase_is_not_a_company_when_it_ends_in_a_short_legal_form(text):
    assert found(text, {"ORG"}) == set()


@pytest.mark.parametrize(
    ("text", "company"),
    [
        ("Dziękujemy Acme Widgets Ltd za współpracę.", "Acme Widgets Ltd"),
        ("Wczoraj Siemens AG ogłosił wyniki.", "Siemens AG"),
        ("Informujemy Acme Ltd o zmianie terminu.", "Acme Ltd"),
        ("Firma Siemens AG wystawiła fakturę.", "Siemens AG"),
        ("Siemens AG\nWerner-von-Siemens-Straße 1", "Siemens AG"),
        ("Siemens AG, München", "Siemens AG"),
        ("Nordvik Konsult AB wystawił fakturę.", "Nordvik Konsult AB"),
    ],
)
def test_a_company_name_is_found_without_the_word_that_opens_the_sentence(text, company):
    assert found(text, {"ORG"}) == {("ORG", company)}


async def test_a_company_gets_one_token_when_a_sentence_opens_on_a_polish_verb():
    text = (
        "Dziękujemy Acme Widgets Ltd za współpracę. "
        "Proszę wystawić korektę dla Acme Widgets Ltd do piątku."
    )
    result = await Anonymizer(default_guards(with_ner=False)).anonymize(text)

    assert result.anonymized == (
        "Dziękujemy <FIRMA_1> za współpracę. Proszę wystawić korektę dla <FIRMA_1> do piątku."
    )


@pytest.mark.parametrize(
    "text",
    [
        "Plan:\n2024 Podsumowanie roku\n2025 Plany\n",
        "Wyniki:\n1500 Sztuk sprzedanych\n",
        "Numer zgłoszenia: ID 12345",
        "Proszę sprawdzić zamówienie OK 54321 jutro.",
        "Kolor tła: #C0C0C0, nagłówek #A1B2C3.",
        "Model A-320 Neo ląduje.",
        "Silnik E-2500 Kompakt na stanie.",
    ],
)
def test_a_year_an_id_or_a_colour_is_not_a_postcode_without_an_address(text):
    assert found(text, {"POSTAL_CODE"}) == set()


async def test_a_year_heading_does_not_mask_the_year_in_the_rest_of_the_text():
    text = "Raport:\n2024 Podsumowanie roku\nW 2024 roku przychody wzrosły o 12%.\n"
    result = await Anonymizer(default_guards(with_ner=False)).anonymize(text)

    assert result.anonymized == text


@pytest.mark.parametrize(
    "text",
    [
        "Zysk netto: +12 345 678 zł",
        "Saldo wzrosło o +25 000 000 zł w 2024 roku.",
        "Wzrost sprzedaży r/r: +12 500 000 zł",
        "Wynik EBITDA: +45 678 901 PLN",
        "Wartość: +1 234 567,89 zł",
    ],
)
def test_a_signed_amount_is_not_a_phone_number(text):
    assert found(text, {"PHONE"}) == set()


@pytest.mark.parametrize(
    "text",
    [
        "Faktura VAT 12345678/2024",
        "Kwota VAT 12 345 678,90 zł",
        "Faktura VAT nr 20240001 z dnia 12.03.2024",
        "Kwota VAT: 1 234 567,89 zł",
        "VAT: 12 345 678 zł",
        "VAT 12345678",
    ],
)
def test_an_amount_or_invoice_number_after_vat_is_not_a_tax_id(text):
    assert found(text, {"TAX_ID"}) == set()


# --- foreign values in the shapes they are printed in ---


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Dear Mrs. McDonald,", "McDonald"),
        ("Dear Ms O'Neill,", "O'Neill"),
        ("Dear Mr. MacLeod,", "MacLeod"),
        ("Sehr geehrte Frau Schmidt-Weber,", "Schmidt-Weber"),
        ("Attn: José Martínez", "José Martínez"),
        ("Attn: Ana María García López", "Ana María García López"),
        ("Kind regards,\nPatrick O'Brien", "Patrick O'Brien"),
        ("Bill to: François Dvořák", "François Dvořák"),
    ],
)
def test_a_foreign_name_is_masked_whole_when_it_has_an_inner_capital_or_accent(text, name):
    assert found(text, {"PERSON"}) == {("PERSON", name)}


def test_a_name_the_rules_cannot_read_whole_is_not_masked_in_part():
    assert found("Dear Mrs. McDONALD,", {"PERSON"}) == set()


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Cordialement,\nJean-Pierre Dubois", "Jean-Pierre Dubois"),
        ("Contact: Hans-Peter Schulz", "Hans-Peter Schulz"),
        ("Contact: François Lefèvre", "François Lefèvre"),
        ("Attn: Jiří Dvořák", "Jiří Dvořák"),
        ("Account manager: Zoë Ångström", "Zoë Ångström"),
        ("Dear Mr van Dijk,", "van Dijk"),
        ("Sehr geehrter Herr von Weizsäcker,", "von Weizsäcker"),
        ("ATTN: OLIVER BENNETT", "OLIVER BENNETT"),
        ("Yours faithfully,\nOLIVER BENNETT", "OLIVER BENNETT"),
        ("Mit freundlichen Grüßen\n\nLukas Meier", "Lukas Meier"),
        ("Mit freundlichen Grüssen\nLukas Meier", "Lukas Meier"),
        ("Kind regards,\n\nOliver Bennett", "Oliver Bennett"),
        ("Best regards, Oliver Bennett", "Oliver Bennett"),
        ("Thanks,\nOliver Bennett", "Oliver Bennett"),
        ("Name: Oliver Bennett", "Oliver Bennett"),
        ("Contact: John Smith Sales Manager", "John Smith"),
    ],
)
def test_a_name_is_found_after_a_label_or_sign_off_in_its_common_variants(text, name):
    assert found(text, {"PERSON"}) == {("PERSON", name)}


@pytest.mark.parametrize(
    "text",
    [
        "Best regards, the Sales Team",
        "Best Practice Guide",
        "ATTN: ACCOUNTS PAYABLE",
        "Product Name: Super Widget",
        "Bank Name: Barclays Bank",
        "Name: Acme Ltd",
        "Monsieur le Président,",
    ],
)
def test_a_desk_a_heading_or_a_company_after_a_label_is_not_a_person(text):
    assert found(text, {"PERSON"}) == set()


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("Bankgironummer: 5833-3543", "5833-3543"),
        ("BG-nr: 5833-3543", "5833-3543"),
        ("PlusGirot 123456-7", "123456-7"),
        ("Sort-Code: 20-45-77", "20-45-77"),
        ("Sort code 204577", "204577"),
        ("Acc No: 41365870", "41365870"),
        ("Acct. No. 41365870", "41365870"),
        ("Kontonr. 1234.56.78901", "1234.56.78901"),
        ("Account 12345678. 2024 total", "12345678"),
    ],
)
def test_an_account_number_is_found_behind_a_common_label_spelling(text, value):
    assert found(text, {"BANK_ACCOUNT"}) == {("BANK_ACCOUNT", value)}


def test_a_sort_code_label_does_not_vouch_for_a_longer_number():
    assert found("Sort code 2045771234", {"BANK_ACCOUNT"}) == set()


@pytest.mark.parametrize(
    ("text", "company"),
    [
        ("Supplier: HARBOUR LANE SUPPLIES LIMITED", "HARBOUR LANE SUPPLIES LIMITED"),
        ("Bill to: ACME WIDGETS INC.", "ACME WIDGETS INC."),
        ("Lieferant: MÜLLER MASCHINENBAU GMBH", "MÜLLER MASCHINENBAU GMBH"),
        ("PAY TO ACME LTD", "ACME LTD"),
        ("Fournisseur: Transports Martin SA", "Transports Martin SA"),
        ("Lieferant: Alpenblick Treuhand SA, Genève", "Alpenblick Treuhand SA"),
        ("Odbiorca: K&S Logistik GmbH", "K&S Logistik GmbH"),
        ("Client: Institute of Applied Robotics Ltd", "Institute of Applied Robotics Ltd"),
    ],
)
def test_a_foreign_company_is_found_in_capitals_with_a_bare_sa_or_an_inner_connector(text, company):
    assert found(text, {"ORG"}) == {("ORG", company)}


@pytest.mark.parametrize(
    "text", ["THIS WARRANTY IS LIMITED", "QUANTITIES ARE LIMITED", "TIME LIMITED"]
)
def test_capitalised_prose_ending_in_limited_is_not_a_company(text):
    assert found(text, {"ORG"}) == set()


@pytest.mark.parametrize(
    ("text", "street"),
    [
        ("Lieferadresse: Berliner Straße 12, 10115 Berlin", "Berliner Straße 12"),
        ("Frankfurter Allee 1", "Frankfurter Allee 1"),
        ("Kölner Str. 5", "Kölner Str. 5"),
        ("Wiedner Hauptstraße 63", "Wiedner Hauptstraße 63"),
        ("Konrad-Adenauer-Straße 1", "Konrad-Adenauer-Straße 1"),
        ("Ship to: 350 5th Avenue, New York, NY 10118", "350 5th Avenue"),
        ("20 W 34th Street", "20 W 34th Street"),
    ],
)
def test_a_street_is_found_with_its_adjective_or_ordinal(text, street):
    assert found(text, {"ADDRESS"}) == {("ADDRESS", street)}


@pytest.mark.parametrize(
    ("text", "postcode"),
    [
        ("Adresse: Hauptstraße 5, 10115 Berlin", "10115"),
        ("Kungsgatan 12, 111 22 Stockholm", "111 22"),
        ("Rue de Rivoli 99\n75001 PARIS\n", "75001"),
        ("Hauptstraße 5\n10115 BERLIN\n", "10115"),
        ("Damrak 1\n1012 LG Amsterdam\n", "1012 LG"),
        ("Rua Augusta 100\n1100-053 Lisboa\n", "1100-053"),
        ("Bahnhofstrasse 21\n8001 Zürich, Switzerland\n", "8001"),
        ("Bahnhofstrasse 21\n\n8001 Zürich\n", "8001"),
        ("Hauptstraße 5\r\n10115 Berlin\r\n", "10115"),
        ("Rua Augusta 100, P-1100-053 Lisboa", "P-1100-053"),
        ("Ottawa, ON K1A0B1", "K1A0B1"),
    ],
)
def test_a_foreign_postcode_is_found_beside_or_below_its_street(text, postcode):
    assert found(text, {"POSTAL_CODE"}) == {("POSTAL_CODE", postcode)}


@pytest.mark.parametrize(
    ("text", "phone"),
    [
        ("Tél. : 01 23 45 67 89", "01 23 45 67 89"),
        ("Telefonnummer: 030 12345678", "030 12345678"),
        ("Rufnummer: 030 12345678", "030 12345678"),
        ("Mobilnummer: 070-123 45 67", "070-123 45 67"),
        ("Tel. 0171/1234567", "0171/1234567"),
        ("Tel.-Nr.: 030 1234567", "030 1234567"),
    ],
)
def test_a_phone_number_is_found_behind_a_french_or_german_label(text, phone):
    assert found(text, {"PHONE"}) == {("PHONE", phone)}


async def test_a_labelled_foreign_phone_is_masked_whole_when_it_starts_like_a_polish_postcode():
    text = "Telefonnummer: 08-123 456 78"
    result = await Anonymizer(default_guards(with_ner=False)).anonymize(text)

    assert result.anonymized == "Telefonnummer: <TELEFON_1>"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Amtsgericht Charlottenburg, HRB 12345", ("COMPANY_ID", "12345")),
        ("Handelsregister: HRB 98765 B", ("COMPANY_ID", "98765 B")),
        ("Siemens AG, HRB 6684 AG München", ("COMPANY_ID", "6684")),
        ("Steuernummer: 12/345/67890", ("TAX_ID", "12/345/67890")),
        ("St.-Nr. 143/123/45678", ("TAX_ID", "143/123/45678")),
        ("IČO: 12345678", ("COMPANY_ID", "12345678")),
        ("Ondernemingsnummer: 0123.456.789", ("COMPANY_ID", "0123.456.789")),
    ],
)
def test_a_register_or_tax_number_is_found_in_its_national_shape(text, expected):
    assert expected in found(text)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("Numéro de TVA intracommunautaire: FR 12345678901", "FR 12345678901"),
        ("N° TVA intracommunautaire : FR 12 345 678 901", "FR 12 345 678 901"),
        ("Numéro d'entreprise: BE 0123.456.789", "BE 0123.456.789"),
        ("ATU 12345678", "ATU 12345678"),
        ("DK 12 34 56 78", "DK 12 34 56 78"),
        ("DIČ: CZ 12345678", "CZ 12345678"),
        ("IČ DPH: SK 1234567890", "SK 1234567890"),
        ("Our VAT number is SE 556677-8899 01", "SE 556677-8899 01"),
        ("VAT: NL 8012.34.567.B01", "NL 8012.34.567.B01"),
    ],
)
def test_an_eu_vat_number_is_found_with_its_national_spacing(text, value):
    assert found(text, {"TAX_ID"}) == {("TAX_ID", value)}


def test_spaced_digit_pairs_without_a_country_prefix_are_not_a_vat_number():
    assert found("Order 12 34 56 78 shipped", {"TAX_ID"}) == set()


@pytest.mark.parametrize("token", ["Ab-", "Ab'", "A-"])
def test_detection_stays_linear_on_a_long_hyphen_or_apostrophe_joined_token(token):
    """A pasted base64url blob used to make every "-X" a new start that rescanned the token."""
    text = "Załącznik: " + token * (20_000 // len(token))
    started = time.perf_counter()
    GUARD.detect(text, ALL)
    assert time.perf_counter() - started < 1.0


def test_a_long_line_of_letters_above_a_postcode_is_scanned_in_linear_time():
    """The street-line rule was retried at every letter: 40 KB took half a minute."""
    started = time.perf_counter()
    Anonymizer(default_guards(with_ner=False)).anonymize_sync("A" * 40_000 + "\n12345 Berlin")
    assert time.perf_counter() - started < 3
