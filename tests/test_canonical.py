"""The canonical form a stable pseudonym is computed from: one value, however it is written.

The forms below are written by hand, not generated with the library's own declension tables, so a
mistake in those tables shows up here instead of being checked against itself.
"""

import pytest

from polanon._canonical import base_form, canonical, scheme_digest

NBSP = chr(0x00A0)

#: A detected name as it stands in a sentence, and its nominative, written by hand.
DECLINED = [
    ("Paweł Wróbel", "Paweł Wróbel"),
    ("Pawła Wróbla", "Paweł Wróbel"),
    ("Piotrem Mazurem", "Piotr Mazur"),
    ("Piotrze Mazurze", "Piotr Mazur"),
    ("Jerzego Zająca", "Jerzy Zając"),
    ("Antonim Pawlakiem", "Antoni Pawlak"),
    ("Kuby Sikory", "Kuba Sikora"),
    ("Bartka Wójcika", "Bartek Wójcik"),
    ("Michałowi Kotowi", "Michał Kot"),
    ("Tomasza Kowalczyka", "Tomasz Kowalczyk"),
    ("Tomaszem Kowalczykiem", "Tomasz Kowalczyk"),
    ("Józefowi Mickiewiczowi", "Józef Mickiewicz"),
    ("Aleksandrze Bednarku", "Aleksander Bednarek"),
    ("Agnieszki Dąbrowskiej", "Agnieszka Dąbrowska"),
    ("Małgorzacie Wiśniewskiej", "Małgorzata Wiśniewska"),
    ("Ewą Kamińską", "Ewa Kamińska"),
    ("Ewie Kamińskiej", "Ewa Kamińska"),
    ("Marii Nowak", "Maria Nowak"),
    ("Zofią Lis", "Zofia Lis"),
    ("Alicji Nowak-Kowalskiej", "Alicja Nowak-Kowalska"),
    ("Jana Nowaka-Kowalskiego", "Jan Nowak-Kowalski"),
    ("Oldze Szymańskiej", "Olga Szymańska"),
    ("Joanną Dudą", "Joanna Duda"),
    ("Joanną Duda", "Joanna Duda"),
    ("Beacie Zielińskiej", "Beata Zielińska"),
    ("Markiem Lewandowskim", "Marek Lewandowski"),
    ("Stanisława Gila", "Stanisław Gil"),
    ("Anną Mazur", "Anna Mazur"),
    ("Jana Mazura", "Jan Mazur"),
    ("Kasi Kowalskiej", "Kasia Kowalska"),
]


@pytest.mark.parametrize(("written", "nominative"), DECLINED)
def test_a_name_in_any_case_has_the_canonical_form_of_its_nominative(written, nominative):
    assert canonical(written, "PERSON") == canonical(nominative, "PERSON")
    assert base_form(written, "PERSON") == nominative


@pytest.mark.parametrize(
    "spelling",
    [
        "JANA KOWALSKIEGO",
        "jana kowalskiego",
        "Pan Jan Kowalski",
        "dr Jan Kowalski",
        "Kowalski Jan",
        "Jan" + NBSP + "Kowalski",
        "Janem  Kowalskim",
    ],
)
def test_letter_case_titles_word_order_and_spaces_do_not_count(spelling):
    assert canonical(spelling, "PERSON") == canonical("Jan Kowalski", "PERSON")
    assert base_form(spelling, "PERSON") == "Jan Kowalski"


@pytest.mark.parametrize(
    "people",
    [
        ["Jan Kowalski", "Janina Kowalska", "Jana Kowalska", "Anna Kowalska"],
        ["Aleksander Nowak", "Aleksandra Nowak"],
        ["Jan Nowak", "Anna Nowak"],
        ["Marek Bednarek", "Marka Bednarka-Nowak"],
    ],
)
def test_different_people_keep_different_canonical_forms(people):
    forms = [canonical(person, "PERSON") for person in people]
    assert len(set(forms)) == len(forms)


def test_the_surname_tells_a_mans_genitive_from_a_womans_nominative():
    """ "Aleksandra" is a woman or the genitive of Aleksander; "Nowaka" says it is him."""
    assert canonical("Aleksandra Nowaka", "PERSON") == canonical("Aleksander Nowak", "PERSON")
    assert canonical("Aleksandra Nowak", "PERSON") != canonical("Aleksander Nowak", "PERSON")


@pytest.mark.parametrize(
    "written",
    [
        "Aleksandrze Dudzie",  # Aleksander or Aleksandra: the form does not say
        "Macieja Krawca",  # a surname with a vowel that drops, not on the list
        "Jana Kozła",
        "Jana Kowalska",  # the genitive of Jan beside a woman's surname: nothing agrees
        "Jana Polskiego",  # an adjective of nationality, not a surname
        "Jana Budynku",  # a noun the guard knows is no surname
        "Jana Firma",  # a word that is never a name
        "Anna Maria Nowak",  # more than two words: read only as written
        "Jan İnce",  # a letter that lowercases to two
    ],
)
def test_a_name_the_rules_cannot_read_for_certain_is_taken_as_written(written):
    assert canonical(written, "PERSON") == canonical(written.upper(), "PERSON")
    assert base_form(written, "PERSON") == written


def test_a_title_stays_out_of_a_foreign_name_the_rules_do_not_decline():
    assert canonical("Mr Oliver Bennett", "PERSON") == canonical("Oliver Bennett", "PERSON")
    assert base_form("Mr Oliver Bennett", "PERSON") == "Oliver Bennett"


@pytest.mark.parametrize(
    ("written", "nominative"),
    [
        ("Kowalskiemu", "Kowalski"),
        ("Kowalskiej", "Kowalska"),
        ("Nowakiem", "Nowak"),
        ("Mickiewicza", "Mickiewicz"),
        ("Bednarkowi", "Bednarek"),
        ("Kasi", "Kasia"),
    ],
)
def test_a_surname_or_a_first_name_alone_is_read_in_the_nominative(written, nominative):
    assert canonical(written, "PERSON") == canonical(nominative, "PERSON")
    assert base_form(written, "PERSON") == nominative


def test_spellings_the_conversation_search_tells_apart_stay_apart():
    """The next turn looks for the spelling the mapping holds, so each needs a token of its own."""
    assert canonical("Łukasza Wiśniewskiego", "PERSON") == "łukasz wiśniewski"
    assert canonical("Lukasza Wisniewskiego", "PERSON") == "lukasz wisniewski"
    assert canonical("wróbel", "PERSON") != canonical("Wróbel", "PERSON")
    assert canonical("orlen", "ORG") != canonical("Orlen", "ORG")
    assert base_form("Lukasza Wisniewskiego", "PERSON") == "Lukasz Wisniewski"


@pytest.mark.parametrize(
    ("entity_type", "spellings"),
    [
        ("PESEL", ["50810112354", "508 101 123 54", "508-101-123-54"]),
        (
            "NIP",
            ["5260000005", "526-000-00-05", "526 000 00 05", "PL5260000005", "PL 526-000-00-05"],
        ),
        ("REGON", ["123456785", "123-456-785"]),
        ("CREDIT_CARD", ["4111111111111111", "4111 1111 1111 1111", "4111-1111-1111-1111"]),
        (
            "IBAN",
            [
                "PL61 1090 1014 0000 0712 1981 2874",
                "PL61109010140000071219812874",
                "61109010140000071219812874",
                "pl61 1090 1014 0000 0712 1981 2874",
            ],
        ),
        (
            "PHONE",
            ["600 100 200", "600-100-200", "+48 600 100 200", "0048600100200", "(+48) 600100200"],
        ),
        ("EMAIL", ["Jan.Kowalski@Example.com", "jan.kowalski [at] example [dot] com"]),
        ("LINK", ["https://www.linkedin.com/in/jkowalski/", "linkedin.com/in/jkowalski"]),
        ("MAC_ADDRESS", ["00:1A:2B:3C:4D:5E", "00-1a-2b-3c-4d-5e"]),
        ("ORG", ["Firma Kot Sp. z o.o.", "FIRMA KOT SP. Z O. O.", "Firma  Kot Sp. z o.o."]),
        ("VIN", ["WVWZZZ1JZXW000001", "wvwzzz1jzxw000001"]),
    ],
)
def test_an_identifier_is_one_value_however_it_is_written(entity_type, spellings):
    assert len({canonical(spelling, entity_type) for spelling in spellings}) == 1


@pytest.mark.parametrize(
    ("entity_type", "first", "second"),
    [
        ("PHONE", "+48 600 100 200", "+44 600 100 200"),
        ("SECRET", "Kot2024!", "kot2024!"),
        ("EMPLOYEE_ID", "EMP-11111", "emp-11111"),
        ("NIP", "5260000005", "5260000006"),
    ],
)
def test_values_that_differ_keep_different_canonical_forms(entity_type, first, second):
    assert canonical(first, entity_type) != canonical(second, entity_type)


def test_only_a_name_has_a_base_form_other_than_its_spelling():
    assert base_form(" 526-000-00-05 ", "NIP") == "526-000-00-05"
    assert base_form("Firma Kot Sp. z o.o.", "ORG") == "Firma Kot Sp. z o.o."


def test_the_rules_and_name_lists_of_the_first_scheme_are_unchanged():
    """Tokens of scheme 1 are computed from these rules, so changing them changes tokens.

    Every index built with scheme 1 would stop matching new questions. A change to the name lists,
    the declension or the rules here needs a new scheme in polanon.pseudonyms, with the old rules
    kept for it, not a new digest pasted in.
    """
    assert scheme_digest() == "2630a8b1c72663d1a0c93fbc7fd5f050bfd8b16ce27cbba5d6a1c77f9a952e59"
