"""Polish personal names in lowercase and casual text, found without a model."""

import pytest

from polanon import Anonymizer, PersonNameGuard, default_guards
from polanon.guards.names_pl import (
    AMBIGUOUS_FORMS,
    NAME_FORMS,
    decline_feminine,
    decline_masculine,
    is_common_adjective,
    person_name_parts,
    surname_reading,
)

PERSON = {"PERSON"}


def _found(text: str, enabled=PERSON) -> list[str]:
    return [text[s.start : s.end] for s in PersonNameGuard().detect(text, enabled)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Mam sprawę sądową z panem kamilem bednarkiem i on nie chciał zapłacić",
            ["kamilem bednarkiem"],
        ),
        ("napisz maila do ani kowalskiej z księgowości", ["ani kowalskiej"]),
        ("jan nowak dzwonił wczoraj", ["jan nowak"]),
        ("pani anna wiśniewska prosi o fakturę", ["anna wiśniewska"]),
        ("KAMIL BEDNAREK zadzwoni jutro", ["KAMIL BEDNAREK"]),
        (
            "Umowa z Kamilem Bednarkiem i Anną Nowak-Kowalską",
            ["Kamilem Bednarkiem", "Anną Nowak-Kowalską"],
        ),
        ("przekaż to annie marii kowalskiej", ["annie marii kowalskiej"]),
    ],
)
def test_a_first_name_takes_the_surname_that_agrees_with_it(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("napisz do tomka, że spotkanie jest przesunięte", ["tomka"]),
        ("zapytaj michala o raport dla pawla", ["michala", "pawla"]),
        ("dzień dobry panie kamilu, pozdrawiam kasia", ["kamilu", "kasia"]),
        ("Olu, wyślij to Maćkowi", ["Olu", "Maćkowi"]),
    ],
)
def test_a_first_name_is_masked_in_every_case_and_without_polish_letters(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("spotkanie z panem bednarkiem", ["bednarkiem"]),
        ("zostaw paczkę u pani wróbel", ["wróbel"]),
        ("pan prezes kowalski i pani dyrektor nowak", ["kowalski", "nowak"]),
        ("dr hab. Kowalski przyjmuje we wtorek", ["Kowalski"]),
        ("list od Pani Zenobii Nowak", ["Zenobii Nowak"]),
        ("umów mnie z panem zającem", ["zającem"]),
    ],
)
def test_a_form_of_address_marks_the_surname_after_it(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "nie ma ani polskiej, ani angielskiej wersji",
        "klienci mają polską wersję strony",
        "pola formularza są puste, pole tekstowe też",
        "nowa marka polskiego producenta, 10 marek w ofercie",
        "umowę podpisano 1 maja 2024 r.",
        "ani wiek, ani płeć nie mają znaczenia",
        "kurs franka szwajcarskiego spadł",
        "pan ma rację, a pani, która dzwoniła, zostawiła numer",
        "Pola Formularza",
        "sprawdź pole w polu wyboru",
    ],
)
def test_words_that_only_look_like_names_are_left_alone(text):
    assert _found(text) == []


@pytest.mark.parametrize("name", ["maja", "marek", "róża", "kalina"])
@pytest.mark.parametrize(
    "surname", ["nowak", "wójcik", "woźniak", "kowalczyk", "kaczmarek", "krawczyk", "mazur"]
)
def test_common_surnames_settle_ambiguous_first_names(name, surname):
    person = f"{name} {surname}"
    assert _found(f"{person} dzwoni") == [person]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("roza wojcik dzwoni", ["roza wojcik"]),
        ("kalina wozniak dzwoni", ["kalina wozniak"]),
        ("lista: nowak marek", ["nowak marek"]),
        ("sprawa marka mazura", ["marka mazura"]),
        ("rozmowa z mają mazur", ["mają mazur"]),
    ],
)
def test_common_surnames_keep_case_folding_order_and_grammatical_agreement(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "10 marek w ofercie",
        "róża kwitnie w ogrodzie",
        "kalina rośnie w ogrodzie",
        "marek młotek",
        "róża kot",
        "nowak wójcik woźniak kowalczyk kaczmarek krawczyk mazur",
    ],
)
def test_common_surnames_do_not_make_ordinary_words_or_standalone_surnames_people(text):
    assert _found(text) == []


@pytest.mark.parametrize("person", ["maja nowak", "marek nowak", "kalina mazur"])
@pytest.mark.parametrize("with_ner", [False, True])
async def test_common_surnames_are_masked_whole_through_the_public_pipeline(person, with_ner):
    if with_ner:
        pytest.importorskip("pl_core_news_md")
    result = await Anonymizer(default_guards(with_ner=with_ner)).anonymize(f"{person} dzwoni")
    assert result.anonymized == "<OSOBA_1> dzwoni"
    assert result.mapping == {"<OSOBA_1>": person}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Przyjmuje lek. med. Kowalski.", ["Kowalski"]),
        ("Przyjmuje lek. med. Zielińska.", ["Zielińska"]),
        ("Przyjmuje dr n. med. Kowalski.", ["Kowalski"]),
        ("Przyjmuje lek. dent. Nowak.", ["Nowak"]),
        ("Przyjmuje lek med Kowalski.", ["Kowalski"]),
        ("Tekst zredagował red. Nowak.", ["Nowak"]),
    ],
)
def test_an_abbreviated_title_marks_the_surname_after_its_full_stops(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Czy mogę brać lek Apap razem z lekiem Ibuprom?",
        "Lekarz przepisał nowy lek. Jutro idę do apteki.",
        "Colour: red. Size: M. Qty: 2",
        "Ordered a red Tesla Model 3.",
    ],
)
def test_an_abbreviation_that_is_also_a_word_is_left_alone_without_a_surname(text):
    assert _found(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "umówiłem się z panem hydraulikiem na jutro",
        "zadzwoń do pana mechanika w sprawie auta",
        "pan elektryk przyjdzie o 10",
        "rozmawiałem z panem naczelnikiem urzędu skarbowego",
        "Pan Prezes Zarządu poprosił o raport.",
        "Pani Księgowa wysłała przelew.",
    ],
)
def test_an_occupation_after_a_form_of_address_is_left_alone(text):
    assert _found(text) == []


def test_a_surname_after_an_occupation_is_still_addressed():
    assert _found("pan hydraulik Wróbel przyjdzie") == ["Wróbel"]
    assert _found("Pani Księgowa Nowak wysłała przelew.") == ["Nowak"]
    assert _found("Pan Prezes Zarządu Kowalski poprosił o raport.") == ["Kowalski"]


@pytest.mark.parametrize(
    "text",
    [
        "Invoice date: 15 Jan 2024, due 14 Feb 2024",
        "Rechnungsdatum: 15. Mai 2024",
        "Date de facture : 15 mai 2024",
        "Invoice date: Jan 15, 2024",
        "Invoice date: 15-Jan-2024",
        "Invoice date: 15/Jan/2024",
        "Invoice date: 2024-Jan-15",
        "Rechnungsdatum: 15-Mai-2024",
    ],
)
def test_a_month_spelled_like_a_first_name_is_left_alone_in_a_date(text):
    assert _found(text) == []


def test_a_first_name_after_a_number_is_still_a_name_outside_a_date():
    assert _found("o 15 Jan przyjdzie") == ["Jan"]
    assert _found("Jan Kowalski przyjdzie") == ["Jan Kowalski"]
    assert _found("z Janem") == ["Janem"]


@pytest.mark.parametrize("separator", ["-", "/", "."])
@pytest.mark.parametrize(
    "order", ["15{sep}Jan{sep}2024", "2024{sep}Jan{sep}15", "Jan{sep}15{sep}2024"]
)
def test_a_punctuated_date_uses_the_same_separator_on_both_sides(separator, order):
    assert _found(order.format(sep=separator)) == []


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Nowak Anna 03/2024", "Nowak Anna"),
        ("Nowak Anna 12.2024", "Nowak Anna"),
        ("Sprawa: Nowak Anna 15/2024", "Nowak Anna"),
        ("15.03.2024 Anna 8 h", "Anna"),
        ("W 2023 Anna 2 razy była na urlopie.", "Anna"),
        ("15 Jan/2024", "Jan"),
        ("15/Jan-2024", "Jan"),
        ("2024/Jan-15", "Jan"),
        ("Jan-15/2024", "Jan"),
        ("2024 Jan 15", "Jan"),
    ],
)
async def test_a_name_next_to_numbers_is_masked_without_spacy(text, name):
    assert _found(text) == [name]
    result = await Anonymizer(default_guards(with_ner=False)).anonymize(text)
    assert result.anonymized == text.replace(name, "<OSOBA_1>")
    assert result.mapping == {"<OSOBA_1>": name}


def test_a_word_that_is_also_a_name_needs_a_surname_or_a_form_of_address():
    assert _found("do 3 maja przyjdzie") == []
    assert _found("pani maja przyjdzie") == ["maja"]
    assert _found("maja kowalska przyjdzie") == ["maja kowalska"]
    assert _found("Nowa marka Marek Zając") == ["Marek Zając"]


@pytest.mark.parametrize(
    ("text", "score"),
    [
        ("Wakacje na Kubie były super.", 0.8),
        ("Lecę na Kubę w maju.", 0.8),
        ("Kuba Nowak przyjdzie.", 0.9),
        ("Rozmawiałem z panem Kubą.", 0.9),
        ("Podpisał J. Nowak.", 0.9),
        ("Anna Maria przyjdzie.", 0.9),
    ],
)
def test_a_first_name_alone_scores_below_the_language_model(text, score):
    """Alone, "Kubie" is as likely Cuba as a man, so a model's place outranks it; with a surname,
    an initial or a form of address it is a person and outranks the model."""
    (span,) = PersonNameGuard().detect(text, PERSON)
    assert span.score == score


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Our offices in Asia and Europe.", []),
        ("Asia przyjdzie jutro.", []),
        ("Asia Kowalska przyjdzie jutro.", ["Asia Kowalska"]),
        ("Dzwoniła pani Asia.", ["Asia"]),
        ("Napisz do Asi.", ["Asi"]),
    ],
)
def test_asia_alone_is_the_continent(text, expected):
    assert _found(text) == expected


def test_a_capitalised_word_after_a_first_name_is_taken_as_the_surname():
    assert _found("Pisała do niej Anna Curie") == ["Anna Curie"]
    assert _found("dzwoniła pani Zenobia") == ["Zenobia"]
    assert _found("pani Zenobia przyszła") == ["Zenobia"]


def test_at_most_two_job_titles_sit_between_the_address_and_the_name():
    assert _found("pan prezes dyrektor nowak") == ["nowak"]
    assert _found("pan prezes dyrektor kierownik nowak") == []


def test_a_surname_written_before_the_first_name_is_included():
    assert _found("lista: kowalski jan, nowak anna") == ["kowalski jan", "nowak anna"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Uczestnik: Konieczny Jan, ur. 1990", ["Konieczny Jan"]),
        ("Uczestnik: Wójcik Marek, ur. 1990", ["Wójcik Marek"]),
        ("Uczestnik: Nowak Wiktoria, ur. 1990", ["Nowak Wiktoria"]),
        ("Uczestnik: Zając Mikołaj, ur. 1990", ["Zając Mikołaj"]),
        ("Uczestnik: KONIECZNY JAN", ["KONIECZNY JAN"]),
    ],
)
def test_a_surname_before_the_first_name_follows_the_rules_of_the_usual_order(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Wczoraj Jan dzwonił", ["Jan"]),
        ("Jasne Kasia, zrobię to", ["Kasia"]),
        ("Re: Super Kasia", ["Kasia"]),
    ],
)
def test_a_word_before_a_first_name_stays_out_outside_a_labelled_field(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("JAN KONIECZNY, PESEL 50810112354", ["JAN KONIECZNY"]),
        ("PAN JAN KONIECZNY, PESEL 50810112354", ["JAN KONIECZNY"]),
        ("ANNA GOŁĄB, PESEL 50810112354", ["ANNA GOŁĄB"]),
        ("KATARZYNA ŚLIWA, PESEL 50810112354", ["KATARZYNA ŚLIWA"]),
        ("Imię i nazwisko: JAN KONIECZNY PESEL 50810112354", ["JAN KONIECZNY"]),
        ("Jan KONIECZNY, dyrektor", ["Jan KONIECZNY"]),
        ("PAN KONIECZNY, tel. 600 100 200", ["KONIECZNY"]),
        ("MAREK ZAJĄC zadzwoni", ["MAREK ZAJĄC"]),
    ],
)
def test_a_surname_in_capitals_is_taken_where_the_name_ends(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Proszę o kontakt z panem Janem\nKowalskim w sprawie faktury.",
            ["Janem\nKowalskim"],
        ),
        ("Pozdrawiam\nAnna\r\nWiśniewska", ["Anna\r\nWiśniewska"]),
        ("Podpisała Anna\nNowak.", ["Anna\nNowak"]),
    ],
)
def test_a_surname_on_the_next_line_of_a_wrapped_text_joins_its_first_name(text, expected):
    """A hard-wrapped e-mail put the surname on its own line, and it went out in clear."""
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("imie;nazwisko\nJan;Kowalski\nAnna;Wiśniewska", ["Jan", "Kowalski", "Anna", "Wiśniewska"]),
        ("| Jan | Kowalski | 50810112354 |", ["Jan", "Kowalski"]),
        ("Jan,Kowalski,Kraków", ["Jan", "Kowalski"]),
        ("Marek, Zając", ["Marek", "Zając"]),
    ],
)
def test_a_surname_in_the_next_cell_is_masked_as_a_value_of_its_own(text, expected):
    """A CSV export or a Markdown table kept the surname; one token for both cells would also
    swallow a column, so the surname gets its own."""
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Imię: Jan\nNazwisko: Kowalski", ["Jan"]),
        ("Jan;Anna;Piotr", ["Jan", "Anna", "Piotr"]),
        ("Jan;Kraków;Polska", ["Jan"]),
        ("Uczestnicy: Jan\nDział: sprzedaż", ["Jan"]),
        ("Anna\n\nNowak przejmuje projekt.", ["Anna"]),
        ("jan;nowak", ["jan"]),
    ],
)
def test_a_field_label_a_first_name_or_a_town_in_the_next_cell_stays_apart(text, expected):
    """Across a line or a cell the next word is often the next field, so it needs a surname
    reading, and a paragraph break or a weak surname in lowercase is no join."""
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ANNA PROSI O FAKTURĘ DO PIĄTKU", ["ANNA"]),
        ("wyślij Kasi PDF z umową", ["Kasi"]),
        ("mieszkam przy al. Jana Pawła II 43A", ["Jana Pawła"]),
    ],
)
def test_capitals_mid_sentence_or_after_an_oblique_case_are_not_a_surname(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cześć tomaszu wróbel, podeślesz pesel?", ["tomaszu wróbel"]),
        ("hej tomaszu wrobel, podeslesz pesel?", ["tomaszu wrobel"]),
        ("szanowny panie pawle nowak, przesyłam umowę.", ["pawle nowak"]),
        ("cześć jacku duda", ["jacku duda"]),
        ("hej krzysztofie stępień", ["krzysztofie stępień"]),
        ("cześć piotrze wieczorek", ["piotrze wieczorek"]),
        ("dzień dobry panie nowak", ["nowak"]),
        ("panie wróbel, gdzie faktura?", ["wróbel"]),
    ],
)
def test_a_mans_nominative_surname_is_taken_after_a_vocative(text, expected):
    """Polish addresses a man by his surname in the nominative: "Panie Nowak", "Pawle Wróbel"."""
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Umowę podpisał J. Kowalski, PESEL 50810112354.", ["J. Kowalski"]),
        ("Umowę podpisała A. Zielińska.", ["A. Zielińska"]),
        ("Z poważaniem\nM. Wiśniewski\nDział Kadr", ["M. Wiśniewski"]),
        ("Podpisali A.M. Nowak i J. K. Wróbel", ["A.M. Nowak", "J. K. Wróbel"]),
    ],
)
def test_an_initial_takes_the_surname_after_it(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    "text", ["P.P.H.U. Kowalski", "Plan B. Wtorek", "witamina C. Rynek", "Dz. U. z 2020 r."]
)
def test_an_abbreviation_or_a_sentence_end_is_not_an_initial(text):
    assert _found(text) == []


def test_a_name_does_not_continue_into_a_company_on_the_next_line():
    assert _found("Pozdrawiam\nKamil\nNowak Transport") == ["Kamil"]
    assert _found("Pozdrawiam\nKamil\nNowak Sp. z o.o.") == ["Kamil"]


def test_nothing_is_returned_when_person_masking_is_off():
    assert _found("jan nowak dzwonił", enabled={"PESEL"}) == []


def test_every_ambiguous_form_is_a_form_of_a_listed_name():
    assert set(AMBIGUOUS_FORMS) <= set(NAME_FORMS)


def test_masculine_names_follow_polish_alternations():
    assert {"pawła", "pawłowi", "pawłem", "pawle"} <= set(decline_masculine("paweł"))
    assert {"marka", "markowi", "markiem", "marku"} <= set(decline_masculine("marek"))
    assert {"maćka", "maćkiem"} <= set(decline_masculine("maciek"))
    assert "piotrze" in decline_masculine("piotr")
    assert "robercie" in decline_masculine("robert")
    assert "dawidzie" in decline_masculine("dawid")
    assert {"jerzego", "jerzemu", "jerzym"} <= set(decline_masculine("jerzy"))
    assert {"jasia", "jasiu"} <= set(decline_masculine("jaś"))


def test_feminine_names_follow_polish_alternations():
    assert "kindze" in decline_feminine("kinga")
    assert "marcie" in decline_feminine("marta")
    assert "barbarze" in decline_feminine("barbara")
    assert "mai" in decline_feminine("maja")
    assert "marii" in decline_feminine("maria")
    assert {"kasi", "kasiu"} <= set(decline_feminine("kasia"))
    assert "urszuli" in decline_feminine("urszula")
    assert "wioletcie" in decline_feminine("wioletta")


def test_a_surname_ending_carries_gender_and_case():
    bednarkiem = surname_reading("bednarkiem")
    assert bednarkiem is not None
    assert bednarkiem.strong
    assert ("m", "ins") in bednarkiem.tags
    kowalskiej = surname_reading("kowalskiej")
    assert kowalskiej is not None
    assert kowalskiej.tags == {("f", "gen"), ("f", "dat"), ("f", "loc")}
    assert surname_reading("polskiej") is None
    assert surname_reading("pracownik") is None
    assert surname_reading("pracownika") is None
    assert surname_reading("Pani") is None


def test_a_common_adjective_is_recognised_in_any_form():
    assert is_common_adjective("polskiej")
    assert is_common_adjective("Angielskimi")
    assert is_common_adjective("polską")
    assert not is_common_adjective("kowalskiej")
    assert not is_common_adjective("Warszawa")


async def test_the_reported_message_is_masked_without_a_language_model():
    text = "Mam sprawę sądową z panem kamilem bednarkiem i on nie chciał zapłacić co teraz?"
    result = await Anonymizer(default_guards(with_ner=False)).anonymize(text)
    assert result.anonymized == (
        "Mam sprawę sądową z panem <OSOBA_1> i on nie chciał zapłacić co teraz?"
    )
    assert result.mapping == {"<OSOBA_1>": "kamilem bednarkiem"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("rozmawiałem z mają kowalską-nowak", ["mają kowalską-nowak"]),
        ("marek kowalski-nowak dzwonił", ["marek kowalski-nowak"]),
        ("Z Mają Kowalską-Nowak", ["Mają Kowalską-Nowak"]),
    ],
)
def test_a_double_surname_is_as_strong_as_its_stronger_part(text, expected):
    """Read by its last part only, "kowalską-nowak" was weak, and the whole person leaked."""
    assert _found(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Dzień dobry, Pana przesyłka została nadana wczoraj.",
        "Uprzejmie informujemy, że Pana pożyczka została zatwierdzona.",
        "Pana paczka czeka w paczkomacie.",
        "Pana składka wzrośnie od stycznia.",
        "Czy Pana usterka została zgłoszona?",
    ],
)
def test_a_noun_in_ka_after_pana_is_not_a_surname(text):
    """Feminine nouns in -ka read as the genitive of an -ek surname: Bednarek, Bednarka."""
    assert _found(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("rozmawiałem z panem bednarkiem", ["bednarkiem"]),
        ("list do pana bednarka", ["bednarka"]),
    ],
)
def test_a_lowercase_ek_surname_after_a_form_of_address_is_still_masked(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Oliwier Nowak", ["Oliwier Nowak"]),
        ("Praktyki u Oliwiera Nowaka", ["Oliwiera Nowaka"]),
        ("Opiekunka: Ksenia Lis-Kowalska", ["Ksenia Lis-Kowalska"]),
        ("Matylda Nowak", ["Matylda Nowak"]),
        ("rozmowa z Maurycym Kowalskim", ["Maurycym Kowalskim"]),
        ("Celina Zając", ["Celina Zając"]),
        ("Melania Kot", ["Melania Kot"]),
        ("Otylia Wiśniewska", ["Otylia Wiśniewska"]),
        ("Cyprian Lis", ["Cyprian Lis"]),
        ("Lilianna Kowalczyk", ["Lilianna Kowalczyk"]),
    ],
)
def test_popular_first_names_missing_from_the_list_are_known(text, expected):
    """Oliwier, Ksenia and Matylda are common names, and "Oliwier Nowak" went out whole."""
    assert _found(text) == expected


@pytest.mark.parametrize(
    ("name", "parts"),
    [
        ("Jan Kowalski", ["Kowalski"]),
        ("JAN KOWALSKI", ["KOWALSKI"]),
        ("Kowalski Jan", ["Kowalski"]),
        ("Jana Kowalskiego", ["Kowalskiego"]),
        ("dr Jan Kowalski", ["Kowalski"]),
        ("Dyrektor Jan Kowalski", ["Kowalski"]),
        ("Anna Nowak-Kępa", ["Nowak-Kępa", "Nowak", "Kępa"]),
        ("Anna-Maria Nowak", ["Nowak"]),
        ("Jan", []),
    ],
)
def test_the_parts_of_a_name_that_point_to_the_person_are_its_surname(name, parts):
    assert person_name_parts(name) == parts


def test_a_name_with_no_known_first_name_gives_its_parts_only_when_trusted():
    """A model that reads "Politechnika Krakowska" as a person must not mask "Politechnika"."""
    assert person_name_parts("Politechnika Krakowska") == ["Politechnika"]
    assert person_name_parts("Politechnika Krakowska", needs_first_name=True) == []
    assert person_name_parts("Jan Kowalski", needs_first_name=True) == ["Kowalski"]
