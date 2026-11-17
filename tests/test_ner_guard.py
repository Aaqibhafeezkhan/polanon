"""Polish NER guard. Needs the spaCy model, so every test here is marked slow."""

import pytest

from polanon.guards.ner import LABEL_MAP, NerGuard, _is_only_labels

pytestmark = pytest.mark.slow

SENTENCE = "Wczoraj Jan Kowalski podpisał umowę w Warszawie dla firmy Orlen."


def _model_available() -> bool:
    try:
        import spacy

        spacy.load("pl_core_news_md")
    except (ImportError, OSError):
        return False
    return True


needs_model = pytest.mark.skipif(not _model_available(), reason="pl_core_news_md is not installed")


@pytest.fixture(scope="module")
def guard() -> NerGuard:
    return NerGuard()


@needs_model
def test_detects_person_location_and_organisation(guard):
    found = {
        (s.entity_type, SENTENCE[s.start : s.end])
        for s in guard.detect(SENTENCE, set(LABEL_MAP.values()))
    }
    assert ("PERSON", "Jan Kowalski") in found
    assert ("LOCATION", "Warszawie") in found
    assert ("ORG", "Orlen") in found


@needs_model
def test_only_returns_the_enabled_entity_types(guard):
    types = {s.entity_type for s in guard.detect(SENTENCE, {"PERSON"})}
    assert types == {"PERSON"}


@needs_model
def test_disabled_guard_returns_nothing(guard):
    assert guard.detect(SENTENCE, set()) == []


@needs_model
def test_blank_input_returns_nothing(guard):
    assert guard.detect("   ", {"PERSON"}) == []


class _Entity:
    def __init__(self, text: str, word: str, label: str = "placeName") -> None:
        self.start_char = text.index(word)
        self.end_char = self.start_char + len(word)
        self.label_ = label


class _Document:
    def __init__(self, ents: list[_Entity]) -> None:
        self.ents = ents

    def __iter__(self):
        return iter(())  # no words: these tests read the filters, not the ML filter


def test_a_lowercase_adjective_tagged_as_a_place_is_dropped():
    """Typed in lowercase, the Polish model tagged "polskiej" and "angielskiej" as places."""
    text = "nie ma ani polskiej, ani angielskiej wersji dla Polski"
    document = _Document([_Entity(text, word) for word in ("polskiej", "angielskiej", "Polski")])

    guard = NerGuard()
    guard._nlp = lambda _text: document
    found = [text[s.start : s.end] for s in guard.detect(text, {"LOCATION"})]
    assert found == ["Polski"]


def test_a_business_abbreviation_tagged_as_a_company_is_dropped():
    """ "co to jest SEO?" went out as "co to jest <FIRMA_1>?"."""
    text = "SEO, PPC i CRM dla klienta Orlen"
    document = _Document(
        [_Entity(text, word, "orgName") for word in ("SEO", "PPC", "CRM", "Orlen")]
    )

    guard = NerGuard()
    guard._nlp = lambda _text: document
    assert [text[s.start : s.end] for s in guard.detect(text, {"ORG"})] == ["Orlen"]


ENGLISH_AND_POLISH = (
    "Safety: NOT every result goes to The Workspace, so ask Maria Lindqvist for the results.\n"
    "Zadzwoń do pani nowak w sprawie umowy.\n"
    "Prosze o kontakt z panem kowalskim w sprawie faktury."
)


def _found(guard: NerGuard, text: str, entities: list[_Entity]) -> list[str]:
    guard._nlp = lambda _text: _Document(entities)
    return [text[s.start : s.end] for s in guard.detect(text, {"PERSON", "LOCATION", "ORG"})]


def test_on_an_english_line_the_polish_model_keeps_only_names_in_mid_sentence():
    """An agent's English system prompt came back with 89 tokens, for "result", "NOT", "Safety".

    The Polish model reads English words as names. On a plainly English line only a capitalised
    name in mid-sentence is kept; Polish lines, in lowercase or without diacritics, stay as they
    were.
    """
    text = ENGLISH_AND_POLISH
    entities = [
        _Entity(text, "Safety", "orgName"),
        _Entity(text, "NOT", "orgName"),
        _Entity(text, "result", "orgName"),
        _Entity(text, "The Workspace"),
        _Entity(text, "Maria Lindqvist", "persName"),
        _Entity(text, "results", "orgName"),
        _Entity(text, "nowak", "persName"),
        _Entity(text, "kowalskim", "persName"),
    ]
    assert _found(NerGuard(), text, entities) == ["Maria Lindqvist", "nowak", "kowalskim"]


def test_after_a_colon_on_an_english_line_a_name_is_kept():
    text = "Questions about the invoice go to: John Smith, who will answer them."
    entities = [_Entity(text, "Questions", "orgName"), _Entity(text, "John Smith", "persName")]
    assert _found(NerGuard(), text, entities) == ["John Smith"]


def test_on_a_line_with_no_language_a_lowercase_word_is_no_name():
    """An agent's tool had a parameter "replace_all"; the model tagged the word as a person.

    The tool schema holds it as a key, which cannot be masked, so the gateway's last check refused
    every call. A line too short to show a language keeps capitalised spans, and lowercase ones
    only when they hold a Polish first name. "Max" of "Max results" is a product's tier, and
    no name since the tiers are dropped.
    """
    text = (
        "replace_all\nnext\nMax results\nSkills for\njan_kowalski\nKOWALSKI JAN\nanna nowak\n"
        "use `jan`\nUstaw volume_level na pół głośności."
    )
    entities = [
        _Entity(text, "replace_all", "persName"),
        _Entity(text, "next", "placeName"),
        _Entity(text, "Max", "persName"),
        _Entity(text, "Skills for", "orgName"),
        _Entity(text, "jan_kowalski", "persName"),
        _Entity(text, "KOWALSKI JAN", "persName"),
        _Entity(text, "anna nowak", "persName"),
        _Entity(text, "`jan`", "persName"),
        _Entity(text, "volume_level", "persName"),
    ]
    assert _found(NerGuard(), text, entities) == [
        "jan_kowalski",
        "KOWALSKI JAN",
        "anna nowak",
    ]


def test_an_english_line_quoting_polish_is_english_and_so_are_short_lines_of_an_english_text():
    """An agent's model reasons in English and quotes the user in Polish: "The user asks" and
    "the model" stood on lines a Polish letter in the quote made Polish. Its headings, "Possible
    causes" or "Max", stood on lines too short to show a language.

    A line is now Polish by its Polish words, letters included, not by one letter; and in a text
    that is plainly English as a whole, a line that shows no language is English too. A Polish
    text keeps such a line as it was.
    """
    english = (
        "Possible causes\n"
        "The user asks „czy światło w kuchni jest włączone” and the model has to check it.\n"
        "The service may be down, or the token is no longer valid for this account.\n"
        "Look at the last error from Anna Lindqvist before you restart the whole server.\n"
        "Max"
    )
    entities = [
        _Entity(english, "Possible causes", "persName"),
        _Entity(english, "The user asks", "orgName"),
        _Entity(english, "the model", "orgName"),
        _Entity(english, "Anna Lindqvist", "persName"),
        _Entity(english, "Max", "persName"),
    ]
    assert _found(NerGuard(), english, entities) == ["Anna Lindqvist"]
    polish = "Jutro spotkanie z zespołem o dziesiątej w biurze.\nMarek\nProszę o potwierdzenie."
    assert _found(NerGuard(), polish, [_Entity(polish, "Marek", "persName")]) == ["Marek"]


def test_a_quoted_word_of_code_is_no_name_on_a_polish_line():
    """A smart home's state printed as JSON in a tool's result is a Polish line by its device
    names, and the Polish model took its keys, "attributes" and "context", for places. Masked
    once, they were masked in every key of every later result.
    """
    text = (
        '{"entity_id": "light.salon", "state": "on", "attributes": {"friendly_name": '
        '"Światło w salonie", "city": "Gdańsk"}, "context": {"user_id": null}}'
    )
    entities = [
        _Entity(text, "attributes"),
        _Entity(text, "context"),
        _Entity(text, "Gdańsk"),
    ]
    assert _found(NerGuard(), text, entities) == ["Gdańsk"]


def test_a_piece_of_an_identifier_and_a_traceback_are_no_names():
    """In an agent's tool results "service" of `ha_call_service` was a company, "File" of a Python
    traceback a place, and "while" in a Polish sentence a place too."""
    text = (
        "Użyj `ha_call_service` do włączenia światła w salonie, while trwa remont.\n"
        '  File "/srv/app/main.py", line 12, in Wroclaw_handler\n'
        "Spotkanie w Gdańsku."
    )
    entities = [
        _Entity(text, "service", "orgName"),
        _Entity(text, "while"),
        _Entity(text, "File"),
        _Entity(text, "Gdańsku"),
    ]
    assert _found(NerGuard(), text, entities) == ["Gdańsku"]
    escaped = "Zgłoszenie od klientki:\\nAnna Lindqvist z Gdańska"
    assert _found(NerGuard(), escaped, [_Entity(escaped, "Anna Lindqvist", "persName")]) == [
        "Anna Lindqvist"
    ]


def test_a_span_opening_with_an_english_small_word_is_no_name_on_a_short_line():
    """A tool's description, "That's the limit", is too short to show a language, and the model
    took "That's" for a person in it: 64 times in one agent's call, once it was known."""
    text = "That's the limit\nThis Week\nOla Kowalska"
    entities = [
        _Entity(text, "That's", "persName"),
        _Entity(text, "This Week", "orgName"),
        _Entity(text, "Ola Kowalska", "persName"),
    ]
    assert _found(NerGuard(), text, entities) == ["Ola Kowalska"]


def test_a_place_or_a_company_in_lowercase_is_no_name_even_on_a_polish_line():
    """Polish manuals in an agent's tool results gave the places "text" and "context" and the
    company "service", masked 54, 25 and 38 times in one call once known. A place or a company of
    one word written in lowercase is dropped on every line now: "gdyni" in a chat goes out as
    written, while a person in lowercase, "kasi", is still the rules' and the model's to mask.
    """
    text = "Jadę do gdyni, potem do Sopotu; pole text i service są puste, daj znać kasi."
    entities = [
        _Entity(text, "gdyni"),
        _Entity(text, "Sopotu"),
        _Entity(text, "text"),
        _Entity(text, "service", "orgName"),
        _Entity(text, "kasi", "persName"),
    ]
    assert _found(NerGuard(), text, entities) == ["Sopotu", "kasi"]


def test_a_span_opening_with_or_holding_an_english_small_word_is_no_name_on_any_line():
    """In an agent's call "That's" stood mid-sentence on an English line, after a dash, and was
    masked 76 times once known. A Polish manual quoting English gave "Before" opening a sentence,
    "the model" and "tokens and passwords". A small word inside a span ends a name, so "Maria
    Lindqvist at Northwind Traders" gives two. "Will" and "May" are first names too, so a span
    they open is still read as the line's language says.
    """
    text = (
        "Retries stop after five attempts — That's what the manual says about it.\n"
        "Send the report to Maria Lindqvist at Northwind Traders before the review.\n"
        "Zapisz plik, a przed tym sprawdź ustawienia. Before, wyślij to do the model, potem "
        "tokens and passwords usuń z pliku.\n"
        "Zadzwoń do Will Nowak w sprawie umowy, a potem do Ola Kowalska."
    )
    entities = [
        _Entity(text, "That's", "persName"),
        _Entity(text, "Maria Lindqvist at Northwind Traders", "orgName"),
        _Entity(text, "Before", "persName"),
        _Entity(text, "the model", "orgName"),
        _Entity(text, "tokens and passwords", "orgName"),
        _Entity(text, "Will Nowak", "persName"),
        _Entity(text, "Ola Kowalska", "persName"),
    ]
    assert _found(NerGuard(), text, entities) == [
        "Maria Lindqvist",
        "Northwind Traders",
        "Will Nowak",
        "Ola Kowalska",
    ]


def test_a_word_in_inline_code_is_no_name_unless_it_holds_a_first_name():
    """An agent's Polish answers named a button `Uncheck`, and the Polish model took it for a
    person on line after line: masked 85 times in one call once known. What stands between two
    backticks of its line is code; a login with a Polish first name in it is still a person's.
    """
    text = (
        "Kliknij `Uncheck`, a potem `Zapisz zmiany` w oknie ustawień.\n"
        "Konto `anna.nowak` założyła Ola Kowalska.\n"
        "Krok 3:\\nWybierz `Archive` i potwierdź.\n"
        "Pomiń ` znak, a potem Gdańsk."
    )
    entities = [
        _Entity(text, "Uncheck", "persName"),
        _Entity(text, "Zapisz zmiany", "orgName"),
        _Entity(text, "anna.nowak", "persName"),
        _Entity(text, "Ola Kowalska", "persName"),
        _Entity(text, "Archive", "orgName"),
        _Entity(text, "Gdańsk"),
    ]
    assert _found(NerGuard(), text, entities) == ["anna.nowak", "Ola Kowalska", "Gdańsk"]


def test_a_span_without_a_letter_is_no_name():
    """Under a Python traceback the Polish model took the carets for companies and places, a
    Markdown heading's "####" for a company and the offset of a timestamp for a place."""
    text = (
        "    wynik = licz(dane)\n"
        "            ^^^^^^^^^^\n"
        "#### Konfiguracja w Gdańsku\n"
        "Ostatnia zmiana 2026-01-02T10:00:00.808+00:00 w Sopocie."
    )
    entities = [
        _Entity(text, "^^^^^^^^^^", "orgName"),
        _Entity(text, "####", "orgName"),
        _Entity(text, "Gdańsku"),
        _Entity(text, "808+00:00"),
        _Entity(text, "Sopocie"),
    ]
    assert _found(NerGuard(), text, entities) == ["Gdańsku", "Sopocie"]


def test_the_english_name_of_a_language_is_no_place_or_company():
    """A Polish manual in an agent's tool results spoke of "Polish" and "English" texts, and the
    Polish model took both for places, "Polish names" and "Polish-first" for companies. "German"
    is a Polish first name too, so a person it opens is kept."""
    text = (
        "Wersja 0.3 dodaje tryb English, a Polish zostaje domyślny dla biura w Gdańsku.\n"
        "Lista Polish names obejmuje też nazwiska; dokument jest Polish-first.\n"
        "Spotkanie z German Nowak w Krakowie."
    )
    entities = [
        _Entity(text, "English"),
        _Entity(text, "Polish"),
        _Entity(text, "Gdańsku"),
        _Entity(text, "Polish names", "orgName"),
        _Entity(text, "Polish-first", "orgName"),
        _Entity(text, "German Nowak", "persName"),
        _Entity(text, "Krakowie"),
    ]
    assert _found(NerGuard(), text, entities) == ["Gdańsku", "German Nowak", "Krakowie"]


def test_a_word_after_a_model_number_is_no_person():
    """An agent's tool results listed a laptop's "M4 Max" and a vacuum's "S8 Pro", and the Polish
    model took "Max" for a person, masked 44 times in one call once known. A person that holds a
    Polish first name is kept after such a word too."""
    text = (
        "Laptop ma układ M4 Max, a odkurzacz S8 Pro stoi w biurze u Ola Kowalska.\n"
        "Przy bramie 12A Maja Nowak odbierze paczkę."
    )
    entities = [
        _Entity(text, "Max", "persName"),
        _Entity(text, "Pro", "persName"),
        _Entity(text, "Ola Kowalska", "persName"),
        _Entity(text, "Maja Nowak", "persName"),
    ]
    assert _found(NerGuard(), text, entities) == ["Ola Kowalska", "Maja Nowak"]


def test_a_product_tier_a_code_identifier_and_a_word_with_a_digit_are_no_names():
    """In an agent's calls the Polish model took a vacuum's fan speed "Max" for a person, masked
    57 times once known, Python's `JSONDecodeError` and `HTTPError` too, an invoice number for a
    place and a line of a CSV file for a company. A name with a capital inside it, MacLeod or
    DeVito, is kept."""
    text = (
        "Ustaw moc na Max albo Max+, wersja Pro też ma tryb Ultra.\n"
        "Błąd JSONDecodeError, potem HTTPError z serwera w Gdańsku.\n"
        "Faktura FV/2026/10/014;Studio Lumen;615,00;4012736604 jest w pliku.\n"
        "Projekt prowadzi Anna MacLeod z Danny DeVito."
    )
    entities = [
        _Entity(text, "Max", "persName"),
        _Entity(text, "Max+", "persName"),
        _Entity(text, "Pro", "orgName"),
        _Entity(text, "Ultra", "orgName"),
        _Entity(text, "JSONDecodeError", "persName"),
        _Entity(text, "HTTPError", "persName"),
        _Entity(text, "Gdańsku"),
        _Entity(text, "FV/2026/10/014;Studio Lumen;615,00;4012736604", "orgName"),
        _Entity(text, "Anna MacLeod", "persName"),
        _Entity(text, "Danny DeVito", "persName"),
    ]
    assert _found(NerGuard(), text, entities) == [
        "Gdańsku",
        "Studio Lumen",
        "Anna MacLeod",
        "Danny DeVito",
    ]
    paranoid = _found(NerGuard(mode="paranoid"), text, entities)
    assert {"Max", "JSONDecodeError", "FV/2026/10/014"} <= set(paranoid)
    assert not {"615,00", "4012736604"} & set(paranoid)


def test_a_word_with_one_capital_inside_may_be_a_handle_and_stays_a_name():
    """A capital after a small letter marks code, `MediaPlayerEntity` or `HTTPServer`, but a
    person's handle or a small company's name is written so too, "AnnaLis" or "ZielonyDom":
    dropped, it went out in the clear. Only a word with two capitals inside, or an abbreviation
    run into a word, is taken for code, and never one with a Polish first name in it."""
    text = (
        "Klasa MediaPlayerEntity i HTTPServer są w kodzie.\n"
        "Konto AnnaLis prowadzi sklep ZielonyDom, a OlaNowakDesign robi zdjęcia."
    )
    entities = [
        _Entity(text, "MediaPlayerEntity", "orgName"),
        _Entity(text, "HTTPServer", "orgName"),
        _Entity(text, "AnnaLis", "persName"),
        _Entity(text, "ZielonyDom", "orgName"),
        _Entity(text, "OlaNowakDesign", "orgName"),
    ]
    assert _found(NerGuard(), text, entities) == ["AnnaLis", "ZielonyDom", "OlaNowakDesign"]


def test_the_english_of_software_is_no_name():
    """An agent's answers named buttons and settings in English inside Polish sentences, and the
    Polish model took them for people and companies: "Uncheck", "Authorization", "Privacy
    Policy". A span made of such words is dropped; a person whose name is also an English word,
    Hugh Grant, is kept, and so is one with a Polish first name."""
    text = (
        "W ustawieniach kliknij Uncheck, potem Authorization i Privacy Policy.\n"
        "Internal Guide jest w zakładce Scheduling, a Switch obok.\n"
        "Film z Hugh Grant poleciła nam Ola Kowalska."
    )
    entities = [
        _Entity(text, "Uncheck", "persName"),
        _Entity(text, "Authorization", "persName"),
        _Entity(text, "Privacy Policy", "orgName"),
        _Entity(text, "Internal Guide", "orgName"),
        _Entity(text, "Scheduling", "persName"),
        _Entity(text, "Switch", "orgName"),
        _Entity(text, "Hugh Grant", "persName"),
        _Entity(text, "Ola Kowalska", "persName"),
    ]
    assert _found(NerGuard(), text, entities) == ["Hugh Grant", "Ola Kowalska"]
    paranoid = _found(NerGuard(mode="paranoid"), text, entities)
    assert {"Uncheck", "Privacy Policy", "Scheduling", "Switch"} <= set(paranoid)


def test_the_english_of_software_in_front_of_a_person_is_cut_off():
    """The Polish model read the label of a field as a first name before a surname: "Login
    Kowalski", "Email Nowaka" and "Admin: Ola Kowalska" came back as one person each, and the
    label went to the model inside the token. Such a word is cut off and the person is kept. The
    list leaves out words that are also names, so Grant of "Grant Nowak" stays, and the paranoid
    mode keeps the span whole."""
    text = (
        "Login Kowalski nie działa, a Email Nowaka jest nieaktualny.\n"
        "Admin: Ola Kowalska zatwierdziła zmianę, a konto Login jkowalski wygasło.\n"
        "Raport przysłał Grant Nowak."
    )
    entities = [
        _Entity(text, "Login Kowalski", "persName"),
        _Entity(text, "Email Nowaka", "persName"),
        _Entity(text, "Admin: Ola Kowalska", "persName"),
        _Entity(text, "Login jkowalski", "persName"),
        _Entity(text, "Grant Nowak", "persName"),
    ]
    assert _found(NerGuard(), text, entities) == [
        "Kowalski",
        "Nowaka",
        "Ola Kowalska",
        "jkowalski",
        "Grant Nowak",
    ]
    assert "Login Kowalski" in _found(NerGuard(mode="paranoid"), text, entities)


def test_chinese_text_is_no_name_and_its_punctuation_is_not_part_of_one():
    """A page in Chinese in an agent's results gave the Polish model a "person" of Chinese
    characters, and "Home Assistant。" with the full stop, which the list of well-known names no
    longer matched. A span written in Chinese, Japanese or Korean alone is dropped, and their
    punctuation is trimmed off the ends of a span as spaces are. "Home Assistant" is made of the
    English of software, so only the paranoid mode keeps it."""
    text = "## 安装\n在 HACS 中搜索并安装，然后重启 Home Assistant。\n联系人：Ola Kowalska。"  # noqa: RUF001
    entities = [
        _Entity(text, "中搜索并安装", "persName"),
        _Entity(text, "Home Assistant。", "orgName"),
        _Entity(text, "：Ola Kowalska。", "persName"),  # noqa: RUF001
    ]
    assert _found(NerGuard(), text, entities) == ["Ola Kowalska"]
    assert _found(NerGuard(mode="paranoid"), text, entities) == [
        "中搜索并安装",
        "Home Assistant",
        "Ola Kowalska",
    ]


def test_the_parties_of_a_contract_are_no_names():
    """A lease read by an agent writes its parties with a capital letter, as defined terms, and
    the Polish model took "Najemcy" for a company. A party named by its role alone is dropped;
    the person who is it is kept."""
    text = (
        "Najemca zapłaci Wynajmującemu czynsz do 10. dnia miesiąca.\n"
        "Zleceniobiorca, pani Ola Kowalska, odda dzieło Zamawiającemu, a Strony podpiszą protokół."
    )
    entities = [
        _Entity(text, "Najemca", "orgName"),
        _Entity(text, "Wynajmującemu", "persName"),
        _Entity(text, "Zleceniobiorca", "orgName"),
        _Entity(text, "Ola Kowalska", "persName"),
        _Entity(text, "Zamawiającemu", "orgName"),
        _Entity(text, "Strony", "orgName"),
    ]
    assert _found(NerGuard(), text, entities) == ["Ola Kowalska"]


def test_in_paranoid_mode_every_name_the_model_finds_is_kept():
    """The filters for English and code let a little through by design: a name opening an English
    sentence, a word in inline code, a place typed in lowercase. With `mode="paranoid"` every
    span with a letter in it is kept, so the carets under a traceback are still dropped."""
    text = (
        "Erik Holm from our team will measure the room on Monday.\n"
        "Kliknij `Uncheck`, a jutro jedziemy do sopotu.\n"
        "            ^^^^^^^^^^"
    )
    entities = [
        _Entity(text, "Erik Holm", "persName"),
        _Entity(text, "Uncheck", "persName"),
        _Entity(text, "sopotu"),
        _Entity(text, "^^^^^^^^^^", "orgName"),
    ]
    assert _found(NerGuard(), text, entities) == []
    assert _found(NerGuard(mode="paranoid"), text, entities) == ["Erik Holm", "Uncheck", "sopotu"]


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="'normal' or 'paranoid'"):
        NerGuard(mode="strict")  # type: ignore[arg-type]


def test_a_number_a_quote_and_an_english_word_are_no_names():
    """In an agent's tool results the Polish model tagged "2026" as a company, `File "` of a
    Python traceback, Python's "None" as a person, and "has" and "before" on lines it read as
    Polish.
    """
    text = (
        "Zamówienie z 2026 roku.\n"
        'Plik "raport.txt" nie ma nagłówka.\n'
        "Pole has i pole before są puste, pisze Zielona Góra.\n"
        "Stan: None"
    )
    entities = [
        _Entity(text, "None", "persName"),
        _Entity(text, "2026", "orgName"),
        _Entity(text, 'Plik "', "orgName"),
        _Entity(text, "has", "orgName"),
        _Entity(text, "before", "placeName"),
        _Entity(text, "Zielona Góra"),
    ]
    assert _found(NerGuard(), text, entities) == ["Zielona Góra"]


def test_a_text_read_before_is_not_read_again():
    """An agent sends its system prompt, its tools and the whole history with every call.

    Masking a call of 105 messages took 28 seconds, 20 of them in spaCy reading texts it had read
    in the call before. The guard remembers what it found in a text, under the text's digest only.
    """
    text = "Spotkanie z Janem Kowalskim."
    read = []

    def nlp(piece):
        read.append(piece)
        return _Document([_Entity(piece, "Janem Kowalskim", "persName")])

    guard = NerGuard()
    guard._nlp = nlp
    first = guard.detect(text, {"PERSON"})
    assert guard.detect(text, {"PERSON"}) == first
    assert len(read) == 1
    first.clear()
    assert guard.detect(text, {"PERSON"}) != []
    guard.detect(text, {"PERSON", "ORG"})
    assert len(read) == 2
    assert "Kowalsk" not in repr(guard._found)

    small = NerGuard(cache_texts=1)
    small._nlp = nlp
    for _ in range(2):
        small.detect(text, {"PERSON"})
        small.detect("Spotkanie z Janem Kowalskim, jutro.", {"PERSON"})
    assert len(read) == 6
    off = NerGuard(cache_texts=0)
    off._nlp = nlp
    off.detect(text, {"PERSON"})
    off.detect(text, {"PERSON"})
    assert len(read) == 8


def test_an_english_model_is_trusted_on_english_lines():
    class _EnglishModel:
        lang = "en"

        def __init__(self, document: _Document) -> None:
            self._document = document

        def __call__(self, _text: str) -> _Document:
            return self._document

    text = ENGLISH_AND_POLISH
    guard = NerGuard(spacy_model="en_core_web_sm")
    guard._nlp = _EnglishModel(_Document([_Entity(text, "Safety", "ORG")]))
    assert [text[s.start : s.end] for s in guard.detect(text, {"ORG"})] == ["Safety"]


@needs_model
def test_ordinary_english_words_are_not_names(guard):
    text = (
        "You are a helpful assistant that runs on the user's own computer.\n"
        "Safety: Verify every claim before you act, and explain what you changed.\n"
        "Grounding: quote the documentation when you rely on it.\n"
        "Please send the signed contract to Maria Lindqvist at Northwind Traders today.\n"
        "Proszę zadzwonić do pani Anny Nowak z firmy Orlen w poniedziałek."
    )
    found = {text[s.start : s.end] for s in guard.detect(text, set(LABEL_MAP.values()))}
    assert not found & {"the user's own", "Safety", "Grounding"}
    assert any("Maria Lindqvist" in fragment for fragment in found)
    assert {"Anny Nowak", "Orlen"} <= found


def test_missing_model_raises_an_actionable_error():
    pytest.importorskip("spacy")  # without spaCy the error names the extra instead, tested apart
    with pytest.raises(RuntimeError, match="spacy download"):
        NerGuard(spacy_model="pl_does_not_exist").detect("Jan Kowalski", {"PERSON"})


@needs_model
def test_end_to_end_masking_of_a_polish_sentence():
    import asyncio

    from polanon import Anonymizer, RegexGuard

    anonymizer = Anonymizer([RegexGuard(), NerGuard()])
    result = asyncio.run(
        anonymizer.anonymize("Jan Kowalski, PESEL 50810112354, mail jan@example.com")
    )
    assert "Jan Kowalski" not in result.anonymized
    assert "<OSOBA_1>" in result.anonymized
    assert "<PESEL_1>" in result.anonymized
    assert Anonymizer.restore(result.anonymized, result.mapping).startswith("Jan Kowalski")


@needs_model
def test_entities_never_span_a_line_break(guard):
    """A detection that runs across a newline is split, so it cannot swallow the next line."""
    text = "Anna Wiśniewska\nPESEL: 45831523452"
    for span in guard.detect(text, {"PERSON", "LOCATION", "ORG"}):
        assert "\n" not in text[span.start : span.end]


@needs_model
def test_email_beats_an_overlapping_name_detection():
    """Regression: a multi-line NER span used to swallow the address inside it."""
    import asyncio

    from polanon import Anonymizer, RegexGuard

    text = "Anna Wiśniewska\ne-mail: anna.wisniewska@poczta.example\nPESEL: 45831523452"
    result = asyncio.run(Anonymizer([RegexGuard(), NerGuard()]).anonymize(text))
    assert "<EMAIL_1>" in result.anonymized
    assert "<PESEL_1>" in result.anonymized
    assert "anna.wisniewska@poczta.example" not in result.anonymized


@needs_model
def test_a_double_surname_becomes_one_person(guard):
    """ "Maria Nowak-Kowalska" came back as <OSOBA_1>-<OSOBA_2>: one person shown as two."""
    text = "Umowa z panią Marią Nowak-Kowalską została podpisana."
    spans = guard.detect(text, {"PERSON"})
    found = [text[s.start : s.end] for s in spans]
    assert any("Nowak-Kowalską" in fragment for fragment in found), found
    assert all(not fragment.endswith("Nowak") for fragment in found), found


@needs_model
def test_a_heading_made_only_of_form_words_is_not_a_name(guard):
    text = "NOTATKA WEWNĘTRZNA\n\nProwadzi: Anna Kowalska"
    found = [text[s.start : s.end] for s in guard.detect(text, {"PERSON", "ORG", "LOCATION"})]
    assert "NOTATKA WEWNĘTRZNA" not in found
    assert any("Anna" in fragment for fragment in found), found


@pytest.mark.parametrize(
    ("fragment", "only_titles"),
    [
        ("n.", True),
        ("dr n. med.", True),
        ("prof. dr hab. inż.", True),
        ("lek. dent.", True),
        ("dr n. med. Ewa Zawadzka", False),
        ("Ewa", False),
    ],
)
def test_a_span_of_academic_titles_alone_is_dropped(fragment, only_titles):
    """The Polish model tagged the "n." of "dr n. med." as a person, and it went out as a token."""
    assert _is_only_labels(fragment) is only_titles


@needs_model
def test_the_title_of_a_doctor_is_not_a_person(guard):
    text = "Ordynator oddziału: dr n. med. Ewa Zawadzka"
    found = [text[s.start : s.end] for s in guard.detect(text, {"PERSON"})]
    assert "n." not in found
    assert any("Ewa Zawadzka" in fragment for fragment in found), found


@needs_model
def test_the_word_bearer_is_not_a_place(guard):
    text = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc"
    found = [text[s.start : s.end] for s in guard.detect(text, {"LOCATION", "ORG", "PERSON"})]
    assert "Bearer" not in found


def test_a_long_text_is_cut_at_line_breaks_into_pieces_spacy_accepts():
    """spaCy refuses texts over a million characters; a long export was a crash, not a mask."""
    from polanon.guards.ner import _chunks

    text = "\n".join(["wiersz z danymi " * 4] * 60)
    pieces = list(_chunks(text, 200))
    assert "".join(piece for _, piece in pieces) == text
    assert all(len(piece) <= 200 for _, piece in pieces)
    assert all(text[offset : offset + len(piece)] == piece for offset, piece in pieces)
    assert all(piece.endswith("\n") for _, piece in pieces[:-1])


def test_a_line_longer_than_the_limit_is_cut_anyway():
    from polanon.guards.ner import _chunks

    assert [len(piece) for _, piece in _chunks("x" * 450, 200)] == [200, 200, 50]


def test_the_model_loads_once_when_threads_ask_for_it_together(monkeypatch):
    """Eight threads on a fresh guard each loaded their own copy: 1.45 GB instead of one."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    loads = []
    lock = threading.Lock()

    def slow_load(self):
        with lock:
            loads.append(self)
        time.sleep(0.05)
        return lambda text: SimpleNamespace(ents=[])

    monkeypatch.setattr(NerGuard, "_load", slow_load)
    fresh = NerGuard()
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: fresh.detect("Jan Kowalski", {"PERSON"}), range(8)))
    assert len(loads) == 1


def test_a_long_text_reaches_spacy_in_pieces_with_spans_mapped_back(monkeypatch):
    """A stand-in model that refuses long pieces, as spaCy does over its limit."""
    from types import SimpleNamespace

    from polanon.guards import ner

    monkeypatch.setattr(ner, "CHUNK_CHARS", 60)
    pieces = []

    def model(piece):
        if len(piece) > ner.CHUNK_CHARS:
            raise ValueError("[E088] text too long")
        pieces.append(piece)
        start = piece.find("Jan Kowalski")
        entity = SimpleNamespace(label_="persName", start_char=start, end_char=start + 12)
        return SimpleNamespace(ents=[entity] if start >= 0 else [])

    fresh = NerGuard(ml_filter=False)
    monkeypatch.setattr(fresh, "_get_pipeline", lambda: model)
    text = "Spotkanie w biurze o dziewiątej.\n" * 5 + "Wczoraj dzwonił Jan Kowalski.\n"
    spans = fresh.detect(text, {"PERSON"})
    assert len(pieces) > 1
    assert [text[span.start : span.end] for span in spans] == ["Jan Kowalski"]


def test_a_single_line_text_is_cut_after_a_sentence_not_inside_a_name():
    """Cut at exactly the limit, a name fell across two pieces and spaCy missed both halves."""
    from polanon.guards.ner import _chunks

    text = "Zamówienie wysłano. Rozmawiałem z Marleną Brzęczyszczykiewicz w sprawie umowy. " * 3
    pieces = list(_chunks(text, 100))
    assert "".join(piece for _, piece in pieces) == text
    assert all(len(piece) <= 100 for _, piece in pieces)
    assert all(piece.endswith(". ") for _, piece in pieces[:-1])
    assert all("Marleną Brzęczyszczykiewicz" in piece for _, piece in pieces)


def test_a_single_line_text_without_sentences_is_cut_at_a_space():
    from polanon.guards.ner import _chunks

    text = "słowo " * 40
    pieces = list(_chunks(text, 50))
    assert "".join(piece for _, piece in pieces) == text
    assert all(piece.endswith(" ") for _, piece in pieces)
