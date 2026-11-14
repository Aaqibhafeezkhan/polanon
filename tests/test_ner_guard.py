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
    only when they hold a Polish first name.
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
        "Max",
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

    fresh = NerGuard()
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
