"""Polish NER guard. Needs the spaCy model, so every test here is marked slow."""

import pytest

from polanon.guards.ner import LABEL_MAP, NerGuard

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
