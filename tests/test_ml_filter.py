"""The ML filter wired into the NER guard, on a made-up model's answer.

The filter's decision is replaced by one these tests can predict, so they check the wiring: what
reaches it, when it runs, and that what it drops is gone.
"""

import re

import pytest

from polanon._detection import detect
from polanon.guards import _span_features
from polanon.guards.base import Guard
from polanon.guards.ner import NerGuard
from polanon.types import Span

TEXT = "Pisze Anna Nowak, a Zenon Wrzosek milczy."


class _Token:
    def __init__(self, text: str, idx: int) -> None:
        self.text = text
        self.idx = idx
        self.pos_ = "PROPN" if text[:1].isupper() else "VERB"
        self.lemma_ = ""
        self.has_vector = True
        self.is_space = False


class _Entity:
    def __init__(self, word: str) -> None:
        self.start_char = TEXT.index(word)
        self.end_char = self.start_char + len(word)
        self.label_ = "persName"


class _Document:
    def __init__(self) -> None:
        self.ents = [_Entity("Anna Nowak"), _Entity("Zenon Wrzosek")]
        self._tokens = [_Token(m.group(), m.start()) for m in re.finditer(r"[^\s,]+", TEXT)]

    def __iter__(self):
        return iter(self._tokens)


class _Rule(Guard):
    name = "rule"

    def detect(self, text, enabled):
        start = text.index("Anna Nowak")
        return [Span(start, start + len("Anna Nowak"), "PERSON", 1.0, self.name)]


def _found(guard: NerGuard) -> list[str]:
    guard._nlp = lambda _text: _Document()
    spans = detect(TEXT, None, {"PERSON"}, [_Rule(), guard])
    return [TEXT[span.start : span.end] for span in spans if span.source == "spacy_ner"]


@pytest.fixture
def seen(monkeypatch) -> list[dict[str, int]]:
    """Let the filter keep a span only where a rule found something too, and record it."""
    features: list[dict[str, int]] = []

    def keeps(found: dict[str, int], entity_type: str, vector=(), context=()) -> bool:
        features.append(found)
        return bool(found["rules_found_it"])

    monkeypatch.setattr(_span_features, "ml_filter_keeps", keeps)
    return features


def test_the_ml_filter_is_on_by_default_and_reviews_knowing_what_the_rules_found(seen):
    assert _found(NerGuard()) == ["Anna Nowak"]
    assert len(seen) == 2
    assert all(features["pos_a_proper_noun"] for features in seen)


def test_switched_off_or_in_the_paranoid_mode_nothing_is_reviewed(seen):
    assert _found(NerGuard(ml_filter=False)) == ["Anna Nowak", "Zenon Wrzosek"]
    assert _found(NerGuard(mode="paranoid")) == ["Anna Nowak", "Zenon Wrzosek"]
    assert seen == []


def test_only_the_words_inside_the_model_s_names_are_remembered_and_never_as_written():
    """A guard remembers thousands of texts, by digest only: the filter keeps what the model read
    in the words of the names, where they stand, and not a letter of them."""
    guard = NerGuard()
    guard._nlp = lambda _text: _Document()
    guard.detect(TEXT, {"PERSON"})
    [words] = guard._words.values()
    names = ("Anna", "Nowak", "Zenon", "Wrzosek")
    assert [TEXT[start:end] for start, end, _ in words] == list(names)
    assert not any(name in repr(guard._words) for name in names)


def test_a_text_whose_words_are_forgotten_keeps_every_span(seen):
    guard = NerGuard()
    guard._nlp = lambda _text: _Document()
    spans = guard.detect(TEXT, {"PERSON"})
    guard._words.clear()
    assert guard._review(TEXT, spans, []) == spans
    assert seen == []


def test_a_review_reads_the_text_and_each_line_once_however_many_spans_they_hold(seen, monkeypatch):
    """Read once for every span, a text of 100 KB took 13 s to mask instead of 2.5 s, and a text
    on one line, as a tool's JSON is, read its line as often."""
    from polanon.guards import ner

    guard = NerGuard()
    guard._nlp = lambda _text: _Document()
    spans = guard.detect(TEXT, {"PERSON"})
    reads: list[str] = []
    for name in ("_is_english_text", "_language"):
        read = getattr(ner, name)
        monkeypatch.setattr(ner, name, lambda text, read=read: reads.append(text) or read(text))
    guard._review(TEXT, spans, [])
    assert len(spans) == 2
    assert reads == [TEXT, TEXT]
    assert [(f["text_english"], f["line_polish"]) for f in seen] == [(0, 0), (0, 0)]


class _Pipeline:
    """A made-up spaCy pipeline that names its model as a real one does."""

    lang = "pl"

    def __init__(self, name: str) -> None:
        self.meta = {"lang": "pl", "name": name}

    def __call__(self, _text: str) -> _Document:
        return _Document()


@pytest.mark.parametrize(
    ("name", "reviewed"), [("core_news_md", 2), ("core_news_lg", 0), ("core_news_sm", 0)]
)
def test_the_ml_filter_reviews_only_with_the_model_it_was_fitted_on(seen, name, reviewed):
    """Its weights read the vectors of pl_core_news_md: another model's are another space, and
    the small model has none, so there the hand-written filters decide alone."""
    guard = NerGuard()
    guard._nlp = _Pipeline(name)
    detect(TEXT, None, {"PERSON"}, [_Rule(), guard])
    assert len(seen) == reviewed


def test_the_switch_takes_true_or_false_only():
    with pytest.raises(TypeError, match="ml_filter is True or False, not 'after'"):
        NerGuard(ml_filter="after")  # type: ignore[arg-type]


def test_the_shipped_weights_score_every_feature_they_name():
    context = ("before:", "after:", "end:", "end2:")
    names = {
        name.split(":")[-1]
        for name in _span_features.fitted.WEIGHTS
        if not name.startswith(context)
    }
    assert names <= set(_span_features.FEATURES)
    assert len(_span_features.fitted.VECTOR) == 300
    assert set(_span_features.fitted.THRESHOLDS) == {"PERSON", "LOCATION", "ORG"}
    everything = dict.fromkeys(_span_features.FEATURES, 1)
    assert 0 < _span_features.ml_filter_score(everything, "PERSON") < 1


def test_default_guards_hand_the_switch_to_the_spacy_guard():
    pytest.importorskip("spacy")
    from polanon import default_guards
    from polanon.types import NerModelUnavailableError

    try:
        on, off = default_guards(), default_guards(ml_filter=False)
    except NerModelUnavailableError:
        pytest.skip("pl_core_news_md is not installed")
    assert [guard.ml_filter for guard in on if isinstance(guard, NerGuard)] == [True]
    assert [guard.ml_filter for guard in off if isinstance(guard, NerGuard)] == [False]


def test_the_shipped_filter_drops_a_month_and_keeps_a_name():
    """On a text like the benchmarks', with the shipped weights and thresholds: an English month
    the model reads as a person goes, a Polish name stays."""
    pytest.importorskip("spacy")
    try:
        guard = NerGuard()
        guard._get_pipeline()
    except RuntimeError:
        pytest.skip("pl_core_news_md is not installed")
    text = "Data: February. Umowę podpisał Jan Kowalski z Gdańska."
    spans = detect(text, None, {"PERSON", "LOCATION", "ORG"}, [guard])
    found = {text[span.start : span.end] for span in spans}
    assert "Jan Kowalski" in found
    assert "February" not in found
    without = NerGuard(ml_filter=False)
    assert "February" in {text[s.start : s.end] for s in without.detect(text, {"PERSON"})}
