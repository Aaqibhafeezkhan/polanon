"""The learned verifier wired into the NER guard, on a made-up model's answer.

The verifier's decision is replaced by one these tests can predict, so they check the wiring:
what reaches it, when it runs, and that what it drops is gone.
"""

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
        self._tokens = [_Token(word, TEXT.index(word)) for word in TEXT.replace(",", "").split()]

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
    """Let the verifier keep a span only where a rule found something too, and record it."""
    features: list[dict[str, int]] = []

    def keeps(found: dict[str, int], entity_type: str, vector=()) -> bool:
        features.append(found)
        return bool(found["rules_found_it"])

    monkeypatch.setattr(_span_features, "verifier_keeps", keeps)
    return features


@pytest.mark.parametrize("verifier", ["after", "hybrid"])
def test_the_verifier_reviews_the_model_s_spans_knowing_what_the_rules_found(seen, verifier):
    assert _found(NerGuard(verifier=verifier)) == ["Anna Nowak"]
    assert len(seen) == 2
    assert all(features["pos_a_proper_noun"] for features in seen)


def test_without_the_verifier_or_in_the_paranoid_mode_nothing_is_reviewed(seen):
    assert _found(NerGuard()) == ["Anna Nowak", "Zenon Wrzosek"]
    assert _found(NerGuard(mode="paranoid", verifier="after")) == ["Anna Nowak", "Zenon Wrzosek"]
    assert seen == []


def test_a_text_whose_words_are_forgotten_keeps_every_span(seen):
    guard = NerGuard(verifier="after")
    guard._nlp = lambda _text: _Document()
    spans = guard.detect(TEXT, {"PERSON"})
    guard._words.clear()
    assert guard._review(TEXT, spans, []) == spans
    assert seen == []


def test_an_unknown_verifier_is_refused():
    with pytest.raises(ValueError, match="verifier must be one of off, after, hybrid"):
        NerGuard(verifier="maybe")  # type: ignore[arg-type]


def test_the_shipped_weights_score_every_feature_they_name():
    names = {name.split(":")[-1] for name in _span_features.fitted.WEIGHTS}
    assert names <= set(_span_features.FEATURES)
    assert set(_span_features.fitted.THRESHOLDS) == {"PERSON", "LOCATION", "ORG"}
    everything = dict.fromkeys(_span_features.FEATURES, 1)
    assert 0 < _span_features.verifier_score(everything, "PERSON") < 1
