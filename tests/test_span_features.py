"""The features the ML filter weighs, and the labels its training data gets.

Stdlib only: the spans are made up, as the language model would return them.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from ml_filter_data import kpwr_part, label_of, openpii_part  # noqa: E402
from run import GoldEntity  # noqa: E402

from polanon.guards._span_features import (  # noqa: E402
    FEATURES,
    span_context,
    span_features,
    span_vector,
)
from polanon.types import Span  # noqa: E402


def _features(
    text: str, fragment: str, entity_type: str = "PERSON", *, vocab=None, **options: bool
) -> dict:
    start = text.index(fragment)
    span = Span(start, start + len(fragment), entity_type, 0.85, "spacy_ner")
    found_it = options.get("rules_found_it", False)
    return span_features(text, span, rules_found_it=found_it, vocab=vocab)


def test_every_feature_is_there_in_its_order_and_is_zero_or_one():
    found = _features("Wczoraj Jan Kowalski podpisał umowę.", "Jan Kowalski")
    assert tuple(found) == FEATURES
    assert set(found.values()) <= {0, 1}


def test_a_polish_name_reads_as_one():
    found = _features(
        "Umowę podpisał pan Jan Kowalski z Gdańska.", "Jan Kowalski", rules_found_it=True
    )
    assert found["type_person"] == 1
    assert found["two_words"] == 1
    assert found["every_word_capitalised"] == 1
    assert found["a_first_name"] == 1
    assert found["a_surname"] == 1
    assert found["line_polish"] == 1
    assert found["after_a_form_of_address"] == 1
    assert found["rules_found_it"] == 1
    assert found["rule_software_english"] == 0


def test_the_signs_the_filters_read_are_features():
    text = "Kliknij `Uncheck`, a potem Login Kowalski i M4 Max."
    assert _features(text, "Uncheck")["rule_inline_code"] == 1
    assert _features(text, "Uncheck")["a_software_word"] == 1
    login = _features(text, "Login Kowalski")
    assert login["a_software_word"] == 1
    assert login["a_surname"] == 1
    assert _features(text, "Max")["rule_after_a_model_number"] == 1


def test_the_place_of_a_span_in_its_sentence_is_a_feature():
    text = "Kontakt: Ola Nowak. Faktura dla Nowaka."
    assert _features(text, "Ola Nowak")["after_a_colon"] == 1
    assert _features(text, "Nowaka")["after_a_preposition"] == 1
    assert _features(text, "Ola Nowak")["opens_a_sentence"] == 0


def test_a_span_over_a_value_is_one_over_another_name_ignored_and_over_nothing_zero():
    person = GoldEntity(10, 22, "Jan Kowalski", "PERSON")
    country = GoldEntity(30, 36, "Polska", "nam_loc_gpe_country")
    assert label_of(Span(10, 22, "PERSON", 0.85, "spacy_ner"), [person], [country]) == 1
    assert label_of(Span(14, 30, "PERSON", 0.85, "spacy_ner"), [person], [country]) == 1
    assert label_of(Span(30, 36, "LOCATION", 0.85, "spacy_ner"), [person], [country]) is None
    assert label_of(Span(40, 45, "ORG", 0.85, "spacy_ner"), [person], [country]) == 0


def test_the_ml_filter_learns_a_sign_from_its_features():
    """A feature that every name has and one that every non-name has: the fitted weights tell
    them apart."""
    np = pytest.importorskip("numpy")
    from ml_filter_train import fit_logistic

    x = np.array([[1.0, 0.0]] * 20 + [[0.0, 1.0]] * 20)
    y = np.array([1.0] * 20 + [0.0] * 20)
    weights, bias = fit_logistic(x, y, 1e-3)
    scores = 1 / (1 + np.exp(-(x @ weights + bias)))
    assert scores[0] > 0.9 > 0.1 > scores[-1]


class _Vocab:
    """Vectors for a few words, as spaCy's vocabulary gives them."""

    def __init__(self) -> None:
        self.vectors = {"anna": [1.0, 0.0], "Anna": [1.0, 0.0], "nieokreślona": [0.0, 2.0]}

    def has_vector(self, word: str) -> bool:
        return word in self.vectors

    def get_vector(self, word: str) -> list[float]:
        return self.vectors.get(word, [0.0, 0.0])


def test_a_capitalised_common_word_a_month_and_the_vector_of_a_span():
    text = "Płeć: Nieokreślona. Anna pisała w February."
    vocab = _Vocab()
    assert _features(text, "Nieokreślona", vocab=vocab)["a_capitalised_common_word"] == 1
    assert _features(text, "Anna", vocab=vocab)["a_capitalised_common_word"] == 0
    assert _features(text, "Anna", vocab=vocab)["every_word_in_vocabulary"] == 1
    assert _features(text, "February", vocab=vocab)["a_month_or_day"] == 1
    assert span_vector("Nieokreślona", vocab) == [0.0, 1.0]
    assert span_vector("Brzęczyszczykiewicz", vocab) == []


def test_the_context_of_a_span_is_the_word_before_the_word_after_and_its_ending():
    text = "Mieszka przy ul. Fabrycznej 5, a pisze do Kowalskiego."
    start = text.index("Fabrycznej")
    span = Span(start, start + len("Fabrycznej"), "LOCATION", 0.85, "spacy_ner")
    assert span_context(text, span) == ["before:ul", "after:5", "end:nej", "end2:ej"]
    start = text.index("Kowalskiego")
    span = Span(start, start + len("Kowalskiego"), "PERSON", 0.85, "spacy_ner")
    assert span_context(text, span) == ["before:do", "after:$", "end:ego", "end2:go"]
    span = Span(0, len("Mieszka"), "PERSON", 0.85, "spacy_ner")
    assert span_context(text, span)[0] == "before:^"


def test_the_parts_keep_the_benchmarks_and_the_final_part_apart():
    assert [openpii_part(i) for i in (0, 999, 1000, 1999, 2000, 7999, 8000)] == [
        "dev",
        "dev",
        "test",
        "test",
        "train",
        "train",
        "final",
    ]
    assert [kpwr_part(d) for d in range(11)] == [
        "final",
        "train",
        "train",
        "train",
        "train",
        "dev",
        "train",
        "train",
        "train",
        "train",
        "final",
    ]
