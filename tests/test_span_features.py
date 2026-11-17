"""The features a learned verifier would weigh, and the labels its training data gets.

Stdlib only: the spans are made up, as the language model would return them.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from run import GoldEntity  # noqa: E402
from verifier_data import label_of  # noqa: E402

from polanon.guards._span_features import FEATURES, span_features  # noqa: E402
from polanon.types import Span  # noqa: E402


def _features(text: str, fragment: str, entity_type: str = "PERSON", **options: bool) -> dict:
    start = text.index(fragment)
    span = Span(start, start + len(fragment), entity_type, 0.85, "spacy_ner")
    return span_features(text, span, rules_found_it=options.get("rules_found_it", False))


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
