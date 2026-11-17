"""What a span the language model found looks like, as numbers a learned verifier can weigh.

The Polish spaCy model tags words that name nobody: English of software, code, a product's tier,
a label before a colon. The normal mode drops them by rules written one at a time. A learned
verifier would weigh the same signs, and more, with weights fitted on annotated text instead.

``benchmarks/verifier_data.py`` writes these features for every span the model finds in the
benchmarks' texts, with whether the span covers an annotated value, ``verifier_train.py`` fits a
logistic regression on them and writes its weights to :mod:`polanon.guards._verifier_weights`, and
:func:`verifier_keeps` reads those when the NER guard asks for its verifier. Every feature is 0
or 1, so a weight reads as how much the sign counts, and :data:`FEATURES` fixes their order.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from polanon._canonical import _surname
from polanon.guards import _verifier_weights as fitted
from polanon.guards import ner
from polanon.guards._software_english import SOFTWARE_WORDS
from polanon.guards.names_pl import HONORIFICS, fold, is_common_adjective
from polanon.types import Span

#: Words before a name that ask for one: "z", "do", "dla" and the like, in lowercase.
_PREPOSITIONS = frozenset(
    {"z", "ze", "do", "od", "dla", "u", "o", "w", "we", "na", "przez", "przy", "po", "za"}
    | {"nad", "pod", "przed", "między", "wobec"}
)

#: Parts of speech that name nothing on their own, as the model tags its words.
_FUNCTION_POS = frozenset({"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"})

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Word:
    """What the language model read in one word of a span: its part of speech and lemma."""

    text: str
    pos: str
    lemma: str
    has_vector: bool


_PREVIOUS_WORD = re.compile(r"(\S+)\s*$")

#: The features in a fixed order, as a verifier's weights are stored.
FEATURES: tuple[str, ...] = (
    # what the model called it
    "type_person",
    "type_location",
    "type_org",
    # its shape
    "one_word",
    "two_words",
    "three_words_or_more",
    "every_word_capitalised",
    "a_word_in_lowercase",
    "all_capitals",
    "short",
    "long",
    "a_digit",
    "a_hyphen",
    "a_dot",
    "a_polish_letter",
    # its words
    "a_first_name",
    "a_surname",
    "a_common_adjective",
    "a_software_word",
    "an_english_word",
    "mostly_english_words",
    "a_polish_word",
    # where it stands
    "line_polish",
    "line_english",
    "line_without_language",
    "text_english",
    "opens_a_sentence",
    "after_a_form_of_address",
    "after_a_preposition",
    "after_a_colon",
    "repeated_in_the_text",
    # the hand-written filters of the normal mode
    "rule_code",
    "rule_inline_code",
    "rule_inside_a_word",
    "rule_lowercase_place",
    "rule_language_named_in_english",
    "rule_after_a_model_number",
    "rule_product_or_code",
    "rule_software_english",
    "rule_east_asian",
    "rule_opens_with_an_english_word",
    "rule_named_in_english",
    # the rules of polanon agree
    "rules_found_it",
    # what the model read in its words
    "pos_every_word_proper",
    "pos_a_proper_noun",
    "pos_a_common_noun",
    "pos_an_adjective",
    "pos_a_verb",
    "pos_a_function_word",
    "pos_other",
    "lemma_in_lowercase",
    "a_word_without_vector",
)


def span_features(
    text: str, span: Span, *, rules_found_it: bool, words: Sequence[Word] = ()
) -> dict[str, int]:
    """Return the features of a span the model found in `text`, each 0 or 1.

    Args:
        text: the whole text the model read.
        span: one of its spans, as the paranoid mode keeps it.
        rules_found_it: whether a rule of polanon, not the model, found something overlapping it.
        words: what the model read in the span's words. A name keeps its capital in its lemma,
            "Kowalskiego" is "Kowalski", while a word of a form written with a capital does not:
            "Nieokreślona" is "nieokreślony".
    """
    fragment = text[span.start : span.end]
    tags = {word.pos for word in words}
    capitalised = [word for word in words if word.text[:1].isupper()]
    written = _WORD.findall(fragment)
    bare = [ner._bare(word) for word in written]
    folded = [word.lower() for word in bare if word]
    english = sum(word in ner._ENGLISH_WORDS or word in SOFTWARE_WORDS for word in folded)
    line_start = text.rfind("\n", 0, span.start) + 1
    line_end = text.find("\n", span.start)
    line = text[line_start : None if line_end < 0 else line_end]
    language = ner._language(line)
    before = _PREVIOUS_WORD.search(text, max(0, span.start - 40), span.start)
    previous = before[1] if before is not None else ""
    found = {
        "type_person": span.entity_type == "PERSON",
        "type_location": span.entity_type == "LOCATION",
        "type_org": span.entity_type == "ORG",
        "one_word": len(written) == 1,
        "two_words": len(written) == 2,
        "three_words_or_more": len(written) >= 3,
        "every_word_capitalised": bool(written) and all(ner._capitalised(w) for w in written),
        "a_word_in_lowercase": any(word[:1].islower() for word in bare),
        "all_capitals": fragment.isupper(),
        "short": len(fragment) <= 3,
        "long": len(fragment) >= 25,
        "a_digit": any(char.isdigit() for char in fragment),
        "a_hyphen": "-" in fragment,
        "a_dot": "." in fragment,
        "a_polish_letter": ner._POLISH_LETTERS.search(fragment) is not None,
        "a_first_name": ner._holds_a_first_name(fragment),
        "a_surname": any(word and _surname(word) for word in bare),
        "a_common_adjective": any(is_common_adjective(word) for word in folded),
        "a_software_word": any(word in SOFTWARE_WORDS for word in folded),
        "an_english_word": any(word in ner._ENGLISH_WORDS for word in folded),
        "mostly_english_words": bool(folded) and 2 * english > len(folded),
        "a_polish_word": any(word in ner._POLISH_WORDS for word in folded),
        "line_polish": language == ner._POLISH,
        "line_english": language == ner._ENGLISH,
        "line_without_language": language == ner._UNKNOWN,
        "text_english": ner._is_english_text(text),
        "opens_a_sentence": ner._opens_a_sentence(text, span.start),
        "after_a_form_of_address": fold(previous.rstrip(".").lower()) in HONORIFICS,
        "after_a_preposition": previous in _PREPOSITIONS,
        "after_a_colon": previous.endswith(":"),
        "repeated_in_the_text": text.count(fragment) > 1,
        "rule_code": ner._is_code(fragment),
        "rule_inline_code": ner._in_inline_code(text, span),
        "rule_inside_a_word": ner._inside_a_word(text, span),
        "rule_lowercase_place": ner._lowercase_place(span, fragment),
        "rule_language_named_in_english": ner._language_named_in_english(span, fragment),
        "rule_after_a_model_number": ner._after_a_model_number(text, span, fragment),
        "rule_product_or_code": ner._product_or_code(fragment),
        "rule_software_english": ner._software_english(fragment),
        "rule_east_asian": ner._east_asian(fragment),
        "rule_opens_with_an_english_word": (
            bool(written) and ner._opens_with_an_english_word(fragment)
        ),
        "rule_named_in_english": ner._named_in_english(text, span),
        "rules_found_it": rules_found_it,
        "pos_every_word_proper": bool(tags - {"PUNCT"}) and tags - {"PUNCT"} <= {"PROPN"},
        "pos_a_proper_noun": "PROPN" in tags,
        "pos_a_common_noun": "NOUN" in tags,
        "pos_an_adjective": "ADJ" in tags,
        "pos_a_verb": "VERB" in tags,
        "pos_a_function_word": bool(tags & _FUNCTION_POS),
        "pos_other": bool(tags - {"PROPN", "NOUN", "ADJ", "VERB", "PUNCT"} - _FUNCTION_POS),
        "lemma_in_lowercase": any(word.lemma[:1].islower() for word in capitalised),
        "a_word_without_vector": any(not word.has_vector for word in words),
    }
    return {name: int(bool(found[name])) for name in FEATURES}


def verifier_score(features: Mapping[str, int], entity_type: str) -> float:
    """Return the learned verifier's probability that a span with these features is a name.

    A feature counts with its own weight and with the weight it has for spans of `entity_type`.
    """
    z = fitted.BIAS + sum(
        fitted.WEIGHTS.get(name, 0.0) + fitted.WEIGHTS.get(f"{entity_type}:{name}", 0.0)
        for name, value in features.items()
        if value
    )
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def verifier_keeps(features: Mapping[str, int], entity_type: str) -> bool:
    """Return whether the verifier keeps a span: its probability reaches its type's threshold."""
    return verifier_score(features, entity_type) >= fitted.THRESHOLDS.get(entity_type, 0.5)
