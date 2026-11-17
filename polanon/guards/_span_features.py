"""What a span the language model found looks like, as numbers a learned verifier can weigh.

The Polish spaCy model tags words that name nobody: English of software, code, a product's tier,
a label before a colon. The normal mode drops them by rules written one at a time. A learned
verifier would weigh the same signs, and more, with weights fitted on annotated text instead.

This module describes a span; it decides nothing. ``benchmarks/verifier_data.py`` writes these
features for every span the model finds in the benchmarks' texts, with whether the span covers an
annotated value, and the masking reads none of it yet. Every feature is 0 or 1, so a weight reads
as how much the sign counts, and :data:`FEATURES` fixes their order.
"""

from __future__ import annotations

import re

from polanon._canonical import _surname
from polanon.guards import ner
from polanon.guards._software_english import SOFTWARE_WORDS
from polanon.guards.names_pl import HONORIFICS, fold, is_common_adjective
from polanon.types import Span

#: Words before a name that ask for one: "z", "do", "dla" and the like, in lowercase.
_PREPOSITIONS = frozenset(
    {"z", "ze", "do", "od", "dla", "u", "o", "w", "we", "na", "przez", "przy", "po", "za"}
    | {"nad", "pod", "przed", "między", "wobec"}
)

_WORD = re.compile(r"\S+")
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
)


def span_features(text: str, span: Span, *, rules_found_it: bool) -> dict[str, int]:
    """Return the features of a span the model found in `text`, each 0 or 1.

    Args:
        text: the whole text the model read.
        span: one of its spans, as the paranoid mode keeps it.
        rules_found_it: whether a rule of polanon, not the model, found something overlapping it.
    """
    fragment = text[span.start : span.end]
    words = _WORD.findall(fragment)
    bare = [ner._bare(word) for word in words]
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
        "one_word": len(words) == 1,
        "two_words": len(words) == 2,
        "three_words_or_more": len(words) >= 3,
        "every_word_capitalised": bool(words) and all(ner._capitalised(w) for w in words),
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
            bool(words) and ner._opens_with_an_english_word(fragment)
        ),
        "rule_named_in_english": ner._named_in_english(text, span),
        "rules_found_it": rules_found_it,
    }
    return {name: int(bool(found[name])) for name in FEATURES}
