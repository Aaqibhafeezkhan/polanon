"""What a span the language model found looks like, as numbers the ML filter can weigh.

The Polish spaCy model tags words that name nobody: English of software, code, a product's tier,
a label before a colon. The normal mode drops them by rules written one at a time. The ML
filter weighs the same signs, and more, with weights fitted on annotated text instead.

``benchmarks/ml_filter_data.py`` writes these features for every span the model finds in the
benchmarks' texts, with whether the span covers an annotated value, ``ml_filter_train.py`` fits a
logistic regression on them and writes its weights to :mod:`polanon.guards._ml_filter_weights`, and
:func:`ml_filter_keeps` reads those when the NER guard's ML filter is on. Every feature is 0
or 1, so a weight reads as how much the sign counts, and :data:`FEATURES` fixes their order.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from polanon._canonical import _surname
from polanon.guards import _ml_filter_weights as fitted
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
_LETTERS = re.compile(r"[^\W\d_]+")

#: Months and days of the week, in Polish and English: written with a capital, they name nobody.
#: "Maj" and "maja" are left out, since Maja is a first name and Maj a surname.
_MONTHS_AND_DAYS_LISTED = """
    january february march april may june july august september october november december
    monday tuesday wednesday thursday friday saturday sunday
    styczeń luty marzec kwiecień czerwiec lipiec sierpień wrzesień październik listopad
    grudzień stycznia lutego marca kwietnia czerwca lipca sierpnia września października
    listopada grudnia poniedziałek wtorek środa czwartek piątek sobota niedziela
    """
_MONTHS_AND_DAYS = frozenset(_MONTHS_AND_DAYS_LISTED.split())


class Vocab(Protocol):
    """The word vectors of a spaCy pipeline, as ``nlp.vocab`` gives them."""

    def has_vector(self, word: str) -> bool:
        """Whether the pipeline has a vector for `word`, as written."""

    def get_vector(self, word: str) -> Sequence[float]:
        """The vector of `word`, as written; zeros when it has none."""


@dataclass(frozen=True)
class Word:
    """What the language model read in one word of a span, without the word itself.

    The NER guard remembers it for thousands of texts, which it otherwise remembers by digest
    only, so no letter of the word is kept: its part of speech, whether the model has a vector
    for it, and whether a word written with a capital has a lemma in lowercase.
    """

    pos: str
    has_vector: bool
    lowercase_lemma: bool

    @classmethod
    def of(cls, token: object) -> Word:
        """What a spaCy token tells of its word."""
        text, lemma = str(getattr(token, "text", "")), str(getattr(token, "lemma_", ""))
        return cls(
            str(getattr(token, "pos_", "")),
            bool(getattr(token, "has_vector", False)),
            text[:1].isupper() and lemma[:1].islower(),
        )


_PREVIOUS_WORD = re.compile(r"(\S+)\s*$")
_NEXT_WORD = re.compile(r"\s*(\S+)")

#: The features in a fixed order, as the ML filter's weights are stored.
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
    # what the pipeline's vocabulary knows of its words
    "a_capitalised_common_word",
    "a_month_or_day",
    "every_word_in_vocabulary",
)


def span_features(
    text: str,
    span: Span,
    *,
    rules_found_it: bool,
    words: Sequence[Word] = (),
    vocab: Vocab | None = None,
    text_english: bool | None = None,
    repeated: bool | None = None,
    line_language: str | None = None,
) -> dict[str, int]:
    """Return the features of a span the model found in `text`, each 0 or 1.

    Args:
        text: the whole text the model read.
        span: one of its spans, as the paranoid mode keeps it.
        rules_found_it: whether a rule of polanon, not the model, found something overlapping it.
        words: what the model read in the span's words. A name keeps its capital in its lemma,
            "Kowalskiego" is "Kowalski", while a word of a form written with a capital does not:
            "Nieokreślona" is "nieokreślony".
        vocab: the pipeline's word vectors. A word it knows only in lowercase, "nieokreślona"
            but not "Nieokreślona", is a common word written with a capital; a name it knows as
            written, or not at all.
        text_english: whether the whole text is plainly English, when the caller already knows;
            read from `text` otherwise. Reading it for every span of a long text made masking
            it take time in the square of its length.
        repeated: whether the span's words stand in the text more than once, when the caller
            already knows; counted in `text` otherwise.
        line_language: the language of the span's line, when the caller already knows; read
            from the line otherwise. A text on one line, such as a tool's JSON, is one line.
    """
    fragment = text[span.start : span.end]
    if text_english is None:
        text_english = ner._is_english_text(text)
    if repeated is None:
        repeated = text.count(fragment) > 1
    letters = _LETTERS.findall(text[span.start : span.end])
    known = vocab.has_vector if vocab is not None else lambda _word: False
    tags = {word.pos for word in words}
    written = _WORD.findall(fragment)
    bare = [ner._bare(word) for word in written]
    folded = [word.lower() for word in bare if word]
    english = sum(word in ner._ENGLISH_WORDS or word in SOFTWARE_WORDS for word in folded)
    if line_language is None:
        line_language = ner._language(text[slice(*line_bounds(text, span.start))])
    language = line_language
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
        "text_english": text_english,
        "opens_a_sentence": ner._opens_a_sentence(text, span.start),
        "after_a_form_of_address": fold(previous.rstrip(".").lower()) in HONORIFICS,
        "after_a_preposition": previous in _PREPOSITIONS,
        "after_a_colon": previous.endswith(":"),
        "repeated_in_the_text": repeated,
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
        "lemma_in_lowercase": any(word.lowercase_lemma for word in words),
        "a_word_without_vector": any(not word.has_vector for word in words),
        "a_capitalised_common_word": any(
            word[:1].isupper() and not known(word) and known(word.lower()) for word in letters
        ),
        "a_month_or_day": any(word.lower() in _MONTHS_AND_DAYS for word in letters),
        "every_word_in_vocabulary": bool(letters) and all(known(word) for word in letters),
    }
    return {name: int(bool(found[name])) for name in FEATURES}


def line_bounds(text: str, index: int) -> tuple[int, int]:
    """Return where the line holding `index` begins and ends in `text`, its break left out."""
    line_end = text.find("\n", index)
    return text.rfind("\n", 0, index) + 1, len(text) if line_end < 0 else line_end


def span_vector(fragment: str, vocab: Vocab) -> list[float]:
    """Return the direction of a span's meaning: the mean vector of its words, of length one.

    A word the vocabulary knows only in lowercase counts with that vector. Names lie in one part
    of the space, common words, months and the words of forms in others, which is what the
    ML filter's weights for these numbers read. A span with no known word gives zeros.
    """
    vectors = []
    for word in _LETTERS.findall(fragment):
        if vocab.has_vector(word):
            vectors.append(vocab.get_vector(word))
        elif vocab.has_vector(word.lower()):
            vectors.append(vocab.get_vector(word.lower()))
    if not vectors:
        return []
    mean = [sum(column) / len(vectors) for column in zip(*vectors, strict=True)]
    length = math.sqrt(sum(value * value for value in mean)) or 1.0
    return [float(value) / length for value in mean]


def span_context(text: str, span: Span) -> list[str]:
    """Return the word before a span, the word after it and the ending of its last word.

    Each comes as the name of a feature, "before:ul", "after:sp", "end:ski", "end2:ka": the
    ML filter has a weight for those that came up often enough in its training texts, and none
    for the rest. A street follows "ul." and a company comes before "sp.", while a form's field
    comes after its label; Polish surnames and places end in "-ski", "-owo" or "-ice".
    """
    before = _PREVIOUS_WORD.search(text, max(0, span.start - 40), span.start)
    after = _NEXT_WORD.match(text, span.end)
    previous = ner._bare(before[1]).lower() if before is not None else ""
    following = ner._bare(after[1]).lower() if after is not None else ""
    letters = _LETTERS.findall(text[span.start : span.end])
    last = letters[-1].lower() if letters else ""
    return [
        f"before:{previous or '^'}",
        f"after:{following or '$'}",
        f"end:{last[-3:]}",
        f"end2:{last[-2:]}",
    ]


def ml_filter_score(
    features: Mapping[str, int],
    entity_type: str,
    vector: Sequence[float] = (),
    context: Sequence[str] = (),
) -> float:
    """Return the ML filter's probability that a span with these features is a name.

    A feature counts with its own weight and with the weight it has for spans of `entity_type`;
    the span's :func:`span_vector` with the weights of its numbers, and its
    :func:`span_context` with the weights of the words and endings the ML filter knows.
    """
    z = fitted.BIAS + sum(
        fitted.WEIGHTS.get(name, 0.0) + fitted.WEIGHTS.get(f"{entity_type}:{name}", 0.0)
        for name, value in features.items()
        if value
    )
    z += sum(fitted.WEIGHTS.get(name, 0.0) for name in context)
    z += sum(weight * value for weight, value in zip(fitted.VECTOR, vector, strict=False))
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def ml_filter_keeps(
    features: Mapping[str, int],
    entity_type: str,
    vector: Sequence[float] = (),
    context: Sequence[str] = (),
) -> bool:
    """Return whether the ML filter keeps a span: its probability reaches its type's threshold."""
    score = ml_filter_score(features, entity_type, vector, context)
    return score >= fitted.THRESHOLDS.get(entity_type, 0.5)
