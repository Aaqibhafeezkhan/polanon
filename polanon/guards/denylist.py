"""Denylist guard: company names and other strings that must always be masked.

Statistical NER misses in-house names ("Zakład Produkcyjny Nowak", internal project codenames), so
the caller can list them literally. Matching is case-insensitive and whole-word, and it forgives the
two ways a document drifts from how a term was typed: any whitespace matches the space between two
words (a no-break space, a double space, a line wrapped in a PDF), and Polish letters match their
plain counterparts in both directions ("Projekt Żuraw" finds "projekt zuraw"). Polish inflection is
out of scope; list several forms on separate lines instead. Entries that overlap in the text, such
as "Projekt Atlas" and "Atlas Wiśniewski" in "Projekt Atlas Wiśniewski", are masked as one span.

Two more drifts are forgiven, because a list of clients is typed from a register and the text is
not. A company listed with its legal form after the name is found without it too, written as on
the list or in capitals: "Siłownia Testowa Sp. z o.o." finds "Siłownia Testowa" in an e-mail. Only
a name of two words or more is shortened, and never one that is a person's name, so "Budex sp. z
o.o." and "Jan Kowalski Sp. j." are found only whole: one word alone, such as "Partner" or
"Apteka", is in too many sentences that do not name the company, and a person is a person. The
shortened name keeps its letter case, so "moje biuro rachunkowe" is not "Biuro Rachunkowe Sp. z
o.o.". And a number of nine digits or more, such as a NIP, a PESEL or a phone number, is found
however its digits are grouped: "1234567890" finds "123-456-78-90", "123-45-67-890" and "123 456 78
90", and the other way round. A shorter number is matched as it was written, since six or eight
digits grouped otherwise are a date or an amount.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from functools import lru_cache

from polanon._view import as_read
from polanon.guards.base import Guard
from polanon.guards.company_pl import _FOREIGN_FORMS, _SUFFIX_FORMS
from polanon.types import Span

#: A legal form at the end of a listed name, after a space or a comma.
_LEGAL_FORM_AT_END = re.compile(rf"[\s,]+(?:{_SUFFIX_FORMS}|{_FOREIGN_FORMS})\.?\s*$")
#: Words a name needs left without its legal form to be masked on its own.
_MIN_WORDS_LEFT = 2
#: A term that is a number: digits with whitespace, hyphens, dots or slashes between them.
_NUMBER = re.compile(r"\d(?:[\s.\-/]*\d)+")
#: Digits a number needs before its grouping is forgiven.
_MIN_NUMBER_DIGITS = 9
#: What may stand between two digits of such a number in the text: any whitespace, as between the
#: words of a term, or one hyphen, dot or slash with whitespace around it or none.
_DIGIT_GAP = r"(?:\s*[.\-/]\s*|\s+)?"

#: Polish letters to their plain counterparts, one character for one, so that an offset in the
#: folded text is the same offset in the original.
_FOLD = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def _fold(text: str) -> str:
    """Drop Polish diacritics without changing the length of the text."""
    return text.translate(_FOLD)


def _term_pattern(term: str) -> str:
    """Return the pattern of one term: its words, folded and escaped, apart by any whitespace.

    The term is read the way the guards read the text, dashes unified, combining marks composed
    and soft hyphens dropped: a term copied from Word, a PDF or macOS matched nothing, not even
    the same text.
    """
    read = as_read(term)
    if _NUMBER.fullmatch(read.strip()):
        digits = [char for char in read if char.isdigit()]
        if len(digits) >= _MIN_NUMBER_DIGITS:
            return _DIGIT_GAP.join(digits)
    return r"\s+".join(re.escape(word) for word in _fold(read).split())


def _shortened(term: str) -> str | None:
    """The name before a listed company's legal form, or None when it is not to be masked alone."""
    read = as_read(term).strip()
    short = _LEGAL_FORM_AT_END.sub("", read).strip(" ,")
    if not short or short == read or len(short.split()) < _MIN_WORDS_LEFT:
        return None
    return None if _is_a_person(short) else short


def _is_a_person(name: str) -> bool:
    """Whether the name guard reads the whole of `name` as a person's name."""
    spans = _name_guard().detect(name, {"PERSON"})
    return any(span.start == 0 and span.end >= len(name.rstrip(" .")) for span in spans)


@lru_cache(maxsize=1)
def _name_guard() -> Guard:
    from polanon.guards.names_pl import PersonNameGuard  # the name lists load on first use

    return PersonNameGuard()


def _alternation(patterns: set[str], flags: int) -> re.Pattern[str] | None:
    """One pattern for all of `patterns`, whole words only, longest first."""
    if not patterns:
        return None
    # Longest first, so that of two terms starting at the same word the longer one is matched.
    ordered = sorted(patterns, key=lambda pattern: (-len(pattern), pattern))
    return re.compile(r"(?<!\w)(?:" + "|".join(ordered) + r")(?!\w)", flags)


def _matches(pattern: re.Pattern[str], folded: str) -> Iterator[tuple[int, int]]:
    """Every match, restarting one character after the start of each rather than after its end.

    A term that begins inside the previous one is found too, so the union is masked.
    """
    match = pattern.search(folded)
    while match is not None:
        yield match.span()
        match = pattern.search(folded, match.start() + 1)


class DenylistGuard(Guard):
    """Masks the literals it is given as ``ORG``."""

    name = "denylist"

    def __init__(self, terms: Iterable[str]) -> None:
        """Build the guard from a list of literal terms; empty lines are ignored."""
        cleaned = sorted({t.strip() for t in terms if t.strip()}, key=len, reverse=True)
        self.terms = cleaned
        shortened = [short for short in map(_shortened, cleaned) if short is not None]
        patterns = (
            _alternation({_term_pattern(term) for term in cleaned}, re.IGNORECASE),
            # A shortened name keeps its letter case, as written on the list or in capitals.
            _alternation({_term_pattern(form) for s in shortened for form in (s, s.upper())}, 0),
        )
        self._patterns = [pattern for pattern in patterns if pattern is not None]

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return one span per denylisted term found in `text`, overlapping terms joined."""
        if "ORG" not in enabled or not self._patterns:
            return []
        folded = _fold(text)
        found = sorted(span for pattern in self._patterns for span in _matches(pattern, folded))
        spans: list[Span] = []
        for start, end in found:
            if spans and start < spans[-1].end:
                previous = spans[-1]
                spans[-1] = Span(previous.start, max(previous.end, end), "ORG", 1.0, self.name)
            else:
                spans.append(Span(start, end, "ORG", 1.0, self.name))
        return spans
