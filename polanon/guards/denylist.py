"""Denylist guard: company names and other strings that must always be masked.

Statistical NER misses in-house names ("Zakład Produkcyjny Nowak", internal project codenames), so
the caller can list them literally. Matching is case-insensitive and whole-word, and it forgives the
two ways a document drifts from how a term was typed: any whitespace matches the space between two
words (a no-break space, a double space, a line wrapped in a PDF), and Polish letters match their
plain counterparts in both directions ("Projekt Żuraw" finds "projekt zuraw"). Polish inflection is
out of scope; list several forms on separate lines instead. Entries that overlap in the text, such
as "Projekt Atlas" and "Atlas Wiśniewski" in "Projekt Atlas Wiśniewski", are masked as one span.

Two more drifts are forgiven, because a list of clients is typed from a register and the text is
not. A company listed with its legal form after the name is found without it as well: "Siłownia
Testowa Sp. z o.o." finds "Siłownia Testowa" in an e-mail, as long as what is left is two words or
one of five letters or more that is not a word such as "Polska" or "Group", so "Kot Sp. j." does
not mask every "kot" nor "Polska S.A." every "Polska". And a number of six digits
or more, such as a NIP or a phone number, is found however its digits are grouped: "1234567890"
finds "123-456-78-90", "123-45-67-890" and "123 456 78 90", and the other way round.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from polanon._view import as_read
from polanon.guards.base import Guard
from polanon.guards.company_pl import _FOREIGN_FORMS, _SUFFIX_FORMS
from polanon.types import Span

#: A legal form at the end of a listed name, after a space or a comma.
_LEGAL_FORM_AT_END = re.compile(rf"[\s,]+(?:{_SUFFIX_FORMS}|{_FOREIGN_FORMS})\.?\s*$")
#: What must be left of a name without its legal form to be masked on its own.
_MIN_WORDS_LEFT = 2
_MIN_LETTERS_OF_ONE_WORD = 5
#: Words a company's name is often made of that say nothing alone: "Polska S.A." leaves "Polska",
#: which is in every other sentence, so a single word left is masked on its own only when it is
#: none of these.
_GENERIC_WORDS = frozenset(
    {
        "polska",
        "polskie",
        "poland",
        "europe",
        "europa",
        "group",
        "grupa",
        "holding",
        "invest",
        "inwestycje",
        "trade",
        "handel",
        "consulting",
        "international",
        "service",
        "services",
        "serwis",
        "system",
        "systems",
        "software",
        "solutions",
        "studio",
        "design",
        "media",
        "transport",
        "logistics",
        "logistyka",
        "budownictwo",
        "partners",
        "capital",
    }
)
#: A term that is a number: digits with single spaces, hyphens, dots or slashes between them.
_NUMBER = re.compile(r"\d(?:[ .\-/]?\d)+")
#: Digits a number needs before its grouping is forgiven: fewer stand for nothing in particular.
_MIN_NUMBER_DIGITS = 6

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
            return r"[ \u00a0.\-/]?".join(digits)
    return r"\s+".join(re.escape(word) for word in _fold(read).split())


def _variants(term: str) -> list[str]:
    """The term, and the name before its legal form when enough of a name is left."""
    short = _LEGAL_FORM_AT_END.sub("", as_read(term)).strip(" ,")
    if short == term or not short:
        return [term]
    words = short.split()
    letters = sum(char.isalpha() for char in short)
    if len(words) >= _MIN_WORDS_LEFT:
        return [term, short]
    if letters >= _MIN_LETTERS_OF_ONE_WORD and _fold(short).lower() not in _GENERIC_WORDS:
        return [term, short]
    return [term]


class DenylistGuard(Guard):
    """Masks the literals it is given as ``ORG``."""

    name = "denylist"

    def __init__(self, terms: Iterable[str]) -> None:
        """Build the guard from a list of literal terms; empty lines are ignored."""
        cleaned = sorted({t.strip() for t in terms if t.strip()}, key=len, reverse=True)
        self.terms = cleaned
        # Longest first, so that of two terms starting at the same word the longer one is matched.
        patterns = {_term_pattern(variant) for term in cleaned for variant in _variants(term)}
        alternatives = sorted(patterns, key=lambda p: (-len(p), p))
        self._pattern: re.Pattern[str] | None = (
            re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)
            if alternatives
            else None
        )

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return one span per denylisted term found in `text`, overlapping terms joined.

        A search restarts one character after the start of every match rather than after its end,
        so a term that begins inside the previous one is found too and the union is masked.
        """
        if "ORG" not in enabled or self._pattern is None:
            return []
        folded = _fold(text)
        spans: list[Span] = []
        match = self._pattern.search(folded)
        while match is not None:
            if spans and match.start() < spans[-1].end:
                previous = spans[-1]
                end = max(previous.end, match.end())
                spans[-1] = Span(previous.start, end, "ORG", 1.0, self.name)
            else:
                spans.append(Span(match.start(), match.end(), "ORG", 1.0, self.name))
            match = self._pattern.search(folded, match.start() + 1)
        return spans
