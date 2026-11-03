"""Denylist guard: company names and other strings that must always be masked.

Statistical NER misses in-house names ("Zakład Produkcyjny Nowak", internal project codenames), so
the caller can list them literally. Matching is case-insensitive and whole-word, and it forgives the
two ways a document drifts from how a term was typed: any whitespace matches the space between two
words (a no-break space, a double space, a line wrapped in a PDF), and Polish letters match their
plain counterparts in both directions ("Projekt Żuraw" finds "projekt zuraw"). Polish inflection is
out of scope; list several forms on separate lines instead. Entries that overlap in the text, such
as "Projekt Atlas" and "Atlas Wiśniewski" in "Projekt Atlas Wiśniewski", are masked as one span.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from polanon._view import as_read
from polanon.guards.base import Guard
from polanon.types import Span

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
    return r"\s+".join(re.escape(word) for word in _fold(as_read(term)).split())


class DenylistGuard(Guard):
    """Masks the literals it is given as ``ORG``."""

    name = "denylist"

    def __init__(self, terms: Iterable[str]) -> None:
        """Build the guard from a list of literal terms; empty lines are ignored."""
        cleaned = sorted({t.strip() for t in terms if t.strip()}, key=len, reverse=True)
        self.terms = cleaned
        # Longest first, so that of two terms starting at the same word the longer one is matched.
        alternatives = sorted({_term_pattern(t) for t in cleaned}, key=lambda p: (-len(p), p))
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
