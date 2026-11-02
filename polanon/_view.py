"""How the guards read a text, and which characters of the original each one comes from.

The detectors are written for text as a person reads it. Copied text often differs only in what a
person cannot see: a letter spelled as a base letter plus a combining mark, a no-break space between
the groups of a number, an en dash inside it, a zero-width character inside a word. The pipeline,
the allowlist and the output sweep all read through the same view, so a value hidden that way is
recognised, matched and restored like the plain one.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from polanon.types import Span

#: Spaces that Word, PDFs and web pages put between the groups of a number or between words. Polish
#: typography keeps "601 234 567" and "PL61 1090 …" together with them; the guards read a space.
_SPACES = (
    "\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
)
#: Dashes that AutoCorrect and typesetting put inside numbers; the guards read a hyphen.
DASHES = "\u2010\u2011\u2012\u2013\u2212"
#: Invisible characters that PDF extraction, web copy and right-to-left text leave inside words
#: and numbers: soft hyphens, zero-width spaces and joiners, direction marks and controls.
INVISIBLE = frozenset(
    "\u00ad\u034f\u180e\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c"
    "\u202d\u202e\u2060\u2061\u2062\u2063\u2064\u2066\u2067\u2068\u2069\ufeff"
)
_PLAIN = {**dict.fromkeys(_SPACES, " "), **dict.fromkeys(DASHES, "-")}
#: Fullwidth ASCII (U+FF01 to U+FF5E) sits this far above the characters it stands for.
_FULLWIDTH_OFFSET = 0xFEE0
#: A text containing none of these is read by the guards exactly as written, which is nearly always.
_NEEDS_VIEW = re.compile(
    "[\u0300-\u036f\uff01-\uff5e" + _SPACES + DASHES + "".join(sorted(INVISIBLE)) + "]"
)


@dataclass(frozen=True)
class View:
    """The text as the guards read it, and the original range behind each of its characters."""

    text: str
    starts: list[int]
    ends: list[int]

    def to_original(self, span: Span) -> Span:
        """Translate a span over the view into the same characters of the original text."""
        return Span(
            self.starts[span.start],
            self.ends[span.end - 1],
            span.entity_type,
            span.score,
            span.source,
        )


def detection_view(text: str) -> View | None:
    """Return the text as the guards should read it, or None when it is fine as written.

    Text copied from a PDF, from macOS or from a web page often spells a Polish letter as a base
    letter plus a combining mark ("s" + U+0301 for "ś"), keeps the groups of a NIP or an IBAN apart
    with a no-break space, puts an en dash inside a phone number, or leaves a zero-width space in
    the middle of a PESEL. Each of those stopped every guard: a whole IBAN, or the second half of a
    surname, went out in the clear. The guards read a view with letters composed, such spaces and
    dashes made plain and invisible characters dropped. Spans come back in original coordinates, so
    the mapping holds each value as it was first written.
    """
    if _NEEDS_VIEW.search(text) is None:
        return None
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        for char in read_cluster(text[index:end]):
            chars.append(char)
            starts.append(index)
            ends.append(end)
        index = end
    return View("".join(chars), starts, ends)


def read_cluster(cluster: str) -> str:
    """Return how the guards read one character together with the combining marks after it."""
    if len(cluster) > 1:
        return unicodedata.normalize("NFC", cluster)
    if cluster in INVISIBLE:
        return ""
    if "\uff01" <= cluster <= "\uff5e":
        return chr(ord(cluster) - _FULLWIDTH_OFFSET)
    return _PLAIN.get(cluster, cluster)


def as_read(text: str) -> str:
    """Return `text` as the guards read it."""
    view = detection_view(text)
    return text if view is None else view.text
