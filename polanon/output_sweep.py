"""Output sweep: look for surprises in the model's answer after tokens were restored.

Two things are worth flagging. The model may invent a token that was never allocated, which means it
hallucinated a reference. It may also emit brand new personal data that nobody sent it. Neither
blocks the answer; both are logged and shown as a hint.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from polanon._detection import detect
from polanon._matching import Haystack, fold, known_value_ranges, type_of
from polanon._view import detection_view
from polanon.allowlist import _ATTACHED_ENDINGS, Allowlist
from polanon.guards.base import Guard
from polanon.policy import TOKEN_PATTERN
from polanon.types import Span, merge_spans

_TOKEN_RE = re.compile(TOKEN_PATTERN)

#: Endings a restored token may take on without the answer inventing new data: the endings written
#: straight after a name, plus the suffixed forms a longer name declines through ("Kowalskiego").
#: Anything glued on that does not spell one of these leaves the detection reported, so a surname
#: that merely shares a stem with a known one is still flagged.
_SWEEP_ENDINGS = (*_ATTACHED_ENDINGS, "ego", "emu")

#: The kinds of value Polish declines by gluing a case ending onto its right: names only. A letter
#: glued onto any other kind of known value changes the value — "ul. Długa 5a" is another building,
#: "WPI 1234E" another plate — so only these get the extension.
_DECLINED_TYPES = frozenset({"PERSON", "ORG", "LOCATION"})


def _glued_end(text: str, end: int) -> int:
    """Return `end` moved through one case ending glued right onto a known name, else `end`.

    Only the endings in :data:`_SWEEP_ENDINGS` count, matched without case: "Jan Kowalskia" extends
    through "a", "Jan Kowalskiego" through "ego". A longer tail only takes the ending it starts
    with, so "Kowalczyk" keeps its "czyk" and digits never match an ending at all.
    """
    for ending in sorted(_SWEEP_ENDINGS, key=len, reverse=True):
        candidate = text[end : end + len(ending)]
        if candidate and fold(candidate) == ending:
            return end + len(ending)
    return end


@dataclass
class SweepReport:
    """Findings from one sweep of a restored answer."""

    unknown_tokens: list[str] = field(default_factory=list)
    new_spans: list[Span] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True when the answer contained neither invented tokens nor new personal data."""
        return not self.unknown_tokens and not self.new_spans


def sweep(
    restored_text: str,
    mapping: dict[str, str],
    guards: Iterable[Guard],
    enabled: set[str],
    allowlist: Allowlist | None = None,
) -> SweepReport:
    """Scan a restored answer for invented tokens and personal data that was not in the mapping.

    Names on the `allowlist` are not reported: an answer about a campaign on Facebook is expected
    to say "Facebook".

    A detection counts as new only when some letter or digit of it lies outside every known value
    and every token in the answer. Comparing whole strings reported the conversation's own data as
    new whenever a guard read it with a trailing full stop, read part of it, or read two known
    values as one name.

    A known value counts as known exactly as written; only a name — a person, an organisation or a
    place — also counts with the letters Polish glues onto its right when they spell a case ending:
    a restored answer says "Jan Kowalskiego" where the mapping holds "Jan Kowalski", and the ending
    the token took on is not personal data the model invented. A letter glued onto any other kind of
    known value changes it ("ul. Długa 5a" is another building), so those stay reported. The
    extension never crosses digits and never reaches left, so a known number inside a longer one
    and a name that starts inside another ("Joanna" beside "Anna") are still new.

    The answer is read the way :func:`~polanon.anonymize` reads a text: a PESEL with a zero-width
    space in it is still a PESEL, and a known name spelled in capitals or with a no-break space is
    still known.
    """
    tokens = list(_TOKEN_RE.finditer(restored_text))
    unknown = sorted({m.group(0) for m in tokens if m.group(0) not in mapping})
    view = detection_view(restored_text)
    known = [(value, token) for token, value in mapping.items() if value != token]
    covered = bytearray(len(restored_text))
    haystack = Haystack.of(restored_text, view)
    ranges = [match.span() for match in tokens]
    ranges += known_value_ranges(restored_text, haystack, known)
    names = [(value, token) for value, token in known if type_of(token) in _DECLINED_TYPES]
    ranges += [
        (end, _glued_end(restored_text, end))
        for _, end in known_value_ranges(restored_text, haystack, names)
    ]
    for start, end in ranges:
        covered[start:end] = b"\x01" * (end - start)
    detected = detect(restored_text, view, enabled, list(guards))
    if allowlist is not None:
        detected = allowlist.drop_allowed(detected, restored_text)
    new_spans = [
        span
        for span in merge_spans(detected, restored_text)
        if any(
            restored_text[index].isalnum() and not covered[index]
            for index in range(span.start, span.end)
        )
    ]
    return SweepReport(unknown_tokens=unknown, new_spans=new_spans)
