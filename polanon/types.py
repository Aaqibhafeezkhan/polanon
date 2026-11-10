"""Shared data types for the anonymization pipeline.

Keep this module stable: every other module depends on it.
"""

from bisect import bisect_right, insort
from dataclasses import dataclass, field
from enum import StrEnum
from operator import attrgetter


class JudgeFallback(StrEnum):
    """What to do when the stage-2 judge fails (timeout, transport error, bad JSON)."""

    SEND_STAGE1 = "send_stage1"  # continue with the stage-1 result, mark judge_failed=True
    BLOCK = "block"  # raise JudgeError to the caller; nothing is sent


class MaskingUnavailableError(RuntimeError):
    """Masking was asked for and cannot run, so nothing may be sent to the model.

    The message names the failure and never holds the text that was being masked; the error that
    caused it is chained for the logs.
    """


class NerModelUnavailableError(MaskingUnavailableError, ModuleNotFoundError):
    """spaCy or its language model is not installed.

    Still a ModuleNotFoundError, which is what ``default_guards`` raised for it before, so code that
    catches that keeps working.
    """


@dataclass(frozen=True, order=True)
class Span:
    """A detected sensitive span in the ORIGINAL text. `end` is exclusive."""

    start: int
    end: int
    entity_type: str  # canonical name, see policy.ENTITY_TYPES
    score: float = 1.0
    source: str = "guard"  # guard | judge | denylist | manual

    @property
    def length(self) -> int:
        """Number of characters covered by the span."""
        return self.end - self.start


@dataclass
class AnonymizationResult:
    """Outcome of one `anonymize()` call."""

    original: str
    anonymized: str  # text with tokens, safe to send
    spans: list[Span]  # all spans that were replaced, in original coordinates
    mapping: dict[str, str]  # token -> original value, e.g. {"<OSOBA_1>": "Jan Kowalski"}
    judge_hits: int = 0  # number of spans contributed by stage 2
    judge_failed: bool = False  # True if the judge errored/timed out and fallback was applied
    entity_counts: dict[str, int] = field(default_factory=dict)


def merge_spans(spans: list[Span], text: str | None = None) -> list[Span]:
    """Resolve overlapping detections without unmasking anything a detector flagged.

    Detections of the same type that overlap are one entity seen by two detectors, so they become
    their union: a guard that read "kamilem bednarkiem" and a model that stopped at "kamilem" found
    one person, and keeping either extent alone would leave part of the name in the clear.

    Where types disagree, confidence picks the label — a checksum-validated identifier scores 1.0,
    a statistical NER hit 0.85, a judge finding 0.5 — and length breaks ties. The detection that
    loses keeps what it covered outside the winner, under its own type, so a loose name span around
    an e-mail address leaves both the address and the name masked. Given the `text`, such leftovers
    are trimmed to their first and last letter or digit, and one with neither is dropped.

    A secret is the exception: it is masked whole, together with anything detected inside it. A
    password built from an e-mail address and "#!" otherwise went out as "<EMAIL_1>#!", which tells
    the model most of the password, and with no SECRET token left to say it was one.

    Args:
        spans: detections in original coordinates, in any order.
        text: the text the spans point into, used only to trim leftovers.

    Returns:
        Non-overlapping spans sorted by position.
    """
    kept: list[Span] = []
    ranked = sorted(
        _join_same_type(_without_parts_of_secrets(spans)),
        key=lambda item: (-item[1].score, -item[1].length, item[1].start, item[0]),
    )
    for _, candidate in ranked:
        for piece in _uncovered_parts(candidate, kept, text):
            insort(kept, piece)
    return kept


#: Types masked whole, with every detection inside them.
_WHOLE_TYPES = frozenset({"SECRET"})


def _without_parts_of_secrets(spans: list[Span]) -> list[Span]:
    """Drop every detection that lies inside a longer secret, which masks it along with the rest.

    A detection with the secret's exact extent is no part of it: "smtp_login: serwis@firma.example"
    is an e-mail address, and confidence picks its label as it does anywhere else.
    """
    secrets = sorted(
        (span for span in spans if span.entity_type in _WHOLE_TYPES), key=attrgetter("start")
    )
    starts: list[int] = []
    ends: list[int] = []
    for secret in secrets:
        if starts and secret.start <= ends[-1]:
            ends[-1] = max(ends[-1], secret.end)
        else:
            starts.append(secret.start)
            ends.append(secret.end)
    if not starts:
        return spans

    def inside_a_secret(span: Span) -> bool:
        index = bisect_right(starts, span.start) - 1
        if index < 0 or span.end > ends[index]:
            return False
        return (span.start, span.end) != (starts[index], ends[index])

    return [span for span in spans if span.entity_type in _WHOLE_TYPES or not inside_a_secret(span)]


def _join_same_type(spans: list[Span]) -> list[tuple[int, Span]]:
    """Replace every group of overlapping same-type spans with their union.

    Each result carries the position of the earliest detection it came from. A complete tie — same
    confidence, same extent — goes to whichever guard reported first, which is how a nine-digit
    number with a valid REGON checksum stays a REGON rather than becoming a phone number.
    """
    joined: list[tuple[int, Span]] = []
    by_type = sorted(enumerate(spans), key=lambda item: (item[1].entity_type, item[1].start))
    for order, span in by_type:
        if joined:
            first, previous = joined[-1]
            if previous.entity_type == span.entity_type and span.start < previous.end:
                best = previous if previous.score >= span.score else span
                union = Span(
                    previous.start,
                    max(previous.end, span.end),
                    span.entity_type,
                    best.score,
                    best.source,
                )
                joined[-1] = (min(first, order), union)
                continue
        joined.append((order, span))
    return joined


def _uncovered_parts(candidate: Span, kept: list[Span], text: str | None) -> list[Span]:
    """Return the pieces of `candidate` that no kept span covers yet.

    `kept` never overlaps itself and is sorted by position, so it is sorted by `end` too and the
    first span that can touch the candidate is found by bisection.
    """
    ranges: list[tuple[int, int]] = []
    cursor = candidate.start
    index = bisect_right(kept, candidate.start, key=attrgetter("end"))
    while index < len(kept) and kept[index].start < candidate.end:
        blocker = kept[index]
        if blocker.start > cursor:
            ranges.append((cursor, blocker.start))
        cursor = max(cursor, blocker.end)
        index += 1
    if cursor < candidate.end:
        ranges.append((cursor, candidate.end))

    pieces: list[Span] = []
    for start, end in ranges:
        whole = (start, end) == (candidate.start, candidate.end)
        if not whole and text is not None:
            start, end = _trim(text, start, end)
        if start < end:
            pieces.append(
                Span(start, end, candidate.entity_type, candidate.score, candidate.source)
            )
    return pieces


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a range to its first and last letter or digit; empty when it has neither."""
    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not text[end - 1].isalnum():
        end -= 1
    return start, end
