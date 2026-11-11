"""Anonymization pipeline: stage 1 guards -> stage 2 judge -> stable tokens + mapping.

Stage 1 runs deterministic, local detectors and replaces every detected span with a stable token.
Stage 2 asks an LLM judge to review the *already masked* text and flag anything that still
identifies a person or an organisation (context, job titles, rare surnames the guards missed).

Token stability is the contract that makes multi-turn conversations work: within one conversation
the same original value always maps to the same token, a token is never reused for another value,
and every occurrence of a known value is masked — not only the one a guard happened to recognise.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass

from polanon._detection import detect
from polanon._known_parts import known_parts
from polanon._matching import (
    TOKEN_RE,
    Haystack,
    Ranges,
    folds,
    known_value_matches,
    place_known_values,
    type_of,
    value_keys,
)
from polanon._view import detection_view
from polanon.allowlist import Allowlist
from polanon.guards.base import Guard
from polanon.judge.base import Judge, JudgeError, JudgeFinding
from polanon.judge.parsing import MIN_FINDING_CHARS
from polanon.policy import ENTITY_TYPES, token_for
from polanon.types import AnonymizationResult, JudgeFallback, Span, merge_spans


class _TokenAllocator:
    """Assigns stable tokens, reusing the ones already known for a conversation.

    A value keeps one token whatever type a guard gives it next time: "WA 12345" read as a
    postcode in one sentence and as a registration in the next would otherwise reach the model as
    two different things. It also keeps one token however it is spelled: "Jan Kowalski", "JAN
    KOWALSKI" and the same name with a no-break space are one person, and the mapping keeps the
    spelling seen first.

    Token-shaped text the caller wrote is never handed out again, so a template's literal
    "<PESEL_1>" cannot collide with a real PESEL and turn into it on restore. Such a token joins the
    mapping as standing for itself, which carries the reservation into the next turn with the
    mapping the caller already passes along.
    """

    def __init__(self, existing_mapping: dict[str, str], reserved: Iterable[str] = ()) -> None:
        self._mapping: dict[str, str] = dict(existing_mapping)
        self._by_value: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}
        for token, value in existing_mapping.items():
            entity_type = type_of(token)
            if entity_type is None:
                continue
            self._reserve(token)
            if value != token:
                for key in value_keys(value, entity_type):
                    self._by_value.setdefault(key, token)
        for token in reserved:
            self._reserve(token)
            self._mapping.setdefault(token, token)

    def _reserve(self, token: str) -> None:
        """Keep new tokens from reusing the number of `token`."""
        match = TOKEN_RE.fullmatch(token)
        if match is not None:
            label, number = match.group("label"), int(match.group("n"))
            self._counters[label] = max(self._counters.get(label, 0), number)

    def allocate(self, value: str, entity_type: str) -> str:
        """Return the token for `value`, creating one if this value has not been seen yet.

        Any key the value shares with a value seen before means the same value: an identifier
        written once with its groups apart and once together keeps one token.
        """
        keys = value_keys(value, entity_type)
        for key in keys:
            known = self._by_value.get(key)
            if known is not None:
                return known
        label = ENTITY_TYPES.get(entity_type, entity_type)
        number = self._counters.get(label, 0) + 1
        self._counters[label] = number
        token = token_for(entity_type, number)
        for key in keys:
            self._by_value.setdefault(key, token)
        self._mapping[token] = value
        return token

    @property
    def mapping(self) -> dict[str, str]:
        """Full token -> original mapping, including entries inherited from the conversation."""
        return self._mapping


def mask_known_values(text: str, mapping: dict[str, str]) -> str:
    """Replace every occurrence of a value `mapping` already holds with that value's token.

    Meant for text that was masked earlier and is sent again, such as the history of a
    conversation. A value that became sensitive after that text went out — a word somebody asked to
    mask, a denylist entry added since, masking switched back on — is still in it verbatim, and
    sending it again would repeat the leak. The matching rules are the ones :func:`anonymize`
    applies to known values, but no token is allocated: an occurrence inside a longer word takes the
    whole word under the value's own token, because replayed text is never restored. Tokens
    already in the text are left alone, except where token-shaped text is no token: a placeholder
    somebody typed, which the mapping records as standing for itself, or one that is itself a value
    the mapping hides, such as a password written as "<ADMIN_2024>".

    Args:
        text: previously masked text.
        mapping: token -> original value, normally the conversation's whole mapping.

    Returns:
        The text with every known value replaced by its token.
    """
    values = [
        (value, token)
        for token, value in mapping.items()
        if value != token and TOKEN_RE.fullmatch(token)
    ]
    haystack = Haystack.of(text, detection_view(text))
    hidden = {value.strip() for value, _ in values}
    occupied = [
        (match.start(), match.end())
        for match in TOKEN_RE.finditer(text)
        if mapping.get(match[0]) != match[0] and match[0] not in hidden
    ]
    parts: list[str] = []
    cursor = 0
    for start, end, token in sorted(known_value_matches(text, haystack, values, occupied)):
        parts.append(text[cursor:start])
        parts.append(token)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _token_ranges(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _whole_ranges(merged: list[Span], detected: list[Span]) -> set[tuple[int, int]]:
    """Return the ranges of merged spans that are a whole detection, not a piece cut out of one.

    merge_spans keeps what a losing detection covered outside the winner, so nothing a guard flagged
    goes out. Such a piece can be a stray word, though: an address rule that ran from an e-mail
    domain into the next line left the bare word "NIP" behind. Masking it in place costs little;
    treating it as a known value masked every "NIP" in the document, so only whole detections are.
    """
    starts = {(span.entity_type, span.start) for span in detected}
    ends = {(span.entity_type, span.end) for span in detected}
    return {
        (span.start, span.end)
        for span in merged
        if (span.entity_type, span.start) in starts and (span.entity_type, span.end) in ends
    }


def _one_type_per_value(spans: list[Span], text: str) -> list[Span]:
    """Give every occurrence of a value the type of its most confident detection.

    Context decides some types: "WA 12345" after "nr rej." is a registration, and alone it reads as
    a postcode. One value is one entity, so all its occurrences take the label the guards were
    surest of; the first occurrence in reading order wins a tie. Spellings of the value that
    :func:`~polanon._matching.value_keys` reads as one count as one value.
    """
    best: dict[tuple[str, str], Span] = {}
    keys = [value_keys(text[span.start : span.end], span.entity_type)[-1] for span in spans]
    for span, key in zip(spans, keys, strict=True):
        if key not in best or span.score > best[key].score:
            best[key] = span
    unified: list[Span] = []
    for span, key in zip(spans, keys, strict=True):
        entity_type = best[key].entity_type
        if entity_type != span.entity_type:
            span = Span(span.start, span.end, entity_type, span.score, span.source)
        unified.append(span)
    return unified


@dataclass
class _Masking:
    """One anonymization in progress: the outcome of stage 1, open for the judge's findings.

    `plain_masked` and `plain_original` hold where each run of untouched text starts in the masked
    and in the original text, so a range of the masked text can be traced back to the original.
    """

    original: str
    masked: str
    spans: list[Span]
    allocator: _TokenAllocator
    entity_counts: dict[str, int]
    plain_masked: list[int]
    plain_original: list[int]
    judge_hits: int = 0
    judge_failed: bool = False

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """Translate a range of untouched text in `masked` to the same text in `original`."""
        index = bisect_right(self.plain_masked, start) - 1
        offset = self.plain_original[index] - self.plain_masked[index]
        return start + offset, end + offset

    def result(self) -> AnonymizationResult:
        """Freeze the outcome into the public result type."""
        return AnonymizationResult(
            original=self.original,
            anonymized=self.masked,
            spans=merge_spans(self.spans, self.original),
            mapping=self.allocator.mapping,
            judge_hits=self.judge_hits,
            judge_failed=self.judge_failed,
            entity_counts=self.entity_counts,
        )


def _substitute(text: str, spans: list[Span], allocator: _TokenAllocator) -> _Masking:
    """Rebuild the text once, in reading order, substituting each span for its token.

    The spans no longer overlap by this point, so one forward pass is enough: it numbers tokens in
    reading order — the first person in the text is ``<OSOBA_1>`` — and it copies the document once
    instead of once per span, which matters when a long upload carries thousands.
    """
    parts: list[str] = []
    counts: dict[str, int] = {}
    plain_masked: list[int] = []
    plain_original: list[int] = []
    cursor = 0
    written = 0
    for span in sorted(spans):
        plain_masked.append(written)
        plain_original.append(cursor)
        token = allocator.allocate(text[span.start : span.end], span.entity_type)
        parts.append(text[cursor : span.start])
        parts.append(token)
        written += span.start - cursor + len(token)
        cursor = span.end
        _count(counts, token)
    plain_masked.append(written)
    plain_original.append(cursor)
    parts.append(text[cursor:])
    return _Masking(
        text, "".join(parts), sorted(spans), allocator, counts, plain_masked, plain_original
    )


def _count(counts: dict[str, int], token: str) -> None:
    """Count one replacement under the type of the token it received.

    A value keeps its token when a guard reads it as another type, so the count follows the token:
    "WA 12345" that went out as a registration is not counted as a postcode.
    """
    entity_type = type_of(token) or token
    counts[entity_type] = counts.get(entity_type, 0) + 1


#: Detections at or above this confidence come from a rule, a list or the caller, never from a
#: language model (0.85) or the judge (0.5), so the parts of their values are trusted as well.
_TRUSTED_SCORE = 0.9


def _remember(
    known: dict[tuple[str, bool], tuple[str, str]], value: str, entity_type: str, *, trusted: bool
) -> None:
    """Add a value to the known values, with the parts of it that identify on their own."""
    known.setdefault((value.strip(), folds(entity_type)), (value, entity_type))
    for part in known_parts(value, entity_type, trusted=trusted):
        known.setdefault((part, folds(entity_type)), (part, entity_type))


def _mask_stage1(
    text: str,
    detected: list[Span],
    existing_mapping: dict[str, str],
    allowlist: Allowlist | None,
    haystack: Haystack,
) -> _Masking:
    """Resolve the guards' detections, hand out stable tokens and substitute them.

    Shared by :func:`anonymize` and the blocking :meth:`polanon.Anonymizer.anonymize_sync`, so
    both mask a text the same way. Tokens are allocated only once every occurrence of every known
    value has joined the spans, which is what keeps the numbering in reading order.

    An occurrence of a known value takes in what a guard found inside it: a signature that puts
    "Jan" and "Kowalski" on two lines is read as "Jan" by the name rule, and masking that alone
    sent the surname. Such detections give way to the occurrence, which is masked whole. One that
    only partly overlaps a detection, or another known value, is resolved with it as two
    detections are, so no letter of either reaches the model.

    The parts of a value that point to the same thing on their own are known values too: the
    surname of a person, the handle in a link to a profile. See :mod:`polanon._known_parts`.
    """
    if allowlist is not None:
        detected = allowlist.drop_allowed(detected, text)
    merged = merge_spans(detected, text)
    whole = _whole_ranges(merged, detected)
    merged = _one_type_per_value(merged, text)

    known: dict[tuple[str, bool], tuple[str, str]] = {}
    for token, value in existing_mapping.items():
        entity_type = type_of(token)
        if entity_type is not None and value != token:
            _remember(known, value, entity_type, trusted=False)
    for span in merged:
        if (span.start, span.end) in whole:
            value = text[span.start : span.end]
            _remember(known, value, span.entity_type, trusted=span.score >= _TRUSTED_SCORE)
    occupied = [(span.start, span.end) for span in merged]
    placement = place_known_values(
        text, haystack, known.values(), occupied, absorb=True, cross_occupied=True
    )
    covered = Ranges((start, end) for start, end, _ in placement.matches)
    kept = [span for span in merged if not covered.overlaps(span.start, span.end)]
    occurrences = placement.matches + placement.crossing
    spans = merge_spans(
        kept + [Span(start, end, kind, 0.9, "known_value") for start, end, kind in occurrences],
        text,
    )

    reserved = (match.group(0) for match in TOKEN_RE.finditer(text))
    return _substitute(text, spans, _TokenAllocator(existing_mapping, reserved))


def _run_stage1(
    text: str,
    enabled: set[str],
    guards: list[Guard],
    existing_mapping: dict[str, str],
    allowlist: Allowlist | None,
) -> _Masking:
    """Detect, merge, find every known value and substitute: all of stage 1, in one thread."""
    view = detection_view(text)
    detected = detect(text, view, enabled, guards)
    return _mask_stage1(text, detected, existing_mapping, allowlist, Haystack.of(text, view))


def _apply_findings(
    masking: _Masking,
    findings: list[JudgeFinding],
    enabled: set[str],
    allowlist: Allowlist | None,
) -> None:
    """Mask what the judge flagged on top of stage 1, by the rules stage 1 uses for known values.

    A finding is located in the masked text the way a known value is: a short one is never matched
    inside a longer word ("Kos" leaves "Kosztorys" alone), a longer one takes the whole word, and
    nothing inside an existing token is touched. All findings are applied in one pass, a token is
    allocated only for what is actually replaced, and every replacement is recorded in original
    coordinates, so `spans` describes exactly what the anonymized text hides.

    Small models often quote overlapping pieces of one name. Findings that overlap are resolved as
    two detections are, by :func:`~polanon.types.merge_spans`: masking only the longer one sent the
    first name of "Ngozi Okonkwo-Brzęczyszczykiewicz z kancelarii ..." in the clear.
    """
    wanted = [
        (finding.text.strip(), finding.entity_type)
        for finding in findings
        if finding.entity_type in enabled
        and len(finding.text.strip()) >= MIN_FINDING_CHARS
        and TOKEN_RE.search(finding.text) is None
        and (allowlist is None or not allowlist.exempts(finding.entity_type, finding.text))
    ]
    masked = masking.masked
    haystack = Haystack.of(masked, detection_view(masked))
    placement = place_known_values(masked, haystack, wanted, _token_ranges(masked))
    found = [
        Span(start, end, entity_type, 0.5, "judge")
        for start, end, entity_type in placement.matches + placement.crossing
    ]
    replaced = merge_spans(found, masked)
    parts: list[str] = []
    cursor = 0
    for span in replaced:
        token = masking.allocator.allocate(masked[span.start : span.end], span.entity_type)
        parts.append(masked[cursor : span.start])
        parts.append(token)
        cursor = span.end
        _count(masking.entity_counts, token)
        original_start, original_end = masking.to_original(span.start, span.end)
        masking.spans.append(Span(original_start, original_end, span.entity_type, 0.5, "judge"))
    parts.append(masked[cursor:])
    masking.masked = "".join(parts)
    masking.judge_hits += len(replaced)


async def anonymize(
    text: str,
    *,
    enabled: set[str],
    guards: list[Guard],
    judge: Judge | None,
    existing_mapping: dict[str, str],
    judge_fallback: JudgeFallback = JudgeFallback.SEND_STAGE1,
    allowlist: Allowlist | None = None,
) -> AnonymizationResult:
    """Mask sensitive data in `text` and return the masked text plus the token mapping.

    Args:
        text: the original text, as the caller received it.
        enabled: entity types to mask, see `polanon.policy.ENTITY_TYPES`.
        guards: stage-1 detectors, run in order; their spans are merged and de-overlapped.
        judge: optional stage-2 reviewer, or None to skip stage 2.
        existing_mapping: token -> value already allocated in this conversation.
        judge_fallback: what to do when the judge fails.
        allowlist: public names left unmasked when a guard or the judge flags them as a name.

    Returns:
        An `AnonymizationResult` whose `mapping` contains both inherited and newly allocated tokens.

    Raises:
        JudgeError: when the judge fails and `judge_fallback` is `BLOCK`.
    """
    # Stage 1 is synchronous CPU work: regex over the whole document, a language model when the
    # NER guard is on, and a search for every known value. Running it inline would stop the
    # caller's event loop for as long as it takes, which on a shared server means every other
    # request waits behind one long document.
    masking = await asyncio.to_thread(
        _run_stage1, text, enabled, guards, existing_mapping, allowlist
    )
    if judge is None:
        return masking.result()
    try:
        findings = await judge.review(masking.masked)
    except Exception as exc:
        # The fallback is a promise about what a failing judge does, so it has to hold for a judge
        # that breaks its contract and raises something other than JudgeError, too.
        if JudgeFallback(judge_fallback) is JudgeFallback.BLOCK:
            if isinstance(exc, JudgeError):
                raise
            raise JudgeError(f"judge {judge.name!r} failed: {type(exc).__name__}") from None
        masking.judge_failed = True
    else:
        _apply_findings(masking, findings, enabled, allowlist)
    return masking.result()


def _anonymize_stage1(
    text: str,
    *,
    enabled: set[str],
    guards: list[Guard],
    existing_mapping: dict[str, str],
    allowlist: Allowlist | None = None,
) -> AnonymizationResult:
    """Run stage 1 in the calling thread with no event loop, for ``Anonymizer.anonymize_sync``.

    It never touches asyncio, so it works whether or not a loop is running in this thread, which is
    the case in a notebook or an async web handler that calls synchronous code.
    """
    return _run_stage1(text, enabled, guards, existing_mapping, allowlist).result()
