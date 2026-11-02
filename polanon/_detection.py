"""Running the guards over a text, and the checks a guard's output has to pass first.

Shared by the pipeline, the output sweep and the facade, which refuses a guard whose declared types
cannot label a token before it is ever run.
"""

from __future__ import annotations

import re

from polanon._view import View
from polanon.guards.base import Guard
from polanon.policy import ENTITY_TYPES
from polanon.types import Span

#: What a custom entity type may be called. It becomes the label inside its tokens, and
#: TOKEN_PATTERN only recognises uppercase letters and underscores there: a type spelled any other
#: way produced tokens that restore could not find, and the original stayed in the answer.
_CUSTOM_TYPE_RE = re.compile(r"[A-Z][A-Z_]*")


#: Token labels of the built-in types. A custom type spelled like one would share its tokens:
#: a guard's "OSOBA" made "<OSOBA_2>" a second person, and every count filed it as one.
_BUILT_IN_LABELS = frozenset(ENTITY_TYPES.values()) - frozenset(ENTITY_TYPES)


def detect(text: str, view: View | None, enabled: set[str], guards: list[Guard]) -> list[Span]:
    """Run every guard over the text and return their spans, unmerged, in original coordinates.

    `view` is the text's :func:`~polanon._view.detection_view`, computed once by the caller.
    """
    seen = text if view is None else view.text
    detected: list[Span] = []
    for guard in guards:
        for span in guard.detect(seen, enabled):
            if _usable(span, guard, len(seen)):
                detected.append(span if view is None else view.to_original(span))
    return detected


def _usable(span: Span, guard: Guard, length: int) -> bool:
    """Check a span before the pipeline trusts it; an empty one is skipped.

    A guard with an off-by-one would otherwise corrupt the text without a sound: Python slices take
    a negative start from the end, and the document went out with half of it repeated.

    Raises:
        ValueError: when the span lies outside the text, or its type cannot be a token label.
    """
    if not 0 <= span.start <= span.end <= length:
        raise ValueError(
            f"guard {guard.name!r} returned the span {span.start}..{span.end}, "
            f"outside a text of {length} characters"
        )
    problem = custom_type_problem(span.entity_type)
    if problem is not None:
        raise ValueError(f"guard {guard.name!r} returned the entity type {problem}")
    return span.start < span.end


def custom_type_problem(entity_type: str) -> str | None:
    """Return why `entity_type` cannot name a custom type, or None when it can.

    The message completes "returned the entity type" or "declares the entity type", so a guard's
    author reads what to rename and why.
    """
    if entity_type in ENTITY_TYPES:
        return None
    if not _CUSTOM_TYPE_RE.fullmatch(entity_type):
        return (
            f"{entity_type!r}. A custom type becomes the label of its tokens, so it must be "
            "uppercase letters and underscores, such as 'EMPLOYEE_ID'."
        )
    if entity_type in _BUILT_IN_LABELS:
        return (
            f"{entity_type!r}, which is the token label of a built-in type, so its tokens would "
            "read as that type. Name it differently, such as 'EMPLOYEE_ID'."
        )
    return None
