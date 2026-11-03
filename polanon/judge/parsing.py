"""Shared, strict parsing of judge responses.

A judge is a small model, so its output is treated as untrusted: anything that is not a verbatim
quote from the reviewed text is dropped rather than guessed at.
"""

from __future__ import annotations

import json
import re
from typing import Any

from polanon._matching import reading
from polanon.judge.base import JudgeError, JudgeFinding
from polanon.judge.prompt import ALLOWED_TYPES
from polanon.policy import TOKEN_PATTERN

#: Shortest finding worth acting on. A finding is masked wherever it stands as a word, so a one- or
#: two-letter answer from a small model ("i", "w") would mask every such word in the message, and
#: nothing that short identifies a person anyway.
MIN_FINDING_CHARS = 3

#: How many "{" in a reply are tried as the start of its JSON object. Each failed try reads ahead,
#: and a reply is untrusted, so the search is bounded.
_MAX_JSON_STARTS = 50

_TOKEN_RE = re.compile(TOKEN_PATTERN)
#: A Markdown code fence, as chat models wrap JSON even when told not to.
_FENCE = "```"


def parse_findings(raw: object, masked_text: str) -> list[JudgeFinding]:
    """Parse a judge's JSON reply into validated findings.

    Args:
        raw: the model's reply. Anything but text is refused: an OpenAI-compatible endpoint
            answers with no content for a refusal or a reasoning model out of tokens.
        masked_text: the text that was reviewed; findings must quote it verbatim.

    Returns:
        Findings whose `text` occurs in `masked_text`, de-duplicated, longest first. Fragments
        shorter than :data:`MIN_FINDING_CHARS`, fragments that are not text, and fragments that
        contain an existing token are dropped: the first would garble the text, the second is a
        small model filling an empty slot with null, and the third is already masked.

    Raises:
        JudgeError: when the reply is not the documented JSON object.
    """
    if not isinstance(raw, str):
        raise JudgeError("judge returned no text content")
    payload = _decode(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("findings"), list):
        raise JudgeError("judge response must be an object with a 'findings' list")

    findings: dict[str, JudgeFinding] = {}
    for item in payload["findings"]:
        finding = _finding(item, masked_text)
        if finding is not None:
            findings[finding.text] = finding
    return sorted(findings.values(), key=lambda f: len(f.text), reverse=True)


def _decode(raw: str) -> Any:
    """Read the JSON object out of a reply, tolerating a Markdown fence or a sentence before it.

    A judge that fails on every fenced reply silently turns stage 2 off.

    Raises:
        JudgeError: when no JSON can be read. The message gives the length, never the reply: the
            reply is made of the very fragments the judge found identifying, and the error is
            logged.
    """
    body = raw.strip()
    for candidate in (body, _fenced(body)):
        if candidate is None:
            continue
        try:
            return json.loads(candidate)
        except (ValueError, RecursionError):
            pass
    found = _object_with_findings(body)
    if found is not None:
        return found
    raise JudgeError(f"judge returned no readable JSON ({len(raw)} characters)")


def _fenced(body: str) -> str | None:
    """Return what the first Markdown code fence holds, or None when there is none.

    Found with plain string search: the regex this replaced backtracked over the whitespace after
    an unclosed fence, and a reply of a few thousand spaces froze the event loop for seconds.
    """
    opening = body.find(_FENCE)
    if opening == -1:
        return None
    start = opening + len(_FENCE)
    while start < len(body) and body[start].isalpha():  # the language tag: ```json
        start += 1
    closing = body.find(_FENCE, start)
    return body[start:] if closing == -1 else body[start:closing]


def _object_with_findings(body: str) -> Any:
    """Return the first JSON object in `body` that has a "findings" key, or None.

    Every "{" is a candidate, up to a bound, not just the first: a sentence before the JSON may
    hold a brace of its own.
    """
    decoder = json.JSONDecoder()
    start = body.find("{")
    for _ in range(_MAX_JSON_STARTS):
        if start == -1:
            return None
        try:
            payload = decoder.raw_decode(body, start)[0]
        except (ValueError, RecursionError):
            payload = None
        if isinstance(payload, dict) and "findings" in payload:
            return payload
        start = body.find("{", start + 1)
    return None


def _finding(item: Any, masked_text: str) -> JudgeFinding | None:
    """Return the finding one entry of the reply describes, or None when it is unusable."""
    if not isinstance(item, dict):
        return None
    text = item.get("text")
    if not isinstance(text, str):
        return None
    text = text.strip()
    # Compared as the masking reads both, whitespace squeezed: a small model quoting a hard-wrapped
    # line with a plain space was dropped, and the identifying phrase went out with stage 2 "done".
    quoted = len(text) >= MIN_FINDING_CHARS and reading(text) in reading(masked_text)
    if not quoted or _TOKEN_RE.search(text):
        return None
    entity_type = item.get("type")
    entity_type = entity_type.upper() if isinstance(entity_type, str) else "OTHER"
    if entity_type not in ALLOWED_TYPES:
        entity_type = "OTHER"
    reason = item.get("reason")
    return JudgeFinding(text, entity_type, reason if isinstance(reason, str) else "")


def reply_content(body: Any, *path: str | int) -> str:
    """Follow `path` through a decoded HTTP body to the model's reply text.

    A provider that changes its response shape, or a proxy that answers with an error page, must
    end in a JudgeError the fallback policy handles, not in a TypeError that escapes it.

    Raises:
        JudgeError: when the path does not lead to text.
    """
    node = body
    for step in path:
        in_list = isinstance(step, int) and isinstance(node, list) and len(node) > step
        in_dict = isinstance(step, str) and isinstance(node, dict) and step in node
        if not (in_list or in_dict):
            raise JudgeError("judge response has an unexpected shape")
        node = node[step]
    if not isinstance(node, str):
        raise JudgeError("judge returned no text content")
    return node
