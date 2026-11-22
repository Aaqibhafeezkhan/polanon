"""The last check before a call leaves: does any value the mapping hides still go out?

It is independent of how the payload was masked, so an application that builds its own payloads,
replays history or adds attachments can run it on exactly what it is about to send. The gateway
runs it on every request.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from polanon.pipeline import mask_known_values

#: Fields whose value names something in the API or in code rather than saying anything: an id, a
#: type, a role, a model, a tool's name, the parameters a tool requires. The masking leaves them
#: as they are, and so does the check.
_NAMING_FIELDS = frozenset(
    {
        "id",
        "tool_call_id",
        "call_id",
        "item_id",
        "type",
        "role",
        "model",
        "object",
        "name",
        "required",
    }
)
#: A name from code: one word of letters, digits and ``_ . : / -``, with no space.
_CODE_NAME = re.compile(r"[\w.:/-]+")


def leaked_tokens(payload: Any, mapping: Mapping[str, str]) -> list[str]:
    """Return the tokens whose original value would still go out in `payload`.

    The last check before a call leaves the machine, independent of how the payload was masked:
    every text of the payload, whatever field it sits in, is searched for every value of the
    mapping by the rules the masking itself follows, so a short number inside a date or a short
    surname inside a longer word does not count, as it would not be masked either. Only tokens
    are returned, never values, so the result may be logged.

    It reads the payload as the masking does. A text that is a JSON object or array, such as a
    tool's arguments or result, is searched value by value. Names from code are left alone, as
    the masking leaves them: a key written as a field name, ``tool_call_id`` or ``attributes``,
    and an id, a type, a role, a model or a tool's name written without spaces, such as
    ``call_00_Ab3`` or ``reasoning.text``. A key with a space in it, ``"Jan Kowalski"``, is a text
    and is searched.

    Args:
        payload: what is about to be sent: a text, or a JSON-like value such as a request body.
        mapping: token -> original value, the whole mapping the payload was masked with.

    Returns:
        The leaked tokens, sorted; an empty list when nothing leaks.
    """
    values = {token: value for token, value in mapping.items() if value != token}
    if not values:
        return []
    leaked: set[str] = set()
    for text in _texts(payload):
        if text and mask_known_values(text, values) != text:
            leaked.update(_tokens_that_mask(text, values))
    return sorted(leaked)


def _texts(value: Any, field: Any = None) -> Iterable[str]:
    """Every text of a JSON-like value the last check reads; see :func:`leaked_tokens`."""
    if isinstance(value, str):
        if field in _NAMING_FIELDS and _CODE_NAME.fullmatch(value):
            return
        parsed = _json_container(value)
        if parsed is None:
            yield value
        else:
            yield from _texts(parsed)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and not _CODE_NAME.fullmatch(key):
                yield key
            yield from _texts(item, key)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _texts(item, field)


def _json_container(text: str) -> Any:
    """The object or array a text holds as JSON, or None for any other text."""
    if text.lstrip()[:1] not in ("{", "["):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict | list) else None


def _tokens_that_mask(text: str, values: Mapping[str, str]) -> list[str]:
    """The tokens whose values the masking rules would still find in `text`."""
    return [
        token for token, value in values.items() if mask_known_values(text, {token: value}) != text
    ]
