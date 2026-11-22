r"""Restore original values inside a streamed response without breaking tokens across chunks.

A provider streams arbitrary fragments, so ``<OSOBA_1>`` can arrive as ``<OSO`` + ``BA_1>``. The
restorer buffers any trailing fragment that could still grow into a token and emits the rest.

A model may write a person's token with the case the sentence needs after a colon, as in
``<OSOBA_1:wołacz>``; the name then comes back declined, as :func:`restored` describes.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

from polanon._inflection import CASE_NAMES, inflected
from polanon.policy import CASED_TOKEN_PATTERN, LABEL_TO_ENTITY

#: A token as an answer may hold it: ``<OSOBA_1>``, or with a case after a colon,
#: ``<OSOBA_1:wołacz>``.
RESTORED_TOKEN_PATTERN = CASED_TOKEN_PATTERN
_TOKEN_RE = re.compile(RESTORED_TOKEN_PATTERN)
#: A trailing fragment that may still become a token, e.g. "<", "<OSOB", "<OSOBA_1", "<OSOBA_1:wo".
_PARTIAL_RE = re.compile(r"<[A-Z_]*\d*(?::[^\W\d_]*)?$")
#: Longest fragment held back for the built-in labels; beyond this it cannot be a token any more.
#: A custom entity type can have a longer label, so a mapping with longer tokens raises the limit.
_MAX_PARTIAL = 32
#: What a case adds to a token: a colon and the longest name of a case, "miejscownik".
_CASE_ROOM = 1 + max(len(name) for name in CASE_NAMES)


class _Values(Protocol):
    def get(self, token: str, /) -> str | None: ...


def restored(
    text: str,
    mapping: _Values,
    *,
    escape: Callable[[str], str] | None = None,
    as_held: bool = False,
) -> str:
    """Put the original values back in place of the tokens `mapping` knows; others stay.

    A token with a case, ``<OSOBA_1:wołacz>``, gives a person's name declined into it: "Janie
    Kowalski" for "Jana Kowalskiego". A value that cannot be declined for sure, and a case Polish
    does not have, give the value as written, and the case is dropped.

    Args:
        text: an answer, or any text with tokens.
        mapping: token -> original value; anything with ``get``.
        escape: applied to every value put back, such as escaping it for a JSON string.
        as_held: give every value as the mapping holds it and drop the case, for a value that
            code compares rather than a person reads, such as a name a tool looks up in the
            records it was masked from.
    """

    def value(match: re.Match[str]) -> str:
        token = f"<{match['label']}_{match['n']}>"
        original = mapping.get(token)
        if original is None or original == token:
            return match[0]  # unknown, or a placeholder typed as it is: it stays as written
        if match["case"] is not None and not as_held:
            entity_type = LABEL_TO_ENTITY.get(match["label"], match["label"])
            original = inflected(original, entity_type, match["case"]) or original
        return original if escape is None else escape(original)

    return _TOKEN_RE.sub(value, text)


def tokens_written(text: str) -> list[str]:
    """Return the tokens in `text`, each as the mapping holds it: ``<OSOBA_1>`` for a cased one."""
    return [f"<{match['label']}_{match['n']}>" for match in _TOKEN_RE.finditer(text)]


class StreamRestorer:
    """Incrementally replaces tokens with their original values in a stream of text chunks."""

    def __init__(
        self, mapping: dict[str, str], *, escape: Callable[[str], str] | None = None
    ) -> None:
        """Create a restorer for one conversation's token mapping.

        Args:
            mapping: token -> original value.
            escape: applied to every value put back, such as escaping it for a JSON string.
        """
        self.mapping = mapping
        self.buffer = ""
        self._escape = escape
        self._max_partial = max([_MAX_PARTIAL, *(len(token) for token in mapping)])

    def _substitute(self, text: str) -> str:
        return restored(text, self.mapping, escape=self._escape)

    def feed(self, chunk: str) -> str:
        """Consume one chunk and return the text that is safe to emit now."""
        self.buffer += chunk
        partial = _PARTIAL_RE.search(self.buffer)
        if partial is None or self._too_long(partial[0]):
            emitted, self.buffer = self.buffer, ""
        else:
            emitted, self.buffer = self.buffer[: partial.start()], self.buffer[partial.start() :]
        return self._substitute(emitted)

    def _too_long(self, fragment: str) -> bool:
        """Whether a fragment held back has grown past any token, with a case or without."""
        token, colon, case = fragment.partition(":")
        return len(token) > self._max_partial or len(colon + case) > _CASE_ROOM

    def flush(self) -> str:
        """Emit whatever is still buffered at the end of the stream."""
        emitted, self.buffer = self.buffer, ""
        return self._substitute(emitted)
