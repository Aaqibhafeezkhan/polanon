r"""Restore original values inside a streamed response without breaking tokens across chunks.

A provider streams arbitrary fragments, so ``<OSOBA_1>`` can arrive as ``<OSO`` + ``BA_1>``. The
restorer buffers any trailing fragment that could still grow into a token and emits the rest.
"""

from __future__ import annotations

import re

from polanon.policy import TOKEN_PATTERN

_TOKEN_RE = re.compile(TOKEN_PATTERN)
#: A trailing fragment that may still become a token, e.g. "<", "<OSOB", "<OSOBA_1".
_PARTIAL_RE = re.compile(r"<[A-Z_]*\d*$")
#: Longest fragment held back for the built-in labels; beyond this it cannot be a token any more.
#: A custom entity type can have a longer label, so a mapping with longer tokens raises the limit.
_MAX_PARTIAL = 32


class StreamRestorer:
    """Incrementally replaces tokens with their original values in a stream of text chunks."""

    def __init__(self, mapping: dict[str, str]) -> None:
        """Create a restorer for one conversation's token mapping."""
        self.mapping = mapping
        self.buffer = ""
        self._max_partial = max([_MAX_PARTIAL, *(len(token) for token in mapping)])

    def _substitute(self, text: str) -> str:
        return _TOKEN_RE.sub(lambda m: self.mapping.get(m.group(0), m.group(0)), text)

    def feed(self, chunk: str) -> str:
        """Consume one chunk and return the text that is safe to emit now."""
        self.buffer += chunk
        partial = _PARTIAL_RE.search(self.buffer)
        if partial is None or partial.end() - partial.start() > self._max_partial:
            emitted, self.buffer = self.buffer, ""
        else:
            emitted, self.buffer = self.buffer[: partial.start()], self.buffer[partial.start() :]
        return self._substitute(emitted)

    def flush(self) -> str:
        """Emit whatever is still buffered at the end of the stream."""
        emitted, self.buffer = self.buffer, ""
        return self._substitute(emitted)
