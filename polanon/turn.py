"""One call to a model, masked: everything it sends and everything it gets back.

A call is rarely one text. A chat turn sends the history, the new question and the results of the
tools the model asks for, and it gets back an answer, often streamed, and the arguments of those
tool calls. :class:`MaskedTurn` masks all of it with one mapping, so a person keeps one token from
the history to the last tool result, and it refuses with
:class:`~polanon.types.MaskingUnavailableError` instead of letting anything out unmasked.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from polanon._matching import type_of
from polanon.guards.denylist import DenylistGuard
from polanon.output_sweep import SweepReport
from polanon.policy import TOKEN_PATTERN
from polanon.stream_restore import StreamRestorer
from polanon.types import MaskingUnavailableError

if TYPE_CHECKING:
    from polanon import Anonymizer

_TOKEN_RE = re.compile(TOKEN_PATTERN)


class MaskedTurn:
    """Masks everything one call to a model sends and restores everything it gets back.

    One mapping serves the whole turn, so the history, the question and every tool result give a
    value the same token. The mapping grows in place: a restorer taken before a tool result was
    masked still knows that result's tokens. Start one turn per call to the model, or keep one per
    conversation together with its mapping; a turn is not meant to be shared between threads.

    Every ``mask`` method refuses with :class:`~polanon.types.MaskingUnavailableError` when a
    detector fails, and returns nothing half masked.
    """

    def __init__(
        self,
        anonymizer: Anonymizer,
        *,
        known: Iterable[str] = (),
        mapping: Mapping[str, str] | None = None,
    ) -> None:
        """Start a turn.

        Args:
            anonymizer: the guards and the policy to mask with. Build it once per process: loading
                the language model is the expensive part, and every turn reuses its guards.
            known: values the caller already knows are sensitive, such as its clients' names. They
                are masked wherever they appear, as ``ORG``, also when the anonymizer's allowlist
                holds them or its enabled types leave ``ORG`` out.
            mapping: token -> original value from earlier turns of the same conversation. It is
                read, never modified.

        Raises:
            TypeError: when `known` is a single string, which would be read letter by letter.
            ValueError: when the anonymizer has a stage-2 judge, which needs an event loop.
            MaskingUnavailableError: when the known values cannot be made into a guard.
        """
        if isinstance(known, str):
            # The value itself stays out of the message: it is what the caller wants kept hidden.
            raise TypeError("known takes a list of values, not a single string")
        if anonymizer.judge is not None:
            raise ValueError(
                "a MaskedTurn masks without an event loop, so it cannot run the stage-2 judge "
                f"{anonymizer.judge.name!r}: use an Anonymizer without a judge"
            )
        terms = [term for term in known if term.strip()]
        self._anonymizer = _with_known_values(anonymizer, terms) if terms else anonymizer
        self._mapping: dict[str, str] = dict(mapping or {})

    @property
    def mapping(self) -> dict[str, str]:
        """A copy of the turn's token -> original value mapping, to keep for the next turn."""
        return dict(self._mapping)

    def mask(self, text: str) -> str:
        """Mask one text and add the tokens it needed to the turn's mapping.

        Raises:
            TypeError: when `text` is not a string; :meth:`mask_data` takes structured values.
            MaskingUnavailableError: when masking fails. Nothing of the text may be sent then.
        """
        if not isinstance(text, str):
            raise TypeError(f"mask takes a str, not {type(text).__name__}: use mask_data")
        if not text:
            return text
        try:
            result = self._anonymizer.anonymize_sync(text, self._mapping)
        except MaskingUnavailableError:
            raise
        except Exception as exc:
            raise MaskingUnavailableError(_failure(exc)) from exc
        self._mapping.update(result.mapping)
        return result.anonymized

    def mask_data(self, value: Any, *, default: Callable[[Any], Any] | None = None) -> Any:
        """Mask every string inside a JSON-like value, such as the result of a tool.

        Strings are masked, dictionaries value by value, lists and tuples item by item. Keys,
        numbers, booleans and None stay as they are, so personal data belongs in string values.
        Any other object goes through `default` first, as with :func:`json.dumps`.

        Args:
            value: the value to mask; it is not modified.
            default: turns an object that is not JSON-like into one, such as ``str`` for dates.

        Raises:
            TypeError: for an object that is not JSON-like when no `default` is given. Converted
                to text after masking, it would reach the model unmasked.
            MaskingUnavailableError: when masking fails.
        """
        if isinstance(value, str):
            return self.mask(value)
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, Mapping):
            return {key: self.mask_data(item, default=default) for key, item in value.items()}
        if isinstance(value, list):
            return [self.mask_data(item, default=default) for item in value]
        if isinstance(value, tuple):
            return tuple(self.mask_data(item, default=default) for item in value)
        if default is None:
            raise TypeError(
                f"cannot mask a {type(value).__name__}: convert it first, or pass default=str"
            )
        return self.mask_data(default(value))

    def restore(self, text: str) -> str:
        """Put the original values back in place of the turn's tokens; unknown tokens stay."""
        return _TOKEN_RE.sub(lambda match: self._mapping.get(match[0], match[0]), text)

    def restore_data(self, value: Any) -> Any:
        """Restore every string inside a JSON-like value, such as the arguments of a tool call.

        Keys and values that are not strings are left as they are, and so is the value itself.
        """
        if isinstance(value, str):
            return self.restore(value)
        if isinstance(value, Mapping):
            return {key: self.restore_data(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.restore_data(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.restore_data(item) for item in value)
        return value

    def stream_restorer(self) -> StreamRestorer:
        """Return a restorer for one streamed answer, which keeps a token split across chunks whole.

        Take a new one for every response the model streams.
        """
        return StreamRestorer(self._mapping)

    def unknown_tokens(self, text: str) -> list[str]:
        """Return the tokens in `text` the turn never handed out, which the model made up.

        It runs no detector, so it is cheap enough for every answer; :meth:`sweep` also looks for
        personal data the answer added.
        """
        return sorted({match[0] for match in _TOKEN_RE.finditer(text)} - self._mapping.keys())

    def sweep(self, restored_text: str) -> SweepReport:
        """Check a restored answer for invented tokens and personal data nobody sent.

        It runs the turn's detectors on the answer, the language model included.
        """
        return self._anonymizer.sweep(restored_text, self._mapping)

    def counts(self) -> dict[str, int]:
        """How many values of each type the turn masked, such as ``{"PERSON": 2, "NIP": 1}``.

        A placeholder the caller wrote, which stands for itself in the mapping, is not counted.
        """
        return dict(
            Counter(
                type_of(token) or token for token, value in self._mapping.items() if value != token
            )
        )


def _with_known_values(anonymizer: Anonymizer, terms: list[str]) -> Anonymizer:
    """The anonymizer's own guards, so the language model is not loaded again, plus a denylist."""
    from polanon import Anonymizer as Facade  # the facade imports this module

    try:
        denylist = DenylistGuard(terms)
    except Exception as exc:
        raise MaskingUnavailableError(_failure(exc, "the known values")) from exc
    return Facade(
        [*anonymizer.guards, denylist],
        enabled=anonymizer.enabled | {"ORG"},
        allowlist=anonymizer.allowlist,
    )


def _failure(exc: Exception, what: str = "the text") -> str:
    """Name a failure without quoting anything: an error's own message may carry the text."""
    return (
        f"masking {what} failed with {type(exc).__name__}, so nothing may be sent to the model; "
        "the error is chained as the cause"
    )
