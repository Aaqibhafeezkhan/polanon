"""One call to a model, masked: everything it sends and everything it gets back.

A call is rarely one text. A chat turn sends the history, the new question and the results of the
tools the model asks for, and it gets back an answer, often streamed, and the arguments of those
tool calls. :class:`MaskedTurn` masks all of it with one mapping, so a person keeps one token from
the history to the last tool result, and it refuses with
:class:`~polanon.types.MaskingUnavailableError` instead of letting anything out unmasked.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from polanon._matching import type_of
from polanon.guards.denylist import DenylistGuard
from polanon.output_sweep import SweepReport
from polanon.policy import TOKEN_PATTERN
from polanon.stream_restore import StreamRestorer
from polanon.types import MaskingUnavailableError

if TYPE_CHECKING:
    from polanon import Anonymizer

_TOKEN_RE = re.compile(TOKEN_PATTERN)

#: What :meth:`MaskedTurn.mask_messages` does with a block no text mask reaches, such as an image,
#: a document or a model's signed reasoning: refuse the whole list, or leave the block out.
UnknownBlocks = Literal["raise", "drop"]

_Text = Callable[[str], str]


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
        unknown_blocks: UnknownBlocks = "raise",
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
            unknown_blocks: what :meth:`mask_messages` does with a block it cannot mask:
                ``"raise"`` refuses, ``"drop"`` leaves the block out.

        Raises:
            TypeError: when `known` is a single string, which would be read letter by letter.
            ValueError: when the anonymizer has a stage-2 judge, which needs an event loop, or
                for an `unknown_blocks` that is neither "raise" nor "drop".
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
        if unknown_blocks not in ("raise", "drop"):
            raise ValueError(f"unknown_blocks is 'raise' or 'drop', not {unknown_blocks!r}")
        terms = [term for term in known if term.strip()]
        self._anonymizer = _with_known_values(anonymizer, terms) if terms else anonymizer
        self._mapping: dict[str, str] = dict(mapping or {})
        self._unknown_blocks = unknown_blocks

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
        return _walk(value, self.mask, default)

    def mask_messages(self, messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Mask a list of chat messages written for the Anthropic or the OpenAI API.

        The text of every message is masked, and so are the input of a ``tool_use`` block, the
        content of a ``tool_result`` block or of a ``"tool"`` message, the arguments of
        ``tool_calls`` and a ``refusal``. A tool result or arguments written as JSON stay valid
        JSON. Roles, ids, tool names and every other field are copied as they are, and the
        citations of a text block are left out, since what they quote may be personal data. A
        block no text mask reaches, such as an image, a document or a model's signed reasoning, is
        refused or left out, as `unknown_blocks` says.

        A value first recognised in a later message, such as the question, is also masked in the
        earlier ones, where nothing recognised it on its own: a message masked before the mapping
        last grew is masked once more, from its original, with the whole mapping.

        Args:
            messages: plain dictionaries, oldest first. They are not modified.

        Raises:
            MaskingUnavailableError: when masking fails, and for a block that cannot be masked
                while `unknown_blocks` is "raise". Nothing of the list may be sent then.
        """
        originals = list(messages)
        masked, sizes = [], []
        for message in originals:
            masked.append(self._message(message, self.mask))
            sizes.append(len(self._mapping))
        final = len(self._mapping)
        return [
            self._message(original, self.mask) if size < final else done
            for original, size, done in zip(originals, sizes, masked, strict=True)
        ]

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

    def _message(self, message: Mapping[str, Any], text: _Text) -> dict[str, Any]:
        """Copy one message with `text` applied to everything in it that carries data."""
        if not isinstance(message, Mapping):
            raise TypeError(f"a message is a dictionary, not {type(message).__name__}")
        out = dict(message)
        content = message.get("content")
        if content is not None:
            tool = message.get("role") == "tool"
            out["content"] = (
                self._tool_content(content, text) if tool else self._content(content, text)
            )
        if message.get("tool_calls"):
            calls = (self._tool_call(call, text) for call in message["tool_calls"])
            out["tool_calls"] = [call for call in calls if call is not None]
        if isinstance(message.get("function_call"), Mapping):
            out["function_call"] = self._arguments_of(message["function_call"], text)
        if isinstance(message.get("refusal"), str):
            out["refusal"] = text(message["refusal"])
        return out

    def _content(self, content: Any, text: _Text) -> Any:
        """A message's content: a text, or a list of blocks."""
        if isinstance(content, str):
            return text(content)
        if isinstance(content, list):
            blocks = (self._block(block, text) for block in content)
            return [block for block in blocks if block is not None]
        raise TypeError(f"message content is a str or a list, not {type(content).__name__}")

    def _block(self, block: Any, text: _Text) -> dict[str, Any] | None:
        """One content block, or None when it is left out."""
        if not isinstance(block, Mapping):
            raise TypeError(f"a content block is a dictionary, not {type(block).__name__}")
        kind = block.get("type")
        if kind == "text":
            out = {key: item for key, item in block.items() if key != "citations"}
            out["text"] = text(block.get("text", ""))
            return out
        if kind == "tool_use":
            return {**block, "input": _walk(block.get("input") or {}, text)}
        if kind == "tool_result":
            out = dict(block)
            if block.get("content") is not None:
                out["content"] = self._tool_content(block["content"], text)
            return out
        self._leave_out(f"a {kind!r} block")
        return None

    def _tool_content(self, content: Any, text: _Text) -> Any:
        """What a tool returned: a text, often JSON, or a list of blocks."""
        if isinstance(content, str):
            return _json_text(content, text)
        return self._content(content, text)

    def _tool_call(self, call: Any, text: _Text) -> dict[str, Any] | None:
        """One OpenAI tool call, with its arguments masked."""
        if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
            self._leave_out("a tool call that is not a function call")
            return None
        return {**call, "function": self._arguments_of(call["function"], text)}

    def _arguments_of(self, function: Mapping[str, Any], text: _Text) -> dict[str, Any]:
        """A function call with its arguments masked, as JSON text or as a value."""
        out = dict(function)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            out["arguments"] = _json_text(arguments, text)
        elif arguments is not None:
            out["arguments"] = _walk(arguments, text)
        return out

    def _leave_out(self, what: str) -> None:
        """Refuse something that cannot be masked, unless the turn was told to leave it out."""
        if self._unknown_blocks == "raise":
            raise MaskingUnavailableError(
                f"{what} cannot be masked; pass unknown_blocks='drop' to leave such blocks out"
            )


def _walk(value: Any, text: _Text, default: Callable[[Any], Any] | None = None) -> Any:
    """Apply `text` to every string in a JSON-like value, keys and other scalars untouched."""
    if isinstance(value, str):
        return text(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Mapping):
        return {key: _walk(item, text, default) for key, item in value.items()}
    if isinstance(value, list):
        return [_walk(item, text, default) for item in value]
    if isinstance(value, tuple):
        return tuple(_walk(item, text, default) for item in value)
    if default is None:
        raise TypeError(
            f"cannot mask a {type(value).__name__}: convert it first, or pass default=str"
        )
    return _walk(default(value), text)


def _json_text(content: str, text: _Text) -> str:
    """Mask a text that may be a JSON object or array, keeping the JSON valid."""
    try:
        parsed = json.loads(content)
    except ValueError:
        return text(content)
    if isinstance(parsed, dict | list):
        return json.dumps(_walk(parsed, text), ensure_ascii=False)
    return text(content)


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
