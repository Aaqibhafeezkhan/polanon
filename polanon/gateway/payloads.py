"""The payloads of the OpenAI API, masked on the way out and restored on the way back.

Everything here works on plain dictionaries and needs nothing beyond the standard library, so it
is tested without a server. A :class:`~polanon.turn.MaskedTurn` does the masking: one turn per
call, so the history, the question and every tool result give a value the same token, and the
mapping that restores the answer is the one the request was masked with.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from polanon.pipeline import mask_known_values
from polanon.policy import TOKEN_PATTERN
from polanon.stream_restore import StreamRestorer
from polanon.turn import MaskedTurn
from polanon.types import MaskingUnavailableError

_TOKEN_RE = re.compile(TOKEN_PATTERN)

#: Items of a Responses API input that carry nothing to mask: blobs the provider sealed from text
#: it got masked, which no one else can read or change.
_SEALED_ITEMS = frozenset({"compaction"})
#: Content parts of a Responses API message whose text is masked.
_TEXT_PARTS = frozenset({"input_text", "output_text", "text", "summary_text", "reasoning_text"})
#: Fields of a Responses API request that keep the conversation at the provider, so the next call
#: would bring back tokens of values the gateway no longer holds.
_PROVIDER_STATE = ("previous_response_id", "conversation")

#: Roles whose ``name`` field names a participant, which may be a person; for a ``tool`` or a
#: ``function`` message it names the function and stays.
_PARTICIPANT_ROLES = frozenset({"system", "developer", "user", "assistant"})

#: Fields of a chat completion request, other than the messages, that carry text the operator's
#: users wrote: an end user's identifier, the metadata stored with a completion, and the content a
#: predicted output is compared with. Tool schemas and response formats describe the shape of an
#: answer, not data, and pass as they are.
_TEXT_FIELDS = ("user", "safety_identifier", "metadata", "prediction")


class GatewayRequestError(ValueError):
    """A request the gateway refuses to forward, with a message for the client.

    The message names what is wrong and never quotes the payload, which may hold personal data.
    """


def mask_chat_request(
    turn: MaskedTurn, body: Mapping[str, Any], *, instructions: str | None = None
) -> dict[str, Any]:
    """Return a ``/chat/completions`` request with everything that carries data masked.

    The messages are masked by :meth:`MaskedTurn.mask_messages`: text, tool results and the
    arguments of earlier tool calls. A participant's ``name`` is left out, since it may be a
    person's and the API allows no token there. An end user's identifier, stored metadata and a
    predicted output are masked as well. An answer in audio cannot be restored, so a request for
    one is refused.

    Args:
        turn: the turn of this call; its mapping restores the answer afterwards.
        body: the request as the client sent it; it is not modified.
        instructions: a note telling the model what the tokens are, added to the first system
            message or as a new one; None leaves the messages as they are.

    Raises:
        GatewayRequestError: for a request without a list of messages, or one asking for audio.
        MaskingUnavailableError: when masking fails, or for content that cannot be masked, such
            as an image, while the turn refuses it.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GatewayRequestError("a chat completion needs a non-empty list of messages")
    modalities = body.get("modalities")
    if isinstance(modalities, list) and any(item != "text" for item in modalities):
        raise GatewayRequestError("the polanon gateway can restore text answers only, not audio")
    masked = [_without_participant(message) for message in turn.mask_messages(messages)]
    out = {**body, "messages": masked}
    for field in _TEXT_FIELDS:
        if body.get(field) is not None:
            out[field] = turn.mask_data(body[field])
    if instructions:
        out["messages"] = _with_instructions(masked, instructions)
    return out


def mask_embeddings_request(turn: MaskedTurn, body: Mapping[str, Any]) -> dict[str, Any]:
    """Return an ``/embeddings`` request with its input masked, one turn for every text.

    Raises:
        GatewayRequestError: for an input that is not text, such as token ids, which cannot be
            read and so cannot be masked.
        MaskingUnavailableError: when masking fails.
    """
    value = body.get("input")
    if isinstance(value, str):
        masked: str | list[str] = turn.mask(value)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        masked = [turn.mask(item) for item in value]
    else:
        raise GatewayRequestError(
            "the polanon gateway embeds text only: input is a string or a list of strings"
        )
    out = {**body, "input": masked}
    if body.get("user") is not None:
        out["user"] = turn.mask_data(body["user"])
    return out


def mask_responses_request(
    turn: MaskedTurn,
    body: Mapping[str, Any],
    *,
    instructions: str | None = None,
    drop_unsupported: bool = False,
) -> dict[str, Any]:
    """Return a ``/responses`` request with everything that carries data masked.

    The instructions and every item of the input are masked: the text of messages, the arguments
    and the output of tool calls, and the summary of a model's reasoning, which the gateway
    restored on the way back. Encrypted reasoning and compaction pass as they are: the provider
    sealed them from text it got masked. A value first recognised in a later item is also masked
    in the earlier ones, as :meth:`MaskedTurn.mask_messages` does for chat messages.

    The gateway keeps nothing between calls, so the client sends the whole conversation, as
    Hermes Agent does with ``store: false``. A request that leaves it at the provider, with
    ``previous_response_id`` or ``conversation``, or asks for a ``background`` response to fetch
    later, would bring back tokens nobody can restore, and is refused.

    Args:
        turn: the turn of this call; its mapping restores the answer afterwards.
        body: the request as the client sent it; it is not modified.
        instructions: a note telling the model what the tokens are, added to the instructions;
            None leaves them as they are.
        drop_unsupported: leave out content that cannot be masked, such as an image, rather than
            refuse the call.

    Raises:
        GatewayRequestError: for a request that keeps state at the provider, asks for audio, or
            has an input that is neither text nor a list of items.
        MaskingUnavailableError: when masking fails, or for content that cannot be masked while
            `drop_unsupported` is off.
    """
    for field in _PROVIDER_STATE:
        if body.get(field):
            raise GatewayRequestError(
                f"the polanon gateway keeps nothing between calls, so it cannot restore a "
                f"conversation kept by {field}: send the whole conversation in input instead"
            )
    if body.get("background"):
        raise GatewayRequestError(
            "the polanon gateway cannot restore a response fetched later: background is not "
            "supported"
        )
    modalities = body.get("modalities")
    if isinstance(modalities, list) and any(item != "text" for item in modalities):
        raise GatewayRequestError("the polanon gateway can restore text answers only, not audio")
    value = body.get("input")
    if value is not None and not isinstance(value, str | list):
        raise GatewayRequestError("input is a string or a list of items")
    pieces: list[tuple[str, Any]] = []
    if isinstance(body.get("instructions"), str):
        pieces.append(("instructions", body["instructions"]))
    if isinstance(value, str):
        pieces.append(("text", value))
    elif isinstance(value, list):
        pieces.extend(("item", item) for item in value)

    def mask(kind: str, original: Any) -> Any:
        if kind == "item":
            return _item(turn, original, drop_unsupported)
        return turn.mask(original)

    masked, sizes = [], []
    for kind, original in pieces:
        masked.append(mask(kind, original))
        sizes.append(len(turn.mapping))
    final = len(turn.mapping)
    done = [
        mask(kind, original) if size < final else result
        for (kind, original), size, result in zip(pieces, sizes, masked, strict=True)
    ]
    out = dict(body)
    if pieces and pieces[0][0] == "instructions":
        out["instructions"] = done.pop(0)
    if isinstance(value, str):
        out["input"] = done[0]
    elif isinstance(value, list):
        out["input"] = [item for item in done if item is not None]
    for field in ("user", "safety_identifier", "metadata"):
        if body.get(field) is not None:
            out[field] = turn.mask_data(body[field])
    prompt = body.get("prompt")
    if isinstance(prompt, Mapping) and prompt.get("variables") is not None:
        out["prompt"] = {**prompt, "variables": turn.mask_data(prompt["variables"])}
    if instructions:
        given = out.get("instructions")
        out["instructions"] = f"{given}\n\n{instructions}" if given else instructions
    return out


def restore_chat_response(turn: MaskedTurn, body: Any) -> Any:
    """Put the original values back into a ``/chat/completions`` response.

    Every text in it is restored: the content, a refusal, a provider's reasoning text. The
    arguments of a tool call are JSON, so a value is written into them escaped, and they stay
    valid JSON whatever quotes or backslashes the value holds.
    """
    return _restored(body, turn.mapping)


def restore_responses_response(turn: MaskedTurn, body: Any) -> Any:
    """Put the original values back into a ``/responses`` response.

    Every text of every output item is restored, a reasoning summary included, and so is the
    echo of the instructions; the arguments of a function call stay valid JSON.
    """
    return _restored(body, turn.mapping)


def leaked_tokens(payload: Any, mapping: Mapping[str, str]) -> list[str]:
    """Return the tokens whose original value would still go out in `payload`.

    The last check before a call leaves the machine, independent of how the payload was masked:
    every text of the payload, whatever field it sits in, is searched for every value of the
    mapping by the rules the masking itself follows, so a short number inside a date or a short
    surname inside a longer word does not count, as it would not be masked either. Only tokens
    are returned, never values, so the result may be logged.
    """
    values = {token: value for token, value in mapping.items() if value != token}
    if not values:
        return []
    leaked: set[str] = set()
    for text in _texts(payload):
        if text and mask_known_values(text, values) != text:
            leaked.update(_tokens_that_mask(text, values))
    return sorted(leaked)


class ChatStream:
    """Restores the chunks of one streamed ``/chat/completions`` answer.

    A provider cuts the answer anywhere, so a token can arrive as ``<OSO`` and ``BA_1>``. Every
    choice's text, and every tool call's arguments, has its own restorer, which holds back the
    start of a token until the rest arrives; what a choice still holds when it finishes goes out
    with its last chunk.
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        """Start a stream restored with the mapping its request was masked with."""
        self._mapping = dict(mapping)
        self._escaped = {token: _json_escaped(value) for token, value in self._mapping.items()}
        self._texts: dict[tuple[int, str], StreamRestorer] = {}
        self._arguments: dict[tuple[int, int], StreamRestorer] = {}
        self._template: dict[str, Any] | None = None

    def chunk(self, chunk: Any) -> Any:
        """Return one chunk with its text restored as far as the stream allows yet."""
        if not isinstance(chunk, dict):
            return chunk
        self._template = {key: value for key, value in chunk.items() if key != "choices"}
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", 0)
            delta = choice.get("delta")
            if isinstance(delta, dict):
                self._delta(index, delta)
            if choice.get("finish_reason") is not None:
                choice["delta"] = self._finished(index, delta if isinstance(delta, dict) else {})
        return chunk

    def rest(self) -> dict[str, Any] | None:
        """Return a chunk with whatever the stream still holds back, or None when it holds nothing.

        For a stream that ended without saying a choice had finished.
        """
        indices = sorted(
            {index for index, _ in self._texts} | {index for index, _ in self._arguments}
        )
        choices = []
        for index in indices:
            delta = self._finished(index, {})
            if delta:
                choices.append({"index": index, "delta": delta, "finish_reason": None})
        if not choices:
            return None
        return {**(self._template or {"object": "chat.completion.chunk"}), "choices": choices}

    def _delta(self, index: int, delta: dict[str, Any]) -> None:
        for field in ("content", "refusal", "reasoning_content", "reasoning"):
            if isinstance(delta.get(field), str):
                delta[field] = self._text(index, field).feed(delta[field])
        for call in delta.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                restorer = self._argument(index, call.get("index", 0))
                function["arguments"] = restorer.feed(function["arguments"])

    def _finished(self, index: int, delta: dict[str, Any]) -> dict[str, Any]:
        """Add to `delta` whatever choice `index` still holds back, and forget the choice."""
        for text_key in [key for key in self._texts if key[0] == index]:
            held = self._texts.pop(text_key).flush()
            if held:
                delta[text_key[1]] = (delta.get(text_key[1]) or "") + held
        for call_key in [key for key in self._arguments if key[0] == index]:
            held = self._arguments.pop(call_key).flush()
            if held:
                calls = delta.setdefault("tool_calls", [])
                calls.append({"index": call_key[1], "function": {"arguments": held}})
        return delta

    def _text(self, index: int, field: str) -> StreamRestorer:
        return self._texts.setdefault((index, field), StreamRestorer(self._mapping))

    def _argument(self, index: int, call: int) -> StreamRestorer:
        return self._arguments.setdefault((index, call), StreamRestorer(self._escaped))


class ResponsesStream:
    """Restores the events of one streamed ``/responses`` answer.

    Text arrives in ``.delta`` events, each text of each output item in its own sequence, so each
    has its own restorer; the ``.done`` event that closes a sequence carries the whole text, and
    what the restorer still holds goes out first, as one more delta. Every other event, such as
    ``response.completed`` with the whole response, is restored as it is. The events the client
    gets are numbered anew, since a held-back delta adds one.
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        """Start a stream restored with the mapping its request was masked with."""
        self._mapping = dict(mapping)
        self._escaped = {token: _json_escaped(value) for token, value in self._mapping.items()}
        self._texts: dict[tuple[Any, ...], tuple[dict[str, Any], StreamRestorer]] = {}
        self._sequence = 0

    def events(self, event: Any) -> list[Any]:
        """Return the events to send for one event of the provider, restored and numbered."""
        if not isinstance(event, dict):
            return [event]
        kind = str(event.get("type", ""))
        out: list[dict[str, Any]] = []
        if kind.endswith(".delta") and isinstance(event.get("delta"), str):
            restorer = self._restorer(event)
            out.append({**event, "delta": restorer.feed(event["delta"])})
        else:
            if kind.endswith(".done"):
                held = self._held(event)
                if held is not None:
                    out.append(held)
            out.append(_restored(event, self._mapping))
        return [self._numbered(item) for item in out]

    def rest(self) -> list[dict[str, Any]]:
        """Return a delta for every text the stream still holds back, for a stream cut short."""
        out = []
        for template, restorer in self._texts.values():
            held = restorer.flush()
            if held:
                out.append(self._numbered({**template, "delta": held}))
        self._texts.clear()
        return out

    def _restorer(self, event: dict[str, Any]) -> StreamRestorer:
        key = _sequence_key(event, ".delta")
        if key not in self._texts:
            mapping = self._escaped if "arguments" in key[0] else self._mapping
            template = {name: value for name, value in event.items() if name != "delta"}
            self._texts[key] = (template, StreamRestorer(mapping))
        return self._texts[key][1]

    def _held(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """One more delta with what the sequence `event` closes still holds, or None."""
        found = self._texts.pop(_sequence_key(event, ".done"), None)
        if found is None:
            return None
        template, restorer = found
        held = restorer.flush()
        return {**template, "delta": held} if held else None

    def _numbered(self, event: dict[str, Any]) -> dict[str, Any]:
        if "sequence_number" in event:
            event = {**event, "sequence_number": self._sequence}
            self._sequence += 1
        return event


def _sequence_key(event: Mapping[str, Any], suffix: str) -> tuple[Any, ...]:
    """What tells one text's sequence of deltas from another: its kind and where it stands."""
    kind = str(event.get("type", ""))[: -len(suffix)]
    return (
        kind,
        event.get("item_id"),
        event.get("output_index"),
        event.get("content_index"),
        event.get("summary_index"),
    )


def _item(turn: MaskedTurn, item: Any, drop: bool) -> dict[str, Any] | None:
    """One item of a Responses API input, masked, or None when it is left out."""
    if not isinstance(item, Mapping):
        raise TypeError(f"an input item is a dictionary, not {type(item).__name__}")
    kind = item.get("type")
    out = dict(item)
    if kind in (None, "message"):
        content = item.get("content")
        if isinstance(content, str):
            out["content"] = turn.mask(content)
        elif isinstance(content, list):
            parts = (_part(turn, part, drop) for part in content)
            out["content"] = [part for part in parts if part is not None]
        return out
    if kind in ("function_call", "custom_tool_call"):
        field = "arguments" if kind == "function_call" else "input"
        if isinstance(item.get(field), str):
            out[field] = _json_masked(turn, item[field])
        return out
    if kind in ("function_call_output", "custom_tool_call_output"):
        output = item.get("output")
        if isinstance(output, str):
            out["output"] = _json_masked(turn, output)
        elif isinstance(output, list):
            parts = (_part(turn, part, drop) for part in output)
            out["output"] = [part for part in parts if part is not None]
        return out
    if kind == "reasoning":
        for field in ("summary", "content"):
            if isinstance(item.get(field), list):
                parts = (_part(turn, part, drop) for part in item[field])
                out[field] = [part for part in parts if part is not None]
        return out
    if kind in _SEALED_ITEMS:
        return out
    _unsupported(f"a {kind!r} item", drop)
    return None


def _part(turn: MaskedTurn, part: Any, drop: bool) -> dict[str, Any] | None:
    """One content part, masked, or None when it is left out.

    The annotations of an output text are left out: they point into the text by position, and
    what they quote may be personal data.
    """
    if not isinstance(part, Mapping):
        raise TypeError(f"a content part is a dictionary, not {type(part).__name__}")
    kind = part.get("type")
    if kind in _TEXT_PARTS:
        out = {key: value for key, value in part.items() if key != "annotations"}
        out["text"] = turn.mask(part.get("text") or "")
        return out
    if kind == "refusal":
        return {**part, "refusal": turn.mask(part.get("refusal") or "")}
    _unsupported(f"a {kind!r} part", drop)
    return None


def _unsupported(what: str, drop: bool) -> None:
    """Refuse content that cannot be masked, unless the gateway was told to leave it out."""
    if not drop:
        raise MaskingUnavailableError(
            f"{what} cannot be masked; pass unknown_blocks='drop' to leave such blocks out"
        )


def _json_masked(turn: MaskedTurn, text: str) -> str:
    """Mask a text that may be a JSON object or array, keeping the JSON valid."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return turn.mask(text)
    if isinstance(parsed, dict | list):
        return json.dumps(turn.mask_data(parsed), ensure_ascii=False)
    return turn.mask(text)


def _without_participant(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("role") in _PARTICIPANT_ROLES and "name" in message:
        return {key: value for key, value in message.items() if key != "name"}
    return message


def _with_instructions(messages: list[dict[str, Any]], instructions: str) -> list[dict[str, Any]]:
    """Add the note about tokens to the first system message, or before the first message.

    Some chat templates allow one system message only and only first, so the note joins the one
    the client wrote rather than adding a second.
    """
    first = messages[0]
    if first.get("role") in ("system", "developer"):
        content = first.get("content")
        if isinstance(content, str):
            return [{**first, "content": f"{content}\n\n{instructions}"}, *messages[1:]]
        if isinstance(content, list):
            note = {"type": "text", "text": instructions}
            return [{**first, "content": [*content, note]}, *messages[1:]]
    return [{"role": "system", "content": instructions}, *messages]


def _restored(value: Any, mapping: Mapping[str, str], *, arguments: bool = False) -> Any:
    if isinstance(value, str):
        if arguments:
            return _TOKEN_RE.sub(lambda m: _escaped_value(mapping, m[0]), value)
        return _TOKEN_RE.sub(lambda m: mapping.get(m[0], m[0]), value)
    if isinstance(value, dict):
        return {
            key: _restored(item, mapping, arguments=arguments or key == "arguments")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_restored(item, mapping, arguments=arguments) for item in value]
    return value


def _escaped_value(mapping: Mapping[str, str], token: str) -> str:
    value = mapping.get(token)
    return token if value is None else _json_escaped(value)


def _json_escaped(value: str) -> str:
    """The value as it is written inside a JSON string: quotes, backslashes and controls escaped."""
    return json.dumps(value, ensure_ascii=False)[1:-1]


def _texts(value: Any) -> Iterable[str]:
    """Every string in a JSON-like value, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _texts(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _texts(item)


def _tokens_that_mask(text: str, values: Mapping[str, str]) -> list[str]:
    """The tokens whose values the masking rules would still find in `text`."""
    return [
        token for token, value in values.items() if mask_known_values(text, {token: value}) != text
    ]
