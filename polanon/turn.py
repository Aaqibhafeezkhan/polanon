"""One call to a model, masked: everything it sends and everything it gets back.

A call is rarely one text. A chat turn sends the history, the new question and the results of the
tools the model asks for, and it gets back an answer, often streamed, and the arguments of those
tool calls. :class:`MaskedTurn` masks all of it with one mapping, so a person keeps one token from
the history to the last tool result, and it refuses with
:class:`~polanon.types.MaskingUnavailableError` instead of letting anything out unmasked.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from polanon._matching import type_of
from polanon.guards.denylist import DenylistGuard
from polanon.output_sweep import SweepReport
from polanon.stream_restore import StreamRestorer, restored, tokens_written
from polanon.types import MaskingUnavailableError

if TYPE_CHECKING:
    from polanon import Anonymizer

#: What :meth:`MaskedTurn.mask_messages` does with a block no text mask reaches, such as an image,
#: a document or a model's signed reasoning: refuse the whole list, leave the block out, or keep it
#: as it is, unmasked, for a caller who accepts that the model sees it.
UnknownBlocks = Literal["raise", "drop", "keep"]

_Text = Callable[[str], str]

#: How many times, at most, the items of one call are masked. Masking an item again with a grown
#: mapping can make it grow once more: a piece left over from a losing detection becomes a value
#: of its own. Each pass masks again only what was masked before the mapping last grew.
MASKING_PASSES = 4

_Item = TypeVar("_Item")
_Masked = TypeVar("_Masked")


def settled(
    originals: list[_Item], mask: Callable[[_Item], _Masked], size: Callable[[], int]
) -> list[_Masked]:
    """Mask every item, then mask again each one the mapping grew during or after.

    An item is masked once more when the mapping grew while it was masked, not only after: a tool
    result held a name first in a list of tags, where nothing recognised it, and then in a sentence
    that did, so the tags had been masked before the name was known. The guards remember what they
    found in a text, so masking an item again costs little more than the rules.

    Args:
        originals: the items, oldest first; they are not modified.
        mask: masks one item from its original with the turn's whole mapping as it is now.
        size: how many values the turn's mapping holds now.
    """
    masked, sizes = [], []
    for original in originals:
        sizes.append(size())
        masked.append(mask(original))
    for _ in range(MASKING_PASSES - 1):
        final = size()
        stale = [index for index, before in enumerate(sizes) if before < final]
        if not stale:
            break
        for index in stale:
            sizes[index] = size()
            masked[index] = mask(originals[index])
    return masked


#: What the model is told about the tokens, in the language it is asked to answer in.
_INSTRUCTIONS = {
    "pl": (
        "Dane osobowe w tej rozmowie, także w wynikach narzędzi, są zastąpione tokenami "
        "w nawiasach ostrych, takimi jak <OSOBA_1>, <EMAIL_1> albo <FIRMA_1>. Traktuj token jak "
        "prawdziwą wartość. Gdy przekazujesz go do narzędzia albo cytujesz w odpowiedzi, przepisz "
        "go dokładnie tak, jak jest zapisany: bez odmiany, bez zmiany liter i bez spacji w środku. "
        "Token z przypadkiem po dwukropku, taki jak <OSOBA_1:celownik>, to ta sama osoba co "
        "<OSOBA_1>, wymieniona w tym przypadku. "
        "Do tokenu osoby dopisz po dwukropku przypadek, w jakim osoba stoi w Twoim zdaniu, "
        "a imię i nazwisko wrócą odmienione: „Szanowny Panie <OSOBA_1:wołacz>”, „wniosek "
        "<OSOBA_1:dopełniacz>”, „rozmawiałem z <OSOBA_1:narzędnik>”. W argumentach narzędzi "
        "pisz <OSOBA_1:mianownik>. "
        "Nie twórz nowych tokenów i nie zgaduj, co się pod nimi kryje."
    ),
    "en": (
        "Personal data in this conversation, tool results included, is replaced by tokens in "
        "angle brackets, such as <OSOBA_1>, <EMAIL_1> or <FIRMA_1>. Treat a token as the real "
        "value. A person's token may carry a Polish grammatical case after a colon: it is the "
        "same person as the token without it. When you pass one to a tool or quote it in your "
        "answer, copy it exactly as written: do not inflect it, change its letters or put "
        "spaces inside. Do not create new tokens, and never guess the value behind one."
    ),
}


def token_instructions(language: str = "pl") -> str:
    """Return a note for the system prompt that tells the model what the tokens are.

    A model that is not told tends to inflect a token, translate its label or make one up, and the
    answer then restores with a value missing.

    Args:
        language: ``"pl"`` or ``"en"``, the language the model is asked to answer in.

    Raises:
        ValueError: for any other language.
    """
    try:
        return _INSTRUCTIONS[language]
    except KeyError:
        raise ValueError(
            f"the token instructions are written in {' and '.join(_INSTRUCTIONS)}, not {language!r}"
        ) from None


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
                ``"raise"`` refuses, ``"drop"`` leaves the block out, ``"keep"`` sends it as it
                is, unmasked: an image, a document and whatever text they carry.

        Raises:
            TypeError: when `known` is a single string, which would be read letter by letter.
            ValueError: when the anonymizer has a stage-2 judge, which needs an event loop, or
                for an `unknown_blocks` that is not "raise", "drop" or "keep".
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
        if unknown_blocks not in ("raise", "drop", "keep"):
            raise ValueError(f"unknown_blocks is 'raise', 'drop' or 'keep', not {unknown_blocks!r}")
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

        A value recognised in one string is masked in every other one, also in a string that
        comes before it, such as a name in a list of tags ahead of the sentence that names it.
        """
        return settled([value], lambda item: _walk(item, self.mask, default), self._size)[0]

    def mask_messages(self, messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Mask a list of chat messages written for the Anthropic or the OpenAI API.

        The text of every message is masked, and so are the input of a ``tool_use`` block, the
        content of a ``tool_result`` block or of a ``"tool"`` message, the arguments of
        ``tool_calls``, a ``refusal``, and the reasoning a provider such as OpenRouter returns next
        to an answer: ``reasoning``, ``reasoning_content`` and the text and summary of every item
        of ``reasoning_details``, whose encrypted data passes as it is. A tool result or arguments
        written as JSON stay valid JSON. Roles, ids, tool names and every other field are copied
        as they are, and the citations of a text block are left out, since what they quote may be
        personal data. A block no text mask reaches, such as an image, a document or a model's
        signed reasoning, is refused, left out or kept as it is, as `unknown_blocks` says.

        A value first recognised in a later message, such as the question, is also masked in the
        earlier ones, where nothing recognised it on its own, and so is one recognised late in a
        message, such as a tool result naming somebody in its tags before a sentence does: a
        message the mapping grew during or after is masked once more, from its original, with the
        whole mapping, until the mapping stops growing or :data:`MASKING_PASSES` passes are done.

        Args:
            messages: plain dictionaries, oldest first. They are not modified.

        Raises:
            MaskingUnavailableError: when masking fails, and for a block that cannot be masked
                while `unknown_blocks` is "raise". Nothing of the list may be sent then.
        """
        originals = list(messages)
        return settled(originals, lambda message: self._message(message, self.mask), self._size)

    def _size(self) -> int:
        return len(self._mapping)

    def restore(self, text: str, *, as_held: bool = False) -> str:
        """Put the original values back in place of the turn's tokens; unknown tokens stay.

        A person's token with a case after a colon, as the Polish note asks for, comes back
        declined: ``<OSOBA_1:wołacz>`` is "Janie Kowalski" where the text said "Jana Kowalskiego".
        A name the rules cannot decline for sure comes back as written, and so does any other
        value; the case is dropped.

        Args:
            text: an answer, or any text with the turn's tokens.
            as_held: give every value as it was masked, the case dropped, as
                :meth:`restore_data` describes.
        """
        return restored(text, self._mapping, as_held=as_held)

    def restore_data(self, value: Any, *, as_held: bool = False) -> Any:
        """Restore every string inside a JSON-like value, such as the arguments of a tool call.

        Keys and values that are not strings are left as they are, and so is the value itself.

        Args:
            value: the value to restore; it is not modified.
            as_held: give every value as it was masked, and drop a case the model added. A tool
                that looks a name up in the records it was masked from needs it written as those
                records hold it: ``<OSOBA_1:dopełniacz>`` is then "Jan Kowalski" when the records
                said "Jan Kowalski", not "Jana Kowalskiego". A value first written in a message,
                not in the records, is held in that message's case: after "Umów mnie z Janem
                Kowalskim" it is "Janem Kowalskim". For such a value leave `as_held` off: the
                Polish note asks for ``<OSOBA_1:mianownik>`` in tool arguments, which the
                default restores as "Jan Kowalski".
        """
        if isinstance(value, str):
            return self.restore(value, as_held=as_held)
        if isinstance(value, Mapping):
            return {key: self.restore_data(item, as_held=as_held) for key, item in value.items()}
        if isinstance(value, list):
            return [self.restore_data(item, as_held=as_held) for item in value]
        if isinstance(value, tuple):
            return tuple(self.restore_data(item, as_held=as_held) for item in value)
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
        return sorted(set(tokens_written(text)) - self._mapping.keys())

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
        for field in _REASONING_FIELDS:
            if isinstance(message.get(field), str):
                out[field] = text(message[field])
        if isinstance(message.get("reasoning_details"), list):
            out["reasoning_details"] = [
                _reasoning_detail(detail, text) for detail in message["reasoning_details"]
            ]
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
        return self._unmaskable(block, f"a {kind!r} block")

    def _tool_content(self, content: Any, text: _Text) -> Any:
        """What a tool returned: a text, often JSON, or a list of blocks."""
        if isinstance(content, str):
            return _json_text(content, text)
        return self._content(content, text)

    def _tool_call(self, call: Any, text: _Text) -> dict[str, Any] | None:
        """One OpenAI tool call, with its arguments masked."""
        if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
            return self._unmaskable(call, "a tool call that is not a function call")
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

    def _unmaskable(self, value: Any, what: str) -> dict[str, Any] | None:
        """Something no text mask reaches: refused, left out (None), or kept as it is."""
        if self._unknown_blocks == "raise":
            raise MaskingUnavailableError(
                f"{what} cannot be masked; pass unknown_blocks='drop' to leave such blocks out"
            )
        if self._unknown_blocks == "keep" and isinstance(value, Mapping):
            return dict(value)
        return None


#: Fields in which a provider such as OpenRouter or DeepSeek returns a model's reasoning as text,
#: next to the answer. A client that keeps the answer sends them back with the history.
_REASONING_FIELDS = ("reasoning", "reasoning_content")


def _reasoning_detail(detail: Any, text: _Text) -> Any:
    """One item of ``reasoning_details``: its text and summary masked, encrypted data as it is."""
    if not isinstance(detail, Mapping):
        return detail
    out = dict(detail)
    for field in ("text", "summary"):
        if isinstance(detail.get(field), str):
            out[field] = text(detail[field])
    return out


def _walk(value: Any, text: _Text, default: Callable[[Any], Any] | None = None) -> Any:
    """Apply `text` to every string in a JSON-like value, keys and other scalars untouched."""
    if isinstance(value, str):
        return text(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Mapping):
        return {key: _under_key(key, item, text, default) for key, item in value.items()}
    if isinstance(value, list):
        return [_walk(item, text, default) for item in value]
    if isinstance(value, tuple):
        return tuple(_walk(item, text, default) for item in value)
    if default is None:
        raise TypeError(
            f"cannot mask a {type(value).__name__}: convert it first, or pass default=str"
        )
    return _walk(default(value), text)


#: Keys that name a birth date, written without spaces, "_" or "-" and in lowercase.
_BIRTH_DATE_KEYS = frozenset(
    {"birthday", "birthdate", "dateofbirth", "dob", "dataurodzenia", "urodziny", "geburtsdatum"}
)
#: The label a value under such a key is read after: a rule masks a date after it as a birth date.
_BIRTH_DATE_LABEL = "data urodzenia: "


def _under_key(key: Any, item: Any, text: _Text, default: Callable[[Any], Any] | None) -> Any:
    """Mask a dictionary's value, a key that names a birth date read as the label before it.

    A contact book an agent read held a birthday under the key "birthday", and the guards, shown
    the value alone, let the date go out: a date is no birth date without its label. If the
    masking ever touched the label, the value is masked alone, as before.
    """
    if isinstance(key, str) and isinstance(item, str):
        name = "".join(char for char in key if char not in " _-").lower()
        if name in _BIRTH_DATE_KEYS:
            masked = text(_BIRTH_DATE_LABEL + item)
            if masked.startswith(_BIRTH_DATE_LABEL):
                return masked[len(_BIRTH_DATE_LABEL) :]
    return _walk(item, text, default)


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
        pseudonyms=anonymizer.pseudonyms,
    )


def _failure(exc: Exception, what: str = "the text") -> str:
    """Name a failure without quoting anything: an error's own message may carry the text."""
    return (
        f"masking {what} failed with {type(exc).__name__}, so nothing may be sent to the model; "
        "the error is chained as the cause"
    )
