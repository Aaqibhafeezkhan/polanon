"""Masking for search indexes: the documents, the questions, and the store of what tokens hide.

A retrieval-augmented application masks its documents before they are embedded and stored, and
masks every question the same way, so the question finds the passages about the same person. That
takes stable tokens, from an :class:`~polanon.Anonymizer` built with
:class:`~polanon.pseudonyms.Pseudonyms`. The masked passages go to the embedder and the index; what
each token stands for goes to a :class:`MappingStore`, which the application keeps apart from the
index, encrypted. Nothing that names a value is written to the index.

A passage found later is restored with the store, or exactly as written with the mapping kept for
it, and so is a model's answer, also while it streams.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from polanon import _canonical
from polanon._canonical import base_form
from polanon._matching import Ranges, type_of
from polanon._view import detection_view
from polanon.guards.base import Guard
from polanon.policy import TOKEN_PATTERN
from polanon.pseudonyms import PseudonymCollisionError, Pseudonyms
from polanon.types import AnonymizationResult, MaskingUnavailableError, Span

if TYPE_CHECKING:
    from polanon import Anonymizer

_TOKEN_RE = re.compile(TOKEN_PATTERN)

#: Masking passes tried when the store reports collisions; each pass settles every one it reports.
_ATTEMPTS = 3

#: The types a text is looked up in the store for. Identifiers have checksums, labels and formats
#: the rules read anyway; a name in a question often has nothing but the store to go by.
_LOOKED_UP = ("PERSON", "ORG", "LOCATION")
#: The longest run of words looked up as one value.
_LONGEST = 4
_WORD = re.compile(r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*")
_GAP = re.compile(r"[ \t\u00a0]+")


@dataclass(frozen=True)
class StoredValue:
    """What a mapping store keeps for a token.

    Attributes:
        value: the value to show for the token: a person's name in the nominative, anything else
            as it was first written. Encrypt it at rest; it is the personal data.
        fingerprint: :meth:`Pseudonyms.fingerprint` of the value. It names nothing without the key,
            and two values with one token are told apart by it.
    """

    value: str
    fingerprint: str


class MappingStore(Protocol):
    """Where an application keeps what each token stands for, apart from its search index.

    Implement it over an encrypted table with a unique key on the token. ``add`` has to be atomic
    per token, so that two processes indexing at once cannot both claim a token for two values.
    """

    def get(self, tokens: Collection[str]) -> Mapping[str, StoredValue]:
        """Return what the store holds for each of `tokens` it has; leave the others out."""
        ...

    def add(self, entries: Mapping[str, StoredValue]) -> Mapping[str, StoredValue]:
        """Store each entry whose token is not held yet; return what is held for the others."""
        ...


class InMemoryMappingStore:
    """A :class:`MappingStore` in a dict, for tests, examples and a single process.

    It keeps the collision record too, so ``Pseudonyms(key, collisions=store.collisions)`` shares
    it. Everything is lost when the process ends.
    """

    def __init__(self) -> None:
        """Start empty."""
        self._entries: dict[str, StoredValue] = {}
        self._lock = threading.Lock()
        self.collisions: dict[str, str] = {}

    def get(self, tokens: Collection[str]) -> dict[str, StoredValue]:
        """Return what the store holds for each of `tokens` it has."""
        with self._lock:
            return {token: self._entries[token] for token in tokens if token in self._entries}

    def add(self, entries: Mapping[str, StoredValue]) -> dict[str, StoredValue]:
        """Store each entry whose token is not held yet; return what is held for the others."""
        held: dict[str, StoredValue] = {}
        with self._lock:
            for token, entry in entries.items():
                if token in self._entries:
                    held[token] = self._entries[token]
                else:
                    self._entries[token] = entry
        return held

    def __len__(self) -> int:
        """The number of tokens stored."""
        return len(self._entries)


@dataclass(frozen=True)
class MaskedChunks:
    """A document's passages, masked with one mapping.

    Attributes:
        texts: the masked passages, in order: what goes to the embedder and the index.
        mappings: for each passage, token -> the value as written in that passage. Kept with the
            passage (encrypted, outside the index), it restores the passage exactly as it was
            written, including the case a name stood in.
        mapping: every token of the document -> its first spelling.
    """

    texts: list[str]
    mappings: list[dict[str, str]]
    mapping: dict[str, str]


def tokens_in(*texts: str) -> set[str]:
    """Return every token that appears in `texts`, such as the passages a search found."""
    return {match.group(0) for text in texts for match in _TOKEN_RE.finditer(text)}


def stored_mapping(store: MappingStore, *texts: str) -> dict[str, str]:
    """Return token -> value for every token in `texts` that the store holds, to restore them.

    Add the mapping of the question on the left, so a value only the question named restores too:
    ``question.mapping | stored_mapping(store, question.anonymized, *passages)``.
    """
    return {token: entry.value for token, entry in store.get(tokens_in(*texts)).items()}


def mask_chunks(
    anonymizer: Anonymizer, chunks: str | Sequence[str], mapping: Mapping[str, str] | None = None
) -> MaskedChunks:
    """Mask the passages of one document with one mapping; see :meth:`Anonymizer.mask_chunks`."""
    texts = _chunks(chunks)
    _no_judge(anonymizer, "mask_chunks")
    running = dict(mapping or {})
    results: list[AnonymizationResult] = []
    sizes: list[int] = []
    for text in texts:
        result = _mask(anonymizer, text, running)
        running = dict(result.mapping)
        results.append(result)
        sizes.append(len(running))
    final = len(running)
    for index, text in enumerate(texts):
        if sizes[index] < final:
            # A value first recognised in a later passage is masked here too, as in a turn.
            results[index] = _mask(anonymizer, text, running)
            running = dict(results[index].mapping)
    return MaskedChunks(
        texts=[result.anonymized for result in results],
        mappings=[_as_written(result, running) for result in results],
        mapping=running,
    )


def index_chunks(
    anonymizer: Anonymizer,
    chunks: str | Sequence[str],
    store: MappingStore,
    *,
    lookup: bool = True,
) -> MaskedChunks:
    """Mask a document for an index and store its tokens; see :meth:`Anonymizer.index_chunks`."""
    pseudonyms = _pseudonyms(anonymizer, "index_chunks")
    texts = _chunks(chunks)
    for _ in range(_ATTEMPTS):
        masker = _with_store_hits(anonymizer, pseudonyms, texts, store) if lookup else anonymizer
        masked = mask_chunks(masker, texts)
        entries = _entries(pseudonyms, masked.mapping)
        held = store.add(entries)
        if not _record_collisions(pseudonyms, entries, held):
            _record_local_collisions(pseudonyms, masked.mapping)
            return masked
    raise PseudonymCollisionError(_TOO_MANY)


def mask_query(
    anonymizer: Anonymizer,
    text: str,
    store: MappingStore,
    mapping: Mapping[str, str] | None = None,
    *,
    lookup: bool = True,
) -> AnonymizationResult:
    """Mask a question to an index; see :meth:`Anonymizer.mask_query`."""
    pseudonyms = _pseudonyms(anonymizer, "mask_query")
    if not isinstance(text, str):
        raise TypeError(f"mask_query takes a str, not {type(text).__name__}")
    for _ in range(_ATTEMPTS):
        masker = _with_store_hits(anonymizer, pseudonyms, [text], store) if lookup else anonymizer
        result = _mask(masker, text, dict(mapping or {}))
        # Tokens the conversation held before keep their values, collision or not: masking the
        # question again cannot change them, and refusing every later turn would help nobody.
        new = {t: v for t, v in result.mapping.items() if t not in (mapping or {})}
        entries = _entries(pseudonyms, new)
        held = store.get(list(entries))
        if not _record_collisions(pseudonyms, entries, held):
            return result
    raise PseudonymCollisionError(_TOO_MANY)


_TOO_MANY = "the store kept reporting collisions; check that every process uses the same key"


def _chunks(chunks: str | Sequence[str]) -> list[str]:
    texts = [chunks] if isinstance(chunks, str) else list(chunks)
    for text in texts:
        if not isinstance(text, str):
            raise TypeError(f"a passage is a str, not {type(text).__name__}")
    return texts


def _no_judge(anonymizer: Anonymizer, method: str) -> None:
    if anonymizer.judge is not None:
        raise ValueError(
            f"{method} masks without an event loop, so it cannot run the stage-2 judge "
            f"{anonymizer.judge.name!r}: use an Anonymizer without a judge"
        )


def _pseudonyms(anonymizer: Anonymizer, method: str) -> Pseudonyms:
    _no_judge(anonymizer, method)
    if anonymizer.pseudonyms is None:
        raise ValueError(
            f"{method} needs stable tokens: build the Anonymizer with pseudonyms=Pseudonyms(key), "
            "or a question would never match the tokens of the documents"
        )
    return anonymizer.pseudonyms


def _mask(anonymizer: Anonymizer, text: str, mapping: dict[str, str]) -> AnonymizationResult:
    """Mask one text, refusing with MaskingUnavailableError instead of returning it half masked."""
    from polanon.turn import _failure  # the turn names failures the same way

    try:
        return anonymizer.anonymize_sync(text, mapping)
    except MaskingUnavailableError:
        raise
    except Exception as exc:
        raise MaskingUnavailableError(_failure(exc)) from exc


def _as_written(result: AnonymizationResult, mapping: Mapping[str, str]) -> dict[str, str]:
    """Return token -> the value as this text wrote it, for every token in its masked text."""
    written: dict[str, str] = {}
    shift = 0
    for span in result.spans:
        match = _TOKEN_RE.match(result.anonymized, span.start + shift)
        if match is None:  # pragma: no cover - every span was replaced by a token
            raise AssertionError("a masked span is not where its token is")
        written.setdefault(match.group(0), result.original[span.start : span.end])
        shift += len(match.group(0)) - (span.end - span.start)
    for token in tokens_in(result.anonymized):
        held = mapping.get(token)
        if token not in written and held is not None:
            written[token] = held  # a token the text held already, or a placeholder
    return written


def _entries(pseudonyms: Pseudonyms, mapping: Mapping[str, str]) -> dict[str, StoredValue]:
    """Return what the store keeps for each token of a mapping; placeholders are left out."""
    entries: dict[str, StoredValue] = {}
    for token, value in mapping.items():
        entity_type = type_of(token)
        if entity_type is None or value == token:
            continue
        entries[token] = StoredValue(
            base_form(value, entity_type), pseudonyms.fingerprint(value, entity_type)
        )
    return entries


def _record_collisions(
    pseudonyms: Pseudonyms, entries: Mapping[str, StoredValue], held: Mapping[str, StoredValue]
) -> bool:
    """Record every token the store holds for another value; return whether there was one."""
    found = False
    for token, entry in held.items():
        if token in entries and entry.fingerprint != entries[token].fingerprint:
            pseudonyms.record_collision(token, entry.fingerprint)
            found = True
    return found


def _record_local_collisions(pseudonyms: Pseudonyms, mapping: Mapping[str, str]) -> None:
    """Record the collisions one document settled itself, so every other text settles them alike.

    Two values of one document with one token were told apart there: the second got a longer token.
    Without a record, a question naming the second value alone would get the shorter one.
    """
    for token, value in mapping.items():
        entity_type = type_of(token)
        if entity_type is None or value == token:
            continue
        expected = pseudonyms.token(value, entity_type)
        owner = mapping.get(expected)
        if expected != token and owner is not None and owner != expected:
            pseudonyms.record_collision(expected, pseudonyms.fingerprint(owner, entity_type))


class _StoreHits(Guard):
    """Reports the values a text names that the mapping store already holds a token for."""

    name = "mapping_store"

    def __init__(self, hits: Mapping[str, list[Span]]) -> None:
        self._hits = hits

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return the hits found in this text, whatever types are enabled: they are known values."""
        return list(self._hits.get(text, ()))


def _with_store_hits(
    anonymizer: Anonymizer, pseudonyms: Pseudonyms, texts: Sequence[str], store: MappingStore
) -> Anonymizer:
    """Return the anonymizer with one more guard: the values of `texts` the store knows.

    A surname written alone, "Co podpisał Kowalczyk?", is a name no rule reads, and a question
    that sent it in the clear also found nothing. Its token is computed for every run of up to four
    words that could be a name, and the ones the store holds for the same value are masked like
    any detection, so the question gets the token the documents have. A capitalised word alone
    counts only when it is no bare first name, as a lowercase one never does.
    """
    from polanon import Anonymizer as Facade  # the facade imports this module

    wanted: dict[str, list[tuple[str, int, int, str]]] = {}
    for text in texts:
        view = detection_view(text)
        seen = text if view is None else view.text
        for start, end in _candidates(seen):
            for entity_type in _LOOKED_UP:
                token = pseudonyms.token(seen[start:end], entity_type)
                wanted.setdefault(token, []).append((seen, start, end, entity_type))
    held = store.get(list(wanted))
    hits: dict[str, list[Span]] = {}
    for token, entry in held.items():
        for seen, start, end, entity_type in wanted.get(token, ()):
            if pseudonyms.fingerprint(seen[start:end], entity_type) == entry.fingerprint:
                hits.setdefault(seen, []).append(
                    Span(start, end, entity_type, 0.9, _StoreHits.name)
                )
    if not hits:
        return anonymizer
    return Facade(
        [*anonymizer.guards, _StoreHits(hits)],
        enabled=anonymizer.enabled,
        allowlist=anonymizer.allowlist,
        pseudonyms=anonymizer.pseudonyms,
    )


def _candidates(text: str) -> list[tuple[int, int]]:
    """Return every run of one to four words of `text` that could be a name, outside its tokens."""
    tokens = Ranges(match.span() for match in _TOKEN_RE.finditer(text))
    words = [
        match for match in _WORD.finditer(text) if not tokens.overlaps(match.start(), match.end())
    ]
    runs: list[tuple[int, int]] = []
    for first, word in enumerate(words):
        for last in range(first, min(first + _LONGEST, len(words))):
            gap = text[words[last - 1].end() : words[last].start()]
            if last > first and _GAP.fullmatch(gap) is None:
                break
            if last > first or _could_be_named_alone(word.group(0)):
                runs.append((word.start(), words[last].end()))
    return runs


def _could_be_named_alone(word: str) -> bool:
    """Whether one word on its own could be a value: capitalised, and no bare first name."""
    if len(word) < 3 or not word[0].isupper():
        return False
    return bool(_canonical._surname(word)) or not _canonical._first_name(word)
