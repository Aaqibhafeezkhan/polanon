"""Haystack 3 components: mask documents and questions for a RAG pipeline, restore what comes back.

``pip install "polanon[haystack]"``. Documents are masked before the embedder with stable tokens
(see :mod:`polanon.pseudonyms`), so the document store and every embedding hold tokens only, and a
question masked with the same key meets the same tokens. What each token stands for goes to a
:class:`~polanon.MappingStore` the application keeps apart from the document store; it never goes
into a document's ``meta``. The answer of the generator and the documents found are restored for
the reader, also while the answer streams.

Pipelines built from these components in code need nothing more. To load one from YAML, allow the
module: ``Pipeline.loads(text, allowed_modules=["polanon.integrations.haystack"])``.
"""

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.core.errors import SerializationError
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.utils import Secret

from polanon import (
    Anonymizer,
    MappingStore,
    Pseudonyms,
    StreamRestorer,
    default_guards,
    stored_mapping,
)
from polanon.turn import _with_known_values

#: The environment variable the key is read from unless a component is given another Secret.
KEY_VARIABLE = "POLANON_KEY"

#: The type the identifiers of documents are fingerprinted under, so none is an unkeyed hash.
_DOCUMENT_ID = "DOCUMENT_ID"

#: Fields of ``meta`` that hold identifiers of documents, which are replaced rather than masked.
_ID_FIELDS = frozenset({"source_id", "_split_overlap"})

#: The longest token held back in a stream: twelve digits lengthened twice, behind a long label.
_LONGEST_TOKEN = 64


def _key(secret: Secret) -> bytes:
    """Read the key from a Secret holding it in hexadecimal, as ``secrets.token_hex(32)`` writes.

    The messages name the problem, never the value.
    """
    value = secret.resolve_value()
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "the polanon key is not set: put 32 random bytes in hexadecimal into the environment "
            f"variable {KEY_VARIABLE}, such as the output of python -c "
            '"import secrets; print(secrets.token_hex(32))"'
        )
    try:
        return bytes.fromhex(value.strip())
    except ValueError:
        raise ValueError(
            "the polanon key is not hexadecimal: write its bytes as secrets.token_hex(32) does"
        ) from None


def _store_to_dict(store: MappingStore) -> dict[str, Any]:
    to_dict = getattr(store, "to_dict", None)
    if not callable(to_dict):
        raise SerializationError(
            f"the mapping store {type(store).__name__} has no to_dict: build this pipeline in "
            "code, or give the store a to_dict that names where it connects, never what it holds"
        )
    serialized: dict[str, Any] = to_dict()
    return serialized


class _Masker:
    """What both maskers share: the key, the store, the settings and the anonymizer they build."""

    def __init__(
        self,
        store: MappingStore,
        *,
        key: Secret = Secret.from_env_var(KEY_VARIABLE),  # noqa: B008 - Haystack's own idiom
        scope: str = "",
        digits: int = 12,
        with_ner: bool = False,
        spacy_model: str = "pl_core_news_md",
    ) -> None:
        self.store = store
        self.key = key
        self.scope = scope
        self.digits = digits
        self.with_ner = with_ner
        self.spacy_model = spacy_model
        self._anonymizer: Anonymizer | None = None

    def warm_up(self) -> None:
        """Read the key and build the detectors, the language model too when it is asked for.

        It runs once; the pipeline calls it before a run, and ``run`` does when called alone.
        """
        if self._anonymizer is not None:
            return
        pseudonyms = Pseudonyms(
            _key(self.key),
            scope=self.scope,
            digits=self.digits,
            collisions=getattr(self.store, "collisions", None),
        )
        guards = default_guards(with_ner=self.with_ner, spacy_model=self.spacy_model)
        self._anonymizer = Anonymizer(guards, pseudonyms=pseudonyms)

    def _masking(self, known: list[str] | None) -> Anonymizer:
        self.warm_up()
        assert self._anonymizer is not None  # set by warm_up
        terms = [term for term in known or () if term.strip()]
        return _with_known_values(self._anonymizer, terms) if terms else self._anonymizer

    def to_dict(self) -> dict[str, Any]:
        """Serialize the settings; the key only by the name of its variable."""
        return default_to_dict(
            self,
            store=_store_to_dict(self.store),
            key=self.key.to_dict(),
            scope=self.scope,
            digits=self.digits,
            with_ner=self.with_ner,
            spacy_model=self.spacy_model,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Rebuild a masker from :meth:`to_dict`."""
        return default_from_dict(cls, data)


@component
class PolanonDocumentMasker(_Masker):
    """Masks documents before they are embedded and written, and stores what their tokens hide.

    Place it before the document embedder of an indexing pipeline. The passages of one document,
    told by the ``source_id`` a splitter gives them, are masked with one mapping, and every value
    goes to the mapping store with the surname of each person. The documents that come out are new
    ones: masked content, masked strings in ``meta``, an ``id`` computed from the masked content.
    Identifiers of other documents in ``meta`` are replaced the same way, since an unmasked
    document's ``id`` is a plain hash of its text.

    A document with a ``blob`` or an embedding is refused: both were made from the original.

    Usage:
        ``PolanonDocumentMasker(store=mappings)`` with the key in the ``POLANON_KEY`` variable.
    """

    @component.output_types(documents=list[Document])
    def run(
        self, documents: list[Document], known: list[str] | None = None
    ) -> dict[str, list[Document]]:
        """Mask the documents.

        Args:
            documents: the documents, or the passages a splitter made of them.
            known: values known to be sensitive, such as client names, masked wherever they stand.
                Given here rather than to the constructor, they never end up in a saved pipeline.

        Returns:
            ``documents``: the masked documents, in the same order.

        Raises:
            ValueError: for a document with a blob or an embedding.
            MaskingUnavailableError: when masking fails; none of the documents may be written.
        """
        for position, document in enumerate(documents):
            _check_maskable(document, position)
        anonymizer = self._masking(known)
        pseudonyms = anonymizer.pseudonyms
        assert pseudonyms is not None  # every masker is built with them
        contents: list[str | None] = [None] * len(documents)
        metas: list[dict[str, Any]] = [{} for _ in documents]
        for group in _by_source(documents):
            texts = [documents[index].content or "" for index in group]
            meta_texts = [_meta_strings(documents[index].meta) for index in group]
            flat = [text for strings in meta_texts for text in strings]
            result = anonymizer.index_chunks([*texts, *flat], self.store)
            rest = iter(result.texts[len(texts) :])
            masked = result.texts[: len(texts)]
            for index, content, strings in zip(group, masked, meta_texts, strict=True):
                original = documents[index]
                contents[index] = None if original.content is None else content
                metas[index] = _meta_replaced(original.meta, [next(rest) for _ in strings])
        return {"documents": _with_keyed_ids(documents, contents, metas, pseudonyms)}


class PolanonStreamRestorer:
    """Restores a streamed answer chunk by chunk, holding back a token cut between two chunks.

    Pass it as the generator's ``streaming_callback``, and to :class:`PolanonQueryMasker` as
    ``stream`` so it learns what the question's tokens stand for; the rest it looks up in the
    mapping store as the tokens arrive. It shows each restored piece with `on_text`. With
    ``Pipeline.stream()``, feed it the chunks yourself: ``print(stream.feed(chunk.content))``.
    Take a new one for every answer.
    """

    def __init__(self, store: MappingStore, on_text: Callable[[str], object] | None = None) -> None:
        """Take the mapping store and what to do with each restored piece of text.

        Args:
            store: the store the documents were indexed into.
            on_text: called with every piece ready to show, such as ``print`` with ``end=""``.
        """
        self._lookup = _LazyMapping(store)
        self._restorer = StreamRestorer(self._lookup)  # type: ignore[arg-type]
        # Tokens arrive after this is built, so the restorer cannot size what it holds back by them.
        self._restorer._max_partial = _LONGEST_TOKEN
        self.on_text = on_text

    def remember(self, mapping: Mapping[str, str]) -> None:
        """Learn what the tokens of the question stand for."""
        self._lookup.known.update(mapping)

    def feed(self, text: str) -> str:
        """Take one piece of the answer and return what can be shown now, restored."""
        return self._restorer.feed(text)

    def flush(self) -> str:
        """Return what is still held back at the end of the answer, restored."""
        return self._restorer.flush()

    def __call__(self, chunk: StreamingChunk) -> None:
        """Restore a chunk and show it; the last chunk of a reply also shows what was held back."""
        shown = self.feed(chunk.content or "")
        if chunk.finish_reason is not None:
            shown += self.flush()
        if shown and self.on_text is not None:
            self.on_text(shown)


class _LazyMapping(Mapping[str, str]):
    """The question's mapping, and the store asked for each token the first time it appears."""

    def __init__(self, store: MappingStore) -> None:
        self.store = store
        self.known: dict[str, str] = {}
        self._asked: dict[str, str | None] = {}

    def get(self, token: str, default: Any = None) -> Any:
        if token in self.known:
            return self.known[token]
        if token not in self._asked:
            held = self.store.get([token]).get(token)
            self._asked[token] = None if held is None else held.value
        found = self._asked[token]
        return default if found is None else found

    def __getitem__(self, token: str) -> str:
        found = self.get(token)
        if found is None:
            raise KeyError(token)
        result: str = found
        return result

    def __iter__(self) -> Iterator[str]:
        return iter(self.known)

    def __len__(self) -> int:
        return len(self.known)


@component
class PolanonQueryMasker(_Masker):
    """Masks a question with the tokens its documents were indexed under.

    Place it first in a query pipeline and send its ``query`` to the text embedder and the prompt
    builder; its ``mapping`` goes to :class:`PolanonRestorer`. A surname written alone, which no
    rule reads, is found through the mapping store.
    """

    @component.output_types(query=str, mapping=dict[str, str])
    def run(
        self,
        query: str,
        mapping: dict[str, str] | None = None,
        known: list[str] | None = None,
        stream: "PolanonStreamRestorer | None" = None,
    ) -> dict[str, Any]:
        """Mask the question.

        Args:
            query: the question as the user wrote it.
            mapping: token -> original value from earlier turns of the conversation.
            known: values known to be sensitive, masked wherever they stand.
            stream: the restorer of a streamed answer, which learns the question's values.

        Returns:
            ``query``, masked, and ``mapping``, what its tokens stand for.

        Raises:
            MaskingUnavailableError: when masking fails; nothing of the question may be sent.
        """
        result = self._masking(known).mask_query(query, self.store, mapping)
        if stream is not None:
            stream.remember(result.mapping)
        return {"query": result.anonymized, "mapping": dict(result.mapping)}


@component
class PolanonRestorer:
    """Puts the original values back into the replies of a generator and the documents found.

    What comes out is for the reader only: never write these documents back to a document store,
    and never send them to a model. Values come from the mapping store, a person's name in the
    nominative, and from the question's mapping for what only the question named.
    """

    def __init__(self, store: MappingStore) -> None:
        """Take the mapping store the documents were indexed into.

        Args:
            store: the application's store of what tokens hide.
        """
        self.store = store

    @component.output_types(replies=list[ChatMessage], documents=list[Document])
    def run(
        self,
        replies: list[ChatMessage] | None = None,
        documents: list[Document] | None = None,
        mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Restore the replies and the documents.

        Args:
            replies: the generator's replies.
            documents: the documents a retriever found.
            mapping: the question's mapping, from :class:`PolanonQueryMasker`.

        Returns:
            ``replies`` and ``documents``, with the original values in place of the tokens.
        """
        replies, documents = replies or [], documents or []
        texts = [text for reply in replies for text in _message_texts(reply)]
        for document in documents:
            texts.append(document.content or "")
            texts.extend(_meta_strings(document.meta))
        values = (mapping or {}) | stored_mapping(self.store, *texts)

        def restore(text: str) -> str:
            return Anonymizer.restore(text, values)

        return {
            "replies": [_restored_message(reply, restore) for reply in replies],
            "documents": [
                dataclasses.replace(
                    document,
                    content=None if document.content is None else restore(document.content),
                    meta=_meta_replaced(
                        document.meta, [restore(text) for text in _meta_strings(document.meta)]
                    ),
                )
                for document in documents
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the restorer with its store."""
        return default_to_dict(self, store=_store_to_dict(self.store))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolanonRestorer":
        """Rebuild a restorer from :meth:`to_dict`."""
        restorer: PolanonRestorer = default_from_dict(cls, data)
        return restorer


def _check_maskable(document: Document, position: int) -> None:
    """Refuse what masking text cannot cover; the message names the position, not the content."""
    if document.blob is not None:
        raise ValueError(
            f"document {position} has a blob, which holds the original file: mask documents "
            "converted to text, before anything else is attached to them"
        )
    if (
        document.embedding is not None
        or document.sparse_embedding is not None
        or getattr(document, "multivector_embedding", None) is not None
    ):
        raise ValueError(
            f"document {position} is embedded already, from its original text: place the "
            "masker before the embedder"
        )


def _by_source(documents: list[Document]) -> list[list[int]]:
    """Group the positions of documents that are passages of one source, in order."""
    groups: dict[object, list[int]] = {}
    for index, document in enumerate(documents):
        source = document.meta.get("source_id")
        key: object = ("source", source) if isinstance(source, str) and source else ("alone", index)
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def _strings(value: Any) -> Iterator[str]:
    """Every string inside a JSON-like value, in order; keys and other values are left alone."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _strings(item)


def _replaced(value: Any, strings: list[str]) -> Any:
    """Return `value` with its strings, in the order :func:`_strings` gives them, replaced."""
    pending = iter(strings)

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            return next(pending)
        if isinstance(item, Mapping):
            return {key: walk(inner) for key, inner in item.items()}
        if isinstance(item, list):
            return [walk(inner) for inner in item]
        if isinstance(item, tuple):
            return tuple(walk(inner) for inner in item)
        return item

    return walk(value)


def _meta_strings(meta: Mapping[str, Any]) -> list[str]:
    """The strings of a document's ``meta`` that are masked: all but the identifiers."""
    return [
        text for key, value in meta.items() if key not in _ID_FIELDS for text in _strings(value)
    ]


def _meta_replaced(meta: Mapping[str, Any], strings: list[str]) -> dict[str, Any]:
    """Return `meta` with the strings :func:`_meta_strings` gives replaced, in that order."""
    fields = {key: value for key, value in meta.items() if key not in _ID_FIELDS}
    replaced: dict[str, Any] = _replaced(fields, strings)
    return {key: replaced.get(key, value) for key, value in meta.items()}


def _with_keyed_ids(
    originals: list[Document],
    contents: list[str | None],
    metas: list[dict[str, Any]],
    pseudonyms: Pseudonyms,
) -> list[Document]:
    """Build the masked documents, with no identifier in them that is a plain hash of a text.

    A splitter writes the ``id`` of the document it split into ``source_id``, and the ``id`` of
    the neighbouring passages into ``_split_overlap``: each a plain SHA-256 of the original text,
    which a short document can be found from by hashing guesses. A passage masked here is pointed
    to by the ``id`` of its masked twin, any other document by a keyed fingerprint. As the splitter
    does, each ``id`` is computed before the overlaps are written in, so they can point both ways.
    """

    def keyed(identifier: object) -> object:
        if not isinstance(identifier, str):
            return identifier
        return pseudonyms.fingerprint(identifier, _DOCUMENT_ID)

    firsts: list[dict[str, Any]] = []
    for meta in metas:
        first = {key: value for key, value in meta.items() if key != "_split_overlap"}
        if "source_id" in first:
            first["source_id"] = keyed(first["source_id"])
        firsts.append(first)
    ids = [
        Document(content=content, meta=first).id
        for content, first in zip(contents, firsts, strict=True)
    ]
    twins = {original.id: new for original, new in zip(originals, ids, strict=True)}
    documents = []
    for original, content, meta, first, new in zip(
        originals, contents, metas, firsts, ids, strict=True
    ):
        overlap = meta.get("_split_overlap")
        if isinstance(overlap, list):
            first = {
                **first,
                "_split_overlap": [
                    {**item, "doc_id": twins.get(item["doc_id"]) or keyed(item["doc_id"])}
                    if isinstance(item, Mapping) and "doc_id" in item
                    else item
                    for item in overlap
                ],
            }
        documents.append(Document(id=new, content=content, meta=first, score=original.score))
    return documents


def _message_texts(message: ChatMessage) -> list[str]:
    texts = list(message.texts)
    for call in message.tool_calls:
        texts.extend(_strings(call.arguments))
    reasoning = message.reasoning
    if reasoning is not None:
        texts.append(reasoning.reasoning_text)
    return texts


def _restored_message(message: ChatMessage, restore: Callable[[str], str]) -> ChatMessage:
    """Return an assistant's message with its text, tool arguments and reasoning restored."""
    if not message.is_from(ChatRole.ASSISTANT):
        return message
    calls = [
        dataclasses.replace(
            call,
            arguments=_replaced(call.arguments, [restore(t) for t in _strings(call.arguments)]),
        )
        for call in message.tool_calls
    ]
    reasoning = message.reasoning
    return ChatMessage.from_assistant(
        text=None if message.text is None else restore(message.text),
        meta=dict(message.meta),
        name=message.name,
        tool_calls=calls or None,
        reasoning=None
        if reasoning is None
        else dataclasses.replace(reasoning, reasoning_text=restore(reasoning.reasoning_text)),
    )


__all__ = [
    "KEY_VARIABLE",
    "PolanonDocumentMasker",
    "PolanonQueryMasker",
    "PolanonRestorer",
    "PolanonStreamRestorer",
]
