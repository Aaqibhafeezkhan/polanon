"""Haystack 3 components: mask documents and questions for a RAG pipeline, restore what comes back.

``pip install "polanon[haystack]"``. Documents are masked before the embedder with stable tokens
(see :mod:`polanon.pseudonyms`), so the document store and every embedding hold tokens only, and a
question masked with the same key meets the same tokens. What each token stands for goes to a
:class:`~polanon.MappingStore` the application keeps apart from the document store; it never goes
into a document's ``meta``. The answer of the generator and the documents found are restored for
the reader, also while the answer streams.

Pipelines built from these components in code need nothing more. To load one from YAML, allow
this module and the module of the mapping store:
``Pipeline.loads(text, allowed_modules=["polanon.integrations.haystack", "myapp.vault"])``.
"""

import dataclasses
import re
from collections.abc import Callable, Iterator, Mapping
from typing import Any

try:
    from haystack import Document, component, default_from_dict, default_to_dict
    from haystack.core.errors import SerializationError
    from haystack.dataclasses import (
        ChatMessage,
        ChatRole,
        ReasoningContent,
        StreamingChunk,
        TextContent,
        ToolCall,
    )
    from haystack.utils import Secret
except ModuleNotFoundError as missing:  # pragma: no cover - the haystack job has it installed
    if missing.name is None or missing.name.split(".")[0] != "haystack":
        raise
    raise ModuleNotFoundError(
        'the Haystack components need Haystack 3: pip install "polanon[haystack]"',
        name="haystack",
    ) from None

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

#: A document's identifier as Haystack computes it: SHA-256 in hexadecimal, of the text and meta.
_HASH = re.compile(r"[0-9a-f]{64}")

#: What a keyed fingerprint of an identifier starts with, so that it never reads as a hash again.
_KEYED = "polanon:"

#: Fields of ``meta`` that name the document a passage was cut from, in Haystack's splitters.
_SOURCE_FIELDS = ("source_id", "parent_id", "__parent_id")

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
        if isinstance(known, str):
            # The value itself stays out of the message: it is what the caller wants kept hidden.
            raise TypeError("known takes a list of values, not a single string")
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

    Place it before the splitter and the embedder of an indexing pipeline, where it reads whole
    documents: a splitter can cut a value in two, and a piece is not always recognised. Passages
    that come split already are masked together when they name one source in ``source_id``,
    ``parent_id`` or ``__parent_id``. Every value goes to the mapping store, with the surname of
    each person.

    The documents that come out are new ones: tokens in the content and in every string of
    ``meta``, nested keys and the members of sets included; the field names of ``meta`` stay. A
    string shaped like a SHA-256 is taken for the identifier of a document, since Haystack computes
    each ``id`` as a plain hash of the original text: it is pointed to the masked twin of that
    document when the twin is in the same batch, and replaced by a keyed fingerprint otherwise. An
    ``id`` Haystack computed is computed again from the masked document, and one the caller chose is
    masked like text, so writing with ``DuplicatePolicy.OVERWRITE`` and deleting by ``id`` keep
    working. Indexing documents this masker wrote changes nothing, when all the passages of a
    document are in one batch.

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
        meta_texts: list[list[str]] = [[] for _ in documents]
        chosen: list[str | None] = [None] * len(documents)
        for group in _by_source(documents):
            texts = [documents[index].content or "" for index in group]
            pieces = [_masked_pieces(documents[index]) for index in group]
            flat = [text for strings in pieces for text in strings]
            result = anonymizer.index_chunks([*texts, *flat], self.store)
            rest = iter(result.texts[len(texts) :])
            masked = result.texts[: len(texts)]
            for index, content, strings in zip(group, masked, pieces, strict=True):
                original = documents[index]
                contents[index] = None if original.content is None else content
                done = [next(rest) for _ in strings]
                if _chosen_id(original):
                    chosen[index] = done.pop()
                meta_texts[index] = done
        return {"documents": _with_keyed_ids(documents, contents, meta_texts, chosen, pseudonyms)}


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

    def __copy__(self) -> "PolanonStreamRestorer":
        """Stay one object: Haystack copies what each component gets, and this one is shared."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "PolanonStreamRestorer":
        """Stay one object, so the question's mapping reaches the generator's callback."""
        return self

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
    """The store, asked for each token the first time it appears, and the question's mapping.

    The store comes first, with a person's name in the nominative, as :class:`PolanonRestorer`
    gives it; the question's mapping holds what only the question named.
    """

    def __init__(self, store: MappingStore) -> None:
        self.store = store
        self.known: dict[str, str] = {}
        self._asked: dict[str, str | None] = {}

    def get(self, token: str, default: Any = None) -> Any:
        if token not in self._asked:
            self._asked[token] = stored_mapping(self.store, token).get(token)
        found = self._asked[token]
        if found is not None:
            return found
        return self.known.get(token, default)

    def __getitem__(self, token: str) -> str:
        found = self.get(token)
        if found is None:
            raise KeyError(token)
        result: str = found
        return result

    def __iter__(self) -> Iterator[str]:
        return iter(self.known)

    def __deepcopy__(self, memo: dict[int, Any]) -> "_LazyMapping":
        return self

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
            texts.extend(_texts(document.meta, top=True))
        values = (mapping or {}) | stored_mapping(self.store, *texts)

        def restore(text: str) -> str:
            return Anonymizer.restore(text, values)

        return {
            "replies": [_restored_message(reply, restore) for reply in replies],
            "documents": [
                dataclasses.replace(
                    document,
                    content=None if document.content is None else restore(document.content),
                    meta=_rebuilt(
                        document.meta,
                        iter([restore(text) for text in _texts(document.meta, top=True)]),
                        _unchanged,
                        top=True,
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
        source = next(
            (
                document.meta[field]
                for field in _SOURCE_FIELDS
                if isinstance(document.meta.get(field), str) and document.meta[field]
            ),
            None,
        )
        key: object = ("source", source) if source is not None else ("alone", index)
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def _is_identifier(value: str) -> bool:
    """Whether a string is the identifier of a document: a SHA-256, or a keyed fingerprint."""
    return _HASH.fullmatch(value.removeprefix(_KEYED)) is not None


def _texts(value: Any, *, top: bool = False) -> Iterator[str]:
    """Every string of a JSON-like value that is text to mask or restore, in order.

    Identifiers of documents are left to :func:`_rebuilt`. The keys of a nested mapping are text,
    and so are the members of a set; the keys of ``meta`` itself are the names of its fields.
    """
    if isinstance(value, str):
        if not _is_identifier(value):
            yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if not top and isinstance(key, str) and not _is_identifier(key):
                yield key
            yield from _texts(item)
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            yield from _texts(item)


def _rebuilt(
    value: Any, texts: Iterator[str], identifier: Callable[[str], str], *, top: bool = False
) -> Any:
    """Return `value` with its texts, in the order :func:`_texts` gives them, taken from `texts`.

    Identifiers of documents go through `identifier` instead.
    """
    if isinstance(value, str):
        return identifier(value) if _is_identifier(value) else next(texts)
    if isinstance(value, Mapping):
        rebuilt: dict[Any, Any] = {}
        for key, item in value.items():
            if not top and isinstance(key, str):
                key = identifier(key) if _is_identifier(key) else next(texts)
            rebuilt[key] = _rebuilt(item, texts, identifier)
        return rebuilt
    if isinstance(value, list):
        return [_rebuilt(item, texts, identifier) for item in value]
    if isinstance(value, tuple):
        return tuple(_rebuilt(item, texts, identifier) for item in value)
    if isinstance(value, set | frozenset):
        return type(value)(_rebuilt(item, texts, identifier) for item in value)
    return value


def _unchanged(identifier: str) -> str:
    return identifier


def _chosen_id(document: Document) -> bool:
    """Whether the caller chose a document's ``id``: Haystack's own is always a SHA-256."""
    return not _is_identifier(document.id)


def _masked_pieces(document: Document) -> list[str]:
    """The texts of a document's ``meta`` to mask, and last its ``id`` if the caller chose it."""
    pieces = list(_texts(document.meta, top=True))
    if _chosen_id(document):
        pieces.append(document.id)
    return pieces


def _with_keyed_ids(
    originals: list[Document],
    contents: list[str | None],
    meta_texts: list[list[str]],
    chosen: list[str | None],
    pseudonyms: Pseudonyms,
) -> list[Document]:
    """Build the masked documents, with no identifier in them that is a plain hash of a text.

    A splitter writes the ``id`` of the document it split, and of the neighbouring passages, into
    ``meta``: each a plain SHA-256 of an original text, which a short document can be found from
    by hashing guesses. An identifier of a document in the batch points to its masked twin, and any
    other one becomes a keyed fingerprint. As Haystack's splitters do, each ``id`` is computed
    before the identifiers of other documents in the batch are written in, so that they can point
    both ways, and so that indexing the masked documents again gives the same ones.
    """
    inside = {original.id for original in originals}

    def keyed(identifier: str) -> str:
        if identifier.startswith(_KEYED):
            return identifier
        return _KEYED + pseudonyms.fingerprint(identifier, _DOCUMENT_ID)

    def before(identifier: str) -> str:
        return "" if identifier in inside else keyed(identifier)

    ids = [
        given
        if given is not None
        else Document(
            content=content, meta=_rebuilt(original.meta, iter(texts), before, top=True)
        ).id
        for original, content, texts, given in zip(
            originals, contents, meta_texts, chosen, strict=True
        )
    ]
    twins = {original.id: new for original, new in zip(originals, ids, strict=True)}

    def after(identifier: str) -> str:
        return twins.get(identifier) or keyed(identifier)

    return [
        Document(
            id=new,
            content=content,
            meta=_rebuilt(original.meta, iter(texts), after, top=True),
            score=original.score,
        )
        for original, content, texts, new in zip(originals, contents, meta_texts, ids, strict=True)
    ]


def _message_texts(message: ChatMessage) -> list[str]:
    texts: list[str] = []
    for part in message._content:
        if isinstance(part, TextContent):
            texts.append(part.text)
        elif isinstance(part, ToolCall):
            texts.extend(_texts(part.arguments, top=True))
        elif isinstance(part, ReasoningContent):
            texts.append(part.reasoning_text)
    return texts


def _restored_message(message: ChatMessage, restore: Callable[[str], str]) -> ChatMessage:
    """Return an assistant's message with every text, tool argument and reasoning restored.

    Every part is kept, in its order: a reply of several texts, or with an image, keeps them all.
    """
    if not message.is_from(ChatRole.ASSISTANT):
        return message

    def part_restored(part: Any) -> Any:
        if isinstance(part, TextContent):
            return TextContent(restore(part.text))
        if isinstance(part, ToolCall):
            restored = [restore(text) for text in _texts(part.arguments, top=True)]
            arguments = _rebuilt(part.arguments, iter(restored), _unchanged, top=True)
            return dataclasses.replace(part, arguments=arguments)
        if isinstance(part, ReasoningContent):
            return dataclasses.replace(part, reasoning_text=restore(part.reasoning_text))
        return part

    return dataclasses.replace(message, _content=[part_restored(p) for p in message._content])


__all__ = [
    "KEY_VARIABLE",
    "PolanonDocumentMasker",
    "PolanonQueryMasker",
    "PolanonRestorer",
    "PolanonStreamRestorer",
]
