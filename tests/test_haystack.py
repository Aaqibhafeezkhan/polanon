"""The Haystack components: masked documents in the store, masked questions, restored answers.

They run where the haystack extra is installed and are skipped elsewhere. Nothing here uses the
network: the embedders and the generator are Haystack's own stand-ins.
"""

import asyncio
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import pytest

haystack = pytest.importorskip("haystack")

from haystack import Document, Pipeline  # noqa: E402
from haystack.components.builders import ChatPromptBuilder  # noqa: E402
from haystack.components.embedders import MockDocumentEmbedder  # noqa: E402
from haystack.components.generators.chat import MockChatGenerator  # noqa: E402
from haystack.components.preprocessors import DocumentSplitter  # noqa: E402
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever  # noqa: E402
from haystack.components.writers import DocumentWriter  # noqa: E402
from haystack.core.errors import PipelineRuntimeError, SerializationError  # noqa: E402
from haystack.dataclasses import ByteStream, ChatMessage, StreamingChunk  # noqa: E402
from haystack.document_stores.in_memory import InMemoryDocumentStore  # noqa: E402
from haystack.utils import Secret  # noqa: E402

from polanon import InMemoryMappingStore, Pseudonyms, StoredValue, tokens_in  # noqa: E402
from polanon.integrations.haystack import (  # noqa: E402
    PolanonDocumentMasker,
    PolanonQueryMasker,
    PolanonRestorer,
    PolanonStreamRestorer,
)

KEY = bytes(range(32))
CONTRACT = (
    "Umowę podpisał Jan Kowalski, PESEL 50810112354, mail jan.kowalski@example.com. "
    "Pełnomocnikiem Jana Kowalskiego jest Anna Nowak, tel. 600 100 200."
)
META = {"author": "Anna Nowak", "file": "umowa.txt", "tags": ["klient: Jan Kowalski"], "year": 2026}
#: Every value the documents name, in the spellings a leak could take.
RAW = [
    "Kowalsk",
    "KOWALSK",
    "kowalsk",
    "Anna Nowak",
    "Nowak",
    "50810112354",
    "jan.kowalski@example.com",
    "600 100 200",
    "600100200",
]


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("POLANON_KEY", KEY.hex())


@pytest.fixture
def mappings():
    return InMemoryMappingStore()


@pytest.fixture
def documents():
    return InMemoryDocumentStore(index=str(uuid.uuid4()))


def indexing(mappings, documents, splitter=None):
    pipeline = Pipeline()
    if splitter is not None:
        pipeline.add_component("split", splitter)
    pipeline.add_component("mask", PolanonDocumentMasker(store=mappings))
    pipeline.add_component("embed", MockDocumentEmbedder(dimension=8))
    pipeline.add_component("write", DocumentWriter(documents))
    if splitter is not None:
        pipeline.connect("split", "mask")
    pipeline.connect("mask", "embed")
    pipeline.connect("embed", "write")
    return pipeline


def index(mappings, documents, splitter=None, docs=None):
    pipeline = indexing(mappings, documents, splitter)
    first = "split" if splitter is not None else "mask"
    sources = docs or [Document(content=CONTRACT, meta=META)]
    pipeline.run({first: {"documents": sources}})
    return sources


def leaks(text):
    return [raw for raw in RAW if raw in text]


# --- what reaches the document store ---


def test_no_original_value_reaches_the_document_store(mappings, documents, tmp_path):
    splitter = DocumentSplitter(split_by="word", split_length=12, split_overlap=3)
    index(mappings, documents, splitter)
    stored = documents.filter_documents()
    assert len(stored) >= 2
    for document in stored:
        assert leaks(json.dumps(document.to_dict(), ensure_ascii=False)) == []
        assert document.embedding is not None  # embedded after masking, from masked text
    saved = tmp_path / "store.json"
    documents.save_to_disk(str(saved))
    assert leaks(saved.read_text(encoding="utf-8")) == []
    assert leaks(json.dumps(documents.to_dict(), ensure_ascii=False)) == []


def test_no_identifier_in_the_store_is_a_plain_hash_of_the_original_text(mappings, documents):
    """A splitter writes the original's id and its neighbours' ids into meta: SHA-256 of text."""
    sources = [Document(content=CONTRACT, meta=META)]
    passages = DocumentSplitter(split_by="word", split_length=12, split_overlap=3).run(
        documents=sources
    )["documents"]
    originals = {document.id for document in sources + passages}
    index(mappings, documents, docs=passages)
    stored = documents.filter_documents()
    dumped = json.dumps([document.to_dict() for document in stored])
    assert not [identifier for identifier in originals if identifier in dumped]
    ids = {document.id for document in stored}
    for document in stored:
        for overlap in document.meta.get("_split_overlap", []):
            assert overlap["doc_id"] in ids
    assert len({document.meta["source_id"] for document in stored}) == 1


def test_the_mapping_goes_to_the_mapping_store_and_never_into_meta(mappings, documents):
    index(mappings, documents)
    (stored,) = documents.filter_documents()
    tokens = tokens_in(stored.content)
    held = mappings.get(list(tokens))
    assert set(held) == tokens
    assert held[Pseudonyms(KEY).token("Jan Kowalski", "PERSON")].value == "Jan Kowalski"
    meta = json.dumps(stored.meta, ensure_ascii=False)
    for entry in held.values():
        assert entry.value not in meta
        assert entry.fingerprint not in meta


def test_strings_in_meta_are_masked_with_the_tokens_of_the_content(mappings, documents):
    index(mappings, documents)
    (stored,) = documents.filter_documents()
    attorney = Pseudonyms(KEY).token("Anna Nowak", "PERSON")
    assert stored.meta["author"] == attorney
    assert attorney in stored.content
    assert stored.meta["file"] == "umowa.txt"
    assert stored.meta["year"] == 2026


def test_the_documents_given_are_not_changed(mappings):
    given = Document(content=CONTRACT, meta=dict(META))
    before = given.to_dict()
    PolanonDocumentMasker(store=mappings).run(documents=[given])
    assert given.to_dict() == before


def test_known_values_given_at_run_time_are_masked(mappings):
    masker = PolanonDocumentMasker(store=mappings)
    (masked,) = masker.run(
        documents=[Document(content="Faktura dla Klub Orlik.")], known=["Klub Orlik"]
    )["documents"]
    assert "Orlik" not in masked.content


@pytest.mark.parametrize(
    "document",
    [
        Document(content="tekst", blob=ByteStream(b"PESEL 50810112354")),
        Document(content="Jan Kowalski", embedding=[0.1, 0.2]),
    ],
)
def test_a_document_made_from_the_original_in_another_form_is_refused(mappings, document):
    with pytest.raises(ValueError, match="document 0") as raised:
        PolanonDocumentMasker(store=mappings).run(documents=[document])
    assert "Kowalski" not in str(raised.value)
    assert "50810112354" not in str(raised.value)


# --- the question and the answer ---


def answer_from_passages(messages):
    """A stand-in model: names the tokens of the first passage in the prompt."""
    prompt = messages[-1].text
    passage = prompt.split("Fragmenty:", 1)[1]
    names = [token for token in sorted(tokens_in(passage)) if token.startswith("<OSOBA_")]
    return ChatMessage.from_assistant(f"Pełnomocnikiem jest {names[0]}.")


TEMPLATE = [
    ChatMessage.from_user(
        "Pytanie: {{ query }}\nFragmenty:\n{% for d in documents %}{{ d.content }}\n{% endfor %}"
    )
]


def asking(mappings, documents, generator=None):
    pipeline = Pipeline()
    pipeline.add_component("mask", PolanonQueryMasker(store=mappings))
    pipeline.add_component("retrieve", InMemoryBM25Retriever(documents, top_k=1))
    pipeline.add_component("prompt", ChatPromptBuilder(template=TEMPLATE))
    pipeline.add_component("llm", generator or MockChatGenerator(response_fn=answer_from_passages))
    pipeline.add_component("restore", PolanonRestorer(store=mappings))
    pipeline.connect("mask.query", "retrieve.query")
    pipeline.connect("mask.query", "prompt.query")
    pipeline.connect("retrieve.documents", "prompt.documents")
    pipeline.connect("retrieve.documents", "restore.documents")
    pipeline.connect("prompt.prompt", "llm.messages")
    pipeline.connect("llm.replies", "restore.replies")
    pipeline.connect("mask.mapping", "restore.mapping")
    return pipeline


QUESTION = "Kto jest pełnomocnikiem Jana Kowalskiego?"


def test_the_question_finds_the_passage_and_the_answer_comes_back_restored(mappings, documents):
    index(mappings, documents)
    result = asking(mappings, documents).run(
        {"mask": {"query": QUESTION}}, include_outputs_from={"llm", "prompt"}
    )
    prompt = result["prompt"]["prompt"][-1].text
    assert leaks(prompt) == []
    assert Pseudonyms(KEY).token("Jan Kowalski", "PERSON") in prompt
    (reply,) = result["restore"]["replies"]
    assert reply.text in {"Pełnomocnikiem jest Jan Kowalski.", "Pełnomocnikiem jest Anna Nowak."}
    assert leaks(result["llm"]["replies"][0].text) == []
    (found,) = result["restore"]["documents"]
    assert "Jan Kowalski" in found.content
    assert found.meta["author"] == "Anna Nowak"


def test_a_value_only_the_question_named_is_restored_from_its_mapping(mappings, documents):
    index(mappings, documents)
    masker = PolanonQueryMasker(store=mappings)
    question = masker.run(query="Czy Piotr Zieliński też podpisał?")
    stranger = Pseudonyms(KEY).token("Piotr Zieliński", "PERSON")
    assert stranger in question["query"]
    assert mappings.get([stranger]) == {}  # a question writes nothing to the store
    restored = PolanonRestorer(store=mappings).run(
        replies=[ChatMessage.from_assistant(f"Nie wiem nic o {stranger}.")],
        mapping=question["mapping"],
    )
    assert restored["replies"][0].text == "Nie wiem nic o Piotr Zieliński."


def test_tool_arguments_and_messages_from_others_are_handled(mappings, documents):
    index(mappings, documents)
    person = Pseudonyms(KEY).token("Jan Kowalski", "PERSON")
    from haystack.dataclasses import ToolCall

    call = ToolCall(tool_name="find", arguments={"who": person, "n": 3}, id="1")
    replies = [ChatMessage.from_assistant(tool_calls=[call]), ChatMessage.from_user(person)]
    restored = PolanonRestorer(store=mappings).run(replies=replies)["replies"]
    assert restored[0].tool_calls[0].arguments == {"who": "Jan Kowalski", "n": 3}
    assert restored[1].text == person  # only a model's reply is restored


# --- streaming ---


def test_a_streamed_answer_is_restored_through_the_callback(mappings, documents):
    index(mappings, documents)
    shown = []
    stream = PolanonStreamRestorer(mappings, on_text=shown.append)
    asking(mappings, documents).run(
        {"mask": {"query": QUESTION, "stream": stream}, "llm": {"streaming_callback": stream}}
    )
    text = "".join(shown)
    assert text in {"Pełnomocnikiem jest Jan Kowalski.", "Pełnomocnikiem jest Anna Nowak."}


def test_a_token_cut_between_chunks_is_held_back_and_restored(mappings, documents):
    index(mappings, documents)
    person = Pseudonyms(KEY).token("Anna Nowak", "PERSON")
    answer = f"Pełnomocnikiem jest {person}, potwierdzam."
    for cut in range(1, len(answer)):
        shown = []
        stream = PolanonStreamRestorer(mappings, on_text=shown.append)
        stream(StreamingChunk(content=answer[:cut]))
        stream(StreamingChunk(content=answer[cut:], finish_reason="stop"))
        assert "".join(shown) == "Pełnomocnikiem jest Anna Nowak, potwierdzam."


def test_pipeline_stream_can_be_restored_by_feeding_the_chunks(mappings, documents):
    index(mappings, documents)
    stream = PolanonStreamRestorer(mappings)

    async def run():
        handle = asking(mappings, documents).stream({"mask": {"query": QUESTION, "stream": stream}})
        shown = [stream.feed(chunk.content) async for chunk in handle]
        return "".join(shown) + stream.flush()

    assert asyncio.run(run()) in {
        "Pełnomocnikiem jest Jan Kowalski.",
        "Pełnomocnikiem jest Anna Nowak.",
    }


def test_an_unknown_token_streams_through_unchanged(mappings):
    shown = []
    stream = PolanonStreamRestorer(mappings, on_text=shown.append)
    stream(StreamingChunk(content="Nie znam <OSOBA_999999999999>.", finish_reason="stop"))
    assert shown == ["Nie znam <OSOBA_999999999999>."]


# --- the key and serialization ---


class RemoteStore(InMemoryMappingStore):
    """A store that serializes where it connects to, as an application's own would."""

    def __init__(self, url="postgresql://vault"):
        super().__init__()
        self.url = url

    def to_dict(self):
        return {"type": f"{__name__}.RemoteStore", "init_parameters": {"url": self.url}}

    @classmethod
    def from_dict(cls, data):
        return cls(**data["init_parameters"])


def test_a_pipeline_saves_the_key_by_its_variable_only_and_loads_back():
    pipeline = Pipeline()
    pipeline.add_component("mask", PolanonDocumentMasker(store=RemoteStore(), scope="kb"))
    pipeline.add_component("restore", PolanonRestorer(store=RemoteStore()))
    saved = pipeline.dumps()
    assert KEY.hex() not in saved
    assert "POLANON_KEY" in saved
    loaded = Pipeline.loads(saved, allowed_modules=["polanon.integrations.haystack", __name__])
    masker = loaded.get_component("mask")
    assert masker.scope == "kb"
    assert isinstance(masker.store, RemoteStore)


def test_a_store_that_cannot_be_saved_says_so(mappings):
    pipeline = Pipeline()
    pipeline.add_component("mask", PolanonQueryMasker(store=mappings))
    with pytest.raises(SerializationError, match="InMemoryMappingStore"):
        pipeline.dumps()


def test_a_missing_key_is_reported_when_masking_starts(mappings, monkeypatch):
    monkeypatch.delenv("POLANON_KEY")
    masker = PolanonQueryMasker(
        store=mappings, key=Secret.from_env_var("POLANON_KEY", strict=False)
    )
    with pytest.raises(ValueError, match="not set"):
        masker.warm_up()


def test_a_key_that_is_not_hexadecimal_is_refused_without_quoting_it(mappings):
    secret = "to-nie-jest-szesnastkowe-haslo"
    masker = PolanonQueryMasker(store=mappings, key=Secret.from_token(secret))
    with pytest.raises(ValueError, match="hexadecimal") as raised:
        masker.warm_up()
    assert secret not in str(raised.value)


def test_a_failure_in_the_store_stops_the_pipeline_without_quoting_the_document(documents):
    class Broken(InMemoryMappingStore):
        def get(self, tokens):
            raise RuntimeError("the vault is down")

    with pytest.raises(PipelineRuntimeError) as raised:
        index(Broken(), documents)
    assert leaks(str(raised.value)) == []
    assert documents.count_documents() == 0


def test_the_restorer_reads_a_stored_value_it_did_not_write(mappings):
    token = Pseudonyms(KEY).token("Firma Kot", "ORG")
    mappings.add({token: StoredValue("Firma Kot", "fingerprint")})
    restored = PolanonRestorer(store=mappings).run(documents=[Document(content=f"Klient {token}.")])
    assert restored["documents"][0].content == "Klient Firma Kot."


def test_the_question_s_own_values_restore_in_the_stream_and_nothing_else_is_shown_twice(mappings):
    stream = PolanonStreamRestorer(mappings)
    stream.remember({"<OSOBA_123456789012>": "Piotr Zieliński"})
    stream(StreamingChunk(content="Nie znam <OSOBA_1234"))  # no on_text: nothing to show it with
    assert stream.feed("56789012>.") == "Piotr Zieliński."
    lookup = stream._lookup
    assert len(lookup) == 1
    assert list(lookup) == ["<OSOBA_123456789012>"]
    assert lookup["<OSOBA_123456789012>"] == "Piotr Zieliński"
    with pytest.raises(KeyError):
        lookup["<OSOBA_999999999999>"]
    assert lookup.get("<OSOBA_999999999999>") is None  # asked once, remembered as unknown


def test_other_shapes_of_meta_and_reasoning_are_kept_and_restored(mappings):
    masker = PolanonDocumentMasker(store=mappings)
    (masked,) = masker.run(
        documents=[Document(content="Pismo.", meta={"people": ("Jan Kowalski",), "source_id": 7})]
    )["documents"]
    person = Pseudonyms(KEY).token("Jan Kowalski", "PERSON")
    assert masked.meta == {"people": (person,), "source_id": 7}
    reply = ChatMessage.from_assistant("Gotowe.", reasoning=f"Pytano o {person}.")
    (restored,) = PolanonRestorer(store=mappings).run(replies=[reply])["replies"]
    assert restored.reasoning.reasoning_text == "Pytano o Jan Kowalski."


# --- the order in a pipeline, and the examples ---


def test_masking_before_the_splitter_keeps_a_company_whole(mappings, documents):
    """Cut at full stops first, "Firmą Kot Sp. z o.o." fell apart and went out unmasked."""
    pipeline = Pipeline()
    pipeline.add_component("mask", PolanonDocumentMasker(store=mappings))
    pipeline.add_component("split", DocumentSplitter(split_by="period", split_length=1))
    pipeline.add_component("write", DocumentWriter(documents))
    pipeline.connect("mask", "split")
    pipeline.connect("split", "write")
    pipeline.run({"mask": {"documents": [Document(content="Umowa z Firmą Kot Sp. z o.o. trwa.")]}})
    stored = " ".join(document.content for document in documents.filter_documents())
    assert "Kot" not in stored
    assert "Sp." not in stored


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXAMPLES))
    return module


def test_the_offline_example_sends_tokens_only_and_restores_the_answer(capsys):
    outcome = load_example("haystack_rag").main()
    for document in outcome["stored"]:
        assert leaks(json.dumps(document.to_dict(), ensure_ascii=False)) == []
        assert "Kot" not in document.content
    assert leaks(outcome["prompt"]) == []
    assert outcome["answer"] == "Pełnomocnikiem jest Anna Nowak."
    assert "Pełnomocnikiem jest Anna Nowak." in capsys.readouterr().out
    assert any("Jan Kowalski" in document.content for document in outcome["passages"])


def test_the_paid_example_builds_its_pipelines_without_calling_anyone(monkeypatch):
    """It is not run: that would cost money. Building it needs neither a key nor the network."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    example = load_example("haystack_openai")
    indexing, asking = example.build(InMemoryDocumentStore(index="paid"), InMemoryMappingStore())
    assert list(indexing.graph.nodes) == ["split", "mask", "embed", "write"]
    assert "restore" in asking.graph.nodes
