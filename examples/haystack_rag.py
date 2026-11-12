"""A RAG pipeline on Haystack 3 that sees tokens only: masked documents in, a restored answer out.

Runs offline, with Haystack's stand-in embedders and generator:

    pip install "polanon[haystack]"
    python examples/haystack_rag.py

The indexing pipeline masks the documents before the splitter and the embedder, so the document
store and every vector hold tokens such as <OSOBA_548354014016>; what the tokens stand for goes to
the mapping store, which the application keeps apart, encrypted. The query pipeline masks the
question with the same key, so it meets the same tokens, and the answer is restored as it streams.
The stand-in embedder below turns a text into a vector of its tokens, which shows that the masked
question finds the passages about the same person; a real embedder does the same on masked text,
see haystack_openai.py.
"""

import hashlib
import os
import secrets

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import MockDocumentEmbedder, MockTextEmbedder
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from polanon import InMemoryMappingStore, token_instructions, tokens_in
from polanon.integrations.haystack import (
    PolanonDocumentMasker,
    PolanonQueryMasker,
    PolanonRestorer,
    PolanonStreamRestorer,
)

DOCUMENTS = [
    Document(
        content=(
            "Umowę serwisową z Firmą Kot Sp. z o.o. podpisał Jan Kowalski, PESEL 50810112354. "
            "Pełnomocnikiem Jana Kowalskiego w sprawach umowy jest Anna Nowak, tel. 600 100 200."
        ),
        meta={"title": "Umowa serwisowa", "author": "Anna Nowak"},
    ),
    Document(
        content="Notatka: Piotr Zieliński zgłosił awarię drukarki w biurze w Gdańsku.",
        meta={"title": "Zgłoszenie"},
    ),
]
QUESTION = "Kto jest pełnomocnikiem Jana Kowalskiego?"
DIMENSIONS = 64


def token_vector(text: str) -> list[float]:
    """A stand-in embedding: one dimension per token, so texts that share a token are close."""
    vector = [0.0] * DIMENSIONS
    for token in tokens_in(text):
        vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % DIMENSIONS] = 1.0
    return vector


def stand_in_model(messages: list[ChatMessage]) -> ChatMessage:
    """A stand-in model that answers from the passage it was given, with the tokens in it."""
    prompt = messages[-1].text or ""
    question, passages = prompt.split("Fragmenty:", 1)
    asked = tokens_in(question)
    for passage in passages.splitlines():
        people = [t for t in sorted(tokens_in(passage)) if t.startswith("<OSOBA_")]
        others = [person for person in people if person not in asked]
        if asked & set(people) and others:
            return ChatMessage.from_assistant(f"Pełnomocnikiem jest {others[0]}.")
    return ChatMessage.from_assistant("Nie znalazłem tego w dokumentach.")


def build(
    document_store: InMemoryDocumentStore, mappings: InMemoryMappingStore
) -> tuple[Pipeline, Pipeline]:
    """Return the indexing pipeline and the query pipeline, both masking with the same key.

    The masker comes before the splitter: a whole document is masked, so a name or a company is
    read where it stands, and passages cut later carry tokens only. Cut first at full stops,
    "Firmą Kot Sp. z o.o." fell apart and went out unmasked.
    """
    indexing = Pipeline()
    indexing.add_component("split", DocumentSplitter(split_by="period", split_length=1))
    indexing.add_component("mask", PolanonDocumentMasker(store=mappings))
    indexing.add_component("embed", MockDocumentEmbedder(embedding_fn=token_vector))
    indexing.add_component("write", DocumentWriter(document_store))
    indexing.connect("mask", "split")
    indexing.connect("split", "embed")
    indexing.connect("embed", "write")

    template = [
        ChatMessage.from_system(token_instructions("pl")),
        ChatMessage.from_user(
            "Pytanie: {{ query }}\nFragmenty:\n"
            "{% for document in documents %}{{ document.content }}\n{% endfor %}"
        ),
    ]
    asking = Pipeline()
    asking.add_component("mask", PolanonQueryMasker(store=mappings))
    asking.add_component("embed", MockTextEmbedder(embedding_fn=token_vector))
    asking.add_component("retrieve", InMemoryEmbeddingRetriever(document_store, top_k=2))
    asking.add_component("prompt", ChatPromptBuilder(template=template))
    asking.add_component("llm", MockChatGenerator(response_fn=stand_in_model))
    asking.add_component("restore", PolanonRestorer(store=mappings))
    asking.connect("mask.query", "embed.text")
    asking.connect("mask.query", "prompt.query")
    asking.connect("embed.embedding", "retrieve.query_embedding")
    asking.connect("retrieve.documents", "prompt.documents")
    asking.connect("retrieve.documents", "restore.documents")
    asking.connect("prompt.prompt", "llm.messages")
    asking.connect("llm.replies", "restore.replies")
    asking.connect("mask.mapping", "restore.mapping")
    return indexing, asking


def main() -> dict:
    """Index the documents, ask the question and print what each side sees."""
    # A key made up for this run. In production: 32 random bytes per company, in hexadecimal,
    # from a secret manager, and the same in every process that indexes or asks.
    os.environ.setdefault("POLANON_KEY", secrets.token_hex(32))
    document_store = InMemoryDocumentStore()
    mappings = InMemoryMappingStore()  # in production, an encrypted table of yours
    indexing, asking = build(document_store, mappings)

    indexing.run({"mask": {"documents": DOCUMENTS}})
    print("In the document store, tokens only:")
    for document in document_store.filter_documents():
        print("  ", document.content.strip(), "| author:", document.meta.get("author"))

    print("\nThe answer, restored as it streams:")
    stream = PolanonStreamRestorer(mappings, on_text=lambda text: print(text, end="", flush=True))
    result = asking.run(
        {"mask": {"query": QUESTION, "stream": stream}, "llm": {"streaming_callback": stream}},
        include_outputs_from={"prompt"},
    )
    print()
    prompt = result["prompt"]["prompt"][-1].text
    print("\nWhat the model was sent:\n  " + prompt.replace("\n", "\n  "))
    print("Passages, restored for the reader:")
    for document in result["restore"]["documents"]:
        print("  ", document.content.strip())
    return {
        "stored": document_store.filter_documents(),
        "prompt": prompt,
        "answer": result["restore"]["replies"][0].text,
        "passages": result["restore"]["documents"],
    }


if __name__ == "__main__":
    main()
