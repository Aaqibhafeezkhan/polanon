"""PAID: the same RAG pipeline with OpenAI's embedder and chat model. Every run costs money.

    pip install "polanon[haystack]"
    export OPENAI_API_KEY=...   # billed to your OpenAI account
    export POLANON_KEY=...      # python -c "import secrets; print(secrets.token_hex(32))"
    python examples/haystack_openai.py

Only masked text leaves the process: the documents are masked before OpenAI embeds them, and the
question and the passages the model reads hold tokens instead of names, numbers and addresses.
That is the reason to mask before the embedder: a cloud embedder sees the text it is given, and a
vector can be turned back into an approximation of it. The models are Haystack's defaults; pass
`model=` to choose others. This recipe is not run in CI, since it needs a paid account; its offline
twin, haystack_rag.py, is.
"""

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from polanon import InMemoryMappingStore, token_instructions
from polanon.integrations.haystack import (
    PolanonDocumentMasker,
    PolanonQueryMasker,
    PolanonRestorer,
    PolanonStreamRestorer,
)

try:  # run as a script from the repository
    from haystack_rag import DOCUMENTS, QUESTION
except ImportError:  # imported as examples.haystack_openai
    from examples.haystack_rag import DOCUMENTS, QUESTION


def build(
    document_store: InMemoryDocumentStore, mappings: InMemoryMappingStore
) -> tuple[Pipeline, Pipeline]:
    """Return the indexing and the query pipeline, OpenAI's models in place of the stand-ins."""
    indexing = Pipeline()
    indexing.add_component("split", DocumentSplitter(split_by="period", split_length=2))
    indexing.add_component("mask", PolanonDocumentMasker(store=mappings))
    indexing.add_component("embed", OpenAIDocumentEmbedder())
    indexing.add_component("write", DocumentWriter(document_store))
    indexing.connect("mask", "split")
    indexing.connect("split", "embed")
    indexing.connect("embed", "write")

    template = [
        ChatMessage.from_system(
            "Odpowiadasz na podstawie fragmentów dokumentów.\n\n" + token_instructions("pl")
        ),
        ChatMessage.from_user(
            "Fragmenty:\n{% for document in documents %}{{ document.content }}\n{% endfor %}"
            "\nPytanie: {{ query }}"
        ),
    ]
    asking = Pipeline()
    asking.add_component("mask", PolanonQueryMasker(store=mappings))
    asking.add_component("embed", OpenAITextEmbedder())
    asking.add_component("retrieve", InMemoryEmbeddingRetriever(document_store, top_k=3))
    asking.add_component("prompt", ChatPromptBuilder(template=template))
    asking.add_component("llm", OpenAIChatGenerator())
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


def main() -> None:
    """Index, ask and stream the restored answer."""
    document_store = InMemoryDocumentStore()
    mappings = InMemoryMappingStore()
    indexing, asking = build(document_store, mappings)
    indexing.run({"mask": {"documents": DOCUMENTS}})
    stream = PolanonStreamRestorer(mappings, on_text=lambda text: print(text, end="", flush=True))
    asking.run(
        {"mask": {"query": QUESTION, "stream": stream}, "llm": {"streaming_callback": stream}}
    )
    print()


if __name__ == "__main__":
    main()
