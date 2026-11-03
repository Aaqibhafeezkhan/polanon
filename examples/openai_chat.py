"""Mask personal data before a call to any OpenAI-compatible chat endpoint.

    pip install polanon openai
    export OPENAI_API_KEY=... OPENAI_MODEL=...
    python examples/openai_chat.py

The same code talks to OpenAI, OpenRouter, Azure, Mistral, vLLM or a local Ollama: point
``OPENAI_BASE_URL`` at the server (``http://localhost:11434/v1`` for Ollama) and name its model in
``OPENAI_MODEL``. The model only ever sees tokens such as ``<OSOBA_1>``; the mapping that turns them
back stays in this process.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import openai

from polanon import Anonymizer, default_guards

SYSTEM = (
    "Personal data in the user's messages has been replaced by tokens such as <OSOBA_1> or "
    "<PESEL_1>. When you refer to what a token stands for, copy the token exactly as written. "
    "Never guess the value behind it."
)

anonymizer = Anonymizer(default_guards(with_ner=False))


def ask(client: openai.OpenAI, model: str, question: str) -> str:
    """Mask `question`, send it to `model` and return the answer with the originals put back."""
    masked = anonymizer.anonymize_sync(question)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": masked.anonymized},
        ],
    )
    answer = completion.choices[0].message.content or ""
    return Anonymizer.restore(answer, masked.mapping)


def ask_streaming(client: openai.OpenAI, model: str, question: str) -> Iterator[str]:
    """Like :func:`ask`, but yield the restored answer as it streams.

    A token can arrive split across two chunks ("<OSO", "BA_1>"); the restorer holds back whatever
    could still grow into one, so a name is never shown half-restored.
    """
    masked = anonymizer.anonymize_sync(question)
    restorer = Anonymizer.stream_restorer(masked.mapping)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": masked.anonymized},
        ],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield restorer.feed(chunk.choices[0].delta.content)
    yield restorer.flush()


if __name__ == "__main__":
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        sys.exit("Set OPENAI_MODEL to the model the endpoint serves.")
    question = (
        "Napisz krótką odpowiedź do Jana Kowalskiego (PESEL 50810112354, "
        "jan.kowalski@example.com), że korekta faktury jest gotowa."
    )
    for piece in ask_streaming(openai.OpenAI(), model, question):
        print(piece, end="", flush=True)
    print()
