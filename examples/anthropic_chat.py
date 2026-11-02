"""Ask Claude about text that holds personal data, without sending the data.

    pip install polanon anthropic
    export ANTHROPIC_API_KEY=...
    python examples/anthropic_chat.py

Claude only ever sees tokens such as ``<OSOBA_1>``. The mapping that turns them back into names and
numbers stays in this process, and the answer is restored before anyone reads it, also while it
streams.
"""

from __future__ import annotations

from collections.abc import Iterator

import anthropic
from anthropic.types import MessageParam

from polanon import Anonymizer, default_guards, mask_known_values

MODEL = "claude-opus-5-5"
SYSTEM = (
    "Personal data in the user's messages has been replaced by tokens such as <OSOBA_1> or "
    "<PESEL_1>. When you refer to what a token stands for, copy the token exactly as written. "
    "Never guess the value behind it."
)

anonymizer = Anonymizer(default_guards(with_ner=False))


def ask(client: anthropic.Anthropic, question: str) -> str:
    """Mask `question`, send it to Claude and return the answer with the originals put back."""
    masked = anonymizer.anonymize_sync(question)
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": masked.anonymized}],
    )
    answer = "".join(block.text for block in response.content if block.type == "text")
    return Anonymizer.restore(answer, masked.mapping)


def ask_streaming(client: anthropic.Anthropic, question: str) -> Iterator[str]:
    """Like :func:`ask`, but yield the restored answer as it streams.

    A token can arrive split across two events ("<OSO", "BA_1>"); the restorer holds back whatever
    could still grow into one, so a name is never shown half-restored.
    """
    masked = anonymizer.anonymize_sync(question)
    restorer = Anonymizer.stream_restorer(masked.mapping)
    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        system=SYSTEM,
        messages=[{"role": "user", "content": masked.anonymized}],
    ) as stream:
        for text in stream.text_stream:
            yield restorer.feed(text)
    yield restorer.flush()


class PrivateChat:
    """A conversation with Claude whose history holds masked text only.

    One mapping serves the whole conversation, so a person keeps the same token from the first
    message to the last. The history keeps Claude's answers as written, with tokens: sending the
    restored answers back would hand Claude every value the masking kept from it.
    """

    def __init__(self, client: anthropic.Anthropic) -> None:
        """Start an empty conversation."""
        self.client = client
        self.mapping: dict[str, str] = {}
        self.exchanges: list[tuple[str, str]] = []  # (masked question, answer with tokens)

    def send(self, message: str) -> str:
        """Send the next user message and return Claude's restored answer."""
        masked = anonymizer.anonymize_sync(message, self.mapping)
        self.mapping = masked.mapping
        # A value first masked in this message may sit unmasked in an earlier one, where nothing
        # recognised it; mask it there too before the history goes out again.
        self.exchanges = [
            (mask_known_values(question, self.mapping), mask_known_values(answer, self.mapping))
            for question, answer in self.exchanges
        ]
        messages: list[MessageParam] = []
        for question, answer in self.exchanges:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": masked.anonymized})
        response = self.client.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM, messages=messages
        )
        answer = "".join(block.text for block in response.content if block.type == "text")
        self.exchanges.append((masked.anonymized, answer))
        return Anonymizer.restore(answer, self.mapping)


if __name__ == "__main__":
    question = (
        "Napisz krótką odpowiedź do Jana Kowalskiego (PESEL 90010112349, "
        "jan.kowalski@firma.pl), że korekta faktury jest gotowa."
    )
    for piece in ask_streaming(anthropic.Anthropic(), question):
        print(piece, end="", flush=True)
    print()
