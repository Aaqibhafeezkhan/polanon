"""The search-index example in both READMEs, run as written, with a stand-in index and embedder.

It needs no SDK, so it runs everywhere, also against the built wheel.
"""

import os
from pathlib import Path

import pytest

from polanon import Anonymizer, Pseudonyms, tokens_in

ROOT = Path(__file__).resolve().parent.parent
README_RAG = {
    "README.md": "## Masking for RAG and search indexes",
    "README.pl.md": "## Maskowanie dla RAG i indeksów wyszukiwania",
}
KEY = os.urandom(32)
#: Everything the example's documents and question name.
RAW = ("Kot", "Kowalsk", "Anna Nowak", "600 100 200")


class StandInIndex:
    """Keeps what it is given and finds the passages that share a token with the question."""

    def __init__(self):
        self.added = []

    def add(self, text, vector):
        self.added.append((text, vector))

    def search(self, vector):
        return [text for text, _ in self.added if tokens_in(text) & tokens_in(vector)]


def _example(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    section = text[text.index(README_RAG[name]) :]
    return section.split("```python\n", 1)[1].split("\n```", 1)[0]


@pytest.fixture(params=sorted(README_RAG))
def example(request, monkeypatch):
    monkeypatch.setenv("POLANON_KEY", KEY.hex())
    index = StandInIndex()
    namespace = {"index": index, "embed": lambda text: text}
    exec(compile(_example(request.param), request.param, "exec"), namespace)  # noqa: S102
    return namespace, index


def test_the_index_gets_tokens_only(example):
    _, index = example
    assert len(index.added) == 2
    for text, vector in index.added:
        for raw in RAW:
            assert raw not in text
            assert raw not in vector


def test_the_question_finds_the_passages_about_the_same_person(example):
    """Both passages name Jan Kowalski, in the instrumental and in the genitive."""
    namespace, index = example
    assert namespace["passages"] == [text for text, _ in index.added]
    person = Pseudonyms(KEY).token("Jan Kowalski", "PERSON")
    assert person in namespace["question"].anonymized


def test_the_answer_and_the_passage_are_restored(example):
    namespace, _ = example
    pseudonyms = Pseudonyms(KEY)
    attorney = pseudonyms.token("Anna Nowak", "PERSON")
    principal = pseudonyms.token("Jan Kowalski", "PERSON")
    answer = f"Pełnomocnikiem {principal} jest {attorney}."
    restorer = namespace["restorer"]
    shown = "".join(restorer.feed(piece) for piece in (answer[:20], answer[20:31], answer[31:]))
    assert shown + restorer.flush() == "Pełnomocnikiem Jan Kowalski jest Anna Nowak."
    document = namespace["document"]
    assert Anonymizer.restore(document.texts[1], document.mappings[1]) == (
        "Pełnomocnikiem Jana Kowalskiego jest Anna Nowak, tel. 600 100 200."
    )
