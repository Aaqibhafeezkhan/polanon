"""FastPDN, CLARIN-PL's Polish named-entity model, run from an ONNX file without torch.

`python -m polanon.fastpdn` exports the model once, from Hugging Face to a directory: the network
as ONNX, its tokenizer, and the labels. This module only reads that directory, with onnxruntime and
the `tokenizers` package. It reproduces what the Hugging Face pipeline the arbiter was fitted with
returns for a text, ``pipeline("token-classification", aggregation_strategy="first", stride=128)``:
the same chunks of 512 tokens overlapping by 128, the same rule for where a word goes on, the
label of a word's first piece, the same grouping of B- and I- labels, and the same choice between
entities two chunks both found. Nothing here is imported until a guard needs the model.

FastPDN (https://huggingface.co/clarin-pl/FastPDN) is CC-BY-4.0, by CLARIN-PL; the export writes
that attribution next to the model.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The directory `python -m polanon.fastpdn` exports to unless told otherwise.
DEFAULT_DIRECTORY = Path.home() / ".cache" / "polanon" / "fastpdn"

#: The files an exported directory holds.
MODEL_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "config.json"

#: The version of the directory's layout, written into its config.
FORMAT = 1


@dataclass(frozen=True)
class Entity:
    """One entity the model found: where it stands, its KPWr category and its mean score."""

    start: int
    end: int
    group: str
    score: float


def missing_files(directory: Path) -> list[str]:
    """The files an export would have written that `directory` lacks."""
    return [
        name
        for name in (MODEL_FILE, TOKENIZER_FILE, CONFIG_FILE)
        if not (directory / name).is_file()
    ]


@dataclass
class _Piece:
    """A token outside the special ones, with the scores of its labels."""

    start: int
    end: int
    scores: Any
    is_subword: bool


def pieces(
    text: str,
    ids: list[int],
    offsets: list[tuple[int, int]],
    special: list[int],
    scores: Any,
    *,
    token_of: Any,
    unknown_id: int,
    subword_prefix: str | None,
) -> list[_Piece]:
    """The tokens of one chunk, special ones left out, each marked as starting a word or not.

    A tokenizer that marks the pieces inside a word does so with a prefix; this one marks the end
    of a word instead, so the pipeline falls back on the text: a piece goes on the word before it
    unless a space stands on either side of where it starts.
    """
    found = []
    for index, token_id in enumerate(ids):
        if special[index]:
            continue
        start, end = offsets[index]
        if subword_prefix:
            is_subword = len(token_of(token_id)) != len(text[start:end])
        else:
            is_subword = start > 0 and " " not in text[start - 1 : start + 1]
        if token_id == unknown_id:
            is_subword = False
        found.append(_Piece(start, end, scores[index], is_subword))
    return found


def _argmax(row: Any) -> int:
    best = 0
    for index in range(1, len(row)):
        if row[index] > row[best]:
            best = index
    return best


def _tag(label: str) -> tuple[str, str]:
    if label.startswith("B-"):
        return "B", label[2:]
    if label.startswith("I-"):
        return "I", label[2:]
    return "I", label


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def entities_of(
    found: list[_Piece], labels: list[str], mean: Callable[[Sequence[float]], float] = _mean
) -> list[Entity]:
    """Label words by their first piece and group neighbours of one category, as the pipeline does.

    This is the pipeline's "first" aggregation, and the "O" groups are left out. `mean` averages
    the scores of a group's words; the model passes numpy's, in the precision of its scores, so a
    score comes out exactly as the pipeline's does.
    """
    words: list[tuple[int, int, str, Any]] = []
    group: list[_Piece] = []
    for piece in [*found, None]:
        if piece is not None and group and piece.is_subword:
            group.append(piece)
            continue
        if group:
            best = _argmax(group[0].scores)
            words.append((group[0].start, group[-1].end, labels[best], group[0].scores[best]))
        group = [piece] if piece is not None else []
    result: list[Entity] = []
    run: list[tuple[int, int, str, Any]] = []
    for word in [*words, None]:
        if word is not None and run:
            bi, tag = _tag(word[2])
            if tag == _tag(run[-1][2])[1] and bi != "B":
                run.append(word)
                continue
        if run:
            name = run[0][2].split("-", 1)[-1]
            if name != "O":
                result.append(
                    Entity(run[0][0], run[-1][1], name, float(mean([item[3] for item in run])))
                )
        run = [word] if word is not None else []
    return result


def overlapping(found: list[Entity]) -> list[Entity]:
    """Keep one of each pair of entities two overlapping chunks both found, as the pipeline does.

    The longer one stays, or the surer of two equally long.
    """
    if not found:
        return found
    ordered = sorted(found, key=lambda entity: entity.start)
    kept = []
    previous = ordered[0]
    for entity in ordered:
        if previous.start <= entity.start < previous.end:
            length, previous_length = entity.end - entity.start, previous.end - previous.start
            if length > previous_length or (
                length == previous_length and entity.score > previous.score
            ):
                previous = entity
        else:
            kept.append(previous)
            previous = entity
    kept.append(previous)
    return kept


class FastPDN:
    """An exported FastPDN, ready to read texts: entities with their KPWr category and score."""

    def __init__(
        self, directory: str | Path = DEFAULT_DIRECTORY, *, threads: int | None = None
    ) -> None:
        """Load the model and the tokenizer from an exported directory.

        Args:
            directory: where `python -m polanon.fastpdn` exported the model.
            threads: how many threads onnxruntime may use for one text; its default otherwise.

        Raises:
            RuntimeError: when onnxruntime or tokenizers is not installed, or the directory lacks
                a file of the export.
        """
        self.directory = Path(directory)
        missing = missing_files(self.directory)
        if missing:
            raise RuntimeError(
                f"FastPDN is not exported to {self.directory} (missing {', '.join(missing)}): "
                "run python -m polanon.fastpdn"
            )
        try:
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - only without the dependencies
            raise RuntimeError(
                'the arbiter needs onnxruntime and tokenizers: pip install "polanon[arbiter]"'
            ) from exc
        config = json.loads((self.directory / CONFIG_FILE).read_text(encoding="utf-8"))
        self.labels: list[str] = list(config["labels"])
        self.unknown_id = int(config["unknown_id"])
        self.subword_prefix: str | None = config.get("subword_prefix") or None
        tokenizer = Tokenizer.from_file(str(self.directory / TOKENIZER_FILE))
        tokenizer.enable_truncation(
            max_length=int(config["max_length"]), stride=int(config["stride"])
        )
        tokenizer.enable_padding(pad_id=int(config["pad_id"]), pad_token=str(config["pad_token"]))
        self._tokenizer = tokenizer
        options = onnxruntime.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self._session = onnxruntime.InferenceSession(
            str(self.directory / MODEL_FILE), options, providers=["CPUExecutionProvider"]
        )
        # One session reads one text at a time: onnxruntime allows concurrent runs, but a lock
        # keeps the memory of a long text's chunks from being held twice.
        self._lock = threading.Lock()

    def entities(self, text: str) -> list[Entity]:
        """The entities FastPDN finds in `text`, as the pipeline returns them."""
        if not text.strip():
            return []
        import numpy as np

        encoding = self._tokenizer.encode_batch([text])[0]
        chunks = [encoding, *encoding.overflowing]
        found: list[Entity] = []
        for chunk in chunks:
            feed = {
                "input_ids": np.array([chunk.ids], dtype=np.int64),
                "attention_mask": np.array([chunk.attention_mask], dtype=np.int64),
            }
            with self._lock:
                logits = self._session.run(["logits"], feed)[0][0]
            shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
            scores = shifted / shifted.sum(axis=-1, keepdims=True)
            precision = scores.dtype

            def mean(values: Sequence[float], precision: Any = precision) -> float:
                return float(np.mean(np.array(values, dtype=precision)))

            found.extend(
                entities_of(
                    pieces(
                        text,
                        chunk.ids,
                        chunk.offsets,
                        chunk.special_tokens_mask,
                        scores,
                        token_of=self._tokenizer.id_to_token,
                        unknown_id=self.unknown_id,
                        subword_prefix=self.subword_prefix,
                    ),
                    self.labels,
                    mean,
                )
            )
        return overlapping(found) if len(chunks) > 1 else found
