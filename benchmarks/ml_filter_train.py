"""Choose and fit the ML filter by cross-validation on the train parts.

Reads ``~/.cache/polanon/ml-filter/spans.jsonl``, written by ml_filter_data.py, and uses only its
train part: dev chooses the thresholds (ml_filter_tune.py), test is the benchmarks', and final is
measured once, at the end. Every configuration is scored by five-fold cross-validation, the folds
cut by document, so no text has spans on both sides.

A configuration is a set of inputs and a model. The inputs: the features of
:func:`polanon.guards._span_features.span_features`, each also per type; the span's vector
(:func:`~polanon.guards._span_features.span_vector`); and its context
(:func:`~polanon.guards._span_features.span_context`), the words around it and its ending, kept
when they come up at least `MIN_CONTEXT` times. The model: a logistic regression, which the
masking reads in plain Python, at three strengths of the L2 penalty; and, to see what it would
add, a network with one hidden layer.

The ML filter works after the hand-written filters, so it is scored on the spans they keep: how
many of the spans that are no names it drops while dropping at most one name in a thousand. The
best configuration is fitted on the whole train part and written with ``--module``; the
thresholds it is written with are placeholders until ml_filter_tune.py sets them.

    python benchmarks/ml_filter_train.py --module polanon/guards/_ml_filter_weights.py
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polanon.guards._span_features import span_vector

TYPES = ("PERSON", "LOCATION", "ORG")
#: A word around a span or an ending gets a weight when it comes up this often in training.
MIN_CONTEXT = 10
#: Names the ML filter may drop of those the filters keep, in a fold: one in a thousand.
NAMES_DROPPED = 0.001
FOLDS = 5


@dataclass
class Spans:
    """The labelled spans of the train part as arrays."""

    names: list[str]
    binary: np.ndarray
    vectors: np.ndarray
    label: np.ndarray
    kept: np.ndarray
    fold: np.ndarray
    text: np.ndarray


def read(path: Path, parts: tuple[str, ...]) -> list[dict]:
    """The labelled spans of `parts`."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return [row for row in rows if row["label"] is not None and row["part"] in parts]


def columns(rows: list[dict], *, context: bool) -> list[str]:
    """The names of the binary inputs: every feature, per type, and the frequent context."""
    base = sorted(rows[0]["features"])
    names = base + [f"{kind}:{name}" for kind in TYPES for name in base]
    if context:
        counts = Counter(name for row in rows for name in row["context"])
        names += sorted(name for name, count in counts.items() if count >= MIN_CONTEXT)
    return names


def arrays(rows: list[dict], names: list[str], vocab: object) -> Spans:
    """The binary inputs, the vectors and the labels of `rows`."""
    column = {name: index for index, name in enumerate(names)}
    binary = np.zeros((len(rows), len(names)), dtype=np.float32)
    vectors = np.zeros((len(rows), 300), dtype=np.float32)
    for index, row in enumerate(rows):
        on = [name for name, value in row["features"].items() if value]
        for name in on + [f"{row['type']}:{name}" for name in on] + row["context"]:
            if name in column:
                binary[index, column[name]] = 1.0
        vector = span_vector(row["text"], vocab)  # type: ignore[arg-type]
        if vector:
            vectors[index] = vector
    return Spans(
        names,
        binary,
        vectors,
        np.array([row["label"] for row in rows], dtype=np.float32),
        np.array([row["kept_by_filters"] for row in rows]),
        np.array([zlib.crc32(str(row["doc"]).encode()) % FOLDS for row in rows]),
        np.array([row["text"] for row in rows]),
    )


def _adam(
    gradient: Callable[[list[np.ndarray]], list[np.ndarray]],
    params: list[np.ndarray],
    steps: int,
    rate: float,
) -> list[np.ndarray]:
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for step in range(1, steps + 1):
        for i, g in enumerate(gradient(params)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            params[i] = params[i] - rate * (m[i] / (1 - 0.9**step)) / (
                np.sqrt(v[i] / (1 - 0.999**step)) + 1e-8
            )
    return params


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    """A logistic regression, fitted by Adam on the mean logistic loss with an L2 penalty."""

    def gradient(params: list[np.ndarray]) -> list[np.ndarray]:
        weights, bias = params
        p = 1 / (1 + np.exp(-(x @ weights + bias)))
        return [x.T @ (p - y) / len(y) + l2 * weights, np.array([np.mean(p - y)])]

    weights, bias = _adam(gradient, [np.zeros(x.shape[1]), np.zeros(1)], 800, 0.05)
    return weights, float(bias[0])


def fit_network(x: np.ndarray, y: np.ndarray, l2: float, hidden: int = 32) -> Callable:
    """A network with one hidden layer of ReLUs, for comparison; returns its scoring function."""
    rng = np.random.default_rng(2026)
    params = [
        rng.normal(0, np.sqrt(2 / x.shape[1]), (x.shape[1], hidden)),
        np.zeros(hidden),
        rng.normal(0, np.sqrt(1 / hidden), hidden),
        np.zeros(1),
    ]

    def forward(params: list[np.ndarray], data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w1, b1, w2, b2 = params
        h = np.maximum(0, data @ w1 + b1)
        return h, 1 / (1 + np.exp(-(h @ w2 + b2)))

    def gradient(params: list[np.ndarray]) -> list[np.ndarray]:
        w1, _b1, w2, _b2 = params
        h, p = forward(params, x)
        d = (p - y) / len(y)
        dh = np.outer(d, w2) * (h > 0)
        return [x.T @ dh + l2 * w1, dh.sum(0), h.T @ d + l2 * w2, np.array([d.sum()])]

    params = _adam(gradient, params, 600, 0.01)
    return lambda data: forward(params, data)[1]


def dropped_at(scores: np.ndarray, spans: Spans, chosen: np.ndarray) -> tuple[int, int]:
    """Non-names dropped, and names, at the highest cut that drops at most 1 in 1000 names."""
    kept = chosen & spans.kept
    names = np.sort(scores[kept & (spans.label == 1)])
    allowed = int(NAMES_DROPPED * len(names))
    cut = names[allowed] if len(names) > allowed else 1.0
    others = kept & (spans.label == 0)
    return int(np.sum(others & (scores < cut))), int(
        np.sum(kept & (spans.label == 1) & (scores < cut))
    )


def cross_validate(spans: Spans, inputs: str, model: str, l2: float) -> tuple[int, int, int]:
    """Non-names and names dropped over the folds, and non-names the filters keep."""
    x = spans.binary if inputs == "binary" else np.hstack([spans.binary, spans.vectors])
    dropped = names = 0
    for fold in range(FOLDS):
        train, held = spans.fold != fold, spans.fold == fold
        if model == "logistic":
            weights, bias = fit_logistic(x[train], spans.label[train], l2)
            scores = 1 / (1 + np.exp(-(x @ weights + bias)))
        else:
            scores = fit_network(x[train], spans.label[train], l2)(x)
        d, n = dropped_at(scores, spans, held)
        dropped += d
        names += n
    total = int(np.sum(spans.kept & (spans.label == 0)))
    return dropped, names, total


def _literal(name: str) -> str:
    """A feature's name as a string literal, every character but a letter or ASCII escaped."""
    escaped = "".join(
        char if char.isalnum() or (char.isascii() and char not in '"\\') else f"\\u{ord(char):04x}"
        for char in name
    )
    return f'"{escaped}"'


def write_module(path: Path, names: list[str], weights: np.ndarray, bias: float) -> None:
    """Write the weights as the module the NER guard reads; ml_filter_tune.py sets thresholds."""
    count = len(names)
    table = "\n".join(
        f"    {_literal(name)}: {weight:.4f},"
        for name, weight in sorted(zip(names, weights[:count], strict=True))
        if round(float(weight), 4)
    )
    numbers = [f"{weight:.4f}" for weight in weights[count:]]
    vector = "\n".join(
        "    " + ", ".join(numbers[start : start + 8]) + "," for start in range(0, len(numbers), 8)
    )
    path.write_text(
        f'''"""Weights of the ML filter, written by benchmarks/ml_filter_train.py.

A logistic regression chosen by cross-validation on the train parts of OpenPII and KPWr. A weight
named "TYPE:feature" counts for spans of that type only, one named "before:", "after:", "end:" or
"end2:" for a span with that context, and VECTOR weighs the numbers of the span's vector. A span
is kept when the logistic of the bias and its weights reaches the threshold of its type, which
benchmarks/ml_filter_tune.py chooses end to end on the dev parts.
"""

BIAS = {bias:.4f}

WEIGHTS: dict[str, float] = {{
{table}
}}

VECTOR: list[float] = [
{vector}
]

THRESHOLDS: dict[str, float] = {{
    "LOCATION": 0.0,
    "ORG": 0.0,
    "PERSON": 0.0,
}}
''',
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    """Cross-validate the configurations, fit the best and write it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/polanon/ml-filter")
    parser.add_argument("--spacy-model", default="pl_core_news_md")
    parser.add_argument("--module", type=Path, help="write the best configuration's weights")
    parser.add_argument("--network", action="store_true", help="also cross-validate a network")
    args = parser.parse_args()

    import spacy

    vocab = spacy.load(args.spacy_model, exclude=["lemmatizer", "tagger"]).vocab
    rows = read(args.data / "spans.jsonl", ("train",))
    print(f"{len(rows)} labelled spans in the train part, {FOLDS} folds by document\n")
    print("| Inputs | Model | L2 | Non-names dropped | Names dropped |")
    print("|---|---|---:|---:|---:|")
    results = []
    for context in (False, True):
        spans = arrays(rows, columns(rows, context=context), vocab)
        for inputs in ("binary", "binary+vector"):
            label = inputs + ("+context" if context else "")
            for l2 in (1e-5, 3e-5, 1e-4, 1e-3):
                dropped, names, total = cross_validate(spans, inputs, "logistic", l2)
                results.append((dropped, -names, context, inputs, l2))
                print(f"| {label} | logistic | {l2:g} | {dropped} of {total} | {names} |")
    if args.network:
        spans = arrays(rows, columns(rows, context=True), vocab)
        for l2 in (1e-4, 1e-3):
            dropped, names, total = cross_validate(spans, "binary+vector", "network", l2)
            print(f"| binary+vector+context | network | {l2:g} | {dropped} of {total} | {names} |")

    dropped, _, context, inputs, l2 = max(results)
    print(f"\nBest logistic: {inputs}{'+context' if context else ''}, L2 {l2:g}")
    names = columns(rows, context=context)
    spans = arrays(rows, names, vocab)
    x = spans.binary if inputs == "binary" else np.hstack([spans.binary, spans.vectors])
    weights, bias = fit_logistic(x, spans.label, l2)
    ranked = sorted(zip(names, weights[: len(names)], strict=True), key=lambda item: item[1])
    print("Weights against a name:", ", ".join(f"{n} {w:.2f}" for n, w in ranked[:15]))
    print("Weights for a name:", ", ".join(f"{n} {w:.2f}" for n, w in ranked[::-1][:15]))
    if args.module is not None:
        if inputs == "binary":
            weights = np.concatenate([weights, np.zeros(300)])
        write_module(args.module, names, weights, bias)
        print(f"\nModule written to {args.module}; set its thresholds with ml_filter_tune.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
