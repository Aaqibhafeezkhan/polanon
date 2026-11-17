"""Fit a learned verifier on the labelled spans and measure it against the hand-written filters.

Reads ``~/.cache/polanon/verifier/spans.jsonl``, written by verifier_data.py. The verifier is a
logistic regression: a weight per feature, a copy of it per type, a weight per number of the
span's vector (:func:`polanon.guards._span_features.span_vector`) and a bias, fitted by gradient
descent on the train parts with an L2 penalty that keeps a weight small unless the data asks for
it. The masking reads the weights in plain Python; numpy only fits them.

The threshold of each type is chosen on the dev parts, OpenPII's and every fifth span of KPWr's
train part, which then trains nothing: the highest that keeps all the names the filters keep, in
each source, but for people one. KPWr calls "Stwórca" and "Śpiąca Królewna" people too, and with
no such allowance one of them holds the threshold for people down so far that months and the
words of forms are masked as people again. Places and companies get none. The test parts are
measured with those thresholds.

Two ways to use it are measured. "Instead" keeps what the verifier keeps, the filters aside.
"After" keeps what the filters keep and the verifier keeps too, so it can only drop more.

    python benchmarks/verifier_train.py [--module polanon/guards/_verifier_weights.py]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polanon.guards._span_features import span_vector

#: Names of each type a threshold may lose on the dev parts: see the module's description.
TOLERANCE = {"PERSON": 1, "LOCATION": 0, "ORG": 0}

#: The penalty on the weights, and the steps of gradient descent.
L2 = 1e-3
STEPS = 800
RATE = 0.05
TYPES = ("PERSON", "LOCATION", "ORG")


@dataclass
class Spans:
    """The labelled spans as arrays: one row of features each, and what is known of them."""

    names: list[str]
    features: np.ndarray
    vectors: np.ndarray
    label: np.ndarray
    kept: np.ndarray
    source: np.ndarray
    part: np.ndarray
    type: np.ndarray
    text: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        """The features and the vector of every span, side by side."""
        return np.hstack([self.features, self.vectors])


def load(path: Path, vocab: object) -> Spans:
    """Read the spans with a label; KPWr's every fifth train span becomes dev."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    rows = [row for row in rows if row["label"] is not None]
    kpwr = [row for row in rows if row["source"] == "kpwr" and row["part"] == "train"]
    for row in kpwr[::5]:
        row["part"] = "dev"
    base = sorted(rows[0]["features"])
    names = base + [f"{kind}:{name}" for kind in TYPES for name in base]
    column = {name: index for index, name in enumerate(names)}
    features = np.zeros((len(rows), len(names)), dtype=np.float32)
    vectors = np.zeros((len(rows), 300), dtype=np.float32)
    for index, row in enumerate(rows):
        on = [name for name in base if row["features"][name]]
        for name in on + [f"{row['type']}:{name}" for name in on]:
            features[index, column[name]] = 1.0
        vector = span_vector(row["text"], vocab)  # type: ignore[arg-type]
        if vector:
            vectors[index] = vector
    return Spans(
        names,
        features,
        vectors,
        np.array([row["label"] for row in rows], dtype=np.float32),
        np.array([row["kept_by_filters"] for row in rows]),
        np.array([row["source"] for row in rows]),
        np.array([row["part"] for row in rows]),
        np.array([row["type"] for row in rows]),
        np.array([row["text"] for row in rows]),
    )


def fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the weights and the bias by gradient descent with Adam on the mean logistic loss."""
    weights = np.zeros(x.shape[1])
    bias = 0.0
    m = np.zeros(x.shape[1] + 1)
    v = np.zeros(x.shape[1] + 1)
    for step in range(1, STEPS + 1):
        p = 1 / (1 + np.exp(-(x @ weights + bias)))
        gradient = np.concatenate([x.T @ (p - y) / len(y) + L2 * weights, [np.mean(p - y)]])
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient**2
        update = RATE * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        weights -= update[:-1]
        bias -= float(update[-1])
    return weights, bias


def thresholds(spans: Spans, scores: np.ndarray, tolerance: dict[str, int]) -> dict[str, float]:
    """Per type, the highest threshold that keeps all but its `tolerance` of the filters' names."""
    found = {}
    for kind in TYPES:
        best = 0.0
        groups = [
            (spans.part == "dev") & (spans.source == source) & (spans.type == kind)
            for source in ("openpii", "kpwr")
        ]
        for t in np.arange(0.01, 1.0, 0.01):
            if all(
                np.sum(group & (spans.label == 1) & (scores >= t))
                >= np.sum(group & (spans.label == 1) & spans.kept) - tolerance[kind]
                for group in groups
            ):
                best = float(t)
        found[kind] = round(best, 2)
    return found


def write_module(
    path: Path, spans: Spans, weights: np.ndarray, bias: float, cuts: dict[str, float]
) -> None:
    """Write the weights as the module the NER guard reads, rounded."""
    count = len(spans.names)
    table = "\n".join(
        f"    {name!r}: {weight:.4f},"
        for name, weight in sorted(zip(spans.names, weights[:count], strict=True))
        if round(float(weight), 4)
    )
    vector = ", ".join(f"{weight:.4f}" for weight in weights[count:])
    limits = "\n".join(f"    {kind!r}: {cut:.2f}," for kind, cut in sorted(cuts.items()))
    trained = int(np.sum(spans.part == "train"))
    path.write_text(
        f'''"""Weights of the learned verifier, written by benchmarks/verifier_train.py.

Fitted on {trained} spans the Polish spaCy model found in OpenPII's and KPWr's train parts. A
weight named "TYPE:feature" counts for spans of that type only, and VECTOR weighs the numbers of
the span's vector. A span is kept when the logistic of the bias and the weights of its features
reaches the threshold of its type.
"""

BIAS = {bias:.4f}

WEIGHTS: dict[str, float] = {{
{table}
}}

VECTOR: list[float] = [{vector}]

THRESHOLDS: dict[str, float] = {{
{limits}
}}
''',
        encoding="utf-8",
        newline="\n",
    )


def row(name: str, spans: Spans, chosen: np.ndarray, keep: np.ndarray) -> str:
    """One line of a markdown table: names and non-names kept and dropped."""
    names, others = chosen & (spans.label == 1), chosen & (spans.label == 0)
    return (
        f"| {name} | {np.sum(names & keep)} | {np.sum(names & ~keep)} | "
        f"{np.sum(others & keep)} | {np.sum(others & ~keep)} |"
    )


def main() -> int:
    """Fit on the train parts, choose the thresholds on dev, report the test parts."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/polanon/verifier")
    parser.add_argument("--spacy-model", default="pl_core_news_md")
    parser.add_argument("--module", type=Path, help="also write the weights as a module")
    args = parser.parse_args()

    import spacy

    vocab = spacy.load(args.spacy_model, exclude=["lemmatizer", "tagger"]).vocab
    spans = load(args.data / "spans.jsonl", vocab)
    train = spans.part == "train"
    weights, bias = fit(spans.matrix[train], spans.label[train])
    scores = 1 / (1 + np.exp(-(spans.matrix @ weights + bias)))
    cuts = thresholds(spans, scores, TOLERANCE)
    keep = scores >= np.array([cuts[kind] for kind in spans.type])

    print(f"{int(np.sum(train))} spans trained, thresholds {cuts} from dev\n")
    for source in ("openpii", "kpwr"):
        test = (spans.part == "test") & (spans.source == source)
        print(f"### {source}, test part ({int(np.sum(test))} spans)\n")
        print("| Way | Names kept | Names dropped | Non-names kept | Non-names dropped |")
        print("|---|---:|---:|---:|---:|")
        print(row("filters (today)", spans, test, spans.kept))
        print(row("verifier instead", spans, test, keep))
        print(row("verifier after", spans, test, spans.kept & keep))
        print()
        lost = Counter(spans.text[test & spans.kept & ~keep & (spans.label == 1)])
        print("Names the filters keep and the verifier drops:", lost.most_common(10))
        left = Counter(spans.text[test & keep & (spans.label == 0)])
        print("Non-names the verifier keeps:", left.most_common(10), "\n")

    count = len(spans.names)
    ranked = sorted(zip(spans.names, weights[:count], strict=True), key=lambda item: item[1])
    print("Weights against a name:", ", ".join(f"{n} {w:.2f}" for n, w in ranked[:10]))
    print("Weights for a name:", ", ".join(f"{n} {w:.2f}" for n, w in ranked[::-1][:10]))
    if args.module is not None:
        write_module(args.module, spans, weights, bias, cuts)
        print(f"\nModule written to {args.module}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
