"""Fit a learned verifier on the labelled spans and measure it against the hand-written filters.

Reads ``~/.cache/polanon/verifier/spans.jsonl``, written by verifier_data.py. The verifier is a
logistic regression: one weight per feature and a bias, fitted by gradient descent in plain Python
on the train part, with an L2 penalty that keeps a weight small unless the data asks for it. The
threshold is chosen on the dev parts, as the highest that keeps at least as many names as the
filters keep in each source, so that it loses no name a leak would cost: OpenPII's dev part, and
every fifth span of KPWr's train part, which then trains nothing. The test parts are measured
once, with that threshold.

Two ways to use it are measured. "Instead" keeps what the verifier keeps, the filters aside.
"After" keeps what the filters keep and the verifier keeps too, so it can only drop more: the
filters were written for an agent's tool results, which these texts barely show.

    python benchmarks/verifier_train.py [--per-type]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

#: The penalty on the weights, and the steps of gradient descent.
L2 = 1e-3
STEPS = 600
RATE = 0.1


@dataclass(frozen=True)
class Example:
    """One labelled span: the indices of its features that are 1, and whether it is a name."""

    active: tuple[int, ...]
    label: int
    kept_by_filters: bool
    source: str
    part: str
    text: str
    type: str


def load(path: Path, *, per_type: bool) -> tuple[list[str], list[Example]]:
    """Read the spans with a label, and name the features, one copy per type with `per_type`."""
    examples: list[Example] = []
    names: list[str] = []
    index: dict[str, int] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["label"] is None:
            continue
        on = [name for name, value in row["features"].items() if value]
        if per_type:
            on += [f"{row['type']}:{name}" for name in on]
        for name in on:
            if name not in index:
                index[name] = len(names)
                names.append(name)
        examples.append(
            Example(
                tuple(sorted(index[name] for name in on)),
                int(row["label"]),
                bool(row["kept_by_filters"]),
                row["source"],
                row["part"],
                row["text"],
                row["type"],
            )
        )
    return names, examples


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def fit(features: int, examples: Sequence[Example]) -> tuple[list[float], float]:
    """Fit the weights and the bias by gradient descent with Adam on the mean logistic loss.

    Spans with the same features and label are counted once with their number, which makes a pass
    over twenty thousand spans a pass over a few thousand patterns.
    """
    patterns = Counter((example.active, example.label) for example in examples)
    total = sum(patterns.values())
    weights = [0.0] * features
    bias = 0.0
    m = [0.0] * (features + 1)
    v = [0.0] * (features + 1)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for step in range(1, STEPS + 1):
        gradient = [0.0] * (features + 1)
        for (active, label), count in patterns.items():
            error = (_sigmoid(bias + sum(weights[j] for j in active)) - label) * count
            for j in active:
                gradient[j] += error
            gradient[features] += error
        for j in range(features):
            gradient[j] = gradient[j] / total + L2 * weights[j]
        gradient[features] /= total
        for j in range(features + 1):
            m[j] = beta1 * m[j] + (1 - beta1) * gradient[j]
            v[j] = beta2 * v[j] + (1 - beta2) * gradient[j] ** 2
            update = (
                RATE * (m[j] / (1 - beta1**step)) / (math.sqrt(v[j] / (1 - beta2**step)) + epsilon)
            )
            if j == features:
                bias -= update
            else:
                weights[j] -= update
    return weights, bias


def probability(weights: Sequence[float], bias: float, example: Example) -> float:
    """The verifier's probability that a span is a name."""
    return _sigmoid(bias + sum(weights[j] for j in example.active))


@dataclass
class Tally:
    """Names kept and dropped, and spans that are no names kept and dropped."""

    kept: int = 0
    dropped: int = 0
    wrongly_kept: int = 0
    rightly_dropped: int = 0

    def add(self, label: int, keep: bool) -> None:
        """Count one span with its label, kept or dropped."""
        if label == 1:
            if keep:
                self.kept += 1
            else:
                self.dropped += 1
        elif keep:
            self.wrongly_kept += 1
        else:
            self.rightly_dropped += 1


def tally(examples: Sequence[Example], keep: Callable[[Example], bool]) -> Tally:
    """Count what a way of keeping spans keeps and drops."""
    found = Tally()
    for example in examples:
        found.add(example.label, keep(example))
    return found


def threshold_on(dev: Sequence[Example], scores: dict[Example, float]) -> float:
    """The highest threshold that keeps as many names as the filters do in every source of `dev`."""
    sources = {example.source for example in dev}
    by_source = {s: [e for e in dev if e.source == s] for s in sources}
    targets = {s: tally(by_source[s], lambda e: e.kept_by_filters).kept for s in sources}
    best = 0.0
    for step in range(1, 100):
        t = step / 100
        if all(
            tally(by_source[s], lambda e, t=t: scores[e] >= t).kept >= targets[s] for s in sources
        ):
            best = t
    return best


def thresholds_on(dev: Sequence[Example], scores: dict[Example, float]) -> dict[str, float]:
    """One threshold per type, each chosen by :func:`threshold_on` on the spans of its type."""
    return {
        kind: threshold_on([e for e in dev if e.type == kind], scores)
        for kind in sorted({example.type for example in dev})
    }


def write_module(
    path: Path,
    names: Sequence[str],
    weights: Sequence[float],
    bias: float,
    thresholds: dict[str, float],
    trained: int,
) -> None:
    """Write the weights as the module the NER guard reads, rounded and sorted by name."""
    table = "\n".join(
        f"    {name!r}: {weight:.4f},"
        for name, weight in sorted(zip(names, weights, strict=True))
        if round(weight, 4)
    )
    cuts = "\n".join(f"    {kind!r}: {cut:.2f}," for kind, cut in sorted(thresholds.items()))
    path.write_text(
        f'''"""Weights of the learned verifier, written by benchmarks/verifier_train.py.

Fitted on {trained} spans the Polish spaCy model found in OpenPII's and KPWr's train parts. A
weight named "TYPE:feature" counts for spans of that type only. A span is kept when the logistic
of the bias and the weights of its features reaches the threshold of its type.
"""

BIAS = {bias:.4f}

WEIGHTS: dict[str, float] = {{
{table}
}}

THRESHOLDS: dict[str, float] = {{
{cuts}
}}
''',
        encoding="utf-8",
        newline="\n",
    )


def row(name: str, t: Tally) -> str:
    """One line of a markdown table."""
    return f"| {name} | {t.kept} | {t.dropped} | {t.wrongly_kept} | {t.rightly_dropped} |"


def main() -> int:
    """Fit on the train part, choose the threshold on dev, report the test part."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/polanon/verifier")
    parser.add_argument("--per-type", action="store_true", help="a copy of every feature per type")
    parser.add_argument("--threshold-per-type", action="store_true", help="a threshold per type")
    parser.add_argument(
        "--module", type=Path, help="also write the weights as a module, such as the package's"
    )
    args = parser.parse_args()

    names, examples = load(args.data / "spans.jsonl", per_type=args.per_type)
    kpwr_train = [e for e in examples if e.source == "kpwr" and e.part == "train"]
    held = {id(e) for e in kpwr_train[::5]}
    examples = [replace(e, part="dev") if id(e) in held else e for e in examples]
    train = [e for e in examples if e.part == "train"]
    weights, bias = fit(len(names), train)
    scores = {e: probability(weights, bias, e) for e in examples}
    dev = [e for e in examples if e.part == "dev"]
    one = threshold_on(dev, scores)
    by_type = thresholds_on(dev, scores) if args.threshold_per_type else {}
    cut = {e: by_type.get(e.type, one) for e in examples}
    chosen = ", ".join(f"{kind} {t:.2f}" for kind, t in by_type.items()) or f"{one:.2f}"

    print(f"{len(train)} spans trained, {len(names)} features, threshold {chosen} from dev\n")
    for source in ("openpii", "kpwr"):
        test = [e for e in examples if e.part == "test" and e.source == source]
        print(f"### {source}, test part ({len(test)} spans)\n")
        print("| Way | Names kept | Names dropped | Non-names kept | Non-names dropped |")
        print("|---|---:|---:|---:|---:|")
        print(row("filters (today)", tally(test, lambda e: e.kept_by_filters)))
        print(row("verifier instead", tally(test, lambda e: scores[e] >= cut[e])))
        print(
            row("verifier after", tally(test, lambda e: e.kept_by_filters and scores[e] >= cut[e]))
        )
        print(row("keep everything", tally(test, lambda e: True)))
        print()

    ranked = sorted(zip(names, weights, strict=True), key=lambda item: item[1])
    print("Weights against a name:", ", ".join(f"{n} {w:.2f}" for n, w in ranked[:12]))
    print("Weights for a name:", ", ".join(f"{n} {w:.2f}" for n, w in ranked[::-1][:12]))
    print(f"Bias {bias:.2f}\n")

    openpii_test = [e for e in examples if e.part == "test" and e.source == "openpii"]
    still = Counter(
        e.text for e in openpii_test if e.label == 0 and e.kept_by_filters and scores[e] >= cut[e]
    )
    print("Non-names the verifier after the filters still keeps:", still.most_common(15))
    lost = Counter(
        e.text for e in openpii_test if e.label == 1 and e.kept_by_filters and scores[e] < cut[e]
    )
    print("Names it drops that the filters keep:", lost.most_common(15))
    kpwr_test = [e for e in examples if e.part == "test" and e.source == "kpwr"]
    lost = Counter(e.text for e in kpwr_test if e.label == 1 and scores[e] < cut[e])
    print("KPWr names it drops:", lost.most_common(20))

    if args.module is not None:
        thresholds = by_type or dict.fromkeys(("PERSON", "LOCATION", "ORG"), one)
        write_module(args.module, names, weights, bias, thresholds, len(train))
        print(f"\nModule written to {args.module}")
    out = args.data / "weights.json"
    out.write_text(
        json.dumps(
            {
                "features": names,
                "weights": weights,
                "bias": bias,
                "threshold": one,
                "by_type": by_type,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nWeights written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
