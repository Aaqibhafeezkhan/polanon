"""Choose and fit the arbiter by cross-validation on the train parts, and write its weights.

Reads ``~/.cache/polanon/arbiter/texts.jsonl``, written by arbiter_data.py: every candidate md or
FastPDN proposed in OpenPII's and KPWr's texts, with its signs and its label. Only the train parts
are read here: dev chooses the thresholds (arbiter_tune.py), test is the benchmarks', and final is
measured once, at the end. Every strength of the L2 penalty is scored by five-fold
cross-validation, the folds cut by document, as ml_filter_train.py scores the ML filter: how many
of the candidates that are no names it drops while dropping at most one name in a thousand.

KPWr's train part trains too. FastPDN learned it, so it is right there more often than on text it
never saw; trained on OpenPII alone, the arbiter scored KPWr's names so low that no threshold lost
none of them on dev, and it kept every candidate. With both, the thresholds are real ones and the
final part, never read before, lost no value.

    python benchmarks/arbiter_train.py --module polanon/guards/_arbiter_weights.py
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from ml_filter_train import _literal, fit_logistic

from polanon.guards._span_features import span_vector

TYPES = ("PERSON", "LOCATION", "ORG")
#: A word around a span, an ending or a FastPDN category gets a weight when it comes up this often.
MIN_COUNT = 10
#: Names the arbiter may drop in a fold, of those it is asked about: one in a thousand.
NAMES_DROPPED = 0.001
FOLDS = 5


def rows_of(path: Path, parts: tuple[str, ...], sources: tuple[str, ...]) -> list[dict]:
    """The labelled candidates of the texts of `parts` and `sources`."""
    rows = []
    for line in path.open(encoding="utf-8"):
        item = json.loads(line)
        if item["part"] not in parts or item["source"] not in sources:
            continue
        for candidate in item["candidates"]:
            if candidate["label"] is not None:
                candidate["text"] = item["text"][candidate["s"] : candidate["e"]]
                candidate["doc"] = item["doc"]
                rows.append(candidate)
    return rows


def columns(rows: list[dict]) -> list[str]:
    """The names of the binary inputs: every sign, per type, and the frequent context."""
    base = sorted({name for row in rows for name in row["on"]})
    names = base + [f"{kind}:{name}" for kind in TYPES for name in base]
    counts = Counter(name for row in rows for name in row["ctx"] + row["cats"])
    return names + sorted(name for name, count in counts.items() if count >= MIN_COUNT)


def write_module(
    path: Path, names: list[str], weights: list[float], bias: float, thresholds: dict[str, float]
) -> None:
    """Write the weights as the module the arbiter reads; arbiter_tune.py sets the thresholds."""
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
    cuts = "\n".join(f'    "{kind}": {cut:.3f},' for kind, cut in sorted(thresholds.items()))
    path.write_text(
        f'''"""Weights of the arbiter, written by benchmarks/arbiter_train.py.

A logistic regression chosen by cross-validation on the train parts of OpenPII and KPWr, over
the candidates md and FastPDN proposed there. A weight named "TYPE:sign" counts for candidates of
that type only; one named "before:", "after:", "end:" or "end2:" for a candidate with that context;
one named "fpl:" for a candidate FastPDN gave that KPWr category; and VECTOR weighs the numbers of
the candidate's vector in pl_core_news_md. A candidate is kept when the logistic of the bias and its
weights reaches the threshold of its type, which benchmarks/arbiter_tune.py chooses end to end on
the dev parts.
"""

BIAS = {bias:.4f}

WEIGHTS: dict[str, float] = {{
{table}
}}

VECTOR: list[float] = [
{vector}
]

THRESHOLDS: dict[str, float] = {{
{cuts}
}}
''',
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    """Cross-validate the penalties, fit the best and write it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/polanon/arbiter")
    parser.add_argument("--module", type=Path, help="write the best penalty's weights")
    parser.add_argument("--from-json", type=Path, help="write the module from a model.json instead")
    args = parser.parse_args()

    if args.from_json is not None:
        model = json.loads(args.from_json.read_text(encoding="utf-8"))
        names = sorted(model["weights"])
        write_module(
            args.module,
            names,
            [model["weights"][name] for name in names] + list(model["vector"]),
            model["bias"],
            model["thresholds"],
        )
        print(f"Module written to {args.module} from {args.from_json}")
        return 0

    import spacy

    vocab = spacy.load("pl_core_news_md", exclude=["lemmatizer", "tagger"]).vocab
    rows = rows_of(args.data / "texts.jsonl", ("train",), ("openpii", "kpwr"))
    names = columns(rows)
    column = {name: index for index, name in enumerate(names)}
    x = np.zeros((len(rows), len(names) + 300), dtype=np.float32)
    for index, row in enumerate(rows):
        for name in (
            row["on"] + [f"{row['type']}:{sign}" for sign in row["on"]] + row["ctx"] + row["cats"]
        ):
            if name in column:
                x[index, column[name]] = 1.0
        vector = span_vector(row["text"], vocab)
        if vector:
            x[index, len(names) :] = vector
    y = np.array([row["label"] for row in rows], dtype=np.float32)
    fold = np.array([zlib.crc32(str(row["doc"]).encode()) % FOLDS for row in rows])
    print(
        f"{len(rows)} labelled candidates, {len(names)} binary inputs, {FOLDS} folds by document\n"
    )
    print("| L2 | Non-names dropped | Names dropped |\n|---:|---:|---:|")
    results = []
    for l2 in (1e-5, 3e-5, 1e-4, 1e-3):
        dropped = lost = 0
        for held_out in range(FOLDS):
            weights, bias = fit_logistic(x[fold != held_out], y[fold != held_out], l2)
            scores = 1 / (1 + np.exp(-(x @ weights + bias)))
            held = fold == held_out
            names_scores = np.sort(scores[held & (y == 1)])
            allowed = int(NAMES_DROPPED * len(names_scores))
            cut = names_scores[allowed] if len(names_scores) > allowed else 1.0
            dropped += int(np.sum(held & (y == 0) & (scores < cut)))
            lost += int(np.sum(held & (y == 1) & (scores < cut)))
        print(f"| {l2:g} | {dropped} of {int(np.sum(y == 0))} | {lost} |", flush=True)
        results.append((dropped, -lost, l2))
    _, _, l2 = max(results)
    weights, bias = fit_logistic(x, y, l2)
    print(f"\nBest L2 {l2:g}")
    if args.module is not None:
        write_module(args.module, names, list(weights), bias, dict.fromkeys(TYPES, 0.0))
        print(f"Module written to {args.module}; set its thresholds with arbiter_tune.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
