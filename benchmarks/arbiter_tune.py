"""Set the arbiter's thresholds end to end on the dev parts, and measure the final part.

For each text of a part, arbiter_data.py kept what the rules found and every candidate with its
signs; the arbiter's probability for each is computed with the weights module, and a set of
thresholds decides which candidates stay. The text is then scored as benchmarks/run.py scores it:
the spans merged, a value counted when every character of it is covered, a span over no annotated
value counted as outside.

The threshold of each type is the highest on a grid with which no source's dev part masks fewer
values whole than with every candidate kept, the others keeping everything; the three together are
checked the same way, and then multiplied by MARGIN, as ml_filter_tune.py sets the ML filter's.
``--write`` puts them into the weights module. ``--final`` measures the final part, which nothing
else reads, against what the spaCy guard masks there today. Run it once, when the arbiter is done.

    python benchmarks/arbiter_tune.py --write polanon/guards/_arbiter_weights.py
    python benchmarks/arbiter_tune.py --final
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from run import EVALUATED, GoldEntity, evaluate

from polanon.guards import _arbiter_weights as fitted
from polanon.guards._span_features import span_vector
from polanon.guards.arbiter import probability
from polanon.guards.base import Guard
from polanon.types import Span

TYPES = ("PERSON", "LOCATION", "ORG")
#: What the thresholds found on dev are multiplied by.
MARGIN = 0.75
GRID = [step / 100 for step in range(1, 61)]


class _Fixed(Guard):
    """Returns the spans it was given: what another guard found, read once."""

    def __init__(self, name: str, spans: list[Span]) -> None:
        self.name = name
        self.spans = spans

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        return [span for span in self.spans if span.entity_type in enabled]


def texts_of(path: Path, part: str) -> list[dict]:
    """The texts of `part`, each candidate with the arbiter's probability."""
    import spacy

    vocab = spacy.load("pl_core_news_md", exclude=["lemmatizer", "tagger"]).vocab
    texts = []
    for line in path.open(encoding="utf-8"):
        item = json.loads(line)
        if item["part"] != part or (item["source"] == "kpwr" and not item["gold"]):
            continue  # KPWr names more than it is read for: only its values count
        item["scored"] = [
            (
                Span(c["s"], c["e"], c["type"], 0.85, "arbiter_ner"),
                probability(
                    c["on"],
                    c["cats"],
                    c["ctx"],
                    c["type"],
                    span_vector(item["text"][c["s"] : c["e"]], vocab),
                ),
            )
            for c in item["candidates"]
        ]
        texts.append(item)
    return texts


def totals(
    texts: list[dict], cuts: dict[str, float] | None, *, shipped: bool = False
) -> dict[str, tuple[int, int]]:
    """Per source, values masked whole and spans outside, with candidates that reach `cuts`."""
    found: dict[str, tuple[int, int]] = {}
    for item in texts:
        if shipped:
            names = [Span(s, e, t, 0.85, "spacy_ner") for s, e, t in item.get("shipped", [])]
        else:
            names = [
                span
                for span, chance in item["scored"]
                if cuts is None or chance >= cuts.get(span.entity_type, 0.0)
            ]
        rules = [Span(s, e, t, 1.0, "rules") for s, e, t in item["rules"]]
        gold = [GoldEntity(s, e, item["text"][s:e], "VALUE") for s, e in item["gold"]]
        guards: list[Guard] = [_Fixed("rules", rules), _Fixed("names", names)]
        scores, outside = evaluate(item["text"], gold, guards, EVALUATED)
        whole, extra = found.get(item["source"], (0, 0))
        found[item["source"]] = (whole + sum(s.masked for s in scores.values()), extra + outside)
    return found


def no_loss(found: dict[str, tuple[int, int]], base: dict[str, tuple[int, int]]) -> bool:
    """Whether every source masks at least as many values whole as with every candidate kept."""
    return all(found[source][0] >= base[source][0] for source in base)


def tune(texts: list[dict]) -> dict[str, float]:
    """The thresholds: the highest per type with no loss on dev, together, times MARGIN."""
    base = totals(texts, None)
    cuts: dict[str, float] = {}
    for kind in TYPES:
        best = 0.0
        for t in GRID:
            if no_loss(totals(texts, {kind: t}), base):
                best = t
        cuts[kind] = best
    while not no_loss(totals(texts, cuts), base):
        worst = max(cuts, key=lambda kind: cuts[kind])
        cuts[worst] = round(cuts[worst] - 0.01, 2)
    return {kind: round(cut * MARGIN, 3) for kind, cut in cuts.items()}


def report(title: str, texts: list[dict], cuts: dict[str, float]) -> None:
    """Values masked whole and spans outside: the spaCy guard, every candidate, the arbiter."""
    today, union, decided = (
        totals(texts, None, shipped=True),
        totals(texts, None),
        totals(texts, cuts),
    )
    print(f"\n### {title}\n")
    print(
        "| Source | Texts | Whole: spaCy guard | every candidate | arbiter "
        "| Outside: spaCy guard | every candidate | arbiter |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for source in sorted(union):
        count = sum(1 for item in texts if item["source"] == source)
        print(
            f"| {source} | {count} | {today[source][0]} | {union[source][0]} "
            f"| {decided[source][0]} | {today[source][1]} | {union[source][1]} "
            f"| {decided[source][1]} |"
        )


def write(path: Path, cuts: dict[str, float]) -> None:
    """Put the thresholds into the weights module."""
    table = "\n".join(f'    "{kind}": {cut:.3f},' for kind, cut in sorted(cuts.items()))
    source = path.read_text(encoding="utf-8")
    source = re.sub(
        r"THRESHOLDS: dict\[str, float\] = \{.*?\}",
        "THRESHOLDS: dict[str, float] = {\n" + table + "\n}",
        source,
        flags=re.DOTALL,
    )
    path.write_text(source, encoding="utf-8", newline="\n")


def main() -> int:
    """Tune on dev and write, or measure the final part once."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/polanon/arbiter")
    parser.add_argument("--write", type=Path, help="put the thresholds into this module")
    parser.add_argument("--final", action="store_true", help="measure the final part, once")
    args = parser.parse_args()
    path = args.data / "texts.jsonl"
    if args.final:
        report("Final part, measured once", texts_of(path, "final"), fitted.THRESHOLDS)
        return 0
    dev = texts_of(path, "dev")
    cuts = tune(dev)
    print("Thresholds:", cuts)
    report("Dev part, where the thresholds were chosen", dev, cuts)
    if args.write is not None:
        write(args.write, cuts)
        print(f"\nThresholds written to {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
