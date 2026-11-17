"""Set the ML filter's thresholds end to end on the dev parts, and measure the final part.

For each text, the rules' spans and the NER guard's spans after its filters are found once, with
the ML filter's probability for each NER span, computed as the guard computes it. A set of
thresholds then decides which NER spans stay, and the text is scored as benchmarks/run.py scores
it: the spans merged, a value counted when every character of it is covered, a span over no
annotated value counted as outside.

The threshold of each type is the highest on a grid with which no source's dev part masks fewer
values whole than with the filters alone, the others keeping everything; the three together are
checked the same way, and then multiplied by MARGIN, so that a name a little less sure than any on
dev still stays. ``--write`` puts them into the weights module.

``--final`` measures the final part, which nothing else reads: OpenPII's last 1,290 rows and every
tenth document of KPWr's train-tune file. Run it once, when the ML filter is done.

    python benchmarks/ml_filter_tune.py --write polanon/guards/_ml_filter_weights.py
    python benchmarks/ml_filter_tune.py --final
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

from ml_filter_data import kpwr_part, openpii_part
from run import EVALUATED, GoldEntity, evaluate, guards_for
from run_external import (
    SEED,
    kpwr_sentence,
    kpwr_sentences,
    openpii_gold,
    openpii_rows,
)

from polanon.guards import _ml_filter_weights as fitted
from polanon.guards import _span_features as features
from polanon.guards.base import Guard
from polanon.guards.ner import NerGuard
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


@dataclass
class Text:
    """One text of a part, with what the guards found in it, read once."""

    source: str
    text: str
    gold: list[GoldEntity]
    rules: list[Span]
    ner: list[tuple[Span, float]]


def texts_of(data: Path, part: str) -> list[tuple[str, str, list[GoldEntity]]]:
    """(source, text, annotated values) of every text of `part` in OpenPII and KPWr."""
    rows = openpii_rows(data)
    random.Random(SEED).shuffle(rows)  # noqa: S311 - the benchmark's fixed sample, not a secret
    found = [
        ("openpii", row["source_text"], openpii_gold(row))
        for index, row in enumerate(rows)
        if openpii_part(index) == part
    ]
    if part == "test":
        sentences = kpwr_sentences(data, "test")
    else:
        numbers: dict[str, int] = {}
        sentences = [
            sentence
            for sentence in kpwr_sentences(data, "train")
            if kpwr_part(numbers.setdefault(sentence["doc"], len(numbers))) == part
        ]
    for sentence in sentences:
        text, gold = kpwr_sentence(sentence)
        if gold:  # KPWr names more than it is read for: only its values count
            found.append(("kpwr", text, gold))
    return found


def read(data: Path, part: str, guard: NerGuard, rules: list[Guard]) -> list[Text]:
    """Find what the rules and the NER guard find in every text of `part`, and score the spans."""
    read_texts = []
    vocab = guard._get_pipeline().vocab
    for source, text, gold in texts_of(data, part):
        found_by_rules = [span for rule in rules for span in rule.detect(text, EVALUATED)]
        spans = guard.detect(text, EVALUATED)
        scored = [(span, _score(guard, text, span, found_by_rules, vocab)) for span in spans]
        read_texts.append(Text(source, text, gold, found_by_rules, scored))
    return read_texts


def _score(guard: NerGuard, text: str, span: Span, others: list[Span], vocab: object) -> float:
    """The ML filter's probability for one span, computed as the NER guard computes it."""
    digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
    words = guard._words.get(digest, ())
    found = features.span_features(
        text,
        span,
        rules_found_it=any(o.start < span.end and span.start < o.end for o in others),
        words=[word for start, end, word in words if start < span.end and span.start < end],
        vocab=vocab,  # type: ignore[arg-type]
    )
    vector = features.span_vector(text[span.start : span.end], vocab)  # type: ignore[arg-type]
    return features.ml_filter_score(
        found, span.entity_type, vector, features.span_context(text, span)
    )


def totals(texts: list[Text], cuts: dict[str, float] | None) -> dict[str, tuple[int, int]]:
    """Per source, values masked whole and spans outside, keeping NER spans that reach `cuts`."""
    found: dict[str, tuple[int, int]] = {}
    for item in texts:
        kept = [s for s, p in item.ner if cuts is None or p >= cuts.get(s.entity_type, 0.0)]
        guards: list[Guard] = [_Fixed("rules", item.rules), _Fixed("spacy", kept)]
        scores, outside = evaluate(item.text, item.gold, guards, EVALUATED)
        masked = sum(score.masked for score in scores.values())
        whole, extra = found.get(item.source, (0, 0))
        found[item.source] = (whole + masked, extra + outside)
    return found


def no_loss(found: dict[str, tuple[int, int]], base: dict[str, tuple[int, int]]) -> bool:
    """Whether every source masks at least as many values whole as with the filters alone."""
    return all(found[source][0] >= base[source][0] for source in base)


def tune(texts: list[Text]) -> dict[str, float]:
    """The thresholds: the highest per type with no loss on dev, together, times MARGIN."""
    base = totals(texts, None)
    cuts = {}
    for kind in TYPES:
        best = 0.0
        for t in GRID:
            if no_loss(totals(texts, {kind: t}), base):
                best = t
        cuts[kind] = best
    while not no_loss(totals(texts, cuts), base):
        worst = max(cuts, key=lambda kind: cuts[kind])
        cuts[worst] = round(cuts[worst] - 0.01, 2)
    return {kind: round(cut * MARGIN, 2) for kind, cut in cuts.items()}


def report(title: str, texts: list[Text], cuts: dict[str, float]) -> None:
    """Values masked whole and spans outside, with the filters alone and with the ML filter."""
    base, verified = totals(texts, None), totals(texts, cuts)
    print(f"\n### {title}\n")
    print("| Source | Texts | Masked whole: filters | ML filter | Outside: filters | ML filter |")
    print("|---|---:|---:|---:|---:|---:|")
    for source in sorted(base):
        count = sum(1 for item in texts if item.source == source)
        print(
            f"| {source} | {count} | {base[source][0]} | {verified[source][0]} | "
            f"{base[source][1]} | {verified[source][1]} |"
        )


def write(path: Path, cuts: dict[str, float]) -> None:
    """Put the thresholds into the weights module."""
    table = "\n".join(f'    "{kind}": {cut:.2f},' for kind, cut in sorted(cuts.items()))
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
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/polanon/benchmarks")
    parser.add_argument("--write", type=Path, help="put the thresholds into this module")
    parser.add_argument("--final", action="store_true", help="measure the final part, once")
    args = parser.parse_args()

    guard = NerGuard()
    rules = guards_for(False, "")
    if args.final:
        report(
            "Final part, measured once", read(args.data, "final", guard, rules), fitted.THRESHOLDS
        )
        return 0
    dev = read(args.data, "dev", guard, rules)
    cuts = tune(dev)
    print(f"Thresholds from dev, times {MARGIN}: {cuts}")
    report("Dev part", dev, cuts)
    if args.write is not None:
        write(args.write, cuts)
        print(f"\nThresholds written to {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
