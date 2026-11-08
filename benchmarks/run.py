"""Measure the stage-1 guards against the annotated sample documents.

The point of a masking tool is not how many spans it finds but whether the sensitive substrings end
up covered, so that is what is measured here:

``masked``
    every character of the annotated entity ended up inside a mask.
``partial``
    part of it was masked and part was left in the clear. For a masking tool this is a failure, not
    a half-success, and it is counted separately because it is the failure mode worth staring at.
``missed``
    nothing of it was masked.
``type``
    of the entities that were masked, how many carried the expected entity type. A wrong label still
    protects the person, so it is reported apart from coverage.
``extra``
    masked spans that overlap no annotation. Some of these are genuinely wrong; others are
    conservative masking the annotation did not ask for. Reported as a count, not as an error.

The corpus is eleven short synthetic documents (169 annotated occurrences), eight Polish and three
foreign invoices, which makes this a regression check with a readable output, not a research
benchmark. Nobody should quote these numbers as a general accuracy claim for any language.

Usage:
    python benchmarks/run.py              # guards that need no model
    python benchmarks/run.py --ner        # also load the Polish spaCy model
    python benchmarks/run.py --ner --markdown   # table ready to paste into the README
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from polanon import (
    CompanyGuard,
    DenylistGuard,
    Guard,
    InternationalGuard,
    PersonNameGuard,
    RegexGuard,
    SecretsGuard,
    merge_spans,
)
from polanon.policy import ENTITY_TYPES

SAMPLES = Path(__file__).parent / "samples"
GOLD = Path(__file__).parent / "gold"

#: The entity types the library masks by default. DATE is deliberately outside it: a date rarely
#: identifies anyone on its own and masking it makes the answers worse, so the annotations do not
#: mark dates either and the guards are not asked to find them here.
EVALUATED = set(ENTITY_TYPES) - {"DATE"}


@dataclass(frozen=True)
class GoldEntity:
    """One annotated occurrence, in the coordinates of its own document."""

    start: int
    end: int
    value: str
    entity_type: str


@dataclass
class Score:
    """Counts for one entity type, or for a whole run."""

    gold: int = 0
    masked: int = 0
    partial: int = 0
    missed: int = 0
    right_type: int = 0

    @property
    def coverage(self) -> float:
        """Share of annotated entities that were masked completely."""
        return self.masked / self.gold if self.gold else 0.0


def load_gold(name: str) -> tuple[str, list[GoldEntity]]:
    """Return a sample's text and its annotations, expanded to every occurrence.

    Raises:
        ValueError: when an annotated value is absent from the document or two annotations overlap.
    """
    text = (SAMPLES / name).read_text(encoding="utf-8")
    payload = json.loads((GOLD / f"{name.removesuffix('.txt')}.json").read_text(encoding="utf-8"))
    entities: list[GoldEntity] = []
    for entry in payload["entities"]:
        value, entity_type = entry["value"], entry["type"]
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"{name}: unknown entity type {entity_type!r}")
        found = _occurrences(text, value)
        if not found:
            raise ValueError(f"{name}: annotated value {value!r} is not in the document")
        entities.extend(GoldEntity(s, e, value, entity_type) for s, e in found)
    entities.sort(key=lambda entity: entity.start)
    for previous, following in pairwise(entities):
        if previous.end > following.start:
            raise ValueError(f"{name}: annotations overlap: {previous.value!r} {following.value!r}")
    return text, entities


def _occurrences(text: str, value: str) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    start = text.find(value)
    while start != -1:
        found.append((start, start + len(value)))
        start = text.find(value, start + len(value))
    return found


def evaluate(
    text: str,
    gold: list[GoldEntity],
    guards: list[Guard],
    enabled: set[str],
    *,
    union: bool = False,
) -> tuple[dict[str, Score], int]:
    """Score one document and return per-type scores plus the number of extra spans.

    Coverage is what polanon's pipeline masks: the detections merged as it merges them. With
    `union`, it is every character any detection covers, which is at least what another tool's
    own conflict resolution would mask, so a comparison never scores it below its best.
    """
    detected: list = []
    for guard in guards:
        detected.extend(guard.detect(text, enabled))
    spans = merge_spans(detected, text)
    covering = detected if union else spans
    masked_chars = {index for span in covering for index in range(span.start, span.end)}
    type_at = {index: span.entity_type for span in spans for index in range(span.start, span.end)}

    scores: dict[str, Score] = {}
    for entity in gold:
        score = scores.setdefault(entity.entity_type, Score())
        score.gold += 1
        covered = sum(1 for index in range(entity.start, entity.end) if index in masked_chars)
        if covered == entity.end - entity.start:
            score.masked += 1
            if type_at.get(entity.start) == entity.entity_type:
                score.right_type += 1
        elif covered:
            score.partial += 1
        else:
            score.missed += 1

    gold_chars = {index for entity in gold for index in range(entity.start, entity.end)}
    extra = sum(
        1 for span in spans if not any(i in gold_chars for i in range(span.start, span.end))
    )
    return scores, extra


def run(guards: list[Guard], *, union: bool = False) -> tuple[dict[str, Score], int, int]:
    """Evaluate every sample and return merged scores, extra spans and the document count."""
    totals: dict[str, Score] = {}
    extra = 0
    documents = sorted(path.name for path in SAMPLES.glob("*.txt"))
    for name in documents:
        text, gold = load_gold(name)
        scores, document_extra = evaluate(text, gold, guards, EVALUATED, union=union)
        extra += document_extra
        for entity_type, score in scores.items():
            merged = totals.setdefault(entity_type, Score())
            merged.gold += score.gold
            merged.masked += score.masked
            merged.partial += score.partial
            merged.missed += score.missed
            merged.right_type += score.right_type
    return totals, extra, len(documents)


def guards_for(with_ner: bool, spacy_model: str) -> list[Guard]:
    """Build the guard stack for one configuration."""
    guards: list[Guard] = [
        RegexGuard(),
        SecretsGuard(),
        CompanyGuard(),
        PersonNameGuard(),
        InternationalGuard(),
        DenylistGuard([]),
    ]
    if with_ner:
        from polanon.guards.ner import NerGuard

        guards.append(NerGuard(spacy_model=spacy_model))
    return guards


def report(totals: dict[str, Score], extra: int, documents: int, markdown: bool) -> None:
    """Print the score table."""
    overall = Counter(
        {
            "gold": sum(s.gold for s in totals.values()),
            "masked": sum(s.masked for s in totals.values()),
            "partial": sum(s.partial for s in totals.values()),
            "missed": sum(s.missed for s in totals.values()),
            "right_type": sum(s.right_type for s in totals.values()),
        }
    )
    if markdown:
        print("| Entity | Annotated | Masked | Partial | Missed | Right type |")
        print("|---|---:|---:|---:|---:|---:|")
    else:
        print(f"{'entity':<14}{'gold':>6}{'masked':>8}{'partial':>9}{'missed':>8}{'type':>7}")

    for entity_type in sorted(totals, key=lambda key: (-totals[key].gold, key)):
        score = totals[entity_type]
        if markdown:
            print(
                f"| `{entity_type}` | {score.gold} | {score.masked} | "
                f"{score.partial} | {score.missed} | {score.right_type} |"
            )
        else:
            print(
                f"{entity_type:<14}{score.gold:>6}{score.masked:>8}"
                f"{score.partial:>9}{score.missed:>8}{score.right_type:>7}"
            )

    coverage = overall["masked"] / overall["gold"] if overall["gold"] else 0.0
    line = (
        f"{documents} documents, {overall['gold']} annotated occurrences: "
        f"{overall['masked']} masked ({coverage:.0%}), {overall['partial']} partial, "
        f"{overall['missed']} missed, {overall['right_type']} with the expected type. "
        f"Extra spans: {extra}."
    )
    print(f"\n{line}")


def main() -> int:
    """Parse the arguments and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner", action="store_true", help="also load the spaCy model")
    parser.add_argument("--spacy-model", default="pl_core_news_md")
    parser.add_argument("--markdown", action="store_true", help="print a markdown table")
    args = parser.parse_args()

    totals, extra, documents = run(guards_for(args.ner, args.spacy_model))
    print(f"configuration: guards without a model{' + spaCy NER' if args.ner else ''}\n")
    report(totals, extra, documents, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
