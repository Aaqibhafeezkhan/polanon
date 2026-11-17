"""The spans the Polish spaCy model finds in the benchmarks' texts, labelled for a verifier.

Every span the model finds, as the paranoid mode keeps it, becomes one line of
``~/.cache/polanon/verifier/spans.jsonl``: its source and part, its type, whether the hand-written
filters of the normal mode keep it, the features of :mod:`polanon.guards._span_features`, and its
label. The label is 1 when the span covers a name the benchmark annotates (a person, a place, a
street, a company), 0 when it covers nothing annotated, and null when it covers only something
else: an identifier or an e-mail address, which the rules mask whatever the model says, or a name
the benchmark does not read, such as a country or an institution in KPWr. Those train nothing.

The parts keep the benchmarks honest. OpenPII's Polish rows are shuffled as run_external.py
shuffles them: its dev part (the first 1,000) tunes, its test part (the next 1,000) is only
measured, and the 7,290 rows after them train. KPWr's train-tune file trains and its test file is
only measured. Neither source was used to build the spaCy model, which learned from NKJP.

    python benchmarks/verifier_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from run import EVALUATED, GoldEntity, guards_for
from run_external import (
    KPWR_TYPES,
    SAMPLE,
    SEED,
    kpwr_sentence,
    kpwr_sentences,
    openpii_gold,
    openpii_rows,
)

from polanon.guards import ner
from polanon.guards._span_features import FEATURES, Word, span_features
from polanon.types import Span

#: The types the model's spans come in.
NER_TYPES = {"PERSON", "LOCATION", "ORG"}
#: OpenPII's values that are names: the model's spans over them count.
OPENPII_NAMES = {"PERSON", "LOCATION", "ADDRESS"}


def overlaps(span: Span, start: int, end: int) -> bool:
    """Whether a span shares a character with the range from `start` to `end`."""
    return span.start < end and start < span.end


def label_of(span: Span, wanted: Iterable[GoldEntity], other: Iterable[GoldEntity]) -> int | None:
    """1 for a span over a wanted value, None for one over only another name, 0 for neither."""
    if any(overlaps(span, gold.start, gold.end) for gold in wanted):
        return 1
    if any(overlaps(span, gold.start, gold.end) for gold in other):
        return None
    return 0


def documents(data: Path) -> Iterator[tuple[str, str, str, list[GoldEntity], list[GoldEntity]]]:
    """Yield (source, part, text, values the benchmark reads, other names) for every text."""
    rows = openpii_rows(data)
    random.Random(SEED).shuffle(rows)  # noqa: S311 - the benchmark's fixed sample, not a secret
    for index, row in enumerate(rows):
        part = "dev" if index < SAMPLE else "test" if index < 2 * SAMPLE else "train"
        gold = openpii_gold(row)
        wanted = [value for value in gold if value.entity_type in OPENPII_NAMES]
        other = [value for value in gold if value.entity_type not in OPENPII_NAMES]
        yield "openpii", part, row["source_text"], wanted, other
    for part in ("train", "test"):
        for sentence in kpwr_sentences(data, part):
            text, names = kpwr_sentence(sentence, None)
            wanted = [gold for gold in names if gold.entity_type in KPWR_TYPES]
            other = [gold for gold in names if gold.entity_type not in KPWR_TYPES]
            yield "kpwr", part, text, wanted, other


def rows_of(
    source: str,
    part: str,
    text: str,
    wanted: list[GoldEntity],
    other: list[GoldEntity],
    guard: ner.NerGuard,
    rules: list,
) -> list[dict]:
    """The dataset's lines for one text: one per span the model found."""
    found = guard.detect(text, NER_TYPES)
    if not found:
        return []
    kept = ner._without_english_misreadings(text, found)
    by_rules = [span for rule in rules for span in rule.detect(text, EVALUATED)]
    document = guard._get_pipeline()(text)
    tokens = [
        (
            token.idx,
            token.idx + len(token),
            Word(token.text, token.pos_, token.lemma_, token.has_vector),
        )
        for token in document
        if not token.is_space
    ]
    lines = []
    for span in found:
        words = [word for start, end, word in tokens if overlaps(span, start, end)]
        lines.append(
            {
                "source": source,
                "part": part,
                "type": span.entity_type,
                "text": text[span.start : span.end],
                "kept_by_filters": any(overlaps(span, k.start, k.end) for k in kept),
                "label": label_of(span, wanted, other),
                "features": span_features(
                    text,
                    span,
                    rules_found_it=any(overlaps(span, r.start, r.end) for r in by_rules),
                    words=words,
                    vocab=guard._get_pipeline().vocab,
                ),
            }
        )
    return lines


def summary(lines: list[dict]) -> None:
    """How the hand-written filters do on each part: what a verifier has to beat."""
    print(
        "| Source | Part | Spans | Values | Not values | Ignored | Filters keep values | "
        "Filters drop values | Filters keep non-values |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    groups: dict[tuple[str, str], Counter[str]] = {}
    for line in lines:
        count = groups.setdefault((line["source"], line["part"]), Counter())
        count["spans"] += 1
        label, kept = line["label"], line["kept_by_filters"]
        count[{1: "values", 0: "non-values", None: "ignored"}[label]] += 1
        if label == 1:
            count["kept values" if kept else "dropped values"] += 1
        elif label == 0 and kept:
            count["kept non-values"] += 1
    for (source, part), c in sorted(groups.items()):
        print(
            f"| {source} | {part} | {c['spans']} | {c['values']} | {c['non-values']} | "
            f"{c['ignored']} | {c['kept values']} | {c['dropped values']} | "
            f"{c['kept non-values']} |"
        )


def main() -> int:
    """Read every text, write the spans and print how the filters do."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/polanon/benchmarks")
    parser.add_argument("--out", type=Path, default=Path.home() / ".cache/polanon/verifier")
    parser.add_argument("--spacy-model", default="pl_core_news_md")
    args = parser.parse_args()

    guard = ner.NerGuard(spacy_model=args.spacy_model, mode="paranoid")
    rules = guards_for(False, "")
    lines: list[dict] = []
    for count, (source, part, text, wanted, other) in enumerate(documents(args.data), 1):
        lines.extend(rows_of(source, part, text, wanted, other, guard, rules))
        if count % 2000 == 0:
            print(f"{count} texts, {len(lines)} spans", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "spans.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for line in lines:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
    (args.out / "features.json").write_text(json.dumps(list(FEATURES)), encoding="utf-8")
    print(f"{len(lines)} spans written to {path}\n")
    summary(lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
