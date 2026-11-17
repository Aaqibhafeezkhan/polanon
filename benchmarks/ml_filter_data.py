"""The spans the Polish spaCy model finds in the benchmarks' texts, labelled for the ML filter.

Every span the model finds, as the paranoid mode keeps it, becomes one line of
``~/.cache/polanon/ml-filter/spans.jsonl``: its source and part, its type, whether the hand-written
filters of the normal mode keep it, the features of :mod:`polanon.guards._span_features`, and its
label. The label is 1 when the span covers a name the benchmark annotates (a person, a place, a
street, a company), 0 when it covers nothing annotated, and null when it covers only something
else: an identifier or an e-mail address, which the rules mask whatever the model says, or a name
the benchmark does not read, such as a country or an institution in KPWr. Those train nothing.

The parts keep the measurement honest. OpenPII's Polish rows are shuffled as run_external.py
shuffles them: its dev part (the first 1,000) tunes, its test part (the next 1,000) is the
benchmark's, the next 6,000 train, and the last 1,290 are "final": held out until the ML filter is
done and measured once. KPWr's train-tune file is split by document, so no text has sentences on
both sides: every tenth document is final, every tenth from the fifth on is dev, the rest train;
its test file is the benchmark's. Neither source was used to build the spaCy model, which learned
from NKJP.

OpenPII does not annotate every name its texts hold: "Kasia" or "Marek" in a sentence the
language model wrote around the annotated values. A span over a first name the rules know that
covers nothing annotated is no evidence that it is no name, so its label is null too.

    python benchmarks/ml_filter_data.py
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
from polanon.guards._span_features import FEATURES, Word, span_context, span_features
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


#: OpenPII's rows after the benchmark's two parts that train; the rest are final.
OPENPII_TRAIN = 6000


def openpii_part(index: int) -> str:
    """The part of OpenPII's shuffled row `index`."""
    if index < SAMPLE:
        return "dev"
    if index < 2 * SAMPLE:
        return "test"
    return "train" if index < 2 * SAMPLE + OPENPII_TRAIN else "final"


def kpwr_part(document: int) -> str:
    """The part of KPWr's train-tune document `document`, counted in the file's order."""
    return {0: "final", 5: "dev"}.get(document % 10, "train")


def documents(
    data: Path,
) -> Iterator[tuple[str, str, str, str, list[GoldEntity], list[GoldEntity]]]:
    """Yield (source, part, document, text, values the benchmark reads, other names)."""
    rows = openpii_rows(data)
    random.Random(SEED).shuffle(rows)  # noqa: S311 - the benchmark's fixed sample, not a secret
    for index, row in enumerate(rows):
        gold = openpii_gold(row)
        wanted = [value for value in gold if value.entity_type in OPENPII_NAMES]
        other = [value for value in gold if value.entity_type not in OPENPII_NAMES]
        yield "openpii", openpii_part(index), row["uid"], row["source_text"], wanted, other
    for split in ("train", "test"):
        numbers: dict[str, int] = {}
        for sentence in kpwr_sentences(data, split):
            number = numbers.setdefault(sentence["doc"], len(numbers))
            part = kpwr_part(number) if split == "train" else "test"
            text, names = kpwr_sentence(sentence, None)
            wanted = [gold for gold in names if gold.entity_type in KPWR_TYPES]
            other = [gold for gold in names if gold.entity_type not in KPWR_TYPES]
            yield "kpwr", part, f"{split}/{sentence['doc']}", text, wanted, other


def rows_of(
    source: str,
    part: str,
    document: str,
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
    read = guard._get_pipeline()(text)
    tokens = [
        (
            token.idx,
            token.idx + len(token),
            Word.of(token),
        )
        for token in read
        if not token.is_space
    ]
    lines = []
    for span in found:
        words = [word for start, end, word in tokens if overlaps(span, start, end)]
        features = span_features(
            text,
            span,
            rules_found_it=any(overlaps(span, r.start, r.end) for r in by_rules),
            words=words,
            vocab=guard._get_pipeline().vocab,
        )
        label = label_of(span, wanted, other)
        if source == "openpii" and label == 0 and features["a_first_name"]:
            label = None  # a name OpenPII did not annotate
        lines.append(
            {
                "source": source,
                "part": part,
                "doc": document,
                "type": span.entity_type,
                "text": text[span.start : span.end],
                "kept_by_filters": any(overlaps(span, k.start, k.end) for k in kept),
                "label": label,
                "features": features,
                "context": span_context(text, span),
            }
        )
    return lines


def summary(lines: list[dict]) -> None:
    """How the hand-written filters do on each part: what the ML filter has to beat."""
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
    parser.add_argument("--out", type=Path, default=Path.home() / ".cache/polanon/ml-filter")
    parser.add_argument("--spacy-model", default="pl_core_news_md")
    args = parser.parse_args()

    guard = ner.NerGuard(spacy_model=args.spacy_model, mode="paranoid")
    rules = guards_for(False, "")
    lines: list[dict] = []
    for count, (source, part, document, text, wanted, other) in enumerate(documents(args.data), 1):
        lines.extend(rows_of(source, part, document, text, wanted, other, guard, rules))
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
