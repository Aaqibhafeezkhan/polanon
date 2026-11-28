"""The candidates md and FastPDN propose in the benchmarks' texts, labelled for the arbiter.

Every text of OpenPII's Polish rows and of KPWr becomes one line of
``~/.cache/polanon/arbiter/texts.jsonl``: its source and part, its annotated values, what the rules
found, and every candidate the arbiter is asked about, with the names of its signs that are on,
FastPDN's categories over it, its context and its label. The label is ml_filter_data.py's: 1 for a
candidate over a name the benchmark annotates, 0 for one over nothing annotated, and null for one
over only another value, which trains nothing. The parts are ml_filter_data.py's too, so nothing
the benchmarks measure trains the arbiter; texts outside the train part also keep what the spaCy
guard masks today, for arbiter_tune.py to compare with.

The texts are read for training: FastPDN on every line, not only on the Polish ones, and its
organisations proposed too, so the regression learns from every candidate either model would make.
The rules that keep the arbiter from doing worse than the spaCy guard come after it, when it masks:
see polanon/guards/arbiter.py.

It needs FastPDN exported (``python -m polanon.fastpdn``); on a desktop it takes about an hour.

    python benchmarks/arbiter_data.py [--fastpdn-dir DIR] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from ml_filter_data import documents, label_of
from run import EVALUATED, guards_for

from polanon._detection import detect
from polanon.guards._span_features import span_context
from polanon.guards.arbiter import ArbiterGuard, _Text, describe
from polanon.guards.ner import NerGuard


def main() -> int:
    """Read every text and write its candidates."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/polanon/benchmarks")
    parser.add_argument("--out", type=Path, default=Path.home() / ".cache/polanon/arbiter")
    parser.add_argument("--fastpdn-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, help="at most this many texts of each source and part")
    args = parser.parse_args()

    guard = ArbiterGuard(args.fastpdn_dir, cache_texts=0)
    guard._for_training = True  # FastPDN on every line, its organisations proposed too
    shipped = NerGuard(cache_texts=0)
    rules = guards_for(False, "")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "texts.jsonl"
    seen: Counter[tuple[str, str]] = Counter()
    started = time.perf_counter()
    with path.open("w", encoding="utf-8") as out:
        for source, part, document, text, wanted, other in documents(args.data):
            if args.limit and seen[(source, part)] >= args.limit:
                continue
            seen[(source, part)] += 1
            found = [span for rule in rules for span in rule.detect(text, EVALUATED)]
            reading = guard._reading(text, reuse=False)
            vocab = guard._models()[0].vocab
            under_review = _Text(text, reading, found, vocab)
            candidates = []
            for span, info in reading.candidates:
                on, categories = describe(span, info, under_review)
                label = label_of(span, wanted, other)
                if source == "openpii" and label == 0 and "a_first_name" in on:
                    label = None  # a first name OpenPII did not annotate
                candidates.append(
                    {
                        "s": span.start,
                        "e": span.end,
                        "type": span.entity_type,
                        "label": label,
                        "on": on,
                        "cats": categories,
                        "ctx": span_context(text, span),
                        "md": info.get("md"),
                    }
                )
            gold = wanted + other if source == "openpii" else wanted
            line = {
                "source": source,
                "part": part,
                "doc": document,
                "text": text,
                "gold": [[value.start, value.end] for value in gold],
                "rules": [[s.start, s.end, s.entity_type] for s in found],
                "candidates": candidates,
            }
            if part != "train":
                spans = detect(text, None, EVALUATED, [*rules, shipped])
                line["shipped"] = [
                    [s.start, s.end, s.entity_type] for s in spans if s.source == shipped.name
                ]
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            count = sum(seen.values())
            if count % 500 == 0:
                rate = (time.perf_counter() - started) / count
                print(f"{count} texts, {rate * 1000:.0f} ms a text", file=sys.stderr, flush=True)
    print(f"{sum(seen.values())} texts written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
