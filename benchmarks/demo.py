"""Show what the anonymizer does to the synthetic sample documents.

This prints masked text for reading. For scores against the annotations in ``benchmarks/gold``,
run ``benchmarks/run.py`` instead.

Usage:
    python benchmarks/demo.py            # guards that need no model
    python benchmarks/demo.py --ner      # also load the Polish spaCy model
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

from polanon import Anonymizer, default_guards

SAMPLES = Path(__file__).parent / "samples"


async def main(with_ner: bool) -> int:
    """Anonymize every sample and print a summary table."""
    anonymizer = Anonymizer(default_guards(with_ner=with_ner))
    totals: Counter[str] = Counter()

    for path in sorted(SAMPLES.glob("*.txt")):
        result = await anonymizer.anonymize(path.read_text(encoding="utf-8"))
        totals.update(result.entity_counts)
        masked = sum(result.entity_counts.values())
        print(f"\n=== {path.name}: {masked} entities masked ===")
        print(result.anonymized.strip())
        restored = Anonymizer.restore(result.anonymized, result.mapping)
        assert restored == result.original, f"restore is not lossless for {path.name}"

    print("\n=== totals ===")
    for entity_type, count in totals.most_common():
        print(f"{entity_type:<14} {count}")
    print(f"{'TOTAL':<14} {sum(totals.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main("--ner" in sys.argv)))
