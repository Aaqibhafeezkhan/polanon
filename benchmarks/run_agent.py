"""Measure the masking of an AI agent's calls: what reaches the model, what is masked for nothing.

:mod:`run` scores the guards on short documents. An agent sends something else: an English system
prompt, the definitions of its tools, JSON from its tools, code and logs, Polish next to English
on one line, and its whole history again in every call. Each call of :mod:`agent_corpus` is
masked here as the gateway masks it, by :func:`polanon.gateway.payloads.mask_chat_request` in one
:class:`polanon.MaskedTurn`, with the well-known names allowed, and two things are counted:

``leaked``
    occurrences of a marked value in the masked request, read string by string, JSON in a string
    included. Every one of them would reach the model.
``for nothing``
    tokens whose value is no marked value, nor part of one, nor holds one, and how many times the
    masked request uses them. Each of them hides a word the model needed to read, "Uncheck" of a
    button or "Max" of a chip, and the more often it stands, the worse the model's answer.

Usage:
    python benchmarks/run_agent.py              # guards that need no model
    python benchmarks/run_agent.py --ner        # also load the Polish spaCy model
    python benchmarks/run_agent.py --ner --show 20   # list the values masked for nothing
    python benchmarks/run_agent.py --ner --paranoid  # the spaCy guard in mode="paranoid"
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from agent_corpus import Marked, calls

from polanon import Allowlist, Anonymizer, MaskedTurn, default_guards
from polanon.allowlist import WELL_KNOWN_NAMES
from polanon.gateway.payloads import mask_chat_request
from polanon.guards.names_pl import fold
from polanon.policy import TOKEN_PATTERN, Mode


@dataclass
class CallScore:
    """What one call gave."""

    values: int = 0
    occurrences: int = 0
    leaked: int = 0
    leaked_values: Counter[tuple[str, str]] = field(default_factory=Counter)
    tokens: int = 0
    uses: int = 0
    for_nothing: Counter[str] = field(default_factory=Counter)

    @property
    def wasted_uses(self) -> int:
        """How many times the masked request uses a token that hides nothing."""
        return sum(self.for_nothing.values())


def strings(value: Any) -> Iterator[str]:
    """Yield every string of a payload, the strings of a JSON text in it instead of the text."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict | list):
            yield from strings(parsed)
        else:
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def occurrences(texts: list[str], values: list[str]) -> Counter[str]:
    """Count where each value stands, a longer value first, so "Gdynia" is not "Gdyni" too."""
    found: Counter[str] = Counter()
    for text in texts:
        taken: list[tuple[int, int]] = []
        for value in sorted(set(values), key=len, reverse=True):
            start = text.find(value)
            while start >= 0:
                end = start + len(value)
                if not any(start < stop and begin < end for begin, stop in taken):
                    taken.append((start, end))
                    found[value] += 1
                start = text.find(value, start + 1)
    return found


def _related(value: str, marked: list[Marked]) -> bool:
    """Return whether a masked value is a marked one, a part of one, or holds one."""
    key = fold(value.lower())
    return any(key in fold(m.value.lower()) or fold(m.value.lower()) in key for m in marked)


def score_call(anonymizer: Anonymizer, request: dict[str, Any], marked: list[Marked]) -> CallScore:
    """Mask one call as the gateway does and count what leaked and what was masked for nothing."""
    turn = MaskedTurn(anonymizer)
    masked = mask_chat_request(turn, request)
    texts = list(strings(masked))
    required = [m for m in marked if m.required]
    written = occurrences(list(strings(request)), [m.value for m in required])
    score = CallScore(values=len(written), occurrences=sum(written.values()))
    leaked = occurrences(texts, [m.value for m in required])
    types = {m.value: m.entity_type for m in required}
    for value, count in leaked.items():
        score.leaked += count
        score.leaked_values[(types[value], value)] += count
    token = re.compile(TOKEN_PATTERN)
    used = Counter(match[0] for text in texts for match in token.finditer(text))
    for name, value in turn.mapping.items():
        if value == name:
            continue
        score.tokens += 1
        score.uses += used[name]
        if not _related(value, marked):
            score.for_nothing[f"{name.split('_')[0].lstrip('<')} {value}"] += used[name]
    return score


def anonymizer_for(
    *, with_ner: bool, spacy_model: str = "pl_core_news_md", mode: Mode = "normal"
) -> Anonymizer:
    """The guards the gateway masks with, and its list of well-known names."""
    return Anonymizer(
        default_guards(with_ner=with_ner, spacy_model=spacy_model, mode=mode),
        allowlist=Allowlist(WELL_KNOWN_NAMES),
    )


def run(anonymizer: Anonymizer) -> dict[str, CallScore]:
    """Score every call of the corpus."""
    requests, marked = calls()
    return {name: score_call(anonymizer, request, marked) for name, request in requests.items()}


def _report(scores: dict[str, CallScore], *, markdown: bool, show: int) -> None:
    header = ("Call", "Values", "Leaked", "Tokens", "For nothing", "Their uses")
    rows = [
        (
            name,
            str(s.values),
            f"{s.leaked} of {s.occurrences}",
            str(s.tokens),
            str(len(s.for_nothing)),
            f"{s.wasted_uses} of {s.uses}",
        )
        for name, s in scores.items()
    ]
    total = CallScore()
    for s in scores.values():
        total.values += s.values
        total.occurrences += s.occurrences
        total.leaked += s.leaked
        total.leaked_values.update(s.leaked_values)
        total.tokens += s.tokens
        total.uses += s.uses
        total.for_nothing.update(s.for_nothing)
    rows.append(
        (
            "all",
            str(total.values),
            f"{total.leaked} of {total.occurrences}",
            str(total.tokens),
            str(sum(len(s.for_nothing) for s in scores.values())),
            f"{total.wasted_uses} of {total.uses}",
        )
    )
    if markdown:
        print("| " + " | ".join(header) + " |")
        print("|---" + "|---:" * (len(header) - 1) + "|")
        for row in rows:
            print("| " + " | ".join(row) + " |")
    else:
        widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
        for row in [header, *rows]:
            print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))
    if total.leaked_values:
        print("\nLeaked, by type and value:")
        for (entity_type, value), count in total.leaked_values.most_common():
            print(f"  {entity_type:12} x{count:<3} {value}")
    if show and total.for_nothing:
        print(f"\nMasked for nothing, the {show} used most:")
        for value, count in total.for_nothing.most_common(show):
            print(f"  x{count:<3} {value}")


def main() -> None:
    """Print the scores of the corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner", action="store_true", help="also load the spaCy model")
    parser.add_argument("--spacy-model", default="pl_core_news_md")
    parser.add_argument("--markdown", action="store_true", help="print a markdown table")
    parser.add_argument(
        "--show", type=int, default=0, help="list this many values masked for nothing"
    )
    parser.add_argument(
        "--paranoid", action="store_true", help='the spaCy guard in mode="paranoid"'
    )
    args = parser.parse_args()
    mode: Mode = "paranoid" if args.paranoid else "normal"
    anonymizer = anonymizer_for(with_ner=args.ner, spacy_model=args.spacy_model, mode=mode)
    _report(run(anonymizer), markdown=args.markdown, show=args.show)


if __name__ == "__main__":
    main()
