"""Measure an AI agent's calls: what reaches the model, what is masked for nothing, what comes back.

:mod:`run` scores the guards on short documents. An agent sends something else: an English system
prompt, the definitions of its tools, JSON from its tools, code and logs, Polish next to English
on one line, and its whole history again in every call. Each call of :mod:`agent_corpus` is
masked here as the gateway masks it, by :func:`polanon.gateway.payloads.mask_chat_request` or,
for the Responses API, :func:`~polanon.gateway.payloads.mask_responses_request`, in one
:class:`polanon.MaskedTurn` with the well-known names allowed, and three things are counted:

``leaked``
    occurrences of a marked value in the masked request, read string by string, JSON in a string
    included. Every one of them would reach the model.
``for nothing``
    tokens whose value is no marked value, nor part of one, nor holds one, and how many times the
    masked request uses them. Each of them hides a word the model needed to read, "Uncheck" of a
    button or "Max" of a chip, and the more often it stands, the worse the model's answer.
``restored``
    the model's answer of :data:`agent_corpus.ANSWERS`, written with the call's tokens, some with
    a case after them, put back three ways: :meth:`polanon.MaskedTurn.restore` and
    ``restore_data``, the gateway's restore of a whole response, and its restore of a stream cut
    into pieces of 1 to 21 characters. Each line of the answer and each argument of its tool call
    is ``exact`` when it reads as written by hand, ``as written`` when a value came back as the
    call wrote it rather than in the case asked for, and ``wrong`` otherwise, a token left among
    them.

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
from itertools import cycle
from typing import Any

from agent_corpus import ANSWERS, Answer, Marked, calls

from polanon import Allowlist, Anonymizer, MaskedTurn, default_guards
from polanon._inflection import CASE_NAMES, inflected
from polanon.allowlist import WELL_KNOWN_NAMES
from polanon.gateway.payloads import (
    ChatStream,
    mask_chat_request,
    mask_responses_request,
    restore_chat_response,
    restore_responses_response,
)
from polanon.guards.names_pl import fold
from polanon.policy import CASED_TOKEN_PATTERN, Mode

#: ``{{value}}`` or ``{{value|case=form}}`` in an answer.
PLACEHOLDER = re.compile(r"\{\{([^{}|]+)(?:\|([^{}=]+)=([^{}]+))?\}\}")

#: The ways an answer is put back, and how each unit of it came out.
PATHS = ("text", "response", "stream")
OUTCOMES = ("exact", "as written", "wrong")

#: The lengths a stream is cut into, in turn, so a token arrives cut at every place.
PIECES = (1, 2, 3, 5, 8, 13, 21)


@dataclass
class RestoreScore:
    """How the answer to one call came back, by path."""

    units: int = 0
    outcomes: dict[str, Counter[str]] = field(
        default_factory=lambda: {path: Counter() for path in PATHS}
    )
    wrong: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, path: str, got: str, ideal: str, as_written: str) -> None:
        """Score one line or argument that came back by one path."""
        if got == ideal:
            self.outcomes[path]["exact"] += 1
        elif got == as_written:
            self.outcomes[path]["as written"] += 1
        else:
            self.outcomes[path]["wrong"] += 1
            self.wrong.append((path, got, ideal))


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
    restore: RestoreScore = field(default_factory=RestoreScore)

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


def _forms(template: str, tokens: dict[str, str]) -> tuple[str, str, str]:
    """Return what the model writes, what a reader should get back, and the values as written."""

    def model(match: re.Match[str]) -> str:
        value, case, form = match[1], match[2], match[3]
        token = tokens.get(value)
        if token is None:  # sent in the clear, so the model writes it in the right form itself
            return form or value
        base = token.split(":")[0].removesuffix(">") + ">"
        return f"{base[:-1]}:{case}>" if case else token

    ideal = PLACEHOLDER.sub(lambda match: match[3] or match[1], template)
    as_written = PLACEHOLDER.sub(lambda match: match[1], template)
    return PLACEHOLDER.sub(model, template), ideal, as_written


def _with_cases(mapping: dict[str, str], templates: list[str]) -> dict[str, str]:
    """Value -> the token the masked text gave it, a person's with a case included.

    A form of a person in another case is no value of the mapping: it went out as the token of
    the form kept, with its case, ``<OSOBA_1:celownik>``, and the model copies that.
    """
    tokens = {value: token for token, value in mapping.items() if value != token}
    people = [(token, kept) for token, kept in mapping.items() if token.startswith("<OSOBA_")]
    for template in templates:
        for match in PLACEHOLDER.finditer(template):
            value = match[1]
            if value in tokens:
                continue
            for token, kept in people:
                case = next(
                    (name for name in CASE_NAMES if inflected(kept, "PERSON", name) == value), None
                )
                if case is not None and kept != token:
                    tokens[value] = f"{token[:-1]}:{case}>"
                    break
    return tokens


def _pieces(text: str) -> Iterator[str]:
    lengths = cycle(PIECES)
    start = 0
    while start < len(text):
        end = start + next(lengths)
        yield text[start:end]
        start = end


def _streamed(turn: MaskedTurn, content: str, arguments: str | None) -> tuple[str, str]:
    """Restore an answer cut into pieces, as the gateway restores a stream, and join it."""
    stream = ChatStream(turn.mapping)
    chunks: list[dict[str, Any]] = [{"choices": [{"index": 0, "delta": {"role": "assistant"}}]}]
    chunks += [
        {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
        for piece in _pieces(content)
    ]
    if arguments is not None:
        chunks += [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]},
                        "finish_reason": None,
                    }
                ]
            }
            for piece in _pieces(arguments)
        ]
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    text, args = [], []
    for chunk in chunks:
        delta = stream.chunk(chunk)["choices"][0]["delta"]
        text.append(delta.get("content") or "")
        for call in delta.get("tool_calls") or []:
            args.append(call["function"].get("arguments") or "")
    assert stream.rest() is None
    return "".join(text), "".join(args)


def _response(responses_api: bool, content: str, answer: Answer, arguments: str | None) -> Any:
    if responses_api:
        output: list[dict[str, Any]] = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        ]
        if arguments is not None:
            output.append(
                {
                    "type": "function_call",
                    "call_id": "call_answer",
                    "name": answer.tool,
                    "arguments": arguments,
                }
            )
        return {"object": "response", "output": output}
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if arguments is not None:
        message["tool_calls"] = [
            {
                "id": "call_answer",
                "type": "function",
                "function": {"name": answer.tool, "arguments": arguments},
            }
        ]
    return {"object": "chat.completion", "choices": [{"index": 0, "message": message}]}


def score_restore(turn: MaskedTurn, answer: Answer, *, responses_api: bool) -> RestoreScore:
    """Put one answer back every way the gateway does, and score each line and argument."""
    tokens = _with_cases(turn.mapping, [*answer.lines, *(answer.arguments or {}).values()])
    lines = [_forms(line, tokens) for line in answer.lines]
    content = "\n".join(model for model, _, _ in lines)
    argument_forms = {key: _forms(value, tokens) for key, value in (answer.arguments or {}).items()}
    arguments = (
        json.dumps(
            {key: model for key, (model, _, _) in argument_forms.items()}, ensure_ascii=False
        )
        if answer.arguments is not None
        else None
    )
    score = RestoreScore(units=len(lines) + len(argument_forms))

    def add(path: str, text: str, restored_arguments: dict[str, Any] | None) -> None:
        got_lines = text.split("\n")
        for index, (_, ideal, as_written) in enumerate(lines):
            got = got_lines[index] if index < len(got_lines) else ""
            score.add(path, got, ideal, as_written)
        for key, (_, ideal, as_written) in argument_forms.items():
            got = str((restored_arguments or {}).get(key, ""))
            score.add(path, got, ideal, as_written)

    plain_arguments = {key: model for key, (model, _, _) in argument_forms.items()}
    add("text", turn.restore(content), turn.restore_data(plain_arguments))

    restored = (restore_responses_response if responses_api else restore_chat_response)(
        turn, _response(responses_api, content, answer, arguments)
    )
    if responses_api:
        text = restored["output"][0]["content"][0]["text"]
        args_text = restored["output"][1]["arguments"] if arguments is not None else None
    else:
        message = restored["choices"][0]["message"]
        text = message["content"]
        args_text = message["tool_calls"][0]["function"]["arguments"] if arguments else None
    add("response", text, json.loads(args_text) if args_text else None)

    if not responses_api:
        text, args_text = _streamed(turn, content, arguments)
        add("stream", text, json.loads(args_text) if args_text else None)
    return score


def score_call(
    anonymizer: Anonymizer, request: dict[str, Any], marked: list[Marked], answer: Answer | None
) -> CallScore:
    """Mask one call as the gateway does, count what leaked and what hides nothing, restore."""
    turn = MaskedTurn(anonymizer)
    responses_api = "input" in request
    masked = (mask_responses_request if responses_api else mask_chat_request)(turn, request)
    texts = list(strings(masked))
    required = [m for m in marked if m.required]
    written = occurrences(list(strings(request)), [m.value for m in required])
    score = CallScore(values=len(written), occurrences=sum(written.values()))
    leaked = occurrences(texts, [m.value for m in required])
    types = {m.value: m.entity_type for m in required}
    for value, count in leaked.items():
        score.leaked += count
        score.leaked_values[(types[value], value)] += count
    token = re.compile(CASED_TOKEN_PATTERN)
    used = Counter(
        f"<{match['label']}_{match['n']}>" for text in texts for match in token.finditer(text)
    )
    for name, value in turn.mapping.items():
        if value == name:
            continue
        score.tokens += 1
        score.uses += used[name]
        if not _related(value, marked):
            score.for_nothing[f"{name.split('_')[0].lstrip('<')} {value}"] += used[name]
    if answer is not None:
        score.restore = score_restore(turn, answer, responses_api=responses_api)
    return score


def anonymizer_for(
    *,
    with_ner: bool,
    spacy_model: str = "pl_core_news_md",
    mode: Mode = "normal",
    ml_filter: bool = True,
) -> Anonymizer:
    """The guards the gateway masks with, and its list of well-known names."""
    guards = default_guards(
        with_ner=with_ner, spacy_model=spacy_model, mode=mode, ml_filter=ml_filter
    )
    return Anonymizer(guards, allowlist=Allowlist(WELL_KNOWN_NAMES))


def run(anonymizer: Anonymizer) -> dict[str, CallScore]:
    """Score every call of the corpus."""
    requests, marked = calls()
    return {
        name: score_call(anonymizer, request, marked, ANSWERS.get(name))
        for name, request in requests.items()
    }


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]], *, markdown: bool) -> None:
    if markdown:
        print("| " + " | ".join(header) + " |")
        print("|---" + "|---:" * (len(header) - 1) + "|")
        for row in rows:
            print("| " + " | ".join(row) + " |")
        return
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))


def _report(scores: dict[str, CallScore], *, markdown: bool, show: int) -> None:
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
    leaked_values: Counter[tuple[str, str]] = Counter()
    for_nothing: Counter[str] = Counter()
    for s in scores.values():
        leaked_values.update(s.leaked_values)
        for_nothing.update(s.for_nothing)
    rows.append(
        (
            "all",
            str(sum(s.values for s in scores.values())),
            f"{sum(s.leaked for s in scores.values())} of "
            f"{sum(s.occurrences for s in scores.values())}",
            str(sum(s.tokens for s in scores.values())),
            str(sum(len(s.for_nothing) for s in scores.values())),
            f"{sum(s.wasted_uses for s in scores.values())} of "
            f"{sum(s.uses for s in scores.values())}",
        )
    )
    _table(
        ("Call", "Values", "Leaked", "Tokens", "For nothing", "Their uses"), rows, markdown=markdown
    )

    print()
    restore_rows = []
    totals = {path: Counter[str]() for path in PATHS}
    for name, s in scores.items():
        cells = [name, str(s.restore.units)]
        for path in PATHS:
            outcome = s.restore.outcomes[path]
            totals[path].update(outcome)
            cells.append("/".join(str(outcome[o]) for o in OUTCOMES) if outcome else "-")
        restore_rows.append(tuple(cells))
    restore_rows.append(
        (
            "all",
            str(sum(s.restore.units for s in scores.values())),
            *("/".join(str(totals[path][o]) for o in OUTCOMES) for path in PATHS),
        )
    )
    _table(
        ("Restored", "Units", *(f"{path}: exact/as written/wrong" for path in PATHS)),
        restore_rows,
        markdown=markdown,
    )

    if leaked_values:
        print("\nLeaked, by type and value:")
        for (entity_type, value), count in leaked_values.most_common():
            print(f"  {entity_type:12} x{count:<3} {value}")
    wrong = [(name, *item) for name, s in scores.items() for item in s.restore.wrong]
    if wrong:
        print("\nRestored wrong:")
        for name, path, got, ideal in wrong:
            print(f"  {name}/{path}: got {got!r}\n  {' ' * len(name)}  want {ideal!r}")
    if show and for_nothing:
        print(f"\nMasked for nothing, the {show} used most:")
        for value, count in for_nothing.most_common(show):
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
    parser.add_argument(
        "--no-ml-filter", action="store_true", help="the spaCy guard with its ML filter off"
    )
    args = parser.parse_args()
    mode: Mode = "paranoid" if args.paranoid else "normal"
    anonymizer = anonymizer_for(
        with_ner=args.ner, spacy_model=args.spacy_model, mode=mode, ml_filter=not args.no_ml_filter
    )
    _report(run(anonymizer), markdown=args.markdown, show=args.show)


if __name__ == "__main__":
    main()
