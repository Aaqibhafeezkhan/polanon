"""The anonymizer the gateway's command builds from its options.

The gateway's process builds one, and so does each of its masking processes with ``--workers``:
this module is never run as ``__main__``, so a process started to mask finds
:func:`worker_anonymizer` here by its name.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from polanon import (
    WELL_KNOWN_NAMES,
    Allowlist,
    Anonymizer,
    Pseudonyms,
    default_guards,
    ner_available,
)
from polanon.allowlist import NAME_TYPES, PROTECTED_SOURCES
from polanon.policy import Mode


def anonymizer(args: argparse.Namespace, env: Mapping[str, str]) -> tuple[Anonymizer, str]:
    """The anonymizer every call is masked with, and a few words on how it detects.

    Raises:
        ValueError: for a POLANON_KEY that is not 64 hexadecimal characters; the key itself is never
            part of the message.
    """
    with_ner = not args.rules_only and ner_available(args.spacy_model)
    detection = f"rules and {args.spacy_model}" if with_ner else "rules alone"
    if not args.rules_only and not with_ner:
        detection += " (spaCy or the model is not installed: places are not masked)"
    mode: Mode = "paranoid" if args.paranoid else "normal"
    if args.paranoid:
        detection += (
            " in paranoid mode" if with_ner else " (--paranoid needs the spaCy model: no change)"
        )
    pseudonyms = None
    if env.get("POLANON_KEY"):
        try:
            pseudonyms = Pseudonyms(bytes.fromhex(env["POLANON_KEY"]))
        except ValueError:
            raise ValueError(
                "POLANON_KEY must be at least 64 hexadecimal characters, such as the output of "
                'python -c "import secrets; print(secrets.token_hex(32))"'
            ) from None
        detection += ", stable tokens"
    operator = [*args.allow, *names_in(args.allow_file)]
    well_known = () if args.no_well_known_names else WELL_KNOWN_NAMES
    allowlist = OperatorNames(well_known, operator) if operator or well_known else None
    if operator:
        detection += f", {len(operator)} of the operator's names left readable"
    anonymizer = Anonymizer(
        default_guards(with_ner=with_ner, spacy_model=args.spacy_model, mode=mode),
        allowlist=allowlist,
        pseudonyms=pseudonyms,
    )
    return anonymizer, detection


def worker_anonymizer(args: argparse.Namespace) -> Anonymizer:
    """The anonymizer of one masking process, built as the gateway built its own.

    It runs in that process, and reads ``POLANON_KEY`` from the environment the process inherits
    from the gateway: the key is never among the arguments, which are pickled to reach it.
    """
    return anonymizer(args, os.environ)[0]


def names_in(path: str | None) -> list[str]:
    """The names of a file, one per line, without blank lines and comments.

    Raises:
        ValueError: when the file cannot be read; the message names the file, not its content.
    """
    if path is None:
        return []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read the names in {path}: {type(exc).__name__}") from None
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


class OperatorNames(Allowlist):
    """The well-known names, and the operator's own, which stay readable even as a person.

    The list of well-known names never releases a person the name guard found, since a brand on
    it must not uncover somebody of the same name. The operator's names are named on purpose, the
    assistant's own or the one of the person it speaks for, and are left readable whatever found
    them; what somebody asked to mask, a denylist entry, stays masked.
    """

    def __init__(self, well_known: Iterable[str], operator: Iterable[str]) -> None:
        operator = list(operator)
        super().__init__([*well_known, *operator])
        self._operator = Allowlist(operator)

    def exempts(self, entity_type: str, fragment: str, source: str = "") -> bool:
        """Leave an operator's name readable, and anything else as the well-known list does."""
        if (
            entity_type in NAME_TYPES
            and source not in PROTECTED_SOURCES
            and self._operator.allows(fragment)
        ):
            return True
        return super().exempts(entity_type, fragment, source)
