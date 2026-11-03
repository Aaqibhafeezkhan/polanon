"""Command line for the anonymizer.

Text comes from the arguments or from standard input. The masked text goes to stdout and the token
mapping to stderr, so a pipeline can keep the two apart:

    polanon "Jan Kowalski, PESEL 50810112354"
    cat umowa.txt | polanon --ner > masked.txt 2> mapping.json

The mapping is what makes the masking reversible, so treat that second file the way you would treat
the original document.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from polanon import Anonymizer, default_guards

DESCRIPTION = "Mask personal data in text before it goes to an LLM; the mapping goes to stderr."

#: Exit status when the language model was asked for but is not installed.
MISSING_MODEL = 3


def main(argv: Sequence[str] | None = None) -> int:
    """Anonymize text from the command line or standard input.

    Returns:
        0 on success, 2 when there was no text to mask, 3 when --ner is given without spaCy.
    """
    parser = argparse.ArgumentParser(prog="polanon", description=DESCRIPTION)
    parser.add_argument("text", nargs="*", help="text to mask; omit to read standard input")
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="read the text from this file, decoding its bytes as UTF-8 or cp1250",
    )
    parser.add_argument(
        "--ner", action="store_true", help="also load the spaCy model for names and places"
    )
    parser.add_argument(
        "--spacy-model",
        default="pl_core_news_md",
        help="spaCy model to load with --ner (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.file is not None and args.text:
        parser.error("--file cannot be combined with text arguments")

    _write_utf8(sys.stdout)
    _write_utf8(sys.stderr)
    if args.file is not None:
        try:
            text = _decode(Path(args.file).read_bytes())
        except OSError as exc:
            print(f"polanon: cannot read {args.file}: {exc.strerror}", file=sys.stderr)
            return 2
    else:
        text = " ".join(args.text) if args.text else _read_stdin()
    if not text.strip():
        print("nothing to mask: pass text as arguments or on standard input", file=sys.stderr)
        return 2
    try:
        guards = default_guards(with_ner=args.ner, spacy_model=args.spacy_model)
    except ModuleNotFoundError as exc:
        print(f"polanon: {exc}", file=sys.stderr)
        return MISSING_MODEL

    result = Anonymizer(guards).anonymize_sync(text)
    print(result.anonymized)
    print(json.dumps(result.mapping, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


def _read_stdin() -> str:
    """Read standard input as UTF-8, whatever the console's code page.

    Windows decodes piped input with the ANSI code page, so "Wiśniewski" arrived garbled, no guard
    recognised it, and it went out unmasked. Files from older Polish software are cp1250, so that is
    the fallback before replacement characters.
    """
    stream = getattr(sys.stdin, "buffer", None)
    if stream is None:
        return sys.stdin.read()
    return _decode(stream.read())


def _decode(raw: bytes) -> str:
    """Decode a file or pipe as UTF-8, then cp1250, before replacement characters."""
    for encoding in ("utf-8-sig", "cp1250"):
        try:
            return str(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    return str(raw.decode("utf-8", errors="replace"))


def _write_utf8(stream: TextIO) -> None:
    """Write UTF-8 even where the console would not: the mapping must stay valid JSON."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
