"""Command line for the anonymizer.

Text comes from the arguments, a file or standard input. The masked text goes to stdout and the
token mapping to stderr, or to a file with --mapping, so a pipeline can keep the two apart:

    polanon "Jan Kowalski, PESEL 50810112354"
    polanon --file umowa.txt --ner --mapping mapping.json > masked.txt

The mapping is what makes the masking reversible, so treat it the way you would treat the original
document.
"""

import argparse
import codecs
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from polanon import Anonymizer, __version__, default_guards

DESCRIPTION = "Mask personal data in text before it goes to an LLM; the mapping goes to stderr."

#: Exit status when the language model was asked for but is not installed.
MISSING_MODEL = 3

# Longer byte order marks first: the UTF-32 LE mark starts with the UTF-16 LE one.
_BYTE_ORDER_MARKS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


class UndecodableInputError(ValueError):
    """The input bytes are not text in an encoding polanon can tell apart for certain."""


def main(argv: Sequence[str] | None = None) -> int:
    """Anonymize text from the command line, a file or standard input.

    Returns:
        0 on success, 2 when there was no text to mask or it could not be read or decoded, or the
        mapping could not be written, 3 when --ner is given without spaCy.
    """
    parser = argparse.ArgumentParser(prog="polanon", description=DESCRIPTION)
    parser.add_argument("--version", action="version", version=f"polanon {__version__}")
    parser.add_argument("text", nargs="*", help="text to mask; omit to read standard input")
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="read the text from this file: UTF-8, UTF-16 or cp1250, told apart by its bytes",
    )
    parser.add_argument(
        "--mapping",
        metavar="PATH",
        help="write the mapping to this file as UTF-8 JSON instead of to stderr",
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
    try:
        text = _read_input(args.file, args.text)
    except OSError as exc:
        print(f"polanon: cannot read {args.file}: {_reason(exc)}", file=sys.stderr)
        return 2
    except UndecodableInputError as exc:
        print(f"polanon: {exc}; save the text as UTF-8 and try again", file=sys.stderr)
        return 2
    if not text.strip():
        print(
            "nothing to mask: pass text as arguments, with --file or on standard input",
            file=sys.stderr,
        )
        return 2
    try:
        guards = default_guards(with_ner=args.ner, spacy_model=args.spacy_model)
    except ModuleNotFoundError as exc:
        print(f"polanon: {exc}", file=sys.stderr)
        return MISSING_MODEL

    result = Anonymizer(guards).anonymize_sync(text)
    if args.mapping is None:
        print(result.anonymized)
        print(json.dumps(result.mapping, ensure_ascii=False, indent=2), file=sys.stderr)
        return 0
    # The mapping first: masked text that cannot be restored is of no use to anyone.
    try:
        _write_mapping(args.mapping, result.mapping)
    except OSError as exc:
        print(f"polanon: cannot write {args.mapping}: {_reason(exc)}", file=sys.stderr)
        return 2
    print(result.anonymized)
    return 0


def _read_input(path: str | None, words: Sequence[str]) -> str:
    """Return the text to mask from the file, the arguments or standard input, in that order."""
    if path is not None:
        return _decode(Path(path).read_bytes())
    if words:
        return " ".join(words)
    return _read_stdin()


def _read_stdin() -> str:
    """Read standard input from its bytes, whatever the console's code page.

    Windows decodes piped input with the ANSI code page, so "Wiśniewski" arrived garbled, no guard
    recognised it, and it went out unmasked.
    """
    if sys.stdin is None:  # started without a standard input handle, e.g. by a task runner
        return ""
    stream = getattr(sys.stdin, "buffer", None)
    if stream is None:
        return sys.stdin.read()
    return _decode(stream.read())


def _decode(raw: bytes) -> str:
    """Decode a file or a pipe, and refuse bytes whose encoding cannot be told for certain.

    A byte order mark decides first. Without one, bytes with many NULs are UTF-16, as Windows
    PowerShell 5.1 writes with ``>``. Otherwise each line is UTF-8 or else cp1250, so one line from
    older Polish software does not turn a whole UTF-8 document into mojibake. Guessing wrong would
    hide every name and number from the guards and send them out unmasked, so whatever is still in
    doubt is refused.

    Raises:
        UndecodableInputError: The bytes are not text in any of those encodings.
    """
    text = _decode_unicode(raw)
    if text is None:
        text = "".join(_decode_line(line) for line in raw.splitlines(keepends=True))
    if "\x00" in text:
        raise UndecodableInputError("the input holds NUL characters and does not look like text")
    return text


def _decode_unicode(raw: bytes) -> str | None:
    """Decode bytes that carry a byte order mark or look like UTF-16; None for anything else."""
    for mark, encoding in _BYTE_ORDER_MARKS:
        if raw.startswith(mark):
            return _strict(raw, encoding)
    nuls = raw.count(b"\x00")
    if not nuls or nuls * 4 < len(raw):
        return None
    # Latin text in UTF-16 has a NUL in every other byte: the odd ones in little-endian order.
    little = raw[1::2].count(b"\x00") >= raw[0::2].count(b"\x00")
    return _strict(raw, "utf-16-le" if little else "utf-16-be")


def _decode_line(line: bytes) -> str:
    for encoding in ("utf-8", "cp1250"):
        try:
            return line.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UndecodableInputError("the input is neither UTF-8 nor cp1250 text")


def _strict(raw: bytes, encoding: str) -> str:
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
        raise UndecodableInputError(f"the input is not valid {encoding.upper()} text") from None


def _write_mapping(path: str, mapping: Mapping[str, str]) -> None:
    """Write the mapping as UTF-8 JSON, created readable by its owner only where the OS allows."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with open(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _reason(exc: OSError) -> str:
    return exc.strerror or type(exc).__name__


def _write_utf8(stream: TextIO | None) -> None:
    """Write UTF-8 even where the console would not, and leave line endings as they came.

    The mapping must stay valid JSON, and on Windows text mode turned the CRLF of a document into
    CR CR LF.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
