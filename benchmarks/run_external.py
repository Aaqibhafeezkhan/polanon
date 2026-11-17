"""Measure polanon, and Presidio if asked, on public Polish data polanon was never tuned on.

    python benchmarks/run_external.py                      # OpenPII and KPWr, polanon only
    python benchmarks/run_external.py --presidio           # Presidio set up as compare_presidio.py
    python benchmarks/run_external.py --teryt              # every street in GUS's TERYT catalogue

The data is downloaded on first use into ~/.cache/polanon/benchmarks (or --data) and never kept in
this repository:

- ai4privacy/pii-masking-openpii-1m, CC-BY-4.0: the Polish rows of its validation split, about
  9,300 texts written by a language model with annotated values. They are shuffled with a fixed
  seed; the first 1,000 were used to find what polanon missed ("dev"), the next 1,000 never were
  ("test", the default). Most of its numbers are random digits, so identifiers are counted apart
  by whether they pass their checksum, as a real PESEL, NIP, ID card or payment card always does.
  Dates, ages, sex, gender and titles are left out: neither tool masks them on their own.
- CLARIN-PL's KPWr, CC-BY-3.0: the test part, 4,323 sentences of real Polish text annotated with
  named entities; people, cities, streets and companies are read. Sentences are rebuilt from their
  tokens the way the text was written. Only recall counts: KPWr names much more (countries,
  institutions, titles) than these four, so a span outside them is no error.
- GUS's TERYT street catalogue (ULIC), CC-BY-4.0: every street name, written with a house number
  in a form, before a postcode and in a sentence; masked whole means in all three.

Scoring is benchmarks/run.py's: a value counts as masked when every character of it is covered.
"""

from __future__ import annotations

import argparse
import csv
import http.cookiejar
import io
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from run import EVALUATED, GoldEntity, Score, evaluate, guards_for, merge_spans

from polanon import Guard

OPENPII = (
    "https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m/resolve/main/data/"
    "validation.jsonl"
)
KPWR = (
    "https://huggingface.co/datasets/clarin-pl/kpwr-ner/resolve/main/data/kpwr-ner-n82-{part}.iob"
)
#: KPWr's parts as its files name them: the test part, and the part meant for training.
KPWR_PARTS = {"test": "test", "train": "train-tune"}
TERYT = (
    "https://eteryt.stat.gov.pl/eTeryt/rejestr_teryt/udostepnianie_danych/baza_teryt/"
    "uzytkownicy_indywidualni/pobieranie/pliki_pelne.aspx?contrast=default"
)
AGENT = {"User-Agent": "polanon-benchmarks"}
SEED = 2026
SAMPLE = 1000

#: OpenPII's labels as polanon's types; the rest are not identifying on their own.
OPENPII_TYPES = {
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "CITY": "LOCATION",
    "STREET": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "ZIPCODE": "POSTAL_CODE",
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "SOCIALNUM": "PESEL",
    "TAXNUM": "NIP",
    "IDCARDNUM": "PL_ID",
    "PASSPORTNUM": "PASSPORT",
    "DRIVERLICENSENUM": "DRIVING_LICENCE",
}
#: KPWr's labels read here, as the names they are reported under.
KPWR_TYPES = {
    "nam_liv_person": "PERSON",
    "nam_loc_gpe_city": "CITY",
    "nam_fac_road": "STREET",
    "nam_org_company": "COMPANY",
}
#: The register's kinds of street, as an address writes them.
TERYT_PREFIX = {
    "ul.": "ul.",
    "al.": "al.",
    "pl.": "pl.",
    "os.": "os.",
    "rondo": "rondo",
    "skwer": "skwer",
    "park": "park",
    "rynek": "Rynek",
    "bulw.": "bulw.",
    "droga": "droga",
    "ogród": "ogród",
    "inne": "",
}
TERYT_TEMPLATES = (
    "Adres: {street} {house}",
    "{street} {house}, 00-950 Miasto",
    "Mieszka przy {street} {house}.",
)


# ---------- downloads ----------


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = dict(AGENT)
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - fixed https URLs
    with urllib.request.urlopen(request, timeout=300) as r:  # noqa: S310
        return bytes(r.read())


def _size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers=AGENT)  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return int(response.headers["Content-Length"])


def _language_at(offset: int) -> str:
    for line in _get(OPENPII, offset, offset + 400_000).decode("utf-8", "replace").split("\n")[1:]:
        try:
            return str(json.loads(line)["language"])
        except (ValueError, KeyError):
            continue
    raise RuntimeError(f"no row near byte {offset}")


def _block_edge(total: int, *, after: bool) -> int:
    """A byte offset just before the Polish block (or just before what follows it).

    The validation split is sorted by language, so the block is found by bisection on offsets.
    """
    low, high = 0, total - 1
    while high - low > 2_000_000:
        middle = (low + high) // 2
        language = _language_at(middle)
        if (language > "pl") if after else (language >= "pl"):
            high = middle
        else:
            low = middle
    return low


def openpii_rows(data: Path) -> list[dict[str, Any]]:
    """The Polish rows of OpenPII's validation split, downloaded once."""
    path = data / "openpii-pl-validation.jsonl"
    if not path.exists():
        print("downloading the Polish rows of OpenPII's validation split...", file=sys.stderr)
        total = _size(OPENPII)
        start, end = _block_edge(total, after=False), _block_edge(total, after=True) + 2_400_000
        rows: dict[str, dict[str, Any]] = {}
        for offset in range(start, min(end, total), 8_000_000):
            chunk = _get(OPENPII, offset, min(offset + 8_400_000, total - 1))
            for line in chunk.decode("utf-8", "replace").split("\n"):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("language") == "pl":
                    rows[row["uid"]] = row
        data.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()),
            encoding="utf-8",
        )
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def kpwr_sentences(data: Path, part: str = "test") -> list[dict[str, Any]]:
    """KPWr's sentences of one part, "test" or "train", as tokens and IOB tags, downloaded once.

    Each sentence carries the name of the document it comes from under "doc", so that the
    sentences of one text can be kept together.
    """
    name = KPWR_PARTS[part]
    path = data / f"kpwr-ner-n82-{name}.iob"
    if not path.exists():
        print(f"downloading KPWr's {part} part...", file=sys.stderr)
        data.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(KPWR.format(part=name)))
    sentences: list[dict[str, Any]] = []
    tokens: list[str] = []
    tags: list[str] = []
    doc = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("-DOCSTART"):
            if line.startswith("-DOCSTART FILE"):
                doc = line.split(maxsplit=2)[-1]
            continue
        if not line.strip():
            if tokens:
                sentences.append({"tokens": tokens, "ner": tags, "doc": doc})
                tokens, tags = [], []
            continue
        parts = line.split("\t")
        tokens.append(parts[0])
        tags.append(parts[-1])
    if tokens:
        sentences.append({"tokens": tokens, "ner": tags, "doc": doc})
    return sentences


def teryt_streets(data: Path) -> list[tuple[str, str]]:
    """Every (kind, name) in GUS's TERYT street catalogue, downloaded once.

    The portal hands the file out on an ASP.NET postback, so this posts the page's form back
    with the download button as its target, as a browser's click does.
    """
    path = data / "ULIC.csv"
    if not path.exists():
        print("downloading GUS's TERYT street catalogue...", file=sys.stderr)
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        opener.addheaders = list(AGENT.items())
        page = opener.open(TERYT, timeout=60).read().decode("utf-8", "replace")
        fields = dict(
            re.findall(r'<input type="hidden" name="([^"]+)" id="[^"]*" value="([^"]*)"', page)
        )
        fields |= {"__EVENTTARGET": "ctl00$body$BULICUrzedowyPobierz", "__EVENTARGUMENT": ""}
        body = urllib.parse.urlencode(fields).encode()
        archive = opener.open(urllib.request.Request(TERYT, data=body), timeout=300).read()
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            name = next(n for n in zipped.namelist() if n.endswith(".csv"))
            data.mkdir(parents=True, exist_ok=True)
            path.write_bytes(zipped.read(name))
    rows = csv.DictReader(path.open(encoding="utf-8-sig"), delimiter=";")
    names = {
        (row["CECHA"], " ".join(p for p in (row["NAZWA_2"], row["NAZWA_1"]) if p).strip())
        for row in rows
    }
    return sorted(names)


# ---------- gold values ----------


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def pesel_valid(text: str) -> bool:
    """The PESEL checksum over the first ten digits."""
    d = _digits(text)
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    return len(d) == 11 and (
        10 - sum(int(a) * w for a, w in zip(d, weights, strict=False)) % 10
    ) % 10 == int(d[-1])


def nip_valid(text: str) -> bool:
    """The NIP checksum."""
    d = _digits(text)
    weights = (6, 5, 7, 2, 3, 4, 5, 6, 7)
    return len(d) == 10 and sum(int(a) * w for a, w in zip(d, weights, strict=False)) % 11 == int(
        d[-1]
    )


def luhn_valid(text: str) -> bool:
    """The payment card checksum."""
    d = [int(c) for c in _digits(text)][::-1]
    total = sum(x if i % 2 == 0 else (x * 2 - 9 if x > 4 else x * 2) for i, x in enumerate(d))
    return len(d) >= 12 and total % 10 == 0


def id_card_valid(text: str) -> bool:
    """The Polish ID card's check digit, the fourth character."""
    value = re.sub(r"\s", "", text).upper()
    if not re.fullmatch(r"[A-Z]{3}\d{6}", value):
        return False
    numbers = [ord(c) - 55 if c.isalpha() else int(c) for c in value]
    rest = numbers[:3] + numbers[4:]
    return (
        sum(n * w for n, w in zip(rest, (7, 3, 1, 7, 3, 1, 7, 3), strict=True)) % 10 == numbers[3]
    )


CHECKSUMS = {
    "PESEL": pesel_valid,
    "NIP": nip_valid,
    "CREDIT_CARD": luhn_valid,
    "PL_ID": id_card_valid,
}


def openpii_gold(row: dict[str, Any]) -> list[GoldEntity]:
    """A row's values as polanon's types, identifiers split by their checksum."""
    entities = []
    for mask in row["privacy_mask"]:
        kind = OPENPII_TYPES.get(mask["label"])
        if kind is None:
            continue
        if kind in CHECKSUMS:
            kind = f"{kind} {'valid' if CHECKSUMS[kind](mask['value']) else 'invalid'}"
        entities.append(GoldEntity(mask["start"], mask["end"], mask["value"], kind))
    return sorted(entities, key=lambda entity: entity.start)


_NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", ")", "]", "”", "’", "'", "%", "…"}  # noqa: RUF001
_NO_SPACE_AFTER = {"(", "[", "„", "“"}


def _written_together(tokens: list[str], index: int) -> bool:
    """Whether a token was written right after the one before it."""
    word, previous = tokens[index], tokens[index - 1]
    if word == "-":
        # "Chłoń - Domińczak" was one hyphenated surname; a dash between sentences keeps spaces.
        return (
            index + 1 < len(tokens) and previous[:1].isupper() and tokens[index + 1][:1].isupper()
        )
    if previous == "-":
        return index >= 2 and tokens[index - 2][:1].isupper() and word[:1].isupper()
    if previous in ("’", "'") and word[:1].islower():  # noqa: RUF001
        return True  # an apostrophe and the ending after it, as in an English name declined
    return word in _NO_SPACE_BEFORE or previous in _NO_SPACE_AFTER


def kpwr_sentence(
    row: dict[str, Any], types: dict[str, str] | None = KPWR_TYPES
) -> tuple[str, list[GoldEntity]]:
    """A sentence as written, rebuilt from its tokens, and the values of `types` in it.

    With `types` None every name KPWr annotates is returned, under its own label.
    """
    tokens = row["tokens"]
    text, starts = "", []
    for index, token in enumerate(tokens):
        if text and not _written_together(tokens, index):
            text += " "
        starts.append(len(text))
        text += token
    gold: list[list[Any]] = []
    current: list[Any] | None = None
    for index, (token, tag) in enumerate(zip(tokens, row["ner"], strict=True)):
        label = tag[2:]
        kind = None if tag == "O" else label if types is None else types.get(label)
        if tag.startswith("I-") and current and current[2] == kind:
            current[1] = starts[index] + len(token)
            continue
        if current:
            gold.append(current)
            current = None
        if tag.startswith("B-") and kind:
            current = [starts[index], starts[index] + len(token), kind]
    if current:
        gold.append(current)
    return text, [GoldEntity(s, e, text[s:e], k) for s, e, k in gold]


# ---------- scoring ----------


def score(
    documents: list[tuple[str, list[GoldEntity]]],
    guards: list[Guard],
    *,
    union: bool,
    recall_only: bool = False,
) -> tuple[dict[str, Score], int, float]:
    """Totals per type, spans outside the annotations, and seconds.

    A text with nothing annotated counts too: a span in it is a false alarm. With `recall_only`
    such texts are skipped, for data that annotates more than is read from it.
    """
    totals: dict[str, Score] = {}
    extra = 0
    started = time.perf_counter()
    for text, gold in documents:
        if recall_only and not gold:
            continue
        scores, outside = evaluate(text, gold, guards, EVALUATED, union=union)
        extra += outside
        for kind, s in scores.items():
            total = totals.setdefault(kind, Score())
            total.gold += s.gold
            total.masked += s.masked
            total.partial += s.partial
            total.missed += s.missed
    return totals, extra, time.perf_counter() - started


def configurations(
    presidio: bool, without_ml_filter: bool = False
) -> list[tuple[str, list[Guard], bool]]:
    """The rules alone, with the spaCy model and paranoid when it is installed, and Presidio.

    The spaCy model runs with its ML filter, as it does by default; with `without_ml_filter`, also
    without it, for comparison.
    """
    from polanon import ner_available

    found: list[tuple[str, list[Guard], bool]] = [("polanon, rules", guards_for(False, ""), False)]
    if ner_available("pl_core_news_md"):
        found.append(("polanon + pl_core_news_md", guards_for(True, "pl_core_news_md"), False))
        found.append(("polanon paranoid", guards_for(True, "pl_core_news_md", "paranoid"), False))
        if without_ml_filter:
            guards = guards_for(True, "pl_core_news_md", "normal", ml_filter=False)
            found.append(("polanon, ML filter off", guards, False))
    if presidio:
        from compare_presidio import PresidioGuard

        found.append(("Presidio", [PresidioGuard("pl_core_news_md")], True))
    return found


def table(
    title: str,
    documents: list[tuple[str, list[GoldEntity]]],
    setups: list[tuple[str, list[Guard], bool]],
    *,
    recall_only: bool = False,
) -> None:
    """Print one dataset's markdown table, a line per type, and the totals."""
    results = {
        name: score(documents, guards, union=union, recall_only=recall_only)
        for name, guards, union in setups
    }
    kinds: dict[str, int] = {}
    for totals, _, _ in results.values():
        for kind, s in totals.items():
            kinds[kind] = s.gold
    print(f"\n### {title}\n")
    print("| Type | Annotated | " + " | ".join(results) + " |")
    print("|---|---:|" + "---:|" * len(results))
    for kind in sorted(kinds, key=lambda k: -kinds[k]):
        cells = []
        for totals, _, _ in results.values():
            s = totals.get(kind, Score())
            part = f", {s.partial} in part" if s.partial else ""
            cells.append(f"{s.masked} ({100 * s.masked / kinds[kind]:.0f}%){part}")
        print(f"| {kind} | {kinds[kind]} | " + " | ".join(cells) + " |")
    print()
    for name, (totals, extra, seconds) in results.items():
        gold = sum(s.gold for s in totals.values())
        masked = sum(s.masked for s in totals.values())
        outside = "" if recall_only else f"{extra} spans outside the annotations, "
        print(
            f"- {name}: {masked} of {gold} masked whole ({100 * masked / gold:.1f}%), "
            f"{outside}{seconds:.0f} s"
        )


def teryt(streets: list[tuple[str, str]]) -> None:
    """How many street names the rules mask whole in all three ways an address is written."""
    guards = guards_for(False, "")
    by_kind: dict[str, list[int]] = {}
    for index, (kind, name) in enumerate(streets):
        street = f"{TERYT_PREFIX.get(kind, kind)} {name}".strip()
        house = str(1 + index % 120) + ("A" if index % 7 == 0 else "")
        whole = True
        for template in TERYT_TEMPLATES:
            text = template.format(street=street, house=house)
            start = text.index(name)
            end = text.index(house, start + len(name)) + len(house)
            spans = merge_spans([s for g in guards for s in g.detect(text, EVALUATED)], text)
            covered = {
                i for s in spans if s.entity_type == "ADDRESS" for i in range(s.start, s.end)
            }
            whole = whole and all(i in covered for i in range(start, end))
        counts = by_kind.setdefault(kind, [0, 0])
        counts[0] += whole
        counts[1] += 1
    done, total = sum(c[0] for c in by_kind.values()), sum(c[1] for c in by_kind.values())
    print(f"\n### TERYT streets\n\n{done} of {total} masked whole ({100 * done / total:.1f}%)\n")
    print("| Kind | Streets | Masked whole |\n|---|---:|---:|")
    for kind, (ok, n) in sorted(by_kind.items(), key=lambda item: -item[1][1]):
        print(f"| {kind} | {n} | {100 * ok / n:.1f}% |")


def main() -> int:
    """Download what is missing, score and print markdown tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/polanon/benchmarks")
    parser.add_argument("--part", choices=("test", "dev"), default="test")
    parser.add_argument("--presidio", action="store_true", help="also score Presidio")
    parser.add_argument("--teryt", action="store_true", help="also score every TERYT street")
    parser.add_argument(
        "--without-ml-filter", action="store_true", help="also with the ML filter off"
    )
    args = parser.parse_args()
    setups = configurations(args.presidio, args.without_ml_filter)

    rows = openpii_rows(args.data)
    random.Random(SEED).shuffle(rows)  # noqa: S311 - a fixed sample, not a secret
    chosen = rows[:SAMPLE] if args.part == "dev" else rows[SAMPLE : 2 * SAMPLE]
    documents = [(row["source_text"], openpii_gold(row)) for row in chosen]
    table(f"OpenPII, Polish rows, {args.part} part ({len(chosen)} texts)", documents, setups)

    sentences = kpwr_sentences(args.data)
    table(
        f"KPWr, test part ({len(sentences)} sentences)",
        [kpwr_sentence(row) for row in sentences],
        setups,
        recall_only=True,
    )
    if args.teryt:
        teryt(teryt_streets(args.data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
