"""The canonical form of a value: what stays the same however the value is written.

A stable pseudonym is computed from it (see :mod:`polanon.pseudonyms`), so a person keeps one
token across documents that name them in different grammatical cases: "Jan Kowalski", "Jana
Kowalskiego" and "z Janem Kowalskim" all read as "jan kowalski". A title in front of a name and
the order of first name and surname do not count, and an identifier is one number however its
groups are written and with or without its country prefix.

The rules only ever join spellings of one value. A form that could belong to two names, such as
"Aleksandrze Dudzie" (Aleksander or Aleksandra), is taken as written: two tokens for one person
cost a missed search, while one token for two people would show one of them in place of the other.

Spellings that the search for known values tells apart stay apart here as well: a name typed
without Polish letters, and a one-word name in lowercase. Under a shared token, the next turn of a
conversation would not look for the second spelling, and it would go out in the clear.

Changing what this module returns changes every token computed from it. ``tests/test_canonical.py``
pins the rules and the name lists they read; changing either is a new pseudonym scheme.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from functools import cache

from polanon._matching import fold as lowercase
from polanon._matching import folds, reading, value_keys
from polanon.guards import names_pl as names
from polanon.guards.names_pl import CASES, Tag

#: Identifiers that are their digits alone: "526-000-00-05" and "PL5260000005" are one NIP.
_DIGITS_ONLY = frozenset({"PESEL", "NIP", "REGON", "KRS", "CREDIT_CARD"})

_NON_DIGIT = re.compile(r"\D")
_SEPARATOR = re.compile(r"[\W_]")
#: A Polish number written with its country code: "+48 600 100 200" is "600 100 200".
_POLISH_PREFIX = re.compile(r"^(?:\+|00)48(?=\d{9}$)")
_EMAIL_AT = re.compile(r"\s*[\[({]\s*(?:at|małpa|malpa)\s*[\])}]\s*")
_EMAIL_DOT = re.compile(r"\s*[\[({]\s*(?:dot|kropka)\s*[\])}]\s*")
_LINK_START = re.compile(r"^(?:[a-z][a-z0-9+.-]*://)?(?:www\.)?")
#: Spaces inside an abbreviated legal form: "sp. z o. o." is "sp. z o.o.".
_SPACED_ABBREVIATION = re.compile(r"(?<=\b\w\.)\s+(?=\w\.)")

#: Marks a one-word value written all in lowercase, which keeps a token of its own.
_LOWERCASE_WORD = "\x00lowercase"

#: Polish letters to their base letters, in both cases, for a name typed without them.
_PLAIN = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def canonical(value: str, entity_type: str) -> str:
    """Return what makes two spellings of `value` the same value of `entity_type`.

    Args:
        value: a masked value, as written.
        entity_type: its type.
    """
    read = reading(value)
    if not folds(entity_type):
        return read  # a password, or a custom identifier: only as written
    if entity_type == "PERSON":
        return _person(read)
    folded = lowercase(read)
    if entity_type in _DIGITS_ONLY:
        return _NON_DIGIT.sub("", folded)
    if entity_type == "IBAN":
        compact = _SEPARATOR.sub("", folded)
        return "pl" + compact if compact.isdigit() and len(compact) == 26 else compact
    if entity_type == "PHONE":
        compact = re.sub(r"[^\d+]", "", folded)
        return _POLISH_PREFIX.sub("", compact).replace("+", "00", 1)
    if entity_type == "EMAIL":
        return _EMAIL_DOT.sub(".", _EMAIL_AT.sub("@", folded)).replace(" ", "")
    if entity_type == "LINK":
        return _LINK_START.sub("", folded).rstrip("/")
    if entity_type == "MAC_ADDRESS":
        return _SEPARATOR.sub("", folded)
    if entity_type == "ORG":
        read = _SPACED_ABBREVIATION.sub("", read)
    keys = value_keys(read, entity_type)
    if len(keys) > 1:
        return keys[-1][1]  # an identifier with its groups taken out
    kind, key = keys[0]
    return key + _LOWERCASE_WORD if kind == "lowercase word" else key


def base_form(value: str, entity_type: str) -> str:
    """Return the form of `value` to show for its token wherever it came from.

    For a person whose name the rules can read, that is the name in the nominative without a title,
    written with capitals where it was written all in capitals or all in lowercase: "Jan Kowalski"
    for "JANA KOWALSKIEGO". A model's answer puts the token in a sentence of its own, where the
    nominative reads best, and it is the same whichever document named the person first. Any
    other value is returned as written.
    """
    if entity_type != "PERSON":
        return value.strip()
    written = reading(value).split(" ")
    words = _without_titles(written)
    nominative = _nominative(words)
    if nominative is None:
        return " ".join(words) if len(words) < len(written) else value.strip()
    return " ".join(_capitalised(word) for word in nominative.split(" "))


def _person(read: str) -> str:
    words = _without_titles(read.split(" "))
    name = " ".join(words)
    nominative = _nominative(words)
    key = lowercase(name if nominative is None else nominative)
    lowercase_word = value_keys(name, "PERSON")[0][0] == "lowercase word"
    return key + _LOWERCASE_WORD if lowercase_word else key


def _without_titles(words: list[str]) -> list[str]:
    """Drop the forms of address and titles in front of a name: "Pan", "dr", "Frau"."""
    while len(words) > 1 and _is_title(words[0]):
        words = words[1:]
    return words


def _is_title(word: str) -> bool:
    key = names.fold(word.rstrip("."))
    return key in names.HONORIFICS or key in names.FOREIGN_TITLES


def _nominative(words: list[str]) -> str | None:
    """Return a name of one or two words in the nominative, or None when it is not certain."""
    if len(words) == 1:
        word = words[0]
        found = {lemma for lemmas in _first_name(word).values() for lemma in lemmas}
        found |= {lemma for lemmas in _surname(word).values() for lemma in lemmas}
        return found.pop() if len(found) == 1 else None
    if len(words) == 2:
        pairs = _agreeing(words[0], words[1]) | _agreeing(words[1], words[0])
        if len(pairs) == 1:
            first, last = pairs.pop()
            return f"{first} {last}"
    return None


def _agreeing(first: str, last: str) -> set[tuple[str, str]]:
    """Return the nominatives of `first` as a first name and `last` as a surname that agree."""
    first_names = _first_name(first)
    surnames = _surname(last)
    return {
        (first_name, surname)
        for tag in first_names.keys() & surnames.keys()
        for first_name in first_names[tag]
        for surname in surnames[tag]
    }


def _first_name(word: str) -> dict[Tag, set[str]]:
    """Return the readings of a word as a first name, each with the nominative it comes from."""
    readings = [_FIRST_NAMES().get(names.fold(part), ()) for part in word.split("-")]
    return _joined(word, readings)


def _surname(word: str) -> dict[Tag, set[str]]:
    """Return the readings of a word as a surname, each with its nominative, as the guard reads it.

    A double surname reads in the cases its parts agree on: "Nowaka-Kowalskiego" is
    "Nowak-Kowalski".
    """
    parts = word.split("-")
    if names.fold(parts[-1]) in names._NOT_NAMES:
        return {}
    return _joined(word, [_surname_part(part) for part in parts])


def _joined(word: str, readings: Sequence[Iterable[tuple[Tag, str]]]) -> dict[Tag, set[str]]:
    """Combine the readings of the parts of a hyphenated word into readings of the whole of it."""
    parts = word.split("-")
    by_part: list[dict[Tag, set[str]]] = []
    for part, part_readings in zip(parts, readings, strict=True):
        table: dict[Tag, set[str]] = defaultdict(set)
        for tag, lemma in part_readings:
            table[tag].add(_like(lemma, part))
        if not table:
            return {}
        by_part.append(table)
    tags = set.intersection(*(set(table) for table in by_part))
    joined: dict[Tag, set[str]] = {}
    for tag in tags:
        lemmas = [""]
        for table in by_part:
            lemmas = [
                f"{done}-{lemma}" if done else lemma for done in lemmas for lemma in table[tag]
            ]
        joined[tag] = set(lemmas)
    return joined


def _surname_part(part: str) -> list[tuple[Tag, str]]:
    """Return the readings of one word, or one part of a double surname, with its nominative."""
    key = names.fold(part)
    if len(key) != len(part):
        return []
    listed = _LISTED_SURNAMES().get(key)
    if listed is not None:
        return list(listed)
    if len(key) < 4:
        return []
    adjectival = names._ADJECTIVAL.search(key)
    if adjectival is not None:
        stem = adjectival.start("ending")
        if key[:stem] in names._COMMON_ADJECTIVES:
            return []
        return [
            (tag, part[:stem] + _cased("i" if tag[0] == "m" else "a", part))
            for tag in names._ADJECTIVAL_TAGS[adjectival.group("ending")]
        ]
    oblique = names._NOUN_OBLIQUE.search(key)
    if oblique is not None:
        ending = oblique.group("k") or oblique.group("wicz")
        nominative = part[: -len(ending)]
        if oblique.group("k") and names.fold(nominative)[-2:-1] not in names._VOWELS:
            nominative = nominative[:-1] + _cased("ek", part)  # Bednarka -> Bednarek
        if names.fold(nominative) in names._NOT_SURNAMES:
            return []
        return [(tag, nominative) for tag in names._OBLIQUE_TAGS[ending]]
    if names._NOUN_NOMINATIVE.search(key) and key not in names._NOT_SURNAMES:
        return [(tag, part) for tag in names._NOMINATIVE_TAGS]
    return []


def _like(lemma: str, written: str) -> str:
    """Write a nominative in the letter case of `written`, without Polish letters if it has none."""
    if names.fold(written) == lowercase(written):
        lemma = lemma.translate(_PLAIN)
    if written.isupper():
        return lemma.upper()
    if written.islower():
        return lemma.lower()
    return lemma


def _cased(ending: str, written: str) -> str:
    return ending.upper() if written.isupper() else ending


def _capitalised(word: str) -> str:
    """Capitalise a name written all in capitals or all in lowercase, part by part."""
    if not (word.isupper() or word.islower()):
        return word
    return "-".join(part[:1].upper() + part[1:].lower() for part in word.split("-"))


@cache
def _FIRST_NAMES() -> dict[str, frozenset[tuple[Tag, str]]]:  # noqa: N802 - a lazy constant
    """Folded form of a first name -> its readings, each with the nominative, as listed."""
    table: dict[str, set[tuple[Tag, str]]] = defaultdict(set)
    for gender, listed, decline in (
        ("m", names._MALE_NAMES, names.decline_masculine),
        ("f", names._FEMALE_NAMES, names.decline_feminine),
    ):
        for name in listed:
            for form, cases in decline(name.lower()).items():
                table[names.fold(form)].update(((gender, case), name) for case in cases)
    return {form: frozenset(readings) for form, readings in table.items()}


@cache
def _LISTED_SURNAMES() -> dict[str, frozenset[tuple[Tag, str]]]:  # noqa: N802 - a lazy constant
    """Folded form of a listed surname -> its readings, each with the nominative."""
    table: dict[str, set[tuple[Tag, str]]] = defaultdict(set)
    for name in names._NOUN_SURNAMES:
        for form, cases in names.decline_masculine(name.lower()).items():
            table[names.fold(form)].update((("m", case), name) for case in cases)
        table[names.fold(name)].update((("f", case), name) for case in CASES)
    for name in names._A_SURNAMES:
        for form, cases in names.decline_feminine(name.lower()).items():
            for gender in "mf":
                table[names.fold(form)].update(((gender, case), name) for case in cases)
    return {form: frozenset(readings) for form, readings in table.items()}


def scheme_digest() -> str:
    """Return a digest of everything the canonical form of a name depends on, for a test to pin.

    The lists of names and the rules that decline them decide which spellings share a token, so a
    change to any of them changes tokens and has to be noticed as a change of scheme.
    """
    parts: list[str] = []
    for table in (_FIRST_NAMES(), _LISTED_SURNAMES()):
        for form in sorted(table):
            parts.append(form + "=" + ",".join(sorted(f"{g}{c}:{n}" for (g, c), n in table[form])))
    for pattern in (
        names._ADJECTIVAL,
        names._NOUN_OBLIQUE,
        names._NOUN_NOMINATIVE,
        _POLISH_PREFIX,
        _EMAIL_AT,
        _EMAIL_DOT,
        _LINK_START,
        _SPACED_ABBREVIATION,
    ):
        parts.append(pattern.pattern)
    for words in (
        names._COMMON_ADJECTIVES,
        names._NOT_SURNAMES,
        names._NOT_NAMES,
        names.FOREIGN_TITLES,
        frozenset(names.HONORIFICS),
        frozenset(f"{k}={sorted(v)}" for k, v in names._ADJECTIVAL_TAGS.items()),
        frozenset(f"{k}={sorted(v)}" for k, v in names._OBLIQUE_TAGS.items()),
        names._NOMINATIVE_TAGS,
        names._VOWELS,
    ):
        parts.append(" ".join(sorted(map(str, words))))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
