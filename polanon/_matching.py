"""Finding every occurrence of a value a conversation already masks.

A guard reports what it recognised, and a statistical one does not recognise the same string every
time. Once any occurrence of a value has a token, the pipeline masks every other occurrence too:
written exactly as before, and also in capitals, across a line break, with a no-break space or an
invisible character in it, and, for an identifier, with its groups written apart or together. The
exact search keeps every match a letter-for-letter search makes; the other searches only add to it.

Shared by the pipeline, which masks known values and the judge's findings, and by the output sweep,
which needs to know which parts of an answer are known values.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property

from polanon._view import DASHES, INVISIBLE, View, as_read
from polanon.policy import ENTITY_TYPES, LABEL_TO_ENTITY, TOKEN_PATTERN

TOKEN_RE = re.compile(TOKEN_PATTERN)

#: Translates a range of a searched text to the same characters of the original.
_ToOriginal = Callable[[int, int], tuple[int, int]]

#: A known value shorter than this is not masked inside a longer word. Widening "Jan" to "Janusz",
#: or a stray two-letter detection to every word that contains it, would mask half a document.
MIN_EMBEDDED_LENGTH = 4

#: A number shorter than this is left alone inside a longer number, a date, a time or an amount.
SHORT_NUMBER = 6

#: Characters that join the parts of one number: "12.2025", "2025-04-15", "FV/2025/03", "12:30",
#: and the dashes typesetting puts in place of a hyphen.
NUMBER_JOINERS = ".,-/:" + DASHES

#: The shapes of a date, a time, an amount and a span of years. A short number that is part of one
#: is left alone: when the PIN is 2025, "15.04.2025" is a date, and masking its year as the PIN told
#: the model what the PIN was. Any other number joined to it, as in "PIN/PUK 4821/36912874", is no
#: reason to leave the PIN in the clear.
_DATE_TIME_OR_AMOUNT = re.compile(
    r"\d{1,2}[./-]\d{1,2}[./-](?:\d{2}|(?:19|20)\d{2})"
    r"|(?:19|20)\d{2}[./-]\d{1,2}(?:[./-]\d{1,2})?"
    r"|\d{1,2}[./-](?:19|20)\d{2}"
    r"|(?:19|20)\d{2}-(?:19|20)?\d{2}"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
    r"|\d+[.,]\d{1,2}"
    r"|\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?"
)

#: Types whose values differ when only their letter case does: two passwords are two secrets. A
#: custom type counts as one too, since nothing says its identifiers ignore case.
_CASE_SENSITIVE = frozenset({"SECRET"})

#: Identifiers that stay the same number whether their groups are written apart or together:
#: "PL61 1090 1014 ..." and "PL61109010140000071219812874" are one account.
_GROUPED_TYPES = frozenset(
    {
        "IBAN",
        "BANK_ACCOUNT",
        "CREDIT_CARD",
        "PESEL",
        "NIP",
        "REGON",
        "KRS",
        "TAX_ID",
        "COMPANY_ID",
        "PL_ID",
        "PL_PASSPORT",
        "VIN",
        "PHONE",
    }
)

#: Up to this many values are searched for one by one, each in a single regex pass; beyond it,
#: every value is found in one pass over the text.
_FEW_VALUES = 128

#: The one-pass search looks up the first characters of a value at every position, then a longer
#: prefix where those match. Values that share a first name are told apart by the longer prefix,
#: which grows for a group that still shares it across many lengths.
_KEY_CHARS = 4
_PREFIX_CHARS = (8, 16, 32, 64)

#: Values sharing a prefix are looked up by each of their lengths where it occurs. Past this many
#: lengths the group is keyed by a longer prefix, and past the longest one it is searched for one
#: value at a time: a text of repeated characters made the one-pass search ten times slower.
_MAX_LENGTHS = 8

#: Whitespace a known value may be written with in place of its single spaces: a line break, a
#: tab, a run of spaces.
_WHITESPACE_RUN = re.compile(r"\s{2,}|[^\S ]")

#: Spaces and hyphens between the groups of an identifier.
_GROUP_SEPARATOR = re.compile(r"(?<=[^\W_])[ -]+(?=[^\W_])")

#: Folding that str.lower() does not do on its own: the Greek final sigma depends on the letters
#: around it, and the Turkish dotless i has no capital of its own in other languages.
_FOLD_FIX = str.maketrans({chr(0x03C2): chr(0x03C3), chr(0x0131): "i"})

_PLAIN_DASHES = str.maketrans(dict.fromkeys(DASHES, "-"))


def type_of(token: str) -> str | None:
    """Return the entity type a token stands for, or None when the string is not a token."""
    match = TOKEN_RE.fullmatch(token)
    if match is None:
        return None
    label = match.group("label")
    return LABEL_TO_ENTITY.get(label, label)


def fold(text: str) -> str:
    """Lowercase `text` character for character, so positions in the result are positions in it.

    The result does not depend on the letters around a character, so a value and the text it is
    searched in fold alike. A character that lowercases to two, such as the Turkish dotted
    capital I, keeps the first of them.
    """
    folded = text.lower()
    if len(folded) != len(text):
        folded = "".join(char.lower()[0] for char in text)
    return folded.translate(_FOLD_FIX)


def squeeze(text: str) -> str:
    """Return `text` with every line break, tab and run of whitespace read as one space."""
    return _WHITESPACE_RUN.sub(" ", text)


def reading(value: str) -> str:
    """Return a value as the search reads it: as the guards read it, whitespace squeezed."""
    return squeeze(as_read(value)).strip()


def folds(entity_type: str) -> bool:
    """Return whether values of this type are the same value whatever their letter case."""
    return entity_type in ENTITY_TYPES and entity_type not in _CASE_SENSITIVE


def value_keys(value: str, entity_type: str) -> list[tuple[str, str]]:
    """Return what makes two spellings of a value the same value, most specific first.

    Where letter case does not tell values apart, the folded reading is one value: "Jan
    Kowalski", "JAN KOWALSKI" and the name with a no-break space in it are one person, and an
    identifier is one whether its groups are written apart or together. A one-word name written
    all in lowercase is kept apart, because the search leaves lowercase occurrences of a
    capitalised name alone; recorded under its own spelling, it is still found in the next turn.

    A password, or a custom identifier, is one value only as written, and never the same value as
    a name that reads alike: the name keeps a token whose spellings are searched for in any case.
    """
    read = reading(value)
    if not folds(entity_type):
        return [("exact", read)]
    folded = fold(read)
    lowercase_word = read == folded and read.isalpha()
    keys = [("lowercase word" if lowercase_word else "folded", folded)]
    if entity_type in _GROUPED_TYPES:
        keys.append(("grouped", _GROUP_SEPARATOR.sub("", folded)))
    return keys


class Ranges:
    """Non-overlapping ranges kept in order, so an overlap check is a bisection, not a scan.

    A long document carries thousands of masked values. Checking every further occurrence against
    all of them one by one made masking a 1 MB export take minutes.
    """

    def __init__(self, ranges: Iterable[tuple[int, int]] = ()) -> None:
        """Store `ranges`, which must not overlap each other."""
        ordered = sorted(ranges)
        self._starts = [start for start, _ in ordered]
        self._ends = [end for _, end in ordered]

    def overlaps(self, start: int, end: int) -> bool:
        """Return whether the range from `start` to `end` touches any stored range."""
        index = bisect_right(self._ends, start)
        return index < len(self._starts) and self._starts[index] < end

    def only_inside(self, start: int, end: int) -> bool:
        """Return whether every stored range the range touches lies inside it and is shorter."""
        return all(
            start <= first and last <= end and (first, last) != (start, end)
            for first, last in self.touching(start, end)
        )

    def touching(self, start: int, end: int) -> list[tuple[int, int]]:
        """Return the stored ranges the range from `start` to `end` overlaps, in order."""
        first = bisect_right(self._ends, start)
        last = bisect_left(self._starts, end)
        return [(self._starts[i], self._ends[i]) for i in range(first, last)]

    def add(self, start: int, end: int) -> None:
        """Store a range that overlaps none of the stored ones."""
        index = bisect_right(self._starts, start)
        self._starts.insert(index, start)
        self._ends.insert(index, end)


@dataclass(frozen=True)
class _Removal:
    """A text with runs of characters replaced, and the way back to positions in the source."""

    text: str
    #: Positions in `text` right after a replaced run, and how many characters all the runs up to
    #: and including that one removed.
    gaps: list[int]
    removed: list[int]

    @classmethod
    def of(cls, source: str, pattern: re.Pattern[str], replacement: str) -> _Removal:
        """Replace every match of `pattern` in `source` with `replacement`."""
        pieces: list[str] = []
        gaps: list[int] = []
        removed: list[int] = []
        cursor = 0
        total = 0
        for run in pattern.finditer(source):
            pieces.append(source[cursor : run.start()])
            pieces.append(replacement)
            gaps.append(run.start() - total + len(replacement))
            total += run.end() - run.start() - len(replacement)
            removed.append(total)
            cursor = run.end()
        pieces.append(source[cursor:])
        return cls("".join(pieces), gaps, removed)

    def back(self, position: int) -> int:
        """Return where a character of `text` that is not a replacement sits in the source."""
        index = bisect_right(self.gaps, position) - 1
        return position if index < 0 else position + self.removed[index]


class Haystack:
    """A text prepared for the searches beside the exact one, and the way back to the original.

    `text` is the guards' view of the text with every run of whitespace read as one space, and
    `folded` is the same text lowercased character for character.
    """

    def __init__(self, text: str, view: View | None) -> None:
        """Prepare `text`, whose detection view is `view`."""
        seen = text if view is None else view.text
        self._view = view
        self._squeezed = _Removal.of(seen, _WHITESPACE_RUN, " ")
        self.text = self._squeezed.text
        self.folded = fold(self.text)
        #: Whether `text` is the original as written, so a search in it repeats the exact one.
        self.as_written = view is None and not self._squeezed.gaps

    @classmethod
    def of(cls, text: str, view: View | None) -> Haystack:
        """Prepare `text`, whose detection view is `view`."""
        return cls(text, view)

    @cached_property
    def ungrouped(self) -> _Removal:
        """The folded text with the spaces and hyphens between letters and digits taken out."""
        return _Removal.of(self.folded, _GROUP_SEPARATOR, "")

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """Translate a range that starts and ends with a non-space character to the original."""
        first = self._squeezed.back(start)
        last = self._squeezed.back(end - 1)
        if self._view is None:
            return first, last + 1
        return self._view.starts[first], self._view.ends[last]

    def regrouped(self, start: int, end: int) -> tuple[int, int]:
        """Translate a range of `ungrouped` text to the same characters of `text`."""
        return self.ungrouped.back(start), self.ungrouped.back(end - 1) + 1


@dataclass(frozen=True)
class Needle:
    """A known value as it is searched for, and how its occurrences may be masked."""

    #: The value as written, for the exact search in the original text.
    written: str
    #: The value as the search reads it, for the search through the guards' reading.
    read: str
    payload: str
    folds: bool
    #: An identifier with its groups taken out, or None for any other value.
    ungrouped: str | None

    @property
    def is_short_number(self) -> bool:
        """A short run of digits, such as a PIN, is left alone inside a longer number or a date."""
        return self.read.isdigit() and len(self.read) < SHORT_NUMBER

    @property
    def lowercase_is_a_common_word(self) -> bool:
        """Whether an all-lowercase occurrence of this capitalised one-word name is left alone.

        Polish surnames such as Zima, Kot or Wilk are common nouns; "zima" in lowercase is winter.
        A value of more than one word, one with a digit or a sign in it, or one written in
        lowercase itself matches whatever the letter case.
        """
        return self.folds and self.read.isalpha() and not self.read.islower()


def needles(values: Iterable[tuple[str, str]]) -> list[Needle]:
    """Return the values worth searching for, one per spelling and letter-case rule, longest first.

    A payload is an entity type or a token; either tells whether letter case matters. A spelling
    held both as a password and as a name is searched for both ways. Of two values of one length,
    one whose case matters goes first, so a password is placed before a name that differs from it
    only in case.
    """
    found: dict[tuple[str, bool], Needle] = {}
    for value, payload in values:
        read = reading(value)
        if not any(char.isalnum() for char in read):
            continue
        entity_type = type_of(payload) or payload
        folding = folds(entity_type)
        ungrouped = None
        if entity_type in _GROUPED_TYPES:
            ungrouped = _GROUP_SEPARATOR.sub("", fold(read))
        needle = Needle(value.strip(), read, payload, folding, ungrouped)
        found.setdefault((needle.written, folding), needle)
    return sorted(found.values(), key=lambda needle: (-len(needle.read), needle.folds))


def occurrences(haystack: str, wanted: list[str]) -> list[list[int]]:
    """Return where each string occurs in `haystack`, left to right, never overlapping itself.

    A few strings are searched for one by one. Many are found in a single pass that looks up their
    first characters at every position, because searching for each of tens of thousands of values
    in turn made masking a 1 MB export of customer records take most of a minute.
    """
    if len(wanted) <= _FEW_VALUES:
        return [_each(haystack, needle) for needle in wanted]
    found: list[list[int]] = [[] for _ in wanted]
    tables: _Tables = {}
    pending = list(enumerate(wanted))
    for prefix_chars in _PREFIX_CHARS:
        _index(tables, pending, prefix_chars)
        pending = _crowded(tables)
    for index, needle in pending:
        found[index] = _each(haystack, needle)
    _scan(haystack, tables, found)
    return found


#: First characters -> a longer prefix, by how many characters it has -> length -> the string
#: itself -> its positions in the list searched for.
_Tables = dict[int, dict[str, dict[int, dict[str, dict[int, dict[str, list[int]]]]]]]


def _index(tables: _Tables, wanted: list[tuple[int, str]], prefix_chars: int) -> None:
    for index, needle in wanted:
        size = min(len(needle), _KEY_CHARS)
        prefix = needle[:prefix_chars]
        by_prefix = tables.setdefault(size, {}).setdefault(needle[:size], {})
        by_length = by_prefix.setdefault(len(prefix), {}).setdefault(prefix, {})
        by_length.setdefault(len(needle), {}).setdefault(needle, []).append(index)


def _crowded(tables: _Tables) -> list[tuple[int, str]]:
    """Take out of `tables` the strings that share a prefix across many lengths."""
    crowded: list[tuple[int, str]] = []
    for table in tables.values():
        for by_prefix in table.values():
            for prefixes in by_prefix.values():
                full = [key for key, lengths in prefixes.items() if len(lengths) > _MAX_LENGTHS]
                for prefix in full:
                    for texts in prefixes.pop(prefix).values():
                        for needle, indices in texts.items():
                            crowded += [(index, needle) for index in indices]
    return crowded


def _scan(haystack: str, tables: _Tables, found: list[list[int]]) -> None:
    free_from = [0] * len(found)
    for size, table in tables.items():
        for start in range(len(haystack) - size + 1):
            by_prefix = table.get(haystack[start : start + size])
            if by_prefix is None:
                continue
            for prefix_size, prefixes in by_prefix.items():
                lengths = prefixes.get(haystack[start : start + prefix_size])
                if lengths is None:
                    continue
                for length, texts in lengths.items():
                    for index in texts.get(haystack[start : start + length], ()):
                        if start >= free_from[index]:
                            found[index].append(start)
                            free_from[index] = start + length


def _each(haystack: str, needle: str) -> list[int]:
    return [match.start() for match in re.finditer(re.escape(needle), haystack)]


@dataclass(frozen=True)
class _Found:
    """Where each needle occurs: as written, through the guards' reading, and ungrouped."""

    written: list[list[int]]
    read: list[list[int]]
    ungrouped: list[list[int]]


def _find(text: str, haystack: Haystack, found: list[Needle]) -> _Found:
    written = occurrences(text, [needle.written for needle in found])
    read: list[list[int]] = [[] for _ in found]
    folding = [i for i, needle in enumerate(found) if needle.folds]
    exact_case = [
        i
        for i, needle in enumerate(found)
        if not needle.folds and not (haystack.as_written and needle.read == needle.written)
    ]
    for indices, searched, wanted in (
        (folding, haystack.folded, [fold(found[i].read) for i in folding]),
        (exact_case, haystack.text, [found[i].read for i in exact_case]),
    ):
        for index, positions in zip(indices, occurrences(searched, wanted), strict=True):
            read[index] = positions
    ungrouped: list[list[int]] = [[] for _ in found]
    grouped = [i for i, needle in enumerate(found) if needle.ungrouped]
    if grouped:
        wanted = [found[i].ungrouped or "" for i in grouped]
        for index, positions in zip(
            grouped, occurrences(haystack.ungrouped.text, wanted), strict=True
        ):
            ungrouped[index] = positions
    return _Found(written, read, ungrouped)


def known_value_ranges(
    text: str, haystack: Haystack, values: Iterable[tuple[str, str]]
) -> list[tuple[int, int]]:
    """Return every occurrence of every value in original coordinates, inside words or not."""
    found = needles(values)
    where = _find(text, haystack, found)
    ranges: list[tuple[int, int]] = []
    for index, needle in enumerate(found):
        ranges += [(start, start + len(needle.written)) for start in where.written[index]]
        ranges += [haystack.to_original(s, s + len(needle.read)) for s in where.read[index]]
        length = len(needle.ungrouped or "")
        for start in where.ungrouped[index]:
            ranges.append(haystack.to_original(*haystack.regrouped(start, start + length)))
    return ranges


@dataclass(frozen=True)
class Placement:
    """What the occurrences of known values mask, as ``(start, end, payload)`` in the text.

    `matches` never overlap each other or the ranges the search had to step around. `crossing`
    holds the occurrences that run past such a range with a letter or digit of their own: masking
    only the matches would send that part in the clear, so the caller resolves each one with the
    range it crosses, as :func:`~polanon.types.merge_spans` does.
    """

    matches: list[tuple[int, int, str]]
    crossing: list[tuple[int, int, str]]


def known_value_matches(
    text: str,
    haystack: Haystack,
    values: Iterable[tuple[str, str]],
    occupied: Iterable[tuple[int, int]],
    *,
    absorb: bool = False,
) -> list[tuple[int, int, str]]:
    """Find every occurrence of a known value, as ``(start, end, payload)`` in `text`.

    The matches of :func:`place_known_values`, for a caller that cannot resolve an overlap.
    """
    return place_known_values(text, haystack, values, occupied, absorb=absorb).matches


def place_known_values(
    text: str,
    haystack: Haystack,
    values: Iterable[tuple[str, str]],
    occupied: Iterable[tuple[int, int]],
    *,
    absorb: bool = False,
    cross_occupied: bool = False,
) -> Placement:
    """Find every occurrence of a known value, and the ones that cross another range.

    The entity filter is deliberately not applied here. A value the conversation already treats as
    sensitive keeps being masked after the enabled types are narrowed; otherwise a later turn that
    mentions it would send it in the clear.

    An occurrence inside a longer word takes the whole word with it, so "Kowalskiego" is masked
    whole instead of becoming "<OSOBA_1>ego". Leaving it in the clear was worse: once a model took
    "Television" for a company, "TelevisionM12037" still held the value, and a check before sending
    would find it there. Values shorter than :data:`MIN_EMBEDDED_LENGTH` are left as they are.

    Longer values are placed first, each as written, then through the guards' reading, then with
    its groups taken out, so "Anna Wiśniewska" across a line break is not split by a shorter "Anna"
    written exactly. Ranges in `occupied` are never cut into. With `absorb`, an occurrence of the
    value itself may cover ranges of `occupied` whole, as "Jan" + line break + "Kowalski" covers a
    guard's "Jan": the caller drops what it covers and masks the occurrence instead.

    An occurrence that only partly overlaps an occurrence placed before it is returned in
    `crossing`, and with `cross_occupied` so is one that partly overlaps a range of `occupied`,
    such as a guard's detection the caller can still merge with it. Without it, a range of
    `occupied` is a token that must stay whole, and an occurrence crossing it is left out.
    """
    found = needles(values)
    where = _find(text, haystack, found)
    placer = _Placer(Ranges(occupied), text, absorb=absorb, cross_occupied=cross_occupied)

    def as_written(start: int, end: int) -> tuple[int, int]:
        return start, end

    def regrouped(start: int, end: int) -> tuple[int, int]:
        return haystack.to_original(*haystack.regrouped(start, end))

    for index, needle in enumerate(found):
        length = len(needle.written)
        for start in where.written[index]:
            placer.place(text, start, start + length, needle, as_written, through_view=False)
        length = len(needle.read)
        for start in where.read[index]:
            placer.place(haystack.text, start, start + length, needle, haystack.to_original)
        length = len(needle.ungrouped or "")
        for start in where.ungrouped[index]:
            # An identifier read with its groups taken out is masked only where it stands alone in
            # the text as written; widening it would run into the words around it.
            first, end = haystack.regrouped(start, start + length)
            if _stands_alone(haystack.text, first, end):
                placer.place(haystack.text, first, end, needle, haystack.to_original)
    return Placement(placer.matches, placer.crossing)


class _Placer:
    """Decides what each occurrence masks, keeping the ranges placed so far."""

    def __init__(
        self, guarded: Ranges, original: str, *, absorb: bool, cross_occupied: bool
    ) -> None:
        self.placed = Ranges()
        self.guarded = guarded
        self.original = original
        self.absorb = absorb
        self.cross_occupied = cross_occupied
        self.matches: list[tuple[int, int, str]] = []
        self.crossing: list[tuple[int, int, str]] = []
        #: The last word that could not be masked whole, and the text it is a word of. Later
        #: occurrences inside it go straight to the value alone: widening each of them again
        #: rescanned the whole word every time.
        self._blocked_word: tuple[str, int, int] | None = None

    def blocked(self, start: int, end: int, *, whole_word: bool = False) -> bool:
        """Return whether a range in original coordinates may not be masked."""
        if self.placed.overlaps(start, end):
            return True
        # Only the value itself takes in a guard's detection: a word widened around it is a guess,
        # and a secret inside "TelevisionM12037" stays a secret.
        may_absorb = self.absorb and not whole_word
        return self.guarded.overlaps(start, end) and not (
            may_absorb and self.guarded.only_inside(start, end)
        )

    def place(
        self,
        text: str,
        start: int,
        end: int,
        needle: Needle,
        to_original: _ToOriginal,
        *,
        through_view: bool = True,
    ) -> None:
        """Mask an occurrence of `needle` at `start`..`end` of `text`, if it is to be masked."""
        # An occurrence already covered is skipped before it is widened: inside a long run of
        # letters and digits each occurrence would otherwise rescan the whole run.
        bare = to_original(start, end)
        if self.blocked(*bare):
            if self._runs_past(*bare) and _occurrence(text, start, end, needle, through_view):
                self.crossing.append((*bare, needle.payload))
            return
        kind = _occurrence(text, start, end, needle, through_view)
        if kind is None:
            return
        if kind == "inside a word" and not self._inside_blocked_word(text, start, end):
            word = _whole_words(text, start, end)
            widened = to_original(*word)
            if not self.blocked(*widened, whole_word=True):
                self._accept(widened, needle.payload)
                return
            self._blocked_word = (text, *word)
        self._accept(bare, needle.payload)

    def _runs_past(self, start: int, end: int) -> bool:
        """Return whether a blocked occurrence has a letter or digit outside what blocks it.

        "Nowak z Kowalskich" known from an earlier turn meets a guard's "Anna Nowak": skipping the
        occurrence sent "z Kowalskich" in the clear. An occurrence that crosses a range of
        `guarded` counts only with `cross_occupied`, since otherwise that range is a token.
        """
        guarded = self.guarded.touching(start, end)
        if guarded and not self.cross_occupied:
            return False
        cursor = start
        for first, last in sorted(self.placed.touching(start, end) + guarded):
            if first > cursor and _has_word_character(self.original[cursor:first]):
                return True
            cursor = max(cursor, last)
        return cursor < end and _has_word_character(self.original[cursor:end])

    def _inside_blocked_word(self, text: str, start: int, end: int) -> bool:
        word = self._blocked_word
        return word is not None and word[0] is text and word[1] <= start and end <= word[2]

    def _accept(self, span: tuple[int, int], payload: str) -> None:
        self.placed.add(*span)
        self.matches.append((*span, payload))


def _occurrence(text: str, start: int, end: int, needle: Needle, through_view: bool) -> str | None:
    """Return whether an occurrence stands alone or sits inside a word, or None to skip it."""
    if through_view and needle.lowercase_is_a_common_word and text[start:end].islower():
        return None
    if _stands_alone(text, start, end):
        if needle.is_short_number and _in_a_date_time_or_amount(text, start, end):
            return None
        return "alone"
    if len(needle.read) < MIN_EMBEDDED_LENGTH:
        return None
    if needle.is_short_number and _next_to_a_digit(text, start, end):
        return None
    return "inside a word"


def _is_word_character(char: str) -> bool:
    """Letters and digits, the combining marks of decomposed text, and invisible characters."""
    return char.isalnum() or unicodedata.combining(char) > 0 or char in INVISIBLE


def _has_word_character(text: str) -> bool:
    return any(_is_word_character(char) for char in text)


def _stands_alone(text: str, start: int, end: int) -> bool:
    """Return whether a match is not part of a longer word.

    A value that begins or ends with punctuation already stops there: "DD." in "DD.MM.YYYY" is not
    inside a word, although a letter follows it. An accent written as a combining mark after the
    last letter belongs to the word: "Kowalska" plus a combining ogonek is "Kowalską".
    """
    left_clear = (
        start == 0 or not _is_word_character(text[start]) or not _is_word_character(text[start - 1])
    )
    right_clear = (
        end == len(text)
        or not _is_word_character(text[end - 1])
        or not _is_word_character(text[end])
    )
    return left_clear and right_clear


def _whole_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a range over the letters, digits, combining marks and invisible characters."""
    while start > 0 and _is_word_character(text[start - 1]):
        start -= 1
    while end < len(text) and _is_word_character(text[end]):
        end += 1
    return start, end


def _next_to_a_digit(text: str, start: int, end: int) -> bool:
    return (start > 0 and text[start - 1].isdigit()) or (end < len(text) and text[end].isdigit())


def _in_a_date_time_or_amount(text: str, start: int, end: int) -> bool:
    """Return whether digits at `start`..`end` are part of a date, a time or an amount."""
    first, last = start, end
    while first > 0 and (text[first - 1].isdigit() or text[first - 1] in NUMBER_JOINERS):
        first -= 1
    while last < len(text) and (text[last].isdigit() or text[last] in NUMBER_JOINERS):
        last += 1
    run = text[first:last].translate(_PLAIN_DASHES).strip(".,-/:")
    return run != text[start:end] and _DATE_TIME_OR_AMOUNT.fullmatch(run) is not None
