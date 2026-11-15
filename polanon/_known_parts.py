"""The parts of a masked value that give the same thing away on their own.

A surname is the person: once "Jan Kowalski" is masked, "Kowalski" further down the text is the
same man, and it went out in the clear, because no rule reads a lone surname as a name. A handle is
the account: once "github.com/jkowalski" is masked, "jkowalski" in a JSON field, after "@" or in
"jkowalski.github.io" is the same account. The pipeline treats such a part as a known value of the
same type, so it is masked wherever it stands, by the rules every known value follows.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from polanon.allowlist import WELL_KNOWN_NAMES
from polanon.guards.names_pl import fold, person_name_parts
from polanon.guards.recognizers_pl import profile_handle

#: A handle shorter than this is not masked on its own: "jk" or "ab" stand for nobody in prose.
MIN_HANDLE_LENGTH = 4

#: The platforms and brands of the allowlist as handles. "github.com/microsoft/vscode" links to a
#: project, and masking "Microsoft" in every sentence after it hides nobody.
_WELL_KNOWN_HANDLES = frozenset(fold(name).replace(" ", "") for name in WELL_KNOWN_NAMES)


def known_parts(value: str, entity_type: str, *, trusted: bool) -> list[str]:
    """Return the parts of a masked value to mask wherever they appear, under the value's type.

    Args:
        value: the masked value, as written.
        entity_type: its type.
        trusted: whether the detection stands on its own, such as a rule's, a list's or the
            caller's, rather than a language model's guess. A model that reads "Google Ads" as a
            person would otherwise mask "Ads" in every sentence after it, so a name it found gives
            its surname only next to a first name the name guard knows.
    """
    if entity_type == "PERSON":
        return person_name_parts(value, needs_first_name=not trusted)
    if entity_type == "LINK":
        handle = profile_handle(value)
        if (
            handle is not None
            and len(handle) >= MIN_HANDLE_LENGTH
            and any(char.isalpha() for char in handle)
            and fold(handle).replace("-", "") not in _WELL_KNOWN_HANDLES
        ):
            return [handle]
    return []


#: A surname shorter than this gives no login: "nowak" or "kot" stand in too many other words.
MIN_LOGIN_SURNAME = 6

#: A word written as a login or a handle is: small letters and digits, with dots, "_" or "-".
_LOGIN = re.compile(r"(?<![\w.-])[a-z0-9ąćęłńóśźż][a-z0-9ąćęłńóśźż._-]*[a-z0-9ąćęłńóśźż](?![\w-])")
#: What a login adds to a surname: an initial or two before it, then maybe a dot, "_" or "-";
#: after it, maybe one of those and an initial or two, then maybe digits.
_LOGIN_BEFORE = re.compile(r"(?:[a-z]{1,2}[._-]?)?")
_LOGIN_AFTER = re.compile(r"(?:[._-]?[a-z]{1,2})?\d*")


def logins_of(text: str, values: Iterable[tuple[str, str]]) -> list[str]:
    """Return the words of `text` that are logins made of the surname of a known person.

    A backup's settings an agent read held "user.name=Tomasz Malinowski" and the directory
    "/home/tmalinowski": the surname was masked, the login made of it went out. A word in
    lowercase that is a known surname of six letters or more with an initial, a dot, "_", "-" or
    digits around it, "tmalinowski", "t.malinowski", "malinowski85", is the person too. The
    surname alone in lowercase is left as it is, since Zima, Kot or Wilk are ordinary words.

    Args:
        text: the text being masked.
        values: the known values as ``(value, entity_type)``, the parts of names among them.
    """
    surnames = {
        fold(value.lower())
        for value, entity_type in values
        if entity_type == "PERSON" and value.isalpha() and len(value) >= MIN_LOGIN_SURNAME
    }
    if not surnames:
        return []
    found: list[str] = []
    for match in _LOGIN.finditer(text):
        word = fold(match[0])
        for surname in surnames:
            at = word.find(surname)
            if at < 0:
                continue
            before, after = word[:at], word[at + len(surname) :]
            if (
                (before or after)
                and _LOGIN_BEFORE.fullmatch(before)
                and _LOGIN_AFTER.fullmatch(after)
            ):
                found.append(match[0])
                break
    return found
