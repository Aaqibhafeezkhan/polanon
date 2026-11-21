"""The parts of a masked value that give the same thing away on their own.

A surname is the person: once "Jan Kowalski" is masked, "Kowalski" further down the text is the
same man, and it went out in the clear, because no rule reads a lone surname as a name. So is the
surname in another case, also where its stem changes, "Wróbla" after "Paweł Wróbel",
"Koziołowi" after "Adam Kozieł" and "Dudzie" after "Anna Duda", and in the nominative after the
name was masked in another case: "Lis" and "Lisowi" after "Grzegorzem Lisem", "Kozieł" after
"Adama Kozieła". A handle is the account: once "github.com/jkowalski"
is masked, "jkowalski" in a JSON field, after "@" or in "jkowalski.github.io" is the same account.
The pipeline treats such a part as a known value of the same type, so it is masked wherever it
stands, by the rules every known value follows.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from polanon._canonical import base_form
from polanon.allowlist import WELL_KNOWN_NAMES
from polanon.guards.names_pl import (
    declined_surname_nominatives,
    fold,
    forget_forms,
    login_name_parts,
    person_name_parts,
    surname_forms,
)
from polanon.guards.recognizers_pl import profile_handle

#: A handle shorter than this is not masked on its own: "jk" or "ab" stand for nobody in prose.
MIN_HANDLE_LENGTH = 4

#: The platforms and brands of the allowlist as handles. "github.com/microsoft/vscode" links to a
#: project, and masking "Microsoft" in every sentence after it hides nobody.
_WELL_KNOWN_HANDLES = frozenset(fold(name).replace(" ", "") for name in WELL_KNOWN_NAMES)


def known_parts(value: str, entity_type: str, *, trusted: bool, cases: bool = True) -> list[str]:
    """Return the parts of a masked value to mask wherever they appear, under the value's type.

    Args:
        value: the masked value, as written.
        entity_type: its type.
        trusted: whether the detection stands on its own, such as a rule's, a list's or the
            caller's, rather than a language model's guess. A model that reads "Google Ads" as a
            person would otherwise mask "Ads" in every sentence after it, so a name it found gives
            its surname only next to a first name the name guard knows.
        cases: give a surname's cases that do not begin with it as well, "Wróbla" for "Wróbel".
            A store of tokens keeps the surname alone and finds its cases by declining it.
    """
    return list(_known_parts(value, entity_type, trusted, cases))


#: How many values the parts stay worked out for: every text of a call reads the parts of every
#: value of the mapping, and working out a name's nominative for each of them took longer than
#: the rest of the masking of a short text.
_PARTED_VALUES = 65_536


@lru_cache(maxsize=_PARTED_VALUES)
def _known_parts(value: str, entity_type: str, trusted: bool, cases: bool) -> tuple[str, ...]:
    if entity_type == "PERSON":
        parts = person_name_parts(value, needs_first_name=not trusted, cases=cases)
        # A name first masked in another case gives its surname in that case; the nominative,
        # where the rules read it for sure, gives the surname as it stands on its own.
        nominative = base_form(value, entity_type)
        more: list[str] = []
        if nominative != value.strip():
            more = person_name_parts(nominative, needs_first_name=not trusted, cases=cases)
        elif cases and parts:
            # Where they cannot, a first name in no nominative tells the surname's case, and the
            # nominatives it may have are masked with their cases: "Kozieł", "Kozioł" and
            # "Koziołowi" after "Adama Kozieła". Only for masking: a store keeps no guess.
            for surname in declined_surname_nominatives(value):
                more += [surname, *surname_forms(surname)]
        parts += [part for part in dict.fromkeys(more) if part not in parts]
        return tuple(parts or login_name_parts(value))
    if entity_type == "LINK":
        handle = profile_handle(value)
        if (
            handle is not None
            and len(handle) >= MIN_HANDLE_LENGTH
            and any(char.isalpha() for char in handle)
            and fold(handle).replace("-", "") not in _WELL_KNOWN_HANDLES
        ):
            return (handle,)
    return ()


def forget() -> None:
    """Drop the parts worked out for the latest values, and the values with them."""
    _known_parts.cache_clear()
    forget_forms()


def person_parts(value: str, entity_type: str) -> list[str]:
    """Return the parts of a value of another type that name a person, to mask as ``PERSON``.

    An e-mail address whose local part is a login written from a name, "anna.kowalczyk@...",
    names Anna Kowalczyk: the address was masked whole, and the login and the surname went out
    further down the text, next to her token. Its surname joins the known values, so the surname
    written as a name and the logins made of it are masked as well.
    """
    if entity_type != "EMAIL" or "@" not in value:
        return []
    return login_name_parts(value.split("@", 1)[0].lower())


#: A surname shorter than this gives no login glued to an initial or digits: "nowakowie" or
#: "kotek" are ordinary words.
MIN_LOGIN_SURNAME = 6
#: With a dot, "_" or "-" between them, a shorter surname gives a login too: "nowak_j", "j.nowak".
MIN_SEPARATED_LOGIN_SURNAME = 5
_SEPARATORS = (".", "_", "-")

#: A word written as a login or a handle is: small letters and digits, with dots, "_" or "-".
_LOGIN = re.compile(r"(?<![\w.-])[a-z0-9ąćęłńóśźż][a-z0-9ąćęłńóśźż._-]*[a-z0-9ąćęłńóśźż](?![\w-])")
#: What a login adds to a surname: an initial or two before it, then maybe a dot, "_" or "-";
#: after it, maybe one of those and then an initial or two, then maybe digits. Letters glued
#: straight on are left out: "malinowskie" may be an adjective as much as a login.
_LOGIN_BEFORE = re.compile(r"(?:[a-z]{1,2}[._-]?)?")
_LOGIN_AFTER = re.compile(r"(?:[._-][a-z]{1,2})?\d*")


def logins_of(text: str, values: Iterable[tuple[str, str]]) -> list[str]:
    """Return the words of `text` that are logins made of the surname of a known person.

    A backup's settings an agent read held "user.name=Tomasz Malinowski" and the directory
    "/home/tmalinowski": the surname was masked, the login made of it went out. A word in
    lowercase that is a known surname of six letters or more with an initial, a dot, "_", "-" or
    digits around it, "tmalinowski", "t.malinowski", "malinowski85", is the person too, and so is
    one of five letters with a dot, "_" or "-": "nowak_j". The surname alone in lowercase is left
    as it is, since Zima, Kot or Wilk are ordinary words.

    Args:
        text: the text being masked.
        values: the known values as ``(value, entity_type)``, the parts of names among them.
    """
    surnames = {
        fold(value.lower())
        for value, entity_type in values
        if entity_type == "PERSON" and value.isalpha() and len(value) >= MIN_SEPARATED_LOGIN_SURNAME
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
            separated = before.endswith(_SEPARATORS) or after.startswith(_SEPARATORS)
            if len(surname) < MIN_LOGIN_SURNAME and not separated:
                continue
            if (
                (before or after)
                and _LOGIN_BEFORE.fullmatch(before)
                and _LOGIN_AFTER.fullmatch(after)
            ):
                found.append(match[0])
                break
    return found
