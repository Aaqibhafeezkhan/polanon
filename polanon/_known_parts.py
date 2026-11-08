"""The parts of a masked value that give the same thing away on their own.

A surname is the person: once "Jan Kowalski" is masked, "Kowalski" further down the text is the
same man, and it went out in the clear, because no rule reads a lone surname as a name. A handle is
the account: once "github.com/jkowalski" is masked, "jkowalski" in a JSON field, after "@" or in
"jkowalski.github.io" is the same account. The pipeline treats such a part as a known value of the
same type, so it is masked wherever it stands, by the rules every known value follows.
"""

from __future__ import annotations

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
