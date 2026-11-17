"""Entity taxonomy and token naming.

Tokens are Polish-labelled on purpose: people read them in previews of what a model was sent, so
``<OSOBA_1>`` tells the Polish readers this library is built for more than ``<PERSON_1>`` would.
"""

from __future__ import annotations

import re
from typing import Literal

#: Canonical entity type -> token label used in ``<LABEL_n>``.
ENTITY_TYPES: dict[str, str] = {
    "PERSON": "OSOBA",
    "ORG": "FIRMA",
    "LOCATION": "MIEJSCE",
    "PESEL": "PESEL",
    "NIP": "NIP",
    "REGON": "REGON",
    "KRS": "KRS",
    "PL_ID": "DOWOD",
    "PL_PASSPORT": "PASZPORT",
    "DRIVING_LICENCE": "PRAWO_JAZDY",  # a driving licence number after its label, any country
    "IBAN": "IBAN",
    "CREDIT_CARD": "KARTA",
    "EMAIL": "EMAIL",
    "LINK": "LINK",  # a link to a person's profile (LinkedIn, GitHub, ...) or the handle in it
    "PHONE": "TELEFON",
    "POSTAL_CODE": "KOD_POCZTOWY",
    "SECRET": "SEKRET",
    "IP_ADDRESS": "IP",
    "MAC_ADDRESS": "MAC",
    "ADDRESS": "ADRES",
    "PLATE": "REJESTRACJA",
    "VIN": "VIN",
    "DATE": "DATA",
    "BIRTH_DATE": "DATA_URODZENIA",  # a date after a birth label; on even where dates are off
    "TAX_ID": "NR_PODATKOWY",  # a foreign VAT or tax number; Polish ones are NIP
    "COMPANY_ID": "NR_REJESTROWY",  # a foreign company register number; Polish ones are KRS/REGON
    "BANK_ACCOUNT": "KONTO",  # an account number that is not an IBAN: Bankgiro, sort code, ...
    "OTHER": "INNE",  # only produced by the stage-2 judge, for context that identifies someone
}

#: How much of what the language model reads as a name is masked. "normal" leaves out what it
#: reads as a name in English prose, in code and in identifiers, where nearly all of it is
#: ordinary words; "paranoid" masks all of it, for text where a leak costs more than a word the
#: model can no longer read.
Mode = Literal["normal", "paranoid"]
MODES: tuple[Mode, ...] = ("normal", "paranoid")


def check_mode(mode: str) -> None:
    """Raise ValueError for a mode that is neither "normal" nor "paranoid"."""
    if mode not in MODES:
        raise ValueError(f"mode must be 'normal' or 'paranoid', not {mode!r}")


#: Matches a complete token and captures its label and number.
TOKEN_PATTERN = r"<(?P<label>[A-Z_]+)_(?P<n>\d+)>"  # noqa: S105 - a regex, not a secret

#: Matches a token with or without a grammatical case after a colon, ``<OSOBA_1:celownik>``, and
#: captures the case too. Masked text holds such a token where a person is named in another case
#: than the form the mapping keeps, and a model may write one in its answer.
CASED_TOKEN_PATTERN = r"<(?P<label>[A-Z_]+)_(?P<n>\d+)(?::(?P<case>[^\W\d_]+))?>"  # noqa: S105

#: Token label -> canonical entity type, the inverse of :data:`ENTITY_TYPES`.
LABEL_TO_ENTITY: dict[str, str] = {label: entity for entity, label in ENTITY_TYPES.items()}


def token_for(entity_type: str, n: int) -> str:
    """Return the nth token for an entity type, e.g. ``token_for("PERSON", 1) == "<OSOBA_1>"``."""
    return f"<{ENTITY_TYPES.get(entity_type, entity_type)}_{n}>"


def entity_type_for_token(token: str) -> str:
    """Return the canonical entity type encoded in a token, or ``UNKNOWN``."""
    match = re.fullmatch(TOKEN_PATTERN, token)
    if match is None:
        return "UNKNOWN"
    return LABEL_TO_ENTITY.get(match.group("label"), "UNKNOWN")


def enabled_entities(policy_csv: str) -> set[str]:
    """Parse a comma-separated list of entity types, as a settings field stores it.

    Names that are not entity types are dropped, so the result can go straight to
    ``Anonymizer(enabled=...)``.
    """
    return {e.strip() for e in policy_csv.split(",") if e.strip() in ENTITY_TYPES}
