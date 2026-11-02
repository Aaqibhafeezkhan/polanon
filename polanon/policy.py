"""Entity taxonomy and token naming.

Tokens are Polish-labelled on purpose: people read them in previews of what a model was sent, so
``<OSOBA_1>`` tells the Polish readers this library is built for more than ``<PERSON_1>`` would.
"""

from __future__ import annotations

import re

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
    "IBAN": "IBAN",
    "CREDIT_CARD": "KARTA",
    "EMAIL": "EMAIL",
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

#: Matches a complete token and captures its label and number.
TOKEN_PATTERN = r"<(?P<label>[A-Z_]+)_(?P<n>\d+)>"  # noqa: S105 - a regex, not a secret

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
