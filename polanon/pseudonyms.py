"""Stable pseudonyms: one token for one value in every document, every process and every question.

Tokens numbered per call, such as ``<OSOBA_1>``, mean nothing outside the conversation that
handed them out, so a document masked for a search index and a question masked later can call
two different people ``<OSOBA_1>``, and the search finds nothing. A stable pseudonym is computed
instead: HMAC-SHA256 of the value's canonical form (see :mod:`polanon._canonical`) under a key
the application keeps secret, written as ``<OSOBA_482193047556>``. The same key gives the same
token for "Jan Kowalski" and "Jana Kowalskiego" in any process, with no shared state.

The key is what keeps the token from naming the value. A plain hash of a PESEL is undone by
hashing every possible PESEL, which a graphics card does in under a second; the HMAC needs the
key for each guess. Keep one key per company, and per index where the indexes must not be linked
(or give each a `scope`), and keep it out of the index, the logs and the source code. A new key,
scope or number of digits changes every token, so the index has to be built again.

Twelve digits leave a collision between two values of one type unlikely below about a hundred
thousand values, and possible at a million. One that does happen is detected wherever both values
meet, and the value that came second gets four digits more; `collisions` keeps the record of it.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Mapping, MutableMapping
from typing import NoReturn

from polanon._canonical import canonical
from polanon.policy import ENTITY_TYPES

#: The rules a token is computed by. Tokens computed under another scheme do not match these.
SCHEME = "polanon-pseudonyms/1"

#: The shortest key accepted: 256 bits, as many as the HMAC can use.
MIN_KEY_BYTES = 32

#: Digits in a token's number unless the caller asks for others.
DEFAULT_DIGITS = 12

#: Digits added to a token that collides with the token of another value.
COLLISION_STEP = 4

_MIN_DIGITS = 6
_MAX_DIGITS = 24
#: Lengthenings tried before giving up; a third collision in a row means the key is broken.
_LEVELS = 3

#: Stands for the owner of a token the caller wrote as a placeholder, which no value may take.
_PLACEHOLDER = "placeholder"


class PseudonymCollisionError(RuntimeError):
    """Raised when a value collides with others at every length; the message names no value."""


class Pseudonyms:
    """Computes stable tokens from a secret key.

    One instance per key is enough for a process, and threads may share it. It is never pickled or
    copied, so the key does not end up in a task queue, a cache or a crash dump by accident: build
    one from the key wherever it is needed.
    """

    __slots__ = ("_collisions", "_digits", "_key", "_key_id", "_scope")

    def __init__(
        self,
        key: bytes | bytearray | memoryview,
        *,
        scope: str = "",
        digits: int = DEFAULT_DIGITS,
        collisions: MutableMapping[str, str] | None = None,
    ) -> None:
        """Take the key and the settings the tokens depend on.

        Args:
            key: at least 32 random bytes, such as ``os.urandom(32)`` stored in a secret manager.
                It is copied, and never written to a message, a log or a representation.
            scope: a name that keeps the tokens of one index apart from another's under the same
                key, such as ``"kb:contracts"``. The same value gets unrelated tokens in two scopes.
            digits: digits in a token's number, from 6 to 24.
            collisions: the record of collisions found so far, token -> owner's fingerprint, kept
                by the application next to its mapping store and shared by every process. It holds
                tokens and fingerprints only, nothing that names a value. An empty dict when
                omitted, which remembers collisions in this process only.

        Raises:
            TypeError: for a key that is not bytes, such as a str.
            ValueError: for a key shorter than 32 bytes, or digits outside 6 to 24.
        """
        key = _key_bytes(key)
        if len(key) < MIN_KEY_BYTES:
            raise ValueError(
                f"the key has {len(key)} bytes; a stable pseudonym needs at least "
                f"{MIN_KEY_BYTES} random ones, such as os.urandom(32)"
            )
        if not isinstance(scope, str):
            raise TypeError(f"scope is a str, not {type(scope).__name__}")
        if isinstance(digits, bool) or not isinstance(digits, int):
            raise TypeError(f"digits is an int, not {type(digits).__name__}")
        if not _MIN_DIGITS <= digits <= _MAX_DIGITS:
            raise ValueError(f"digits is from {_MIN_DIGITS} to {_MAX_DIGITS}, not {digits}")
        self._key = key
        self._scope = scope
        self._digits = digits
        self._collisions: MutableMapping[str, str] = {} if collisions is None else collisions
        self._key_id = hmac.new(key, b"polanon key id", hashlib.sha256).hexdigest()[:16]

    @property
    def key_id(self) -> str:
        """A short name for the key, safe to log and to store with an index.

        It tells two keys apart without revealing either: an index built under another key_id
        does not match tokens computed now.
        """
        return self._key_id

    @property
    def settings(self) -> dict[str, str | int]:
        """Everything the tokens depend on except the key itself, to store with an index.

        If the settings an index was built with differ from these, its tokens do not match the
        ones computed now, and it has to be built again.
        """
        return {
            "scheme": SCHEME,
            "key_id": self._key_id,
            "scope": self._scope,
            "digits": self._digits,
        }

    @property
    def collisions(self) -> MutableMapping[str, str]:
        """The record of collisions: token -> fingerprint of the value that keeps it."""
        return self._collisions

    def token(self, value: str, entity_type: str) -> str:
        """Return the stable token of `value` read as `entity_type`, such as ``<OSOBA_4821...>``.

        It does not depend on the spelling: "Jana Kowalskiego" and "Jan Kowalski" give one token.
        Use it to find every passage of an index that names someone, for instance to delete them
        when the person asks to be erased.

        Raises:
            PseudonymCollisionError: when the value collides at every length.
        """
        return self.allocate(value, entity_type, lambda token: None)

    def fingerprint(self, value: str, entity_type: str) -> str:
        """Return a fingerprint of `value`: two spellings of one value have the same one.

        It is the full HMAC, from which no value can be read without the key, so a mapping store
        may keep it unencrypted beside each token and compare values by it.
        """
        return self._digest(self._message(value, entity_type), 0).hex()

    def allocate(self, value: str, entity_type: str, taken: Callable[[str], str | None]) -> str:
        """Return the token of `value`, lengthened past any token another value holds.

        Args:
            value: the value, as written.
            entity_type: its type.
            taken: returns the fingerprint of the value that holds a token where the caller
                knows of one, such as in the mapping of the current conversation, or None.
        """
        message = self._message(value, entity_type)
        owner_of_value = self._digest(message, 0).hex()
        label = ENTITY_TYPES.get(entity_type, entity_type)
        for level in range(_LEVELS):
            digits = self._digits + level * COLLISION_STEP
            number = int.from_bytes(self._digest(message, level), "big") % 10**digits
            token = f"<{label}_{number:0{digits}d}>"
            owner = self._collisions.get(token) or taken(token)
            if owner is None or owner == owner_of_value:
                return token
        raise PseudonymCollisionError(
            f"a {entity_type} value collides with another at {_LEVELS} lengths of its token, "
            "which a working HMAC key does not do: check the key and the collision record"
        )

    def record_collision(self, token: str, owner: str) -> str:
        """Record that the value with fingerprint `owner` keeps `token`; return who keeps it.

        A token already recorded keeps its first owner, so two processes that meet the same
        collision agree on it.
        """
        return self._collisions.setdefault(token, owner)

    def _message(self, value: str, entity_type: str) -> bytes:
        label = ENTITY_TYPES.get(entity_type, entity_type)
        return "\x00".join((SCHEME, self._scope, label, canonical(value, entity_type))).encode()

    def _digest(self, message: bytes, level: int) -> bytes:
        if level:
            message += b"\x00level %d" % level
        return hmac.digest(self._key, message, "sha256")

    def __repr__(self) -> str:
        """Name the key by its id only."""
        return f"Pseudonyms(key_id={self._key_id!r}, scope={self._scope!r}, digits={self._digits})"

    def __reduce_ex__(self, protocol: object) -> NoReturn:
        """Refuse pickling and copying, which would write the key out."""
        raise TypeError(
            "Pseudonyms is not pickled or copied, so its key cannot leak that way: "
            "build one from the key in each process"
        )


def _key_bytes(key: object) -> bytes:
    """Return the key as bytes; anything else, such as a password typed as text, is refused."""
    if not isinstance(key, bytes | bytearray | memoryview):
        raise TypeError(
            f"the key is bytes, such as os.urandom(32), not {type(key).__name__}: "
            "a password typed as text is far easier to guess"
        )
    return bytes(key)


def owner_lookup(
    pseudonyms: Pseudonyms, mapping: Mapping[str, str], type_of: Callable[[str], str | None]
) -> Callable[[str], str | None]:
    """Return a `taken` callback for :meth:`Pseudonyms.allocate` that reads a mapping.

    A token the mapping holds belongs to the value it holds, and a token that stands for itself is
    a placeholder the caller wrote, which no value may take. The mapping may grow between calls,
    so only an answer that names an owner is kept.
    """
    owners: dict[str, str] = {}

    def taken(token: str) -> str | None:
        if token in owners:
            return owners[token]
        held = mapping.get(token)
        entity_type = type_of(token)
        if held is None or entity_type is None:
            return None
        owner = _PLACEHOLDER if held == token else pseudonyms.fingerprint(held, entity_type)
        owners[token] = owner
        return owner

    return taken
