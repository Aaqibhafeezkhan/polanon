r"""Technical secrets guard: API keys, tokens, passwords and private keys pasted into prompts.

Employees paste logs, config files and code snippets into chat. Those carry credentials that are not
personal data but must never reach a third-party provider either.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from polanon.guards.base import Guard
from polanon.types import Span

#: The header of a private key in any armor: PEM ("RSA", "EC", "ENCRYPTED", "OPENSSH", none) and
#: OpenPGP, whose header reads "PGP PRIVATE KEY BLOCK".
_PRIVATE_KEY = r"(?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"

#: Patterns whose whole match is the secret.
_WHOLE_MATCH: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bsk-or-[A-Za-z0-9_-]{20,}"),
    # OpenAI project, service-account and admin keys: a second prefix, then "-" and "_" inside.
    re.compile(r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}"),  # Stripe secret and restricted keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"),  # GitHub fine-grained tokens
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"),  # GitLab personal access tokens
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    # A JWT starts where a base64url run starts: "\b" also let a match start after every "-" in a
    # long "eyJ-eyJ-..." run, and each start scanned to the end of the run, quadratically.
    re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    # A block never holds a second BEGIN, and stopping there keeps a paste of many headers without
    # an END from rescanning the rest of the text once per header.
    re.compile(
        rf"-----BEGIN {_PRIVATE_KEY}(?:(?!-----BEGIN ).)*?-----END {_PRIVATE_KEY}", re.DOTALL
    ),
    # A key pasted without its END line leaks its body all the same, so the BEGIN line is taken
    # together with every following line that is base64, an armor header or blank.
    re.compile(
        rf"-----BEGIN {_PRIVATE_KEY}"
        r"(?:\r?\n(?:[A-Za-z][A-Za-z-]*: [^\r\n]*|[A-Za-z0-9+/=]+[ \t]*|[ \t]*)(?=\r?\n|\Z))*"
    ),
)

#: Credentials written into a URL: ``postgres://user:password@host``. The host is often a bare
#: service name or an address, which no e-mail rule recognises.
_URL_CREDENTIALS = re.compile(r"(?i)\b[a-z][a-z0-9+.-]{0,30}://[^\s:/@]*:(?P<value>[^\s@/]+)@")

#: The token in an ``Authorization: Bearer ...`` header or a ``Bearer ...`` line from a log.
_BEARER = re.compile(r"(?i)\bBearer\s+(?P<value>[A-Za-z0-9._~+/-]{12,}=*)")

#: The base64 ``user:password`` pair of an ``Authorization: Basic ...`` header.
_BASIC_AUTH = re.compile(r"(?i)\bAuthorization\s*:\s*Basic\s+(?P<value>[A-Za-z0-9+/]{8,}=*)")

#: The words that name a password-like value. They may name what it opens, "hasło do Wi-Fi:".
_CREDENTIAL_LABELS = r"password|passwd|has[łl]o|has[łl]a|pin|kod\s+dost[ęe]pu"

#: The value after a label. A quoted value is taken whole, spaces included, because passphrases
#: are what password policies now ask for. A bare one ends at whitespace or at one of the `stop`
#: characters, and leaves out one ".", ",", ";", ")" or "]" that closes the sentence: "hasło:
#: Zima2026." masked with its full stop gave a value no later bare "Zima2026" matched, so the
#: repeat went out in clear.
_VALUE = (
    r"(?:\"(?P<dq>[^\"\n]{{1,200}})\"|'(?P<sq>[^'\n]{{1,200}})'"
    r"|[\"']?(?P<bare>[^\s\"'{stop}]{{{shortest},}}?)[.,;)\]]?(?![^\s\"'{stop}]))"
)

#: Assignments such as `password: hunter2` or `DB_PASSWORD="abc"`. The value is a named group.
#: The left boundary is a lookbehind rather than ``\b`` so a prefixed name still matches: in
#: ``DB_PASSWORD`` the character before the keyword is an underscore, which ``\b`` treats as part of
#: the same word and would therefore skip. For the same reason "key" and "pass" count as labels
#: only as the last part of a compound name, ``SECRET_KEY``, ``APP_KEY``, ``DB_PASS``, the way
#: settings and .env files name them; alone they are ordinary words. After an explicit ":" or "="
#: a bare value runs to whitespace, so a password with "," or ";" in it is not cut short.
_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    rf"(?:(?:{_CREDENTIAL_LABELS})(?:\s+(?:do|dla|od)\s+[\w-]{{1,20}})?"
    r"|pwd|secret|token|api[_-]?key|access[_-]?key|client[_-]?secret"
    r"|(?<=[_-])key|(?<=[_-])pass"
    r"|klucz|login|u[żz]ytkownik)"
    r"(?:\s+to)?\s*[:=]\s*" + _VALUE.format(stop="", shortest=4)
)

#: Polish and English phrasings that introduce a credential without a colon or an equals sign:
#: "hasło Zima2026", "hasło to Zima2026", "password Zima2026". The word alone is not enough —
#: see :func:`_looks_like_a_secret`.
_LABELLED = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:has[łl]o|has[łl]em|password|pin|kod\s+dost[ęe]pu)(?:\s+to)?\s+"
    + _VALUE.format(stop=",;:", shortest=6)
)

#: A credential named by one or two words of the same line before its colon: "Hasło
#: administratora:", "hasło użytkownika admin:", "Hasło do panelu klienta:". Those words could
#: also be a sentence that reaches a colon later, so the value must look like a password, and not
#: like the date or time such a sentence ends in, "Hasło ważne do: 2026-05-31"; the line break
#: and the word count keep the label from running into prose.
_NAMED_ASSIGNMENT = re.compile(
    rf"(?i)(?<![A-Za-z0-9])(?:{_CREDENTIAL_LABELS})(?:[ \t]+(?:do|dla|od))?"
    r"(?:[ \t]+[^\W\d_][\w-]{0,19}){1,2}[ \t]*:[ \t]*" + _VALUE.format(stop="", shortest=4)
)

#: A PIN is four to eight digits, shorter than any other labelled secret, so it gets its own rule:
#: "PIN 4321", "PIN do karty: 4321", "kod PIN to 4321".
_PIN = re.compile(
    r"(?i)(?<![A-Za-z0-9])PIN(?:-?(?:u|em|y))?(?:\s+(?:do\s+[\w-]+|karty|kod|code|to))*"
    r"\s*[:=]?\s*(?P<value>\d{4,8})(?!\d)"
)


def _looks_like_a_secret(value: str) -> bool:
    """Return whether a word following "hasło" is plausibly the password rather than the sentence.

    Without a colon there is nothing to say where the credential starts, so "hasło zostało
    zmienione" would mask the next ordinary Polish word. A password carries a digit or a symbol
    almost without exception; a word in a sentence does not.
    """
    return any(character.isdigit() or not character.isalnum() for character in value)


#: A date or a time of day, written with dashes, dots, slashes or colons.
_DATE_OR_TIME = re.compile(r"\d{1,4}(?:[-./:]\d{1,4}){1,2}")


def _looks_like_a_named_secret(value: str) -> bool:
    """Return whether the value after "Hasło administratora:" is the password, not a date."""
    return _looks_like_a_secret(value) and _DATE_OR_TIME.fullmatch(value) is None


def _looks_like_a_token(value: str) -> bool:
    """Return whether the word after "Bearer" is a token rather than English prose.

    "bearer responsibilities" is long enough to pass for a token; a word has no digit, no symbol
    and no capital letter inside it, while a random token almost always has one of them.
    """
    return _looks_like_a_secret(value) or any(character.isupper() for character in value[1:])


#: Rules that mask a captured value rather than the whole match: the pattern, the score of a hit,
#: and an optional check that the value is a credential rather than the sentence around it.
#: A credential in a URL or an auth header is certain from its structure. An explicitly labelled
#: assignment ("hasło: ...") is more certain than a statistical name guess, so it outranks the NER
#: guard (0.85) when the two overlap. A password introduced by a word rather than by punctuation
#: is still labelled, but the value is only a guess at where it ends, so it scores below that.
_VALUE_RULES: tuple[tuple[re.Pattern[str], float, Callable[[str], bool] | None], ...] = (
    (_URL_CREDENTIALS, 1.0, None),
    (_BEARER, 1.0, _looks_like_a_token),
    (_BASIC_AUTH, 1.0, None),
    (_ASSIGNMENT, 0.9, None),
    (_NAMED_ASSIGNMENT, 0.9, _looks_like_a_named_secret),
    (_PIN, 0.9, None),
    (_LABELLED, 0.88, _looks_like_a_secret),
)


def _value_span(match: re.Match[str]) -> tuple[int, int]:
    """Return the extent of whichever value group took part in the match."""
    for name in ("value", "dq", "sq", "bare"):
        if name in match.re.groupindex and match.group(name) is not None:
            return match.span(name)
    return match.span()


class SecretsGuard(Guard):
    """Detects credentials and private keys. Entity type is always ``SECRET``."""

    name = "secrets"

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return spans covering credential-like substrings."""
        if "SECRET" not in enabled:
            return []
        # A key block taken up to its last base64 line ends in a newline; the span stops before
        # it so the token does not swallow the paragraph break.
        spans = [
            Span(m.start(), m.start() + len(m.group().rstrip()), "SECRET", 1.0, self.name)
            for pattern in _WHOLE_MATCH
            for m in pattern.finditer(text)
        ]
        for pattern, score, plausible in _VALUE_RULES:
            for m in pattern.finditer(text):
                start, end = _value_span(m)
                if plausible is None or plausible(text[start:end]):
                    spans.append(Span(start, end, "SECRET", score, self.name))
        return spans
