"""The secrets guard on its own: every credential format it knows, the phrasings that introduce
one, and the prose it must leave alone.

Synthetic keys are assembled from parts so that the secret scanners of code hosts do not take this
file for a leak.
"""

import time

import pytest

from polanon import Anonymizer
from polanon.guards.secrets import SecretsGuard
from polanon.types import merge_spans

BODY = "Xq7vR2mK9pL4tW8nB3cF6hJ1dS5gA0zE"
DASHED = BODY + "-YuIo_PaQwErTy7UiOpAsDf"


def masked(text: str) -> list[str]:
    spans = merge_spans(SecretsGuard().detect(text, {"SECRET"}), text)
    return [text[s.start : s.end] for s in spans]


@pytest.mark.parametrize(
    "key",
    [
        "sk-" + "ant-api03-" + DASHED,
        "sk-" + "or-v1-" + DASHED,
        "sk-" + BODY + "abcdEFGH12345678",  # a legacy OpenAI key
        "sk-" + "proj-" + DASHED,
        "sk-" + "svcacct-" + DASHED,
        "sk-" + "admin-" + DASHED,
        "sk_" + "live_" + BODY[:24],
        "rk_" + "test_" + BODY[:24],
        "ghp" + "_" + BODY + "abcd",
        "github" + "_pat_11AXq7vR2mK9_" + BODY,
        "glpat" + "-" + BODY[:20],
        "xoxb" + "-1234567890-" + BODY,
        "https://hooks.slack.com/services/" + "T0000000/B0000000/" + BODY[:24],
        "AKIA" + "IOSFODNN7EXAMPLE",
        "AIza" + BODY + "abc",
        "eyJhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxIn0." + "dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1",
    ],
)
def test_key_is_masked_whole_when_written_in_running_text(key):
    text = f"Mój klucz to {key}, nie udostępniaj."
    assert masked(text) == [key]


PEM_BODY = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\nAmEwggJdAgEAAoGBAKj34GkxFhD90vcN"


@pytest.mark.parametrize(
    ("begin", "end"),
    [
        ("-----BEGIN PRIVATE KEY-----", "-----END PRIVATE KEY-----"),
        ("-----BEGIN RSA PRIVATE KEY-----", "-----END RSA PRIVATE KEY-----"),
        ("-----BEGIN EC PRIVATE KEY-----", "-----END EC PRIVATE KEY-----"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----", "-----END OPENSSH PRIVATE KEY-----"),
        ("-----BEGIN ENCRYPTED PRIVATE KEY-----", "-----END ENCRYPTED PRIVATE KEY-----"),
        ("-----BEGIN PGP PRIVATE KEY BLOCK-----\n", "=AbCd\n-----END PGP PRIVATE KEY BLOCK-----"),
    ],
)
def test_private_key_block_is_masked_whole_for_every_header(begin, end):
    block = f"{begin}\n{PEM_BODY}\n{end}"
    assert masked(f"Klucz serwera:\n{block}\nDzięki!") == [block]


def test_private_key_body_is_masked_when_the_end_line_is_missing():
    """A key cut off while pasting still leaks its body; the masked part stops before the prose."""
    key = f"-----BEGIN OPENSSH PRIVATE KEY-----\n{PEM_BODY}"
    assert masked(f"Wklejam klucz:\n{key}\n...\nDzięki, ąę!") == [key]


def test_private_key_match_stays_linear_when_headers_repeat_without_an_end():
    started = time.perf_counter()
    SecretsGuard().detect("-----BEGIN RSA PRIVATE KEY-----\n" * 5000, {"SECRET"})
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    ("text", "password"),
    [
        ("DATABASE_URL=postgres://app:S3cretPass@db:5432/app", "S3cretPass"),
        ("Połącz się: postgresql://admin:S3cretPass@localhost/crm", "S3cretPass"),
        ("mysql://root:toor1234@10.0.0.5:3306/app", "toor1234"),
        ("REDIS_URL=redis://:Pa55word@cache:6379/0", "Pa55word"),
        ("amqp://guest:Gu3stPass@rabbitmq:5672/", "Gu3stPass"),
    ],
)
def test_password_in_a_connection_url_is_masked_whatever_the_host(text, password):
    assert masked(text) == [password]


@pytest.mark.parametrize(
    ("text", "token"),
    [
        (f'curl -H "Authorization: Bearer {BODY}" https://api.firma.pl', BODY),
        (f"GET /api 401, header bearer {BODY[:20]}", BODY[:20]),
        (
            "curl -H 'Authorization: Basic dXNlcjpTM2NyZXQ=' https://api.firma.pl",
            "dXNlcjpTM2NyZXQ=",
        ),
    ],
)
def test_token_in_an_authorization_header_is_masked(text, token):
    assert masked(text) == [token]


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("Hasło do Wi-Fi: Zima2026!", "Zima2026!"),
        ("Hasło do poczty: Zima2026!", "Zima2026!"),
        ("Hasło to Zima2026!", "Zima2026!"),
        ("Twoje hasło to: Zima2026!", "Zima2026!"),
        ("Kod dostępu: 9f2a11bc", "9f2a11bc"),
        ("PIN: 123456 do bramki", "123456"),
        ("Klient podał PIN 4321 do karty.", "4321"),
        ("PIN do karty: 4321", "4321"),
    ],
)
def test_credential_is_masked_when_introduced_by_an_everyday_phrase(text, secret):
    assert masked(text) == [secret]


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("password = 'correct horse battery 7'", "correct horse battery 7"),
        ('DB_PASSWORD="Ala ma kota 2026"', "Ala ma kota 2026"),
        ('hasło: "Zima 2026!"', "Zima 2026!"),
    ],
)
def test_quoted_passphrase_is_masked_whole_when_it_contains_spaces(text, secret):
    assert masked(text) == [secret]


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("Nowe hasło: Zima2026.", "Zima2026"),
        ("hasło: Tajne123!. Proszę zmienić.", "Tajne123!"),
        ("password=Zima2026) i potem", "Zima2026"),
        ("hasło Zima2026! (tymczasowe)", "Zima2026!"),
        ("hasło: S3cret.Pass", "S3cret.Pass"),  # a dot inside the value stays in it
    ],
)
def test_password_value_stops_before_punctuation_that_closes_the_sentence(text, secret):
    assert masked(text) == [secret]


async def test_password_is_masked_again_when_repeated_after_a_full_stop():
    """The value used to carry the full stop, so the bare repeat matched nothing and went out."""
    text = "Nowe hasło: Zima2026. Jeśli Zima2026 nie działa, zadzwoń."
    result = await Anonymizer([SecretsGuard()]).anonymize(text)
    assert result.anonymized == "Nowe hasło: <SEKRET_1>. Jeśli <SEKRET_1> nie działa, zadzwoń."


@pytest.mark.parametrize(
    "text",
    [
        "Klucz do sukcesu: ciężka praca i regularność.",
        "Hasło zostało zmienione, PIN również.",
        "Ten pin 12 w złączu jest uszkodzony.",
        "Bearer instruments and bearer responsibilities are described in chapter 3.",
        "Zobacz https://firma.pl:8443/panel albo http://host/a@b, port 8443.",
        "token: nie wiem, sprawdź w konfiguracji",
    ],
)
def test_prose_is_left_alone_when_it_only_mentions_credentials(text):
    assert masked(text) == []


def test_nothing_is_reported_when_secrets_are_disabled():
    assert SecretsGuard().detect("hasło: Zima2026!", {"EMAIL"}) == []
