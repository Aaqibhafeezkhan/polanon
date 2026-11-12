"""Stable pseudonyms: one token for one value in every instance, process and spelling."""

import copy
import hashlib
import hmac
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pytest

import polanon
from polanon import (
    TOKEN_PATTERN,
    Anonymizer,
    MockJudge,
    PseudonymCollisionError,
    Pseudonyms,
    RegexGuard,
    default_guards,
)
from polanon._matching import type_of
from polanon.guards.names_pl import _MALE_NAMES
from polanon.pseudonyms import owner_lookup

KEY = bytes(range(32))
OTHER_KEY = bytes(range(1, 33))
TOKEN = re.compile(TOKEN_PATTERN)
#: The shape PolSales recognises a token by: a label and a number.
POLSALES_TOKEN = re.compile(r"<[A-Z][A-Z_]*_\d+>")


def stable(key=KEY, **options):
    return Anonymizer(default_guards(with_ner=False), pseudonyms=Pseudonyms(key, **options))


def tokens(text):
    return [match.group(0) for match in TOKEN.finditer(text)]


def pesel(index):
    """A made-up PESEL with a valid checksum: born in 1980, a different one for every index."""
    month, day, serial = 1 + index // 10_000 % 12, 1 + index // 120_000 % 28, index % 10_000
    digits = f"80{month:02d}{day:02d}{serial:04d}"
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    check = (10 - sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 10) % 10
    return digits + str(check)


# --- one value, one token ---


def test_two_instances_with_one_key_give_one_token():
    text = "Umowę podpisał Jan Kowalski, PESEL 50810112354, NIP 526-000-00-05."
    first = stable().anonymize_sync(text)
    second = stable().anonymize_sync(text)
    assert first.anonymized == second.anonymized
    assert len(tokens(first.anonymized)) == 3


@pytest.mark.parametrize("hash_seed", ["0", "12345", "random"])
def test_another_process_gives_the_same_token(hash_seed):
    """Nothing in a token may depend on the process: not hash(), not the order of a set."""
    code = (
        "from polanon import Anonymizer, Pseudonyms, default_guards\n"
        "p = Pseudonyms(bytes(range(32)))\n"
        "a = Anonymizer(default_guards(with_ner=False), pseudonyms=p)\n"
        "print(a.anonymize_sync('Pismo Jana Kowalskiego, PESEL 50810112354.').anonymized)\n"
    )
    library = str(Path(polanon.__file__).parent.parent)
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": library}
    done = subprocess.run(  # noqa: S603 - the interpreter running these tests
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    here = stable().anonymize_sync("Pismo Jana Kowalskiego, PESEL 50810112354.").anonymized
    assert done.stdout.strip() == here


@pytest.mark.parametrize(
    "text",
    [
        "Umowę podpisał Jan Kowalski.",
        "Pełnomocnictwo dla Jana Kowalskiego.",
        "Spotkanie z Janem Kowalskim.",
        "Przekazano Janowi Kowalskiemu.",
        "Pismo do Pana Jana Kowalskiego.",
        "DANE: JAN KOWALSKI",
        "co podpisał jan kowalski",
    ],
)
def test_a_name_in_any_case_gets_the_token_of_its_nominative(text):
    expected = stable().anonymize_sync("Jan Kowalski").anonymized
    assert tokens(stable().anonymize_sync(text).anonymized) == [expected]


def test_one_text_keeps_the_first_spelling_for_the_token_it_shares():
    result = stable().anonymize_sync("Jan Kowalski, potem Jana Kowalskiego.")
    (token,) = set(tokens(result.anonymized))
    assert result.anonymized == f"{token}, potem {token}."
    assert result.mapping == {token: "Jan Kowalski"}


@pytest.mark.parametrize(
    "spellings",
    [
        ["NIP 5260000005", "NIP 526-000-00-05", "NIP PL5260000005"],
        ["konto PL61 1090 1014 0000 0712 1981 2874", "konto 61109010140000071219812874"],
        ["tel. +48 600 100 200", "tel. 600-100-200"],
    ],
)
def test_an_identifier_gets_one_token_however_it_is_written(spellings):
    found = {tuple(tokens(stable().anonymize_sync(text).anonymized)) for text in spellings}
    assert len(found) == 1


def test_different_people_get_different_tokens():
    people = ["Jan Kowalski", "Janina Kowalska", "Anna Kowalska", "Jan Nowak"]
    found = [stable().anonymize_sync(f"Pismo od: {name}.").anonymized for name in people]
    assert len({tuple(tokens(text)) for text in found}) == len(people)


def test_another_key_scope_or_length_gives_other_tokens():
    text = "Jan Kowalski, PESEL 50810112354."
    results = [
        stable().anonymize_sync(text),
        stable(OTHER_KEY).anonymize_sync(text),
        stable(scope="kb:umowy").anonymize_sync(text),
        stable(digits=14).anonymize_sync(text),
    ]
    found = [set(tokens(result.anonymized)) for result in results]
    for index, these in enumerate(found):
        for others in found[index + 1 :]:
            assert not these & others


def test_a_token_has_the_shape_every_restorer_reads():
    result = stable().anonymize_sync("Jan Kowalski, PESEL 50810112354, mail jan@example.com.")
    for token in tokens(result.anonymized):
        assert POLSALES_TOKEN.fullmatch(token)
        assert len(TOKEN.fullmatch(token).group("n")) == 12


def test_no_two_values_share_a_token_in_a_large_synthetic_set():
    """At 12 digits two of 30,000 values of one type collide with a chance of about 1 in 2,000."""
    pseudonyms = Pseudonyms(KEY)
    identifiers = {pseudonyms.token(pesel(serial), "PESEL") for serial in range(20_000)}
    assert len(identifiers) == 20_000
    names = {f"{first} Kowal{serial}ski" for first in _MALE_NAMES[:50] for serial in range(200)}
    assert len({pseudonyms.token(name, "PERSON") for name in names}) == len(names)


def test_the_token_is_the_hmac_of_the_canonical_form_as_the_scheme_says():
    """Written out here so the scheme is documented and cannot change without this test noticing."""
    message = "\x00".join(("polanon-pseudonyms/1", "", "PESEL", "50810112354")).encode()
    number = int.from_bytes(hmac.digest(KEY, message, "sha256"), "big") % 10**12
    assert Pseudonyms(KEY).token("508 101 123 54", "PESEL") == f"<PESEL_{number:012d}>"
    assert Pseudonyms(KEY).fingerprint("50810112354", "PESEL") == (
        hmac.digest(KEY, message, "sha256").hex()
    )


def test_tokens_of_the_first_scheme_are_pinned():
    pseudonyms = Pseudonyms(KEY)
    assert {
        value: pseudonyms.token(value, entity_type)
        for value, entity_type in [
            ("Jana Kowalskiego", "PERSON"),
            ("526-000-00-05", "NIP"),
            ("Firma Kot Sp. z o.o.", "ORG"),
            ("Hasło2024!", "SECRET"),
        ]
    } == {
        "Jana Kowalskiego": "<OSOBA_548354014016>",
        "526-000-00-05": "<NIP_448788630316>",
        "Firma Kot Sp. z o.o.": "<FIRMA_167437650533>",
        "Hasło2024!": "<SEKRET_469588147612>",
    }


# --- the key ---


@pytest.mark.parametrize("key", ["a" * 64, 12345, None, list(range(32))])
def test_a_key_that_is_not_bytes_is_refused(key):
    with pytest.raises(TypeError, match="bytes"):
        Pseudonyms(key)


def test_a_short_key_is_refused_without_repeating_it():
    secret = b"sekretny-klucz-31-bajtow-xxxxxx"
    with pytest.raises(ValueError, match="31 bytes") as raised:
        Pseudonyms(secret)
    message = str(raised.value)
    assert secret.decode() not in message
    assert secret.hex() not in message


@pytest.mark.parametrize("digits", [5, 25, 0])
def test_digits_out_of_range_are_refused(digits):
    with pytest.raises(ValueError, match="digits"):
        Pseudonyms(KEY, digits=digits)


@pytest.mark.parametrize(("option", "value"), [("digits", True), ("digits", 12.0), ("scope", 7)])
def test_settings_of_the_wrong_type_are_refused(option, value):
    with pytest.raises(TypeError, match=option):
        Pseudonyms(KEY, **{option: value})


def test_the_key_appears_in_no_representation_and_is_never_pickled_or_copied():
    pseudonyms = Pseudonyms(KEY, scope="kb")
    shown = repr(pseudonyms) + str(pseudonyms) + repr(pseudonyms.settings)
    for written in (KEY.hex(), repr(KEY), KEY.decode("latin-1")):
        assert written not in shown
    assert pseudonyms.key_id in shown
    for leak in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError, match="key"):
            leak(pseudonyms)


def test_the_key_is_copied_so_changing_the_callers_buffer_changes_no_token():
    buffer = bytearray(KEY)
    pseudonyms = Pseudonyms(buffer)
    before = pseudonyms.token("50810112354", "PESEL")
    buffer[0] ^= 0xFF
    assert pseudonyms.token("50810112354", "PESEL") == before


def test_the_key_id_tells_keys_apart_and_does_not_reveal_them():
    assert Pseudonyms(KEY).key_id == Pseudonyms(KEY, scope="other").key_id
    assert Pseudonyms(KEY).key_id != Pseudonyms(OTHER_KEY).key_id
    assert (
        Pseudonyms(KEY).key_id
        == (hmac.new(KEY, b"polanon key id", hashlib.sha256).hexdigest()[:16])
    )


# --- collisions ---


def colliding_pair(pseudonyms):
    """Two made-up identifiers whose shortest tokens are the same, found by search."""
    seen = {}
    for serial in range(100_000):
        value = f"{serial:011d}"
        token = pseudonyms.token(value, "PESEL")
        if token in seen:
            return seen[token], value, token
        seen[token] = value
    raise AssertionError("no collision found")


def test_two_values_with_one_token_in_one_text_get_two_tokens():
    pseudonyms = Pseudonyms(KEY, digits=6)
    first, second, token = colliding_pair(pseudonyms)
    guard = FixedValues(first, second)
    anonymizer = Anonymizer([guard], enabled={"PESEL"}, pseudonyms=pseudonyms)
    result = anonymizer.anonymize_sync(f"Numery {first} i {second}.")
    found = tokens(result.anonymized)
    assert found[0] == token
    assert len(found[1]) == len(token) + 4
    assert Anonymizer.restore(result.anonymized, result.mapping) == f"Numery {first} i {second}."


def test_a_recorded_collision_lengthens_the_second_value_wherever_it_stands_alone():
    pseudonyms = Pseudonyms(KEY, digits=6)
    first, second, token = colliding_pair(pseudonyms)
    record = {token: pseudonyms.fingerprint(first, "PESEL")}
    recorded = Pseudonyms(KEY, digits=6, collisions=record)
    assert recorded.token(first, "PESEL") == token
    assert len(recorded.token(second, "PESEL")) == len(token) + 4
    assert recorded.collisions is record


def test_the_owner_of_a_token_in_a_mapping_is_its_value_or_a_placeholder():
    pseudonyms = Pseudonyms(KEY)
    mapping = {"<PESEL_1>": "50810112354", "<NIP_1>": "<NIP_1>"}
    taken = owner_lookup(pseudonyms, mapping, type_of)
    owner = pseudonyms.fingerprint("50810112354", "PESEL")
    assert taken("<PESEL_1>") == owner
    assert taken("<PESEL_1>") == owner  # asked again
    assert taken("<NIP_1>") != owner
    assert taken("<PESEL_2>") is None
    mapping["<PESEL_2>"] = "45831523452"  # the mapping grows while a text is masked
    assert taken("<PESEL_2>") == pseudonyms.fingerprint("45831523452", "PESEL")


def test_a_collision_keeps_its_first_owner():
    pseudonyms = Pseudonyms(KEY)
    assert pseudonyms.record_collision("<PESEL_1>", "first") == "first"
    assert pseudonyms.record_collision("<PESEL_1>", "second") == "first"


def test_a_value_colliding_at_every_length_is_refused_without_naming_it():
    with pytest.raises(PseudonymCollisionError) as raised:
        Pseudonyms(KEY).allocate("50810112354", "PESEL", lambda token: "someone else")
    assert "50810112354" not in str(raised.value)


def test_a_placeholder_the_caller_wrote_is_never_handed_to_a_value():
    pseudonyms = Pseudonyms(KEY)
    token = pseudonyms.token("50810112354", "PESEL")
    result = Anonymizer([RegexGuard()], pseudonyms=pseudonyms).anonymize_sync(
        f"Szablon: {token}. PESEL 50810112354."
    )
    placeholder, value = tokens(result.anonymized)
    assert placeholder == token
    assert result.mapping[token] == token
    assert value != token
    assert result.mapping[value] == "50810112354"


# --- the rest of the library keeps the tokens ---


def test_a_turn_with_known_values_keeps_the_stable_tokens():
    turn = stable().turn(known=["Klub Orlik"])
    masked = turn.mask("Klient Klub Orlik, opiekun Jan Kowalski.")
    expected = Pseudonyms(KEY)
    assert expected.token("Klub Orlik", "ORG") in masked
    assert expected.token("Jan Kowalski", "PERSON") in masked


async def test_a_judge_finding_gets_a_stable_token():
    pseudonyms = Pseudonyms(KEY)
    anonymizer = Anonymizer(
        default_guards(with_ner=False),
        judge=MockJudge({"dyrektor huty": "OTHER"}),
        pseudonyms=pseudonyms,
    )
    result = await anonymizer.anonymize("Pisał dyrektor huty w Stalowej Woli.")
    assert pseudonyms.token("dyrektor huty", "OTHER") in result.anonymized


def test_a_long_token_is_restored_from_a_stream_cut_at_any_point():
    result = stable().anonymize_sync("Umowę podpisał Jan Kowalski, PESEL 50810112354.")
    answer = result.anonymized
    for cut in range(1, len(answer)):
        restorer = Anonymizer.stream_restorer(result.mapping)
        shown = restorer.feed(answer[:cut]) + restorer.feed(answer[cut:]) + restorer.flush()
        assert shown == result.original


class FixedValues(polanon.Guard):
    """Reports each of the given values where it stands, as a PESEL."""

    name = "fixed_values"

    def __init__(self, *values):
        self.values = values

    def detect(self, text, enabled):
        spans = []
        for value in self.values:
            start = text.find(value)
            if start >= 0:
                spans.append(polanon.Span(start, start + len(value), "PESEL", 1.0, self.name))
        return spans
