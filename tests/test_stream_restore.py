"""Streaming restore must be independent of how the provider splits its chunks."""

import pytest

from polanon import Anonymizer, StreamRestorer

MAPPING = {"<OSOBA_1>": "Jan Kowalski", "<FIRMA_2>": "ACME Sp. z o.o."}
TEXT = "Witaj <OSOBA_1>, <FIRMA_2> czeka na odpowiedź od <OSOBA_1>."
EXPECTED = Anonymizer.restore(TEXT, MAPPING)


@pytest.mark.parametrize("split_at", range(len(TEXT) + 1))
def test_any_two_way_split_restores_identically(split_at):
    restorer = StreamRestorer(MAPPING)
    out = restorer.feed(TEXT[:split_at]) + restorer.feed(TEXT[split_at:]) + restorer.flush()
    assert out == EXPECTED


def test_character_by_character_stream_restores_identically():
    restorer = StreamRestorer(MAPPING)
    out = "".join(restorer.feed(c) for c in TEXT) + restorer.flush()
    assert out == EXPECTED


def test_unknown_tokens_are_passed_through_untouched():
    restorer = StreamRestorer(MAPPING)
    assert restorer.feed("kto to <OSOBA_9>?") + restorer.flush() == "kto to <OSOBA_9>?"


def test_text_without_tokens_is_emitted_immediately():
    restorer = StreamRestorer(MAPPING)
    assert restorer.feed("zwykły tekst") == "zwykły tekst"
    assert restorer.buffer == ""


def test_lone_angle_bracket_is_flushed_at_the_end():
    restorer = StreamRestorer(MAPPING)
    assert restorer.feed("a < b") + restorer.flush() == "a < b"


def test_long_non_token_fragment_is_not_held_back_forever():
    restorer = StreamRestorer(MAPPING)
    long_tail = "<" + "A" * 40
    assert restorer.feed(long_tail) == long_tail


def test_a_long_custom_token_split_anywhere_is_restored():
    """A custom type can make a token longer than the built-in labels allow for."""
    mapping = {"<INTERNAL_EMPLOYEE_REFERENCE_NUMBER_1>": "EMP-12345"}
    text = "Sprawdziłem <INTERNAL_EMPLOYEE_REFERENCE_NUMBER_1>, wszystko gra."
    for split_at in range(len(text) + 1):
        restorer = StreamRestorer(mapping)
        out = restorer.feed(text[:split_at]) + restorer.feed(text[split_at:]) + restorer.flush()
        assert out == "Sprawdziłem EMP-12345, wszystko gra.", split_at


def test_built_in_token_split_before_its_end_is_restored_when_the_mapping_grows_later():
    """The built-in limit covers every built-in label, whatever the mapping held at creation."""
    mapping: dict[str, str] = {}
    restorer = StreamRestorer(mapping)
    mapping["<KOD_POCZTOWY_1>"] = "00-950"
    out = restorer.feed("Kod: <KOD_POCZTOWY_1") + restorer.feed("> Warszawa") + restorer.flush()
    assert out == "Kod: 00-950 Warszawa"
