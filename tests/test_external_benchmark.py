"""The pieces of benchmarks/run_external.py that read the downloaded data, offline and on made-up
rows shaped like it: a KPWr sentence rebuilt as written, OpenPII's labels and the checksums."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from run_external import (  # noqa: E402
    id_card_valid,
    kpwr_sentence,
    luhn_valid,
    nip_valid,
    openpii_gold,
    pesel_valid,
)


def test_a_kpwr_sentence_is_rebuilt_as_written_with_its_people_and_places():
    row = {
        "tokens": [
            "Pani",
            "Anna",
            "Chłoń",
            "-",
            "Domińczak",
            "(",
            "ur",
            ".",
            "w",
            "Gdańsku",
            ")",
            "pisała",
            "do",
            "George",
            "’",  # noqa: RUF001
            "owi",
            "W",
            ".",
            "Bushowi",
            ".",
        ],
        "ner": [
            "O",
            "B-nam_liv_person",
            "I-nam_liv_person",
            "I-nam_liv_person",
            "I-nam_liv_person",
            "O",
            "O",
            "O",
            "O",
            "B-nam_loc_gpe_city",
            "O",
            "O",
            "O",
            "B-nam_liv_person",
            "I-nam_liv_person",
            "I-nam_liv_person",
            "I-nam_liv_person",
            "I-nam_liv_person",
            "I-nam_liv_person",
            "O",
        ],
    }
    text, gold = kpwr_sentence(row)
    assert text == "Pani Anna Chłoń-Domińczak (ur. w Gdańsku) pisała do George’owi W. Bushowi."  # noqa: RUF001
    assert [(g.value, g.entity_type) for g in gold] == [
        ("Anna Chłoń-Domińczak", "PERSON"),
        ("Gdańsku", "CITY"),
        ("George’owi W. Bushowi", "PERSON"),  # noqa: RUF001
    ]


def test_openpii_labels_become_polanon_types_and_identifiers_are_split_by_checksum():
    text = "Jan Kowalski, PESEL 50810112354, PESEL 50810112355, tel. 512 345 678, wiek 40"
    row = {
        "source_text": text,
        "privacy_mask": [
            {
                "label": label,
                "start": text.index(value),
                "end": text.index(value) + len(value),
                "value": value,
            }
            for label, value in [
                ("GIVENNAME", "Jan"),
                ("SURNAME", "Kowalski"),
                ("SOCIALNUM", "50810112354"),
                ("SOCIALNUM", "50810112355"),
                ("TELEPHONENUM", "512 345 678"),
                ("AGE", "40"),
            ]
        ],
    }
    assert [(g.value, g.entity_type) for g in openpii_gold(row)] == [
        ("Jan", "PERSON"),
        ("Kowalski", "PERSON"),
        ("50810112354", "PESEL valid"),
        ("50810112355", "PESEL invalid"),
        ("512 345 678", "PHONE"),
    ]


@pytest.mark.parametrize(
    ("check", "valid", "invalid"),
    [
        (pesel_valid, "50810112354", "50810112355"),
        (nip_valid, "123-456-32-18", "123-456-32-19"),
        (luhn_valid, "4111 1111 1111 1111", "4111 1111 1111 1112"),
        (id_card_valid, "ABA300000", "ABA300001"),
    ],
)
def test_the_checksums_tell_a_real_number_from_a_mistyped_one(check, valid, invalid):
    assert check(valid)
    assert not check(invalid)
