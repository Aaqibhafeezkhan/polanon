"""The annotated sample corpus, used as a regression check on detection quality.

The short synthetic documents are not a research benchmark, so the thresholds here are floors
rather than targets: they exist to make a drop in coverage fail the build instead of quietly
shipping. `benchmarks/run.py` prints the same numbers in a readable table.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from run import EVALUATED, guards_for, load_gold, run  # noqa: E402

#: Types found by regex, checksum or label alone; a miss here is a bug, not a model being a model.
DETERMINISTIC = {
    "EMAIL",
    "PHONE",
    "POSTAL_CODE",
    "ADDRESS",
    "NIP",
    "PESEL",
    "REGON",
    "IBAN",
    "PL_ID",
    "SECRET",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "PLATE",
    "VIN",
    "TAX_ID",
    "COMPANY_ID",
    "BANK_ACCOUNT",
}


def _model_available() -> bool:
    try:
        import spacy

        spacy.load("pl_core_news_md")
    except (ImportError, OSError):
        return False
    return True


needs_model = pytest.mark.skipif(not _model_available(), reason="pl_core_news_md is not installed")


@pytest.fixture(scope="module")
def model_free():
    return run(guards_for(with_ner=False, spacy_model=""))


def test_the_annotations_are_consistent_with_the_documents():
    """load_gold raises on a value that moved or on two annotations that overlap."""
    for path in sorted((ROOT / "benchmarks" / "samples").glob("*.txt")):
        text, gold = load_gold(path.name)
        assert gold, f"{path.name} has no annotations"
        for entity in gold:
            assert text[entity.start : entity.end] == entity.value
            assert entity.entity_type in EVALUATED


def test_no_annotation_is_ever_masked_only_halfway(model_free):
    """Half a masked PESEL is a leak, so a partial cover counts as a failure everywhere."""
    totals, _, _ = model_free
    partial = {name: score.partial for name, score in totals.items() if score.partial}
    assert not partial


def test_every_deterministic_type_is_fully_covered_without_a_model(model_free):
    totals, _, _ = model_free
    missed = {
        name: (score.missed, score.gold)
        for name, score in totals.items()
        if name in DETERMINISTIC and score.masked != score.gold
    }
    assert not missed


def test_every_person_is_masked_without_a_model(model_free):
    """The chat sample is lowercase on purpose: there the Polish model stops seeing people. On the
    foreign invoices every name stands after a label, a form of address or a closing formula."""
    totals, _, _ = model_free
    assert totals["PERSON"].masked == totals["PERSON"].gold


def test_the_guards_without_a_model_cover_four_fifths_of_the_corpus(model_free):
    totals, extra, documents = model_free
    gold = sum(score.gold for score in totals.values())
    masked = sum(score.masked for score in totals.values())
    assert documents >= 10
    assert masked / gold >= 0.8
    assert extra == 0


@needs_model
@pytest.mark.slow
def test_adding_the_polish_model_covers_almost_everything():
    totals, extra, _ = run(guards_for(with_ner=True, spacy_model="pl_core_news_md"))
    gold = sum(score.gold for score in totals.values())
    masked = sum(score.masked for score in totals.values())
    partial = sum(score.partial for score in totals.values())
    assert masked / gold >= 0.95
    assert partial == 0
    # The model still reads "ksiegowosci", typed without Polish letters, as a place, and on the
    # foreign invoices it tags country names, "Company Registered in England and Wales", a Swedish
    # line item and the month "August".
    assert extra <= 7


@needs_model
@pytest.mark.slow
def test_places_and_companies_are_what_the_model_adds():
    totals, _, _ = run(guards_for(with_ner=True, spacy_model="pl_core_news_md"))
    assert totals["ORG"].masked == totals["ORG"].gold
    # A Polish model misses one "Kraków", and "Bristol" and one "Zürich" on the foreign invoices.
    assert totals["LOCATION"].masked >= totals["LOCATION"].gold - 3
    assert totals["PERSON"].masked == totals["PERSON"].gold
