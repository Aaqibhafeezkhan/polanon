"""An AI agent's calls as a regression check on what leaks and what is masked for nothing.

The corpus is small and synthetic, so the numbers here are bounds, not targets: a leak more, or a
token more that hides nothing, fails the build instead of quietly shipping, and a change that does
better lowers the bound. `benchmarks/run_agent.py` prints the same numbers in a table.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from agent_corpus import MARK, MARKED_CALLS, calls  # noqa: E402
from run_agent import anonymizer_for, occurrences, run, strings  # noqa: E402

from polanon.policy import ENTITY_TYPES  # noqa: E402

#: Types a rule finds by their shape, checksum or label; a leak of one is a bug.
DETERMINISTIC = {"EMAIL", "PHONE", "POSTAL_CODE", "ADDRESS", "NIP", "PESEL", "IBAN", "SECRET"}


def _model_available() -> bool:
    try:
        import spacy

        spacy.load("pl_core_news_md")
    except (ImportError, OSError):
        return False
    return True


needs_model = pytest.mark.skipif(not _model_available(), reason="pl_core_news_md is not installed")


def _pesel_is_fictional(pesel: str) -> bool:
    """A valid checksum and a birth year before 1860 or after 2100, as CONTRIBUTING.md asks."""
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    check = (10 - sum(int(d) * w for d, w in zip(pesel, weights, strict=False)) % 10) % 10
    year, month = int(pesel[:2]), int(pesel[2:4])
    century = {80: 1800, 40: 2100, 60: 2200}.get(month // 20 * 20)
    born = None if century is None else century + year
    return check == int(pesel[10]) and born is not None and (born < 1860 or born > 2100)


@pytest.fixture(scope="module")
def model_free():
    return run(anonymizer_for(with_ner=False))


def test_every_mark_is_well_formed_and_every_value_is_invented():
    requests, marked = calls()
    assert set(requests) == set(MARKED_CALLS)
    for request in requests.values():
        for text in strings(request):
            assert MARK.search(text) is None
            assert "[[" not in text
    assert marked
    for value in marked:
        assert value.entity_type in ENTITY_TYPES
        assert value.value == value.value.strip()
        if value.entity_type == "PESEL":
            assert _pesel_is_fictional(value.value), value.value
    every_text = [text for request in requests.values() for text in strings(request)]
    for address in re.findall(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)", " ".join(every_text)):
        assert address.endswith("example"), address
    for host in re.findall(r"https?://([^/\s\"']+)", " ".join(every_text)):
        assert host.endswith(".example"), host


def test_every_required_value_stands_in_some_call():
    requests, marked = calls()
    every_text = [text for request in requests.values() for text in strings(request)]
    found = occurrences(every_text, [value.value for value in marked])
    assert all(found[value.value] for value in marked)


def test_no_value_a_rule_finds_leaks_without_a_model(model_free):
    leaked = {
        (entity_type, value)
        for score in model_free.values()
        for entity_type, value in score.leaked_values
        if entity_type in DETERMINISTIC
    }
    assert not leaked


def test_the_rules_alone_mask_nothing_for_nothing(model_free):
    """The rules find a value by its shape or its label, so in an agent's code, logs and English
    they find nothing that is not one. What they leave are the places, the companies without a
    legal form and the foreign names a model is for."""
    assert {name: dict(s.for_nothing) for name, s in model_free.items() if s.for_nothing} == {}
    leaked = sum(score.leaked for score in model_free.values())
    assert leaked <= 20


@needs_model
@pytest.mark.slow
def test_with_the_polish_model_little_leaks_and_little_is_masked_for_nothing():
    """The bounds of the default masking, both ways.

    Two values leak by design: a place typed in lowercase in a chat, "sopotu", and a foreign name
    opening an English sentence, read as a capitalised English word. Masked for nothing are
    mostly "Max", of a chip, a vacuum's fan speed and "Max results" in the tools, and "Dell".
    """
    scores = run(anonymizer_for(with_ner=True))
    leaked = {value for score in scores.values() for _, value in score.leaked_values}
    assert leaked <= {"sopotu", "Erik Holm"}
    wasted = sum(score.wasted_uses for score in scores.values())
    used = sum(score.uses for score in scores.values())
    assert wasted <= 54
    assert wasted / used <= 0.25
