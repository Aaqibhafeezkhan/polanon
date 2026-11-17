"""An AI agent's calls as a regression check on what leaks and what is masked for nothing.

The corpus is small and synthetic, so the numbers here are bounds, not targets: a leak more, or a
token more that hides nothing, fails the build instead of quietly shipping, and a change that does
better lowers the bound. `benchmarks/run_agent.py` prints the same numbers in a table.
"""

import ipaddress
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from agent_corpus import MARK, MARKED_CALLS, calls  # noqa: E402
from run_agent import anonymizer_for, occurrences, run, strings  # noqa: E402

from polanon.policy import ENTITY_TYPES  # noqa: E402

#: What leaks with the default masking: "sopotu" by design, and a name written in Cyrillic,
#: which no guard finds.
KNOWN_LEAKS = {"sopotu", "Олена Коваленко"}

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


def _private_address(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


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
        assert address.endswith("example") or _private_address(address), address
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
    assert leaked <= 35


@needs_model
@pytest.mark.slow
def test_with_the_polish_model_little_leaks_and_little_is_masked_for_nothing():
    """The bounds of the default masking, both ways.

    One value leaks by design, a place typed in lowercase in a chat, "sopotu". A foreign name
    opening an English sentence, read as a capitalised English word, is found as a first name
    of the PESEL register. No guard finds a name written in Cyrillic. Masked for nothing are a
    few German nouns, a speaker's name in a smart home and "thanks" in Ukrainian.
    """
    scores = run(anonymizer_for(with_ner=True))
    leaked = {value for score in scores.values() for _, value in score.leaked_values}
    assert leaked <= KNOWN_LEAKS
    wasted = sum(score.wasted_uses for score in scores.values())
    used = sum(score.uses for score in scores.values())
    assert wasted <= 5
    assert wasted / used <= 0.02


@needs_model
@pytest.mark.slow
def test_in_paranoid_mode_nothing_leaks_and_more_than_half_the_tokens_hide_nothing():
    """What `mode="paranoid"` buys and what it costs: "sopotu" is masked, and so
    are "return", "message" and "notify" of the tools and the agent's English. What no guard
    finds still leaks."""
    scores = run(anonymizer_for(with_ner=True, mode="paranoid"))
    leaked = {value for score in scores.values() for _, value in score.leaked_values}
    assert leaked <= KNOWN_LEAKS - {"sopotu"}
    wasted = sum(score.wasted_uses for score in scores.values())
    used = sum(score.uses for score in scores.values())
    assert wasted <= 386
    assert wasted / used > 0.5


@pytest.mark.parametrize("mode", ["normal", "paranoid"])
@needs_model
@pytest.mark.slow
def test_every_answer_comes_back_by_every_path(mode):
    """A token comes back whole whether the answer is restored as text, as a response or as a
    stream cut anywhere, in a tool's arguments too. A case the model asks for is declined; only
    a name polanon cannot decline for sure comes back as the call wrote it."""
    scores = run(anonymizer_for(with_ner=True, mode=mode))
    wrong = [(name, item) for name, score in scores.items() for item in score.restore.wrong]
    assert wrong == []
    as_written = sum(s.restore.outcomes["response"]["as written"] for s in scores.values())
    assert as_written <= 2


def test_every_answer_comes_back_without_a_model(model_free):
    assert [item for score in model_free.values() for item in score.restore.wrong] == []
