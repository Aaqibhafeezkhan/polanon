"""The facade someone meets right after pip install polanon: defaults, checks, blocking call."""

import asyncio

import pytest

from polanon import (
    DEFAULT_ENABLED,
    WELL_KNOWN_NAMES,
    Allowlist,
    Anonymizer,
    JudgeError,
    MaskingUnavailableError,
    NerModelUnavailableError,
    RegexGuard,
    default_guards,
    ner_available,
)
from polanon.guards.base import Guard
from polanon.judge.mock import MockJudge
from polanon.types import JudgeFallback, Span

TEXT = "Jan Kowalski, PESEL 50810112354, termin 15.03.2024."


class EmployeeGuard(Guard):
    """A third-party guard with a type of its own, declared the documented way."""

    name = "employee"
    entity_types = frozenset({"EMPLOYEE_ID"})

    def detect(self, text, enabled):
        start = text.find("EMP-")
        if start < 0 or "EMPLOYEE_ID" not in enabled:
            return []
        return [Span(start, start + 9, "EMPLOYEE_ID")]


def test_anonymize_sync_gives_what_anonymize_gives():
    anonymizer = Anonymizer(default_guards(with_ner=False))
    assert anonymizer.anonymize_sync(TEXT) == asyncio.run(anonymizer.anonymize(TEXT))


async def test_anonymize_sync_works_inside_a_running_event_loop():
    """A notebook, or an async web handler calling sync code, already has a loop running."""
    result = Anonymizer([RegexGuard()]).anonymize_sync("PESEL 50810112354")
    assert result.anonymized == "PESEL <PESEL_1>"


def test_anonymize_sync_reuses_a_conversation_mapping_without_changing_it():
    mapping = {"<PESEL_1>": "50810112354"}
    result = Anonymizer([RegexGuard()]).anonymize_sync(
        "Ten sam PESEL 50810112354 i nowy 45831523452", mapping
    )
    assert result.anonymized == "Ten sam PESEL <PESEL_1> i nowy <PESEL_2>"
    assert mapping == {"<PESEL_1>": "50810112354"}


def test_anonymize_sync_refuses_a_judge_instead_of_skipping_it():
    with pytest.raises(RuntimeError, match="await"):
        Anonymizer([RegexGuard()], judge=MockJudge()).anonymize_sync("tekst")


def test_dates_stay_readable_unless_asked_for():
    """Masking every deadline left the model unable to answer what the text was sent for."""
    text = "Termin: 15.03.2024, PESEL 50810112354"
    assert Anonymizer([RegexGuard()]).anonymize_sync(text).anonymized == (
        "Termin: 15.03.2024, PESEL <PESEL_1>"
    )
    with_dates = Anonymizer([RegexGuard()], enabled=DEFAULT_ENABLED | {"DATE"})
    assert with_dates.anonymize_sync(text).anonymized == "Termin: <DATA_1>, PESEL <PESEL_1>"


@pytest.mark.parametrize("typo", ["PERSONS", "person", "Pesel"])
def test_an_unknown_entity_type_is_refused_instead_of_masking_nothing(typo):
    with pytest.raises(ValueError, match="unknown entity types"):
        Anonymizer([RegexGuard()], enabled={typo})


def test_a_type_a_custom_guard_declares_is_on_by_default():
    result = Anonymizer([EmployeeGuard()]).anonymize_sync("Pracownik EMP-11111.")
    assert result.anonymized == "Pracownik <EMPLOYEE_ID_1>."


def test_a_type_a_custom_guard_declares_is_accepted_in_enabled():
    anonymizer = Anonymizer([EmployeeGuard()], enabled={"EMPLOYEE_ID"})
    assert anonymizer.anonymize_sync("EMP-11111").anonymized == "<EMPLOYEE_ID_1>"


def test_a_fallback_given_as_text_is_accepted_and_checked():
    assert Anonymizer([], judge_fallback="block").judge_fallback is JudgeFallback.BLOCK
    with pytest.raises(ValueError, match="blokuj"):
        Anonymizer([], judge_fallback="blokuj")


def test_a_single_string_denylist_is_refused():
    """A string is an iterable of letters: "Acme" became the terms "A", "c", "m" and "e"."""
    with pytest.raises(TypeError, match="not a single string") as error:
        default_guards(denylist="Acme", with_ner=False)
    assert "Acme" not in str(error.value)


def test_missing_spacy_is_reported_when_the_guards_are_built(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError, match="with_ner=False"):
        default_guards()


def test_a_missing_model_names_the_command_that_installs_it(monkeypatch):
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: object() if name == "spacy" else None
    )
    with pytest.raises(ModuleNotFoundError, match="spacy download pl_core_news_md"):
        default_guards()


class WordGuard(Guard):
    """Flags fixed words the way the language model flags names."""

    name = "spacy_ner"

    def __init__(self, entity_type: str, *words: str) -> None:
        self.entity_type = entity_type
        self.words = words

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        if self.entity_type not in enabled:
            return []
        return [
            Span(start, start + len(word), self.entity_type, 0.85, self.name)
            for word in self.words
            if (start := text.find(word)) != -1
        ]


def test_birth_date_is_masked_by_default():
    """A deadline identifies nobody; a birth date next to a name does."""
    assert "BIRTH_DATE" in DEFAULT_ENABLED
    result = Anonymizer(default_guards(with_ner=False)).anonymize_sync("Pacjent ur. 15.03.1985")
    assert result.anonymized == "Pacjent ur. <DATA_URODZENIA_1>"


async def test_facade_blocks_when_the_judge_fails_and_the_policy_says_block():
    anonymizer = Anonymizer([RegexGuard()], judge=MockJudge(fail=True), judge_fallback="block")
    with pytest.raises(JudgeError):
        await anonymizer.anonymize("PESEL 50810112354")


def test_anonymize_sync_honours_the_allowlist():
    anonymizer = Anonymizer([WordGuard("ORG", "Canvie")], allowlist=Allowlist(WELL_KNOWN_NAMES))
    assert anonymizer.anonymize_sync("grafiki w Canvie").anonymized == "grafiki w Canvie"


class LabelGuard(Guard):
    """A guard whose own type is spelled like the token label of a built-in one."""

    name = "label"

    def __init__(self, entity_type: str, declared: bool = True) -> None:
        self.kind = entity_type
        self.entity_types = frozenset({entity_type}) if declared else frozenset()

    def detect(self, text, enabled):
        return [Span(0, 3, self.kind)]


@pytest.mark.parametrize("entity_type", ["OSOBA", "DATA", "SEKRET"])
def test_a_declared_type_spelled_like_a_built_in_label_is_refused(entity_type):
    """ "<OSOBA_2>" from such a guard would read as a second person."""
    with pytest.raises(ValueError, match=r"'label' declares the entity type .*token label"):
        Anonymizer([LabelGuard(entity_type)])


def test_a_returned_type_spelled_like_a_built_in_label_is_refused():
    anonymizer = Anonymizer([LabelGuard("OSOBA", declared=False)], enabled={"PERSON"})
    with pytest.raises(ValueError, match="'label' returned the entity type 'OSOBA'"):
        anonymizer.anonymize_sync("KOD-123")


def test_a_declared_type_that_cannot_label_a_token_is_refused():
    with pytest.raises(ValueError, match="uppercase letters and underscores"):
        Anonymizer([LabelGuard("Employee_Id")])


@pytest.mark.parametrize("found", [set(), {"spacy"}], ids=["without spaCy", "without the model"])
def test_a_missing_language_model_is_also_masking_unavailable(monkeypatch, found):
    """An app that refuses to send unmasked text catches one error, whatever is missing."""
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: object() if name in found else None
    )
    with pytest.raises(MaskingUnavailableError) as error:
        default_guards()
    assert isinstance(error.value, NerModelUnavailableError)
    assert isinstance(error.value, ModuleNotFoundError)
    assert "with_ner=False" in str(error.value)


def test_the_language_model_is_reported_available_only_when_it_is_installed(monkeypatch):
    installed = {"spacy", "pl_core_news_md"}
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: object() if name in installed else None
    )
    assert ner_available()
    assert not ner_available("pl_core_news_lg")
    installed.discard("spacy")
    assert not ner_available()
