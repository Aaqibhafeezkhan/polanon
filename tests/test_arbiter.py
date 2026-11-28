"""The arbiter between pl_core_news_md and FastPDN, and the rules that keep it from doing worse.

The spaCy model reads for real, so the tests that run a guard are marked slow; FastPDN is stood in
for by an object that returns the entities each test needs, so neither onnxruntime nor the model
is needed. That the guard masks what the experiment it was fitted in masked was checked on 1,900
of the benchmarks' texts, with the real FastPDN, with no difference.
"""

import pytest

import polanon
from polanon import NerModelUnavailableError, arbiter_available, default_guards
from polanon._detection import detect
from polanon.guards import _arbiter_weights as fitted
from polanon.guards._fastpdn import Entity
from polanon.guards.arbiter import ArbiterGuard, fastpdn_type, polish_runs
from polanon.guards.ner import NerGuard
from polanon.policy import ENTITY_TYPES

ALL = set(ENTITY_TYPES)


class FastPDN:
    """Stands in for FastPDN: entities as (word, category, score), found wherever they stand."""

    def __init__(self, *found):
        self.found = found
        self.read = []

    def entities(self, text):
        self.read.append(text)
        hits = []
        for word, group, score in self.found:
            at = text.find(word)
            if at >= 0:
                hits.append(Entity(at, at + len(word), group, score))
        return hits


def masked(guard, text):
    rules = default_guards(with_ner=False)
    spans = detect(text, None, ALL, [*rules, guard])
    return sorted((text[s.start : s.end], s.entity_type) for s in spans if s.source == guard.name)


def test_fastpdn_s_categories_become_polanon_s_types():
    assert fastpdn_type("nam_liv_person") == "PERSON"
    assert fastpdn_type("nam_loc_gpe_city") == "LOCATION"
    assert fastpdn_type("nam_fac_road") == "LOCATION"
    assert fastpdn_type("nam_org_company") == "ORG"
    assert fastpdn_type("nam_org_nation") is None
    assert fastpdn_type("nam_pro_software") is None


def test_fastpdn_reads_only_the_runs_of_polish_lines():
    text = (
        "The chat app renders a subset of Markdown.\nPisała do mnie wczoraj, że przyjedzie.\n---\n"
    )
    runs = polish_runs(text)
    assert [text[a:b] for a, b in runs] == ["Pisała do mnie wczoraj, że przyjedzie.\n"]


def test_the_settings_it_was_not_fitted_for_are_refused():
    with pytest.raises(ValueError, match="with_ner=True"):
        default_guards(arbiter=True, with_ner=False)
    with pytest.raises(ValueError, match="normal mode"):
        default_guards(arbiter=True, mode="paranoid")
    with pytest.raises(ValueError, match="ML filter"):
        default_guards(arbiter=True, ml_filter=False)
    with pytest.raises(ValueError, match="pl_core_news_md"):
        default_guards(arbiter=True, spacy_model="pl_core_news_lg")


def test_without_its_packages_or_the_export_it_says_what_to_run(tmp_path, monkeypatch):
    installed = {"spacy", "pl_core_news_md"}
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: object() if name in installed else None
    )
    with pytest.raises(NerModelUnavailableError, match=r"onnxruntime.*python -m polanon\.fastpdn"):
        default_guards(arbiter=True, fastpdn_dir=tmp_path)
    installed |= {"onnxruntime", "tokenizers"}
    with pytest.raises(NerModelUnavailableError, match=r"missing model\.onnx"):
        default_guards(arbiter=True, fastpdn_dir=tmp_path)
    assert not arbiter_available(tmp_path)
    for name in ("model.onnx", "tokenizer.json", "config.json"):
        (tmp_path / name).write_bytes(b"")
    assert arbiter_available(tmp_path)
    guards = default_guards(arbiter=True, fastpdn_dir=tmp_path)
    assert isinstance(guards[-1], ArbiterGuard)
    assert guards[-1].fastpdn_dir == tmp_path
    assert not any(isinstance(guard, NerGuard) for guard in guards)


def test_the_arbiter_needs_no_model_until_it_reads():
    guard = ArbiterGuard(fastpdn=FastPDN())
    assert guard.detect("", ALL) == []
    assert guard._review("tekst", [], []) == []
    assert polanon.default_guards is default_guards


@pytest.mark.slow
def test_a_person_the_spacy_guard_masks_stays_masked_whatever_the_arbiter_thinks(monkeypatch):
    monkeypatch.setitem(fitted.THRESHOLDS, "PERSON", 1.01)  # the arbiter would drop every person
    text = "Wczoraj pisała do mnie Renata Kubiak w sprawie umowy najmu."
    guard = ArbiterGuard(fastpdn=FastPDN())
    assert ("Renata Kubiak", "PERSON") in masked(guard, text)


@pytest.mark.slow
def test_a_person_only_fastpdn_finds_on_a_polish_line_is_the_arbiter_s_to_add(monkeypatch):
    text = "Zgłoszenie przyjął wczoraj pomidorek z działu obsługi klienta."
    found = FastPDN(("pomidorek", "nam_liv_person", 0.999))
    monkeypatch.setitem(fitted.THRESHOLDS, "PERSON", 0.0)
    assert ("pomidorek", "PERSON") in masked(ArbiterGuard(fastpdn=found), text)
    monkeypatch.setitem(fitted.THRESHOLDS, "PERSON", 1.01)
    assert ("pomidorek", "PERSON") not in masked(ArbiterGuard(fastpdn=found), text)


@pytest.mark.slow
def test_an_organisation_only_fastpdn_finds_is_never_a_candidate(monkeypatch):
    monkeypatch.setitem(fitted.THRESHOLDS, "ORG", 0.0)
    text = "Codziennie rano słucham muzyki w kwadraciku, a potem idę do pracy."
    guard = ArbiterGuard(fastpdn=FastPDN(("kwadraciku", "nam_org_company", 0.99)))
    assert all(kind != "ORG" for _, kind in masked(guard, text))


@pytest.mark.slow
def test_lines_fastpdn_does_not_read_are_decided_as_the_spacy_guard_decides():
    text = (
        "Use the tool to read the calendar of Anna Lindqvist and reply to Henrik.\n"
        "Return the result as JSON with the keys name, city and date."
    )
    stand_in = FastPDN(("Anna Lindqvist", "nam_liv_person", 0.99))
    arbiter = ArbiterGuard(fastpdn=stand_in)
    rules = default_guards(with_ner=False)
    spacy = [s for s in detect(text, None, ALL, [*rules, NerGuard()]) if s.source == "spacy_ner"]
    assert masked(arbiter, text) == sorted((text[s.start : s.end], s.entity_type) for s in spacy)
    assert stand_in.read == []


@pytest.mark.slow
def test_what_fastpdn_s_pieces_carry_at_an_edge_is_trimmed(monkeypatch):
    monkeypatch.setitem(fitted.THRESHOLDS, "LOCATION", 0.0)
    text = "Paczkę odbierzemy jutro w Gdyni.\nDo zobaczenia."
    guard = ArbiterGuard(fastpdn=FastPDN(("Gdyni.\nDo", "nam_loc_gpe_city", 0.99)))
    found = masked(guard, text)
    assert ("Gdyni", "LOCATION") in found
    assert all(not value.endswith(".") and value != "Do" for value, _ in found)


@pytest.mark.slow
def test_a_text_read_before_is_neither_read_nor_decided_again_and_never_remembered_as_written(
    monkeypatch,
):
    text = "Wczoraj pisała do mnie Renata Kubiak z Gdyni w sprawie umowy."
    stand_in = FastPDN(
        ("Renata Kubiak", "nam_liv_person", 0.99), ("Gdyni", "nam_loc_gpe_city", 0.99)
    )
    guard = ArbiterGuard(fastpdn=stand_in)
    decided = []
    original = guard._decide
    monkeypatch.setattr(guard, "_decide", lambda *args: decided.append(1) or original(*args))
    first = masked(guard, text)
    assert masked(guard, text) == first
    assert len(stand_in.read) == 1
    assert len(decided) == 1
    remembered = repr([vars(reading) for reading in guard._read.values()]) + repr(guard._decided)
    assert "Kubiak" not in remembered
    assert "Gdyni" not in remembered
    forgetful = ArbiterGuard(fastpdn=stand_in, cache_texts=0)
    masked(forgetful, text)
    masked(forgetful, text)
    assert len(stand_in.read) == 3
