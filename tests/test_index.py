"""Masking for a search index: documents, the questions after them, and the store of tokens."""

import re
import threading

import pytest

from polanon import (
    TOKEN_PATTERN,
    Anonymizer,
    InMemoryMappingStore,
    MaskingUnavailableError,
    MockJudge,
    Pseudonyms,
    RegexGuard,
    StoredValue,
    default_guards,
    stored_mapping,
    tokens_in,
)
from polanon.guards.base import Guard
from polanon.types import Span

KEY = bytes(range(32))
TOKEN = re.compile(TOKEN_PATTERN)

#: Two documents that name the same people in other cases, and identifiers written differently.
CONTRACT = [
    "Umowa zawarta między Firmą Kot Sp. z o.o. a Janem Kowalskim, PESEL 50810112354.",
    "Pełnomocnikiem Jana Kowalskiego jest Anna Nowak, tel. +48 600 100 200.",
]
NOTE = ["Notatka: Jan Kowalski dzwonił, prosi o kontakt pod 600-100-200. Anna Nowak też."]


def stable(store=None, digits=12):
    collisions = None if store is None else store.collisions
    pseudonyms = Pseudonyms(KEY, digits=digits, collisions=collisions)
    return Anonymizer(default_guards(with_ner=False), pseudonyms=pseudonyms)


def originals(store):
    return {entry.value for entry in store.get(list(store._entries)).values()}


# --- the store ---


def test_the_store_keeps_the_first_value_of_a_token_and_reports_it_after():
    store = InMemoryMappingStore()
    first = StoredValue("Jan Kowalski", "aa")
    assert store.add({"<OSOBA_1>": first}) == {}
    assert store.add({"<OSOBA_1>": StoredValue("Anna Nowak", "bb")}) == {"<OSOBA_1>": first}
    assert store.get(["<OSOBA_1>", "<OSOBA_2>"]) == {"<OSOBA_1>": first}
    assert len(store) == 1


def test_two_threads_adding_one_token_store_one_value():
    store = InMemoryMappingStore()
    held = []
    barrier = threading.Barrier(8)

    def add(number):
        barrier.wait()
        held.append(store.add({"<OSOBA_1>": StoredValue(f"value {number}", str(number))}))

    threads = [threading.Thread(target=add, args=(number,)) for number in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(store) == 1
    assert sum(1 for result in held if not result) == 1


# --- indexing ---


def test_the_passages_of_a_document_share_one_mapping():
    masked = stable().mask_chunks(CONTRACT)
    person = Pseudonyms(KEY).token("Jan Kowalski", "PERSON")
    assert person in masked.texts[0]
    assert person in masked.texts[1]
    assert masked.mapping[person] == "Janem Kowalskim"


def test_a_value_found_in_a_later_passage_is_masked_in_an_earlier_one():
    """The surname alone is recognised only once the full name has been read."""
    masked = stable().mask_chunks(["Kowalski spóźnił się.", "Jan Kowalski przeprasza."])
    assert "Kowalski" not in masked.texts[0]


def test_each_passage_restores_exactly_as_it_was_written():
    masked = stable().mask_chunks(CONTRACT)
    for text, mapping, original in zip(masked.texts, masked.mappings, CONTRACT, strict=True):
        assert Anonymizer.restore(text, mapping) == original


def test_a_whole_document_may_be_given_as_one_text():
    masked = stable().mask_chunks(NOTE[0])
    assert len(masked.texts) == 1


def test_indexing_writes_nothing_that_names_a_value_to_the_passages():
    store = InMemoryMappingStore()
    masked = stable(store).index_chunks(CONTRACT, store)
    for value in originals(store) | set(masked.mapping.values()):
        for text in masked.texts:
            assert value not in text


def test_the_store_gets_a_name_in_the_nominative_and_a_fingerprint():
    store = InMemoryMappingStore()
    masked = stable(store).index_chunks(CONTRACT, store)
    pseudonyms = Pseudonyms(KEY)
    person = pseudonyms.token("Jan Kowalski", "PERSON")
    assert store.get([person]) == {
        person: StoredValue("Jan Kowalski", pseudonyms.fingerprint("Jan Kowalski", "PERSON"))
    }
    assert set(store.get(list(masked.mapping))) == set(masked.mapping)


def test_the_question_meets_the_tokens_of_every_document_that_names_the_same_people():
    store = InMemoryMappingStore()
    anonymizer = stable(store)
    contract = anonymizer.index_chunks(CONTRACT, store)
    note = anonymizer.index_chunks(NOTE, store)
    question = anonymizer.mask_query("Co podpisał Jan Kowalski i jaki ma numer 600 100 200?", store)
    asked = tokens_in(question.anonymized)
    assert len(asked) == 2
    assert asked <= tokens_in(*contract.texts)
    assert asked <= tokens_in(*note.texts)


def test_a_found_passage_and_the_answer_restore_from_the_store_also_while_streaming():
    store = InMemoryMappingStore()
    anonymizer = stable(store)
    contract = anonymizer.index_chunks(CONTRACT, store)
    question = anonymizer.mask_query("Kto jest pełnomocnikiem Jana Kowalskiego?", store)
    passage = contract.texts[1]
    mapping = question.mapping | stored_mapping(store, question.anonymized, passage)
    assert Anonymizer.restore(passage, mapping) == (
        "Pełnomocnikiem Jan Kowalski jest Anna Nowak, tel. +48 600 100 200."
    )
    person = Pseudonyms(KEY).token("Anna Nowak", "PERSON")
    answer = f"Pełnomocnikiem jest {person}."
    for cut in range(1, len(answer)):
        restorer = Anonymizer.stream_restorer(mapping)
        shown = restorer.feed(answer[:cut]) + restorer.feed(answer[cut:]) + restorer.flush()
        assert shown == "Pełnomocnikiem jest Anna Nowak."


def test_a_passage_from_the_index_masked_again_keeps_its_tokens():
    """A found passage put into a turn is masked again; its tokens must come through whole."""
    store = InMemoryMappingStore()
    anonymizer = stable(store)
    passage = anonymizer.index_chunks(CONTRACT, store).texts[0]
    turn = anonymizer.turn(mapping=stored_mapping(store, passage))
    assert turn.mask(passage) == passage


# --- collisions ---


class Numbers(Guard):
    """Reports every run of eleven digits as a PESEL, checksum or not."""

    name = "numbers"

    def detect(self, text, enabled):
        return [
            Span(m.start(), m.end(), "PESEL", 1.0, self.name) for m in re.finditer(r"\d{11}", text)
        ]


def colliding_pair(pseudonyms):
    seen = {}
    for serial in range(100_000):
        value = f"{serial:011d}"
        token = pseudonyms.token(value, "PESEL")
        if token in seen:
            return seen[token], value, token
        seen[token] = value
    raise AssertionError("no collision found")


def short_tokens(store):
    pseudonyms = Pseudonyms(KEY, digits=6, collisions=store.collisions)
    return Anonymizer([Numbers()], enabled={"PESEL"}, pseudonyms=pseudonyms), pseudonyms


def test_a_collision_with_a_stored_value_gives_the_newcomer_a_longer_token():
    store = InMemoryMappingStore()
    anonymizer, pseudonyms = short_tokens(store)
    first, second, token = colliding_pair(Pseudonyms(KEY, digits=6))
    assert tokens_in(*anonymizer.index_chunks([f"Numer {first}."], store).texts) == {token}
    later = tokens_in(*anonymizer.index_chunks([f"Numer {second}."], store).texts)
    (longer,) = later
    assert len(longer) == len(token) + 4
    assert store.collisions == {token: pseudonyms.fingerprint(first, "PESEL")}
    assert store.get([token])[token].value == first
    assert store.get([longer])[longer].value == second


def test_a_question_about_a_value_that_collides_finds_nothing_of_the_other():
    store = InMemoryMappingStore()
    anonymizer, _ = short_tokens(store)
    first, second, token = colliding_pair(Pseudonyms(KEY, digits=6))
    anonymizer.index_chunks([f"Numer {first}."], store)
    question = anonymizer.mask_query(f"Czyj jest numer {second}?", store)
    assert token not in question.anonymized
    assert anonymizer.mask_query(f"Czyj jest numer {first}?", store).anonymized == (
        f"Czyj jest numer {token}?"
    )


def test_a_collision_inside_one_document_is_recorded_for_the_texts_after_it():
    store = InMemoryMappingStore()
    anonymizer, pseudonyms = short_tokens(store)
    first, second, token = colliding_pair(Pseudonyms(KEY, digits=6))
    masked = anonymizer.index_chunks([f"Numery {first} i {second}."], store)
    longer = TOKEN.findall(masked.texts[0])  # groups: label and number
    assert len(longer) == 2
    assert store.collisions == {token: pseudonyms.fingerprint(first, "PESEL")}
    question = anonymizer.mask_query(f"Numer {second}?", store)
    assert tokens_in(question.anonymized) <= tokens_in(masked.texts[0])
    assert token not in question.anonymized


# --- refusals ---


def test_indexing_needs_stable_tokens():
    store = InMemoryMappingStore()
    plain = Anonymizer(default_guards(with_ner=False))
    with pytest.raises(ValueError, match="pseudonyms"):
        plain.index_chunks(CONTRACT, store)
    with pytest.raises(ValueError, match="pseudonyms"):
        plain.mask_query("Co podpisał Jan Kowalski?", store)


def test_indexing_refuses_a_judge():
    judged = Anonymizer([RegexGuard()], judge=MockJudge({}), pseudonyms=Pseudonyms(KEY))
    with pytest.raises(ValueError, match="judge"):
        judged.mask_chunks(CONTRACT)


@pytest.mark.parametrize("chunks", [[b"bytes"], [None], ["tekst", 7]])
def test_a_passage_that_is_not_text_is_refused(chunks):
    with pytest.raises(TypeError, match="str"):
        stable().mask_chunks(chunks)


def test_a_question_that_is_not_text_is_refused():
    with pytest.raises(TypeError, match="str"):
        stable().mask_query(["Kto?"], InMemoryMappingStore())


class Broken(Guard):
    name = "broken"

    def detect(self, text, enabled):
        raise RuntimeError(f"cannot read {text}")


def test_a_failing_detector_refuses_the_document_without_quoting_it():
    anonymizer = Anonymizer([Broken()], pseudonyms=Pseudonyms(KEY))
    with pytest.raises(MaskingUnavailableError) as raised:
        anonymizer.index_chunks(["Jan Kowalski, PESEL 50810112354."], InMemoryMappingStore())
    assert "50810112354" not in str(raised.value)


class FickleStore(InMemoryMappingStore):
    """Claims to hold every token for some other value, as a store shared under two keys would."""

    def get(self, tokens):
        return {token: StoredValue("someone else", "other") for token in tokens}

    def add(self, entries):
        return {token: StoredValue("someone else", "other") for token in entries}


def test_collisions_that_never_settle_are_refused():
    from polanon import PseudonymCollisionError

    with pytest.raises(PseudonymCollisionError, match="same key"):
        stable().index_chunks(["PESEL 50810112354."], FickleStore())
    with pytest.raises(PseudonymCollisionError, match="same key"):
        stable().mask_query("PESEL 50810112354?", FickleStore())


class Unavailable(Guard):
    name = "unavailable"

    def detect(self, text, enabled):
        raise MaskingUnavailableError("the model is not installed")


def test_a_refusal_from_a_detector_comes_through_as_it_was_raised():
    anonymizer = Anonymizer([Unavailable()], pseudonyms=Pseudonyms(KEY))
    with pytest.raises(MaskingUnavailableError, match="not installed"):
        anonymizer.mask_chunks(["PESEL 50810112354."])


def test_a_placeholder_in_a_passage_stands_for_itself_and_is_not_stored():
    store = InMemoryMappingStore()
    masked = stable(store).index_chunks(["Szablon: <PESEL_1>. Klient Jan Kowalski."], store)
    assert masked.mappings[0]["<PESEL_1>"] == "<PESEL_1>"
    assert store.get(["<PESEL_1>"]) == {}
    assert Anonymizer.restore(masked.texts[0], masked.mappings[0]) == (
        "Szablon: <PESEL_1>. Klient Jan Kowalski."
    )
