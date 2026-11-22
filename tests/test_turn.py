"""One call to a model masked with one object: what it sends, and what it gets back."""

import copy
import datetime
import json
import re
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from polanon import (
    ENTITY_TYPES,
    TOKEN_PATTERN,
    WELL_KNOWN_NAMES,
    Allowlist,
    Anonymizer,
    MaskedTurn,
    MaskingUnavailableError,
    MockJudge,
    RegexGuard,
    default_guards,
    token_instructions,
)
from polanon.guards.base import Guard
from polanon.types import Span

anonymizer = Anonymizer(default_guards(with_ner=False), allowlist=Allowlist(WELL_KNOWN_NAMES))


class BrokenGuard(Guard):
    """A detector with a bug whose error message carries the text it was given."""

    name = "broken"

    def detect(self, text, enabled):
        raise KeyError(text)


def test_one_mapping_serves_the_history_the_question_and_the_tool_results():
    turn = anonymizer.turn()
    assert turn.mask("Wczoraj dzwonił Jan Kowalski.") == "Wczoraj dzwonił <OSOBA_1>."
    assert turn.mask("Jan Kowalski prosi o fakturę.") == "<OSOBA_1> prosi o fakturę."
    result = {"klient": "Jan Kowalski", "pesel": "50810112354", "kwota": 1200}
    assert turn.mask_data(result) == {"klient": "<OSOBA_1>", "pesel": "<PESEL_1>", "kwota": 1200}
    assert turn.mapping == {"<OSOBA_1>": "Jan Kowalski", "<PESEL_1>": "50810112354"}


def test_a_turn_continues_a_conversation_without_changing_its_mapping():
    earlier = {"<OSOBA_1>": "Jan Kowalski"}
    turn = MaskedTurn(anonymizer, mapping=earlier)
    assert turn.mask("Anna Nowak i Jan Kowalski") == "<OSOBA_2> i <OSOBA_1>"
    assert earlier == {"<OSOBA_1>": "Jan Kowalski"}
    assert turn.mapping == {"<OSOBA_1>": "Jan Kowalski", "<OSOBA_2>": "Anna Nowak"}


def test_the_mapping_a_turn_hands_out_is_a_copy():
    turn = anonymizer.turn()
    turn.mask("Jan Kowalski")
    turn.mapping.clear()
    assert turn.restore("<OSOBA_1>") == "Jan Kowalski"


def test_a_known_value_is_masked_even_when_the_allowlist_holds_it():
    assert anonymizer.turn().mask("Zwrot dla Allegro") == "Zwrot dla Allegro"
    assert anonymizer.turn(known=["Allegro"]).mask("Zwrot dla Allegro") == "Zwrot dla <FIRMA_1>"


def test_a_known_value_is_masked_even_when_organisations_are_not_enabled():
    only_pesel = Anonymizer(default_guards(with_ner=False), enabled={"PESEL"})
    turn = only_pesel.turn(known=["Klub Orlik", " "])
    assert turn.mask("Klub Orlik, PESEL 50810112354") == "<FIRMA_1>, PESEL <PESEL_1>"


def test_a_single_string_of_known_values_is_refused():
    with pytest.raises(TypeError, match="not a single string") as error:
        anonymizer.turn(known="Klub Orlik")
    assert "Orlik" not in str(error.value)


def test_a_turn_refuses_an_anonymizer_with_a_judge():
    with pytest.raises(ValueError, match="judge 'mock'"):
        Anonymizer(default_guards(with_ner=False), judge=MockJudge()).turn()


def test_a_failing_detector_refuses_without_quoting_the_text():
    turn = Anonymizer([RegexGuard(), BrokenGuard()]).turn()
    with pytest.raises(MaskingUnavailableError) as error:
        turn.mask_data({"note": "Jan Kowalski, PESEL 50810112354"})
    assert "Kowalski" not in str(error.value)
    assert "50810112354" not in str(error.value)
    assert isinstance(error.value.__cause__, KeyError)
    assert turn.mapping == {}


def test_a_detector_that_refuses_keeps_its_own_message():
    class Unavailable(Guard):
        name = "unavailable"

        def detect(self, text, enabled):
            raise MaskingUnavailableError("the vault holding the client list is locked")

    with pytest.raises(MaskingUnavailableError, match="vault"):
        Anonymizer([Unavailable()]).turn().mask("Jan Kowalski")


def test_a_language_model_that_will_not_load_refuses(monkeypatch):
    from polanon.guards.ner import NerGuard

    load = Mock(side_effect=OSError("damaged model"))
    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=load))
    turn = Anonymizer([NerGuard("synthetic_model")]).turn()
    with pytest.raises(MaskingUnavailableError, match="RuntimeError"):
        turn.mask("Jan Kowalski dzwonił wczoraj.")


def test_known_values_that_cannot_become_a_guard_refuse(monkeypatch):
    def broken(terms):
        raise ValueError(terms)

    monkeypatch.setattr("polanon.turn.DenylistGuard", broken)
    with pytest.raises(MaskingUnavailableError, match="known values") as error:
        anonymizer.turn(known=["Klub Orlik"])
    assert "Orlik" not in str(error.value)


def test_mask_takes_text_only():
    with pytest.raises(TypeError, match="use mask_data"):
        anonymizer.turn().mask({"name": "Jan Kowalski"})  # type: ignore[arg-type]
    assert anonymizer.turn().mask("") == ""


def test_structured_data_keeps_its_shape_keys_and_numbers():
    value = {"Jan Kowalski": ["Anna Nowak", 3, 2.5, True, None], "tags": ("vip", "Anna Nowak")}
    assert anonymizer.turn().mask_data(value) == {
        "Jan Kowalski": ["<OSOBA_1>", 3, 2.5, True, None],
        "tags": ("vip", "<OSOBA_1>"),
    }


def test_an_object_that_is_not_json_like_needs_a_default():
    when = datetime.date(2024, 3, 15)
    with pytest.raises(TypeError, match="default=str"):
        anonymizer.turn().mask_data({"date": when})
    masked = anonymizer.turn().mask_data({"date": when, "by": "Jan Kowalski"}, default=str)
    assert masked == {"date": "2024-03-15", "by": "<OSOBA_1>"}


def test_tool_arguments_and_answers_are_restored():
    turn = anonymizer.turn()
    turn.mask("Jan Kowalski, PESEL 50810112354")
    arguments = {"query": "<OSOBA_1>", "ids": [1, 2], "pesel": ("<PESEL_1>",), "other": "<OSOBA_9>"}
    assert turn.restore_data(arguments) == {
        "query": "Jan Kowalski",
        "ids": [1, 2],
        "pesel": ("50810112354",),
        "other": "<OSOBA_9>",
    }
    assert turn.restore("Dzień dobry <OSOBA_1>") == "Dzień dobry Jan Kowalski"


def test_a_person_comes_back_in_the_case_the_model_asks_for():
    """ "pracownika Jana Kowalskiego" came back as "Szanowny Panie Jana Kowalskiego"."""
    turn = anonymizer.turn()
    turn.mask("Wniosek pracownika Jana Kowalskiego, tel. 600 100 200.")
    answer = (
        "Szanowny Panie <OSOBA_1:wołacz>, wniosek <OSOBA_1:dopełniacz> przyjęto. "
        "Telefon <TELEFON_1:mianownik>, a <OSOBA_1> to jak w tekście."
    )
    assert turn.restore(answer) == (
        "Szanowny Panie Janie Kowalski, wniosek Jana Kowalskiego przyjęto. "
        "Telefon 600 100 200, a Jana Kowalskiego to jak w tekście."
    )
    arguments = {"employee": "<OSOBA_1:mianownik>", "case": "<OSOBA_1:przypadek>"}
    assert turn.restore_data(arguments) == {
        "employee": "Jan Kowalski",
        "case": "Jana Kowalskiego",
    }
    assert Anonymizer.restore(answer, turn.mapping) == turn.restore(answer)
    assert turn.restore("Kto to <OSOBA_9:wołacz>?") == "Kto to <OSOBA_9:wołacz>?"
    assert turn.unknown_tokens("Kto to <OSOBA_9:wołacz> i <OSOBA_1:wołacz>?") == ["<OSOBA_9>"]


def test_a_tool_that_looks_records_up_gets_the_values_as_they_were_masked():
    """PolCRM and PolSales each wrote this: a tool searches the records the tokens were masked
    from, and "Jana Kowalskiego" finds nobody where the records say "Jan Kowalski"."""
    turn = anonymizer.turn()
    turn.mask_data({"client": "Jan Kowalski", "phone": "600 100 200"})
    arguments = {"who": ["<OSOBA_1:dopełniacz>", "<OSOBA_1>"], "phone": ("<TELEFON_1:mianownik>",)}
    assert turn.restore_data(arguments, as_held=True) == {
        "who": ["Jan Kowalski", "Jan Kowalski"],
        "phone": ("600 100 200",),
    }
    assert turn.restore_data(arguments)["who"] == ["Jana Kowalskiego", "Jan Kowalski"]
    assert turn.restore("<OSOBA_1:wołacz> i <OSOBA_9:wołacz>", as_held=True) == (
        "Jan Kowalski i <OSOBA_9:wołacz>"
    )


def test_an_application_checks_what_it_is_about_to_send_with_the_core_s_last_check():
    """PolGate wrote its own check, with the masking's thresholds copied by hand; the gateway's
    check is in the core now, with the rules the masking follows."""
    from polanon import leaked_tokens
    from polanon.gateway.payloads import leaked_tokens as gateway_check

    assert gateway_check is leaked_tokens
    turn = anonymizer.turn()
    messages = turn.mask_messages([{"role": "user", "content": "Pani Ewa Kot, tel. 600 100 200."}])
    assert leaked_tokens(messages, turn.mapping) == []
    replayed = [*messages, {"role": "user", "content": "Jeszcze raz: Ewa Kot, 600 100 200."}]
    assert leaked_tokens(replayed, turn.mapping) == ["<OSOBA_1>", "<TELEFON_1>"]
    # A short surname inside a longer word, and an id from code, are not what the masking masks.
    assert leaked_tokens({"id": "Kot", "content": "Kotwica i kotlet."}, {"<OSOBA_2>": "Kot"}) == []


def test_a_restorer_taken_early_knows_the_tokens_of_later_tool_results():
    turn = anonymizer.turn()
    turn.mask("Jan Kowalski")
    restorer = turn.stream_restorer()
    turn.mask_data({"pesel": "50810112354"})
    chunks = ["Dzień dobry <OSO", "BA_1>, PESEL <PES", "EL_1>."]
    shown = "".join(restorer.feed(chunk) for chunk in chunks) + restorer.flush()
    assert shown == "Dzień dobry Jan Kowalski, PESEL 50810112354."


def test_made_up_tokens_and_new_personal_data_are_reported():
    turn = anonymizer.turn()
    turn.mask("Jan Kowalski")
    assert turn.unknown_tokens("Zadzwoń do <OSOBA_1> i <OSOBA_7>, <OSOBA_7>") == ["<OSOBA_7>"]
    restored = turn.restore("Dzień dobry <OSOBA_1>, rozmawiałam z Anną Nowak.")
    report = turn.sweep(restored)
    assert [restored[span.start : span.end] for span in report.new_spans] == ["Anną Nowak"]


def test_counts_leave_out_placeholders_the_caller_wrote():
    turn = anonymizer.turn()
    masked = turn.mask("Pole <PESEL_1> wypełnia Jan Kowalski, PESEL 50810112354")
    assert masked == "Pole <PESEL_1> wypełnia <OSOBA_1>, PESEL <PESEL_2>"
    assert turn.counts() == {"PERSON": 1, "PESEL": 1}


ANTHROPIC = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Znajdź klienta Jan Kowalski.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Szukam Jana Kowalskiego", "signature": "c2ln"},
            {
                "type": "text",
                "text": "Sprawdzam.",
                "citations": [{"type": "char_location", "cited_text": "Jan Kowalski"}],
            },
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "find_client",
                "input": {"name": "Jan Kowalski", "limit": 5},
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01",
                "content": json.dumps({"client": "Jan Kowalski", "pesel": "50810112354"}),
                "is_error": False,
            }
        ],
    },
]


def test_an_anthropic_history_is_masked_where_it_carries_data():
    before = copy.deepcopy(ANTHROPIC)
    question, answer, result = anonymizer.turn(unknown_blocks="drop").mask_messages(ANTHROPIC)
    assert before == ANTHROPIC
    assert question["content"] == [
        {
            "type": "text",
            "text": "Znajdź klienta <OSOBA_1>.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert answer["content"] == [
        {"type": "text", "text": "Sprawdzam."},
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "find_client",
            "input": {"name": "<OSOBA_1>", "limit": 5},
        },
    ]
    [block] = result["content"]
    assert block["tool_use_id"] == "toolu_01"
    assert block["is_error"] is False
    assert json.loads(block["content"]) == {"client": "<OSOBA_1>", "pesel": "<PESEL_1>"}


def test_a_block_that_cannot_be_masked_is_refused_unless_it_may_be_left_out():
    with pytest.raises(MaskingUnavailableError, match="'thinking' block") as error:
        anonymizer.turn().mask_messages(ANTHROPIC)
    assert "Kowalsk" not in str(error.value)
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}}
    message = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_02", "content": [image]},
            {"type": "text", "text": "Co jest na zdjęciu Anny Nowak?"},
        ],
    }
    with pytest.raises(MaskingUnavailableError, match="'image' block"):
        anonymizer.turn().mask_messages([message])
    [masked] = anonymizer.turn(unknown_blocks="drop").mask_messages([message])
    assert masked["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_02", "content": []},
        {"type": "text", "text": "Co jest na zdjęciu <OSOBA_1>?"},
    ]


def test_a_block_that_cannot_be_masked_goes_as_it_is_when_the_turn_keeps_it():
    """An operator who wants the model to see the images sends them unmasked, and says so."""
    data = "iVBORw0KGgo="
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}
    message = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_02", "content": [image]},
            {"type": "text", "text": "Co jest na zdjęciu Anny Nowak?"},
        ],
    }
    [masked] = anonymizer.turn(unknown_blocks="keep").mask_messages([message])
    assert masked["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_02", "content": [image]},
        {"type": "text", "text": "Co jest na zdjęciu <OSOBA_1>?"},
    ]
    call = {"id": "c9", "type": "custom", "custom": {"name": "sql", "input": "SELECT 1"}}
    history = [{"role": "assistant", "content": None, "tool_calls": [call]}]
    [kept] = anonymizer.turn(unknown_blocks="keep").mask_messages(history)
    assert kept["tool_calls"] == [call]


def test_an_openai_history_keeps_its_arguments_and_tool_results_valid_json():
    messages = [
        {"role": "system", "content": "Jesteś asystentem CRM."},
        {"role": "user", "content": [{"type": "text", "text": "Kim jest Anna Nowak?"}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_contact", "arguments": '{"name": "Anna Nowak"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"phone": "600 100 200"}'},
        {"role": "tool", "tool_call_id": "call_2", "content": "Brak danych o Annie Nowak."},
        {"role": "assistant", "content": None, "refusal": "Nie podam danych Anny Nowak."},
    ]
    system, user, call, result, note, refusal = anonymizer.turn().mask_messages(messages)
    assert system == messages[0]
    assert user["content"] == [{"type": "text", "text": "Kim jest <OSOBA_1>?"}]
    [tool_call] = call["tool_calls"]
    assert tool_call["id"] == "call_1"
    assert tool_call["function"]["name"] == "get_contact"
    assert json.loads(tool_call["function"]["arguments"]) == {"name": "<OSOBA_1>"}
    assert result == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"phone": "<TELEFON_1>"}',
    }
    # "Annie" is the dative and the locative alike; the first case that gives the form is written.
    assert note["content"] == "Brak danych o <OSOBA_1:celownik>."
    assert refusal["refusal"] == "Nie podam danych <OSOBA_1:dopełniacz>."


def test_a_provider_s_reasoning_in_an_assistant_message_is_masked_and_sealed_parts_pass():
    """OpenRouter returns the model's reasoning with its answer; a client that sends the answer
    back with the history sent the reasoning, restored, in the clear."""
    details = [
        {"type": "reasoning.text", "text": "Pyta o Annę Nowak.", "format": "unknown", "index": 0},
        {"type": "reasoning.summary", "summary": "Anna Nowak, telefon.", "index": 1},
        {"type": "reasoning.encrypted", "data": "gAAAA-sealed", "index": 2},
        "unknown",
    ]
    messages = [
        {"role": "user", "content": "Kim jest Anna Nowak?"},
        {
            "role": "assistant",
            "content": "To klientka.",
            "reasoning": "Anna Nowak to klientka.",
            "reasoning_content": "Szukam: Anna Nowak.",
            "reasoning_details": details,
        },
    ]
    _, answer = anonymizer.turn().mask_messages(messages)
    assert answer["reasoning"] == "<OSOBA_1> to klientka."
    assert answer["reasoning_content"] == "Szukam: <OSOBA_1>."
    text, summary, sealed, odd = answer["reasoning_details"]
    assert text == {**details[0], "text": "Pyta o <OSOBA_1:biernik>."}
    assert summary == {**details[1], "summary": "<OSOBA_1>, telefon."}
    assert sealed == details[2]
    assert odd == "unknown"


def test_arguments_given_as_a_value_and_old_function_calls_are_masked():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "f", "arguments": {"who": "Anna Nowak"}},
        },
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_3", "type": "custom", "custom": {"input": "Anna Nowak"}}],
        },
    ]
    with pytest.raises(MaskingUnavailableError, match="not a function call"):
        anonymizer.turn().mask_messages(messages)
    old, custom = anonymizer.turn(unknown_blocks="drop").mask_messages(messages)
    assert old["function_call"] == {"name": "f", "arguments": {"who": "<OSOBA_1>"}}
    assert custom["tool_calls"] == []


@pytest.mark.parametrize(
    ("earlier", "later", "masked"),
    [
        ("Kod 4821 nie działa.", "Mój PIN: 4821", "Kod <SEKRET_1> nie działa."),
        ("Login admin2024 nie działa", "hasło: admin2024", "Login <SEKRET_1> nie działa"),
        (
            "Konieczny Jan spóźnił się.",
            "Uczestnik: Konieczny Jan, ur. 1990",
            "<OSOBA_2> spóźnił się.",
        ),
        (
            "Kowalski dzwonił wczoraj.",
            "Jan Kowalski prosi o fakturę.",
            "<OSOBA_2> dzwonił wczoraj.",
        ),
    ],
)
def test_a_value_found_in_a_later_message_is_masked_in_an_earlier_one(earlier, later, masked):
    assert anonymizer.turn().mask(earlier) != masked  # nothing recognises it on its own
    first, _ = anonymizer.turn().mask_messages(
        [{"role": "user", "content": earlier}, {"role": "user", "content": later}]
    )
    assert first["content"] == masked


def test_a_value_found_late_in_one_message_is_masked_where_it_came_earlier_in_it():
    """An agent's tool returned JSON with a person's name first in a list of tags, where nothing
    recognised it, and then in a sentence that did. The tags went out in the clear, and the
    gateway's last check, finding them, refused the call: the agent got an error, three times."""
    note = "Jan Kowalski prosi o fakturę."
    result = json.dumps({"tags": ["pilne", "Kowalski"], "note": note}, ensure_ascii=False)
    messages = [
        {"role": "tool", "tool_call_id": "call_1", "content": result},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Kowalski?"}, {"type": "text", "text": note}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "notify",
                        "arguments": json.dumps(
                            {"to": "Kowalski", "text": note}, ensure_ascii=False
                        ),
                    },
                },
            ],
        },
    ]
    for message in messages:
        masked = anonymizer.turn().mask_messages([message])
        assert "Kowalski" not in json.dumps(masked, ensure_ascii=False), message["role"]
    data = anonymizer.turn().mask_data({"tags": ["Kowalski"], "note": note})
    assert "Kowalski" not in json.dumps(data, ensure_ascii=False)


class SecondLookGuard(Guard):
    """Finds the model code "Q12" in a text only when it reads that text a second time.

    It stands in for what makes the mapping grow while earlier messages are masked once more:
    in an agent's history of 115 messages a piece left over from a losing detection became a new
    value then, and a message re-masked just before kept it in the clear.
    """

    name = "second_look"

    def __init__(self):
        self.seen = set()

    def detect(self, text, enabled):
        again = text in self.seen
        self.seen.add(text)
        start = text.find("Q12")
        if not again or "Jan Kowalski" not in text or start < 0:
            return []
        return [Span(start, start + 3, "ORG", 0.95, self.name)]


def test_a_value_found_while_messages_are_masked_once_more_is_masked_in_all_of_them():
    second_look = Anonymizer([*default_guards(with_ner=False), SecondLookGuard()])
    messages = [
        {"role": "user", "content": "Mam drukarkę Q12."},
        {"role": "user", "content": "Jan Kowalski ma drukarkę Q12."},
        {"role": "user", "content": "Anna Nowak też."},
    ]
    masked = second_look.turn().mask_messages(messages)
    assert "Q12" not in json.dumps(masked, ensure_ascii=False)


class EveryLookGuard(Guard):
    """Finds one more code in a text each time it reads it again, so the mapping never settles."""

    name = "every_look"

    def __init__(self):
        self.reads = 0

    def detect(self, text, enabled):
        if not text.startswith("Kody"):
            return []
        self.reads += 1
        start = text.find(f"K{self.reads}0")
        return [] if start < 0 else [Span(start, start + 3, "ORG", 0.95, self.name)]


def test_masking_stops_after_its_passes_when_the_mapping_keeps_growing():
    from polanon.turn import MASKING_PASSES

    guard = EveryLookGuard()
    every_look = Anonymizer([*default_guards(with_ner=False), guard])
    messages = [
        {"role": "user", "content": "Kody K10 K30 K50 K70"},
        {"role": "user", "content": "Kody K20 K40 K60 K80"},
    ]
    every_look.turn().mask_messages(messages)
    # Both read in every pass, since the mapping grows while each of them is masked, and no more.
    assert guard.reads == 2 * MASKING_PASSES


def test_messages_that_are_not_chat_messages_are_refused():
    with pytest.raises(TypeError, match="a message is a dictionary"):
        anonymizer.turn().mask_messages(["Jan Kowalski"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="content is a str or a list"):
        anonymizer.turn().mask_messages([{"role": "user", "content": 5}])
    with pytest.raises(TypeError, match="a content block is a dictionary"):
        anonymizer.turn().mask_messages([{"role": "user", "content": ["Jan Kowalski"]}])
    with pytest.raises(ValueError, match="'raise', 'drop' or 'keep'"):
        anonymizer.turn(unknown_blocks="skip")  # type: ignore[arg-type]


def test_empty_tool_results_calls_without_arguments_and_json_scalars_pass_through():
    messages = [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_03"}]},
        {"role": "assistant", "function_call": {"name": "ping"}},
        {"role": "tool", "tool_call_id": "call_4", "content": '"Anna Nowak"'},
    ]
    empty, ping, scalar = anonymizer.turn().mask_messages(messages)
    assert empty == messages[0]
    assert ping == messages[1]
    assert scalar["content"] == '"<OSOBA_1>"'


@pytest.mark.parametrize("language", ["pl", "en"])
def test_the_token_instructions_name_tokens_the_library_hands_out(language):
    note = token_instructions(language)
    tokens = re.findall(TOKEN_PATTERN, note)
    assert len(set(tokens)) == 3
    assert {label for label, _ in tokens} <= set(ENTITY_TYPES.values())


def test_the_polish_note_asks_for_a_person_s_case_and_its_examples_restore():
    note = token_instructions("pl")
    asked = re.findall(r"<OSOBA_1:([^>]+)>", note)
    assert {"wołacz", "dopełniacz", "mianownik"} <= set(asked)
    turn = anonymizer.turn()
    turn.mask("Jana Kowalskiego")
    assert "<OSOBA_1:" not in turn.restore(note)
    assert ":" not in "".join(re.findall(r"<[A-Z_]+_\d+(?::[^>]*)?>", token_instructions("en")))


def test_the_token_instructions_are_polish_unless_asked_otherwise():
    assert token_instructions() == token_instructions("pl")
    assert "przepisz" in token_instructions()
    with pytest.raises(ValueError, match="pl and en, not 'de'"):
        token_instructions("de")


def test_a_date_under_a_key_that_names_a_birthday_is_a_birth_date():
    """A contact book an agent read held a birthday under the key "birthday", and the guards,
    shown the value alone, let the date go out. A key that names a birth date is read as the
    label before its value; a date under any other key stays readable."""
    turn = anonymizer.turn()
    contact = {
        "name": "Irena Nowicka",
        "birthday": "1961-11-21",
        "dateOfBirth": "21.11.1961",
        "created": "2026-11-15",
        "family": [{"birth_date": "1957-11-03", "relation": "mama"}],
    }
    masked = turn.mask_data(contact)
    assert re.fullmatch(r"<DATA_URODZENIA_\d+>", masked["birthday"])
    assert re.fullmatch(r"<DATA_URODZENIA_\d+>", masked["dateOfBirth"])
    assert re.fullmatch(r"<DATA_URODZENIA_\d+>", masked["family"][0]["birth_date"])
    assert masked["created"] == "2026-11-15"
    assert turn.restore_data(masked) == contact
