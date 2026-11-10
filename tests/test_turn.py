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
    assert note["content"] == "Brak danych o <OSOBA_2>."
    assert refusal["refusal"] == "Nie podam danych <OSOBA_3>."


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


def test_messages_that_are_not_chat_messages_are_refused():
    with pytest.raises(TypeError, match="a message is a dictionary"):
        anonymizer.turn().mask_messages(["Jan Kowalski"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="content is a str or a list"):
        anonymizer.turn().mask_messages([{"role": "user", "content": 5}])
    with pytest.raises(TypeError, match="a content block is a dictionary"):
        anonymizer.turn().mask_messages([{"role": "user", "content": ["Jan Kowalski"]}])
    with pytest.raises(ValueError, match="'raise' or 'drop'"):
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
    assert len(tokens) == 3
    assert {label for label, _ in tokens} <= set(ENTITY_TYPES.values())


def test_the_token_instructions_are_polish_unless_asked_otherwise():
    assert token_instructions() == token_instructions("pl")
    assert "przepisz" in token_instructions()
    with pytest.raises(ValueError, match="pl and en, not 'de'"):
        token_instructions("de")
