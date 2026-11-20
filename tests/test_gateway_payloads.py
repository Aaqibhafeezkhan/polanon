"""The payloads of the OpenAI API as the gateway masks and restores them, without a server."""

import copy
import json
import time

import pytest

from polanon import Anonymizer, MaskedTurn, MaskingUnavailableError, default_guards
from polanon.gateway.payloads import (
    ChatStream,
    GatewayRequestError,
    ResponsesStream,
    leaked_tokens,
    mask_chat_request,
    mask_embeddings_request,
    mask_responses_request,
    restore_chat_response,
    restore_responses_response,
)

ANONYMIZER = Anonymizer(default_guards(with_ner=False))
PESEL = "50810112354"


def turn():
    return MaskedTurn(ANONYMIZER)


def chat(*messages, **fields):
    return {"model": "gpt-test", "messages": list(messages), **fields}


def user(content, **fields):
    return {"role": "user", "content": content, **fields}


# --- requests ----------------------------------------------------------------------------------


def test_every_message_of_a_chat_request_is_masked_and_the_request_left_as_it_was():
    body = chat(
        {"role": "system", "content": "Jesteś asystentem kadr."},
        user(f"Jan Kowalski, PESEL {PESEL}, prosi o urlop."),
        {"role": "assistant", "content": "Kiedy Jan Kowalski chce iść na urlop?"},
        user("Od poniedziałku."),
        temperature=0.2,
        tools=[{"type": "function", "function": {"name": "find", "parameters": {}}}],
    )
    original = copy.deepcopy(body)
    one = turn()
    masked = mask_chat_request(one, body)
    assert body == original
    sent = json.dumps(masked, ensure_ascii=False)
    assert "Kowalski" not in sent
    assert PESEL not in sent
    assert masked["messages"][1]["content"] == "<OSOBA_1>, PESEL <PESEL_1>, prosi o urlop."
    assert masked["messages"][2]["content"] == "Kiedy <OSOBA_1> chce iść na urlop?"
    assert masked["temperature"] == 0.2
    assert masked["tools"] == body["tools"]


def test_a_participant_s_name_is_left_out_and_a_function_s_name_stays():
    """The API allows no token in "name", and a user's name there would go out as written."""
    body = chat(
        user("Dzień dobry.", name="jan_kowalski"),
        {"role": "tool", "tool_call_id": "c1", "name": "find", "content": "Brak."},
    )
    masked = mask_chat_request(turn(), body)
    assert "name" not in masked["messages"][0]
    assert masked["messages"][1]["name"] == "find"


def test_an_end_user_s_identifier_and_stored_metadata_are_masked():
    body = chat(
        user("Pytanie."),
        user="jan.kowalski@example.com",
        metadata={"klient": "Jan Kowalski"},
    )
    masked = mask_chat_request(turn(), body)
    assert masked["user"] == "<EMAIL_1>"
    assert masked["metadata"] == {"klient": "<OSOBA_1>"}


@pytest.mark.parametrize(
    "body",
    [
        {"model": "gpt-test"},
        chat(),
        {"model": "gpt-test", "messages": "Jan Kowalski"},
        chat(user("Hej."), modalities=["text", "audio"]),
    ],
)
def test_a_request_without_messages_or_asking_for_audio_is_refused(body):
    with pytest.raises(GatewayRequestError):
        mask_chat_request(turn(), body)


def test_an_image_in_a_message_is_refused():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    body = chat(user([{"type": "text", "text": "Co to?"}, image]))
    with pytest.raises(MaskingUnavailableError):
        mask_chat_request(turn(), body)


def test_the_last_check_sees_a_value_written_with_lookalike_letters():
    """The check before a call leaves found nothing in "Jan Kowalski" with a Cyrillic a and o."""
    lookalike = "J\u0430n K\u043ew\u0430lski"
    payload = {"messages": [{"role": "user", "content": f"Pismo od {lookalike}."}]}
    assert leaked_tokens(payload, {"<OSOBA_1>": "Jan Kowalski"}) == ["<OSOBA_1>"]


def test_the_last_check_sees_a_short_surname_with_a_case_ending():
    """ "Lisowi" holds the surname "Lis", which the masking masks there too now."""
    mapping = {"<OSOBA_1>": "Lis"}
    payload = {"messages": [{"role": "user", "content": "Przekaż Lisowi notatkę."}]}
    assert leaked_tokens(payload, mapping) == ["<OSOBA_1>"]
    listed = {"messages": [{"role": "user", "content": "Lista zadań na listopad."}]}
    assert leaked_tokens(listed, mapping) == []


def test_an_image_and_a_file_go_as_they_are_when_the_turn_keeps_them():
    """The operator chose to let the model see them: the text is masked, they are not."""
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 2_000_000}}
    file = {"type": "file", "file": {"filename": "a.pdf", "file_data": "data:;base64,JVBERi0="}}
    one = MaskedTurn(ANONYMIZER, unknown_blocks="keep")
    body = chat(user([{"type": "text", "text": "Jan Kowalski?"}, image, file]))
    masked = mask_chat_request(one, body)
    assert masked["messages"][0]["content"] == [{"type": "text", "text": "<OSOBA_1>?"}, image, file]
    started = time.perf_counter()
    assert leaked_tokens(masked, one.mapping) == []
    assert time.perf_counter() - started < 5


@pytest.mark.parametrize(
    ("first", "expected"),
    [
        ({"role": "system", "content": "Bądź zwięzły."}, "Bądź zwięzły.\n\nNOTE"),
        (
            {"role": "developer", "content": [{"type": "text", "text": "Bądź zwięzły."}]},
            [{"type": "text", "text": "Bądź zwięzły."}, {"type": "text", "text": "NOTE"}],
        ),
    ],
)
def test_the_note_about_tokens_joins_the_system_message_the_client_wrote(first, expected):
    masked = mask_chat_request(turn(), chat(first, user("Hej.")), instructions="NOTE")
    assert len(masked["messages"]) == 2
    assert masked["messages"][0]["content"] == expected


def test_the_note_about_tokens_comes_first_when_there_is_no_system_message():
    masked = mask_chat_request(turn(), chat(user("Hej.")), instructions="NOTE")
    assert masked["messages"][0] == {"role": "system", "content": "NOTE"}


def test_a_tool_call_from_the_history_and_its_result_are_masked():
    body = chat(
        user("Znajdź umowę: Jan Kowalski."),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "find", "arguments": '{"who": "Jan Kowalski"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"pesel": PESEL})},
    )
    masked = mask_chat_request(turn(), body)
    arguments = masked["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"who": "<OSOBA_1>"}
    assert json.loads(masked["messages"][2]["content"]) == {"pesel": "<PESEL_1>"}


def leave_tool():
    """A tool whose definition holds personal data in every place a schema lets text stand."""
    return {
        "type": "function",
        "function": {
            "name": "send_leave_notice",
            "description": f"Wysyła potwierdzenie urlopu do pana Zielińskiego, PESEL {PESEL}.",
            "parameters": {
                "type": "object",
                "title": "Urlop pana Zielińskiego",
                "properties": {
                    "employee": {
                        "type": "string",
                        "description": "Na przykład Jan Kowalski.",
                        "enum": ["Jan Kowalski", "Anna Nowak"],
                        "default": "Jan Kowalski",
                    },
                    "description": {
                        "type": "string",
                        "examples": ["Urlop dla Anny Nowak"],
                    },
                    "days": {"type": "integer", "minimum": 1, "const": 5},
                },
                "required": ["employee", "description"],
                "$defs": {"note": {"type": "string", "description": "Do wiadomości Anny Nowak."}},
            },
            "strict": True,
        },
    }


def test_the_text_of_a_tool_definition_is_masked_and_its_shape_kept():
    """An agent's tool descriptions went out as written, and with them any value they held."""
    body = chat(user("Potwierdź urlop."), tools=[leave_tool()])
    original = copy.deepcopy(body)
    one = turn()
    masked = mask_chat_request(one, body)
    assert body == original
    sent = json.dumps(masked, ensure_ascii=False)
    for value in ("Kowalski", "Nowak", "Zielińskiego", PESEL):
        assert value not in sent
    assert leaked_tokens(masked, one.mapping) == []
    function = masked["tools"][0]["function"]
    assert function["name"] == "send_leave_notice"
    assert function["strict"] is True
    schema = function["parameters"]
    assert schema["type"] == "object"
    assert schema["required"] == ["employee", "description"]
    assert set(schema["properties"]) == {"employee", "description", "days"}
    assert schema["properties"]["employee"]["enum"] == ["<OSOBA_2>", "<OSOBA_3>"]
    assert schema["properties"]["days"] == {"type": "integer", "minimum": 1, "const": 5}
    call = {"name": "send_leave_notice", "arguments": json.dumps({"employee": "<OSOBA_2>"})}
    answer = {"choices": [{"message": {"role": "assistant", "tool_calls": [{"function": call}]}}]}
    restored = restore_chat_response(one, answer)["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(restored["function"]["arguments"]) == {"employee": "Jan Kowalski"}


def test_a_value_found_in_a_tool_definition_is_masked_in_the_messages_and_back():
    """Only "pana Zielińskiego" reads as a person; "od Zielińskiego" alone does not.

    Wherever the form of address stands, the value is masked everywhere else as well: in the
    messages, in the other tools and in the tool that held it.
    """
    note = {
        "type": "function",
        "function": {"name": "notes", "description": "Notatki od Zielińskiego."},
    }
    body = chat(
        user("Notatka od Zielińskiego jest gotowa."),
        tools=[note, {"type": "function", "function": {"name": "call", "description": "Dzwoni."}}],
    )
    for where in ("tool", "message"):
        found = copy.deepcopy(body)
        if where == "tool":
            found["tools"][1]["function"]["description"] = "Dzwoni do pana Zielińskiego."
        else:
            found["messages"].append(user("Zadzwoń do pana Zielińskiego."))
        one = turn()
        masked = mask_chat_request(one, found)
        assert "Zielińskiego" not in json.dumps(masked, ensure_ascii=False), where
        assert leaked_tokens(masked, one.mapping) == [], where


def test_a_legacy_function_and_a_response_format_are_masked_too():
    schema = {
        "type": "object",
        "properties": {"who": {"type": "string", "description": "Na przykład Jan Kowalski."}},
        "examples": [{"who": "Jan Kowalski", "age": 40}],
    }
    body = chat(
        user("Kto?"),
        functions=[{"name": "find", "description": "Szuka Anny Nowak.", "parameters": schema}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "person", "description": "Dane Anny Nowak.", "schema": schema},
        },
    )
    one = turn()
    masked = mask_chat_request(one, body)
    sent = json.dumps(masked, ensure_ascii=False)
    assert "Kowalski" not in sent
    assert "Nowak" not in sent
    assert masked["functions"][0]["name"] == "find"
    token = {value: token for token, value in one.mapping.items()}["Jan Kowalski"]
    assert masked["functions"][0]["parameters"]["examples"] == [{"who": token, "age": 40}]
    assert masked["response_format"]["json_schema"]["name"] == "person"
    assert set(masked["response_format"]["json_schema"]["schema"]["properties"]) == {"who"}
    assert leaked_tokens(masked, one.mapping) == []


def test_a_tool_that_is_not_a_dictionary_passes_for_the_provider_to_refuse():
    masked = mask_chat_request(turn(), chat(user("Hej."), tools=["find"], functions=[None]))
    assert masked["tools"] == ["find"]
    assert masked["functions"] == [None]


def test_the_input_of_an_embeddings_request_is_masked_with_one_turn():
    one = turn()
    masked = mask_embeddings_request(one, {"model": "e", "input": ["Jan Kowalski", "Kowalski"]})
    assert masked["input"] == ["<OSOBA_1>", "<OSOBA_2>"]
    assert (
        mask_embeddings_request(turn(), {"input": f"PESEL {PESEL}"})["input"] == "PESEL <PESEL_1>"
    )


@pytest.mark.parametrize("value", [[101, 2045], [[101, 2045]], None, 5])
def test_an_embeddings_input_that_is_not_text_is_refused(value):
    """Token ids cannot be read, so they cannot be masked."""
    with pytest.raises(GatewayRequestError):
        mask_embeddings_request(turn(), {"input": value})


def test_an_answer_sent_back_with_its_reasoning_is_masked_and_not_refused():
    """OpenRouter's reasoning came back restored, and the next call that carried it was refused
    by the last check, since nothing masked it."""
    answer = {
        "role": "assistant",
        "content": "Gotowe.",
        "reasoning": "Pisze do: Jan Kowalski.",
        "reasoning_details": [
            {"type": "reasoning.text", "text": "Jan Kowalski, urlop.", "index": 0}
        ],
    }
    one = turn()
    masked = mask_chat_request(
        one, chat(user("Jan Kowalski prosi o urlop."), answer, user("Dzięki."))
    )
    assert leaked_tokens(masked, one.mapping) == []
    assert masked["messages"][1]["reasoning"] == "Pisze do: <OSOBA_1>."


# --- responses ---------------------------------------------------------------------------------


def test_the_answer_and_a_provider_s_reasoning_are_restored():
    one = turn()
    mask_chat_request(one, chat(user("Jan Kowalski prosi o urlop.")))
    response = {
        "id": "r1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<OSOBA_1> dostanie urlop. <OSOBA_9> nie.",
                    "reasoning_content": "Pyta <OSOBA_1>.",
                },
                "finish_reason": "stop",
            }
        ],
    }
    restored = restore_chat_response(one, response)
    message = restored["choices"][0]["message"]
    assert message["content"] == "Jan Kowalski dostanie urlop. <OSOBA_9> nie."
    assert message["reasoning_content"] == "Pyta Jan Kowalski."


def test_the_arguments_of_a_tool_call_stay_valid_json_whatever_the_value_holds():
    one = turn()
    mask_chat_request(one, chat(user('Firma "Kot & Syn" Sp. z o.o., dział C:\\kadry.')))
    token = next(iter(one.mapping))
    arguments = json.dumps({"firma": token, "uwagi": f"dla {token}"})
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": arguments},
                        }
                    ],
                },
            }
        ]
    }
    restored = restore_chat_response(one, response)
    value = one.mapping[token]
    parsed = json.loads(restored["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert parsed == {"firma": value, "uwagi": f"dla {value}"}


# --- the last check ----------------------------------------------------------------------------


def test_a_masked_payload_has_no_leak():
    one = turn()
    masked = mask_chat_request(one, chat(user(f"Jan Kowalski, PESEL {PESEL}.")))
    assert leaked_tokens(masked, one.mapping) == []


def test_a_value_in_a_field_nobody_masked_is_found_wherever_it_stands():
    one = turn()
    masked = mask_chat_request(one, chat(user(f"Jan Kowalski, PESEL {PESEL}.")))
    masked["tools"] = [{"function": {"name": "f", "description": f"Dla PESEL {PESEL}"}}]
    masked["extra"] = {"Jan Kowalski": 1}
    assert leaked_tokens(masked, one.mapping) == ["<OSOBA_1>", "<PESEL_1>"]


def test_the_last_check_follows_the_masking_rules_and_raises_no_false_alarm():
    """ "kot" next to the surname Kot, "Lista" next to Lis and a date next to a PIN are not
    masked, so they are no leak either."""
    one = turn()
    masked = mask_chat_request(
        one,
        chat(user("Anna Kot i Jan Lis. Mój kot. Lista płac. PIN 2025, termin 15.04.2025.")),
    )
    assert leaked_tokens(masked, one.mapping) == []


# --- streams -----------------------------------------------------------------------------------


def _chunks(pieces, *, field="content", finish="stop"):
    chunks = [
        {
            "id": "s1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {field: piece}}],
        }
        for piece in pieces
    ]
    chunks.append({"id": "s1", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    return chunks


def _cut(text, size=3):
    return [text[i : i + size] for i in range(0, len(text), size)]


def _call_chunk(call):
    return {"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}


def test_a_streamed_answer_cut_anywhere_is_restored_exactly():
    one = turn()
    mask_chat_request(one, chat(user(f"Jan Kowalski, PESEL {PESEL}.")))
    stream = ChatStream(one.mapping)
    out = [stream.chunk(chunk) for chunk in _chunks(_cut("Tak, <OSOBA_1> ma PESEL <PESEL_1>."))]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in out)
    assert text == f"Tak, Jan Kowalski ma PESEL {PESEL}."
    assert stream.rest() is None


def test_the_arguments_of_a_streamed_tool_call_are_restored_into_valid_json():
    one = turn()
    mask_chat_request(one, chat(user('Firma "Kot & Syn" Sp. z o.o.')))
    token = next(iter(one.mapping))
    arguments = json.dumps({"firma": token})
    stream = ChatStream(one.mapping)
    pieces = [
        _call_chunk({"index": 0, "id": "c1", "function": {"name": "f", "arguments": ""}}),
        *[_call_chunk({"index": 0, "function": {"arguments": p}}) for p in _cut(arguments, 4)],
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    joined = ""
    for chunk in pieces:
        for call in stream.chunk(chunk)["choices"][0]["delta"].get("tool_calls", []):
            joined += call["function"].get("arguments", "")
    assert json.loads(joined) == {"firma": one.mapping[token]}


def test_a_chat_answer_and_its_arguments_come_back_in_the_case_asked_for():
    one = turn()
    mask_chat_request(one, chat(user("Wniosek pracownika Jana Kowalskiego.")))
    arguments = json.dumps({"employee": "<OSOBA_1:mianownik>"})
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": arguments}}
    message = {"role": "assistant", "content": "Szanowny Panie <OSOBA_1:wołacz>,"}
    message["tool_calls"] = [call]
    restored = restore_chat_response(one, {"choices": [{"index": 0, "message": message}]})
    sent = restored["choices"][0]["message"]
    assert sent["content"] == "Szanowny Panie Janie Kowalski,"
    arguments_sent = sent["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments_sent) == {"employee": "Jan Kowalski"}
    stream = ChatStream(one.mapping)
    joined = ""
    for piece in _cut(arguments, 4):
        chunk = stream.chunk(_call_chunk({"index": 0, "function": {"arguments": piece}}))
        for streamed in chunk["choices"][0]["delta"].get("tool_calls", []):
            joined += streamed["function"].get("arguments", "")
    end = stream.chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    for streamed in end["choices"][0]["delta"].get("tool_calls", []):
        joined += streamed["function"].get("arguments", "")
    assert json.loads(joined) == {"employee": "Jan Kowalski"}


def test_a_stream_that_ends_without_finishing_gives_back_what_it_held():
    one = turn()
    mask_chat_request(one, chat(user("Jan Kowalski.")))
    stream = ChatStream(one.mapping)
    out = stream.chunk({"id": "s1", "choices": [{"index": 0, "delta": {"content": "To <OSOBA"}}]})
    assert out["choices"][0]["delta"]["content"] == "To "
    rest = stream.rest()
    assert rest["id"] == "s1"
    assert rest["choices"][0]["delta"]["content"] == "<OSOBA"
    assert stream.rest() is None


def test_a_chunk_without_choices_such_as_usage_passes_as_it_is():
    stream = ChatStream({})
    usage = {"id": "s1", "choices": [], "usage": {"total_tokens": 3}}
    assert stream.chunk(usage) == usage


def test_an_embeddings_request_masks_the_end_user_s_identifier_too():
    masked = mask_embeddings_request(turn(), {"input": "Hej.", "user": "jan.kowalski@example.com"})
    assert masked["user"] == "<EMAIL_1>"


def test_a_payload_with_nothing_masked_has_nothing_to_leak():
    assert leaked_tokens({"messages": [{"content": "Jan Kowalski"}]}, {}) == []
    assert leaked_tokens({"x": "<OSOBA_1>"}, {"<OSOBA_1>": "<OSOBA_1>"}) == []


def test_a_system_message_of_another_shape_gets_the_note_before_it():
    first = {"role": "system", "content": None}
    masked = mask_chat_request(turn(), chat(first, user("Hej.")), instructions="NOTE")
    assert masked["messages"][0] == {"role": "system", "content": "NOTE"}


def test_odd_chunks_pass_without_breaking_the_stream():
    stream = ChatStream({"<OSOBA_1>": "Jan"})
    assert stream.chunk("keep-alive") == "keep-alive"
    odd = {"choices": ["x", {"index": 0, "delta": None}, {"index": 0, "finish_reason": "stop"}]}
    assert stream.chunk(odd)["choices"][2]["delta"] == {}
    call = {"index": 0, "delta": {"tool_calls": ["x", {"index": 0, "type": "function"}]}}
    assert stream.chunk({"choices": [call]})["choices"][0]["delta"]["tool_calls"][0] == "x"


def test_a_tool_call_held_back_when_the_stream_ends_comes_out_in_the_rest():
    stream = ChatStream({"<OSOBA_1>": 'Jan "J" Kowalski'})
    chunk = _call_chunk({"index": 0, "function": {"arguments": '{"who": "<OSOBA_1'}})
    assert stream.chunk(chunk)["choices"][0]["delta"]["tool_calls"][0]["function"] == {
        "arguments": '{"who": "'
    }
    rest = stream.rest()
    held = rest["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
    assert held == "<OSOBA_1"
    stream = ChatStream({"<OSOBA_1>": 'Jan "J" Kowalski'})
    stream.chunk(_call_chunk({"index": 0, "function": {"arguments": '{"who": "<OSOBA_1'}}))
    finished = stream.chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '>"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    calls = finished["choices"][0]["delta"]["tool_calls"]
    assert json.loads('{"who": "' + "".join(c["function"]["arguments"] for c in calls)) == {
        "who": 'Jan "J" Kowalski'
    }


def test_a_value_hidden_in_a_key_of_the_payload_is_found():
    assert leaked_tokens(
        {"meta": {1: "x", "Jan Kowalski": "y"}}, {"<OSOBA_1>": "Jan Kowalski"}
    ) == ["<OSOBA_1>"]


def test_the_last_check_reads_names_from_code_as_the_masking_does():
    """An agent's history refused every call: words the masking took for names in a tool's JSON
    result, "service" or "text", stood in the keys, ids, types and function names it never masks.

    Those are names from code. Every text and every value, also inside a JSON string, is still
    searched, and a key that reads as text is too.
    """
    mapping = {"<FIRMA_1>": "service", "<SEKRET_1>": "call", "<MIEJSCE_1>": "text"}
    arguments = json.dumps({"service": "<FIRMA_1>", "entity": "light.salon"})
    payload = {
        "model": "vendor/model-v1.0",
        "messages": [
            {
                "role": "assistant",
                "content": "Włączam.",
                "tool_calls": [
                    {
                        "id": "call_00_Ab3",
                        "type": "function",
                        "function": {"name": "ha_call_service", "arguments": arguments},
                    }
                ],
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "<MIEJSCE_1>", "index": 0}
                ],
            },
            {"role": "tool", "tool_call_id": "call_00_Ab3", "content": '{"text": "ok"}'},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ha_call_service",
                    "parameters": {
                        "type": "object",
                        "properties": {"service": {"type": "string"}, "text": {"type": "string"}},
                        "required": ["service", "text"],
                    },
                },
            }
        ],
    }
    assert leaked_tokens(payload, mapping) == []
    for leak in (
        '{"service": "service"}',
        "A sentence with a service in it.",
        "{not JSON, a service}",
        '{"service service": 1}',
    ):
        found = copy.deepcopy(payload)
        found["messages"][1]["content"] = leak
        assert leaked_tokens(found, mapping) == ["<FIRMA_1>"], leak


# --- the Responses API -------------------------------------------------------------------------


def hermes_request():
    """A /responses request shaped the way Hermes Agent sends one, with store off."""
    first = {"type": "input_text", "text": "Znajdź umowę: Jan Kowalski."}
    summary = {"type": "summary_text", "text": "Szukam: Jan Kowalski."}
    answer = {
        "type": "output_text",
        "text": "Jan Kowalski ma umowę.",
        "annotations": [{"type": "url_citation", "title": "Jan Kowalski"}],
    }
    return {
        "model": "gpt-test",
        "store": False,
        "instructions": "Pomagasz działowi kadr. Kierownik: Anna Nowak.",
        "include": ["reasoning.encrypted_content"],
        "input": [
            {"role": "user", "content": [first]},
            {"type": "reasoning", "encrypted_content": "gAAAA-sealed", "summary": [summary]},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "find",
                "arguments": '{"who": "Jan Kowalski"}',
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": json.dumps({"pesel": PESEL}),
            },
            {"type": "message", "role": "assistant", "content": [answer]},
            {"type": "compaction", "encrypted_content": "gAAAA-compacted"},
            {"role": "user", "content": "Dziękuję."},
        ],
        "tools": [{"type": "function", "name": "find", "parameters": {"type": "object"}}],
    }


def test_every_item_of_a_responses_request_is_masked_and_sealed_ones_pass():
    body = hermes_request()
    original = copy.deepcopy(body)
    one = turn()
    masked = mask_responses_request(one, body, instructions="NOTE")
    assert body == original
    sent = json.dumps(masked, ensure_ascii=False)
    assert "Kowalski" not in sent
    assert "Nowak" not in sent
    assert PESEL not in sent
    items = masked["input"]
    assert items[0]["content"][0]["text"] == "Znajdź umowę: <OSOBA_2>."
    assert items[1]["encrypted_content"] == "gAAAA-sealed"
    assert items[1]["summary"][0]["text"] == "Szukam: <OSOBA_2>."
    assert json.loads(items[2]["arguments"]) == {"who": "<OSOBA_2>"}
    assert json.loads(items[3]["output"]) == {"pesel": "<PESEL_1>"}
    assert items[4]["content"] == [{"type": "output_text", "text": "<OSOBA_2> ma umowę."}]
    assert items[5] == {"type": "compaction", "encrypted_content": "gAAAA-compacted"}
    assert masked["instructions"] == "Pomagasz działowi kadr. Kierownik: <OSOBA_1>.\n\nNOTE"
    assert masked["tools"] == body["tools"]
    assert leaked_tokens(masked, one.mapping) == []


def test_a_value_first_found_in_a_later_item_is_masked_in_the_earlier_ones():
    body = {
        "input": [
            {"role": "user", "content": "Kowalski dzwonił."},
            {"role": "user", "content": "To Jan Kowalski."},
        ]
    }
    masked = mask_responses_request(turn(), body)
    assert "Kowalski" not in json.dumps(masked, ensure_ascii=False)


def test_a_text_input_the_instructions_and_the_end_user_are_masked():
    body = {
        "input": f"PESEL {PESEL}",
        "user": "jan.kowalski@example.com",
        "metadata": {"klient": "Jan Kowalski"},
        "prompt": {"id": "p1", "variables": {"klient": "Jan Kowalski"}},
    }
    masked = mask_responses_request(turn(), body, instructions="NOTE")
    assert masked["input"] == "PESEL <PESEL_1>"
    assert masked["instructions"] == "NOTE"
    assert masked["user"] == "<EMAIL_1>"
    assert masked["metadata"] == {"klient": "<OSOBA_1>"}
    assert masked["prompt"] == {"id": "p1", "variables": {"klient": "<OSOBA_1>"}}


@pytest.mark.parametrize(
    "fields",
    [
        {"previous_response_id": "resp_1"},
        {"conversation": "conv_1"},
        {"background": True},
        {"modalities": ["audio"]},
        {"input": 5},
    ],
)
def test_a_responses_request_the_gateway_could_not_restore_is_refused(fields):
    with pytest.raises(GatewayRequestError):
        mask_responses_request(turn(), {"model": "m", "input": "Hej.", **fields})


@pytest.mark.parametrize(
    "item",
    [
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}],
        },
        {"type": "web_search_call", "id": "ws1"},
    ],
)
def test_an_image_or_an_item_the_gateway_does_not_know_is_refused_or_left_out(item):
    body = {"input": [item, {"role": "user", "content": "Jan Kowalski?"}]}
    with pytest.raises(MaskingUnavailableError):
        mask_responses_request(turn(), body)
    masked = mask_responses_request(turn(), body, drop_unsupported=True)
    assert masked["input"][-1]["content"] == "<OSOBA_1>?"
    assert "image_url" not in json.dumps(masked)
    assert "web_search_call" not in json.dumps(masked)


def test_a_responses_request_passes_what_it_cannot_mask_when_told_to():
    image = {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
    item = {"type": "web_search_call", "id": "ws1"}
    text = {"type": "input_text", "text": "Jan Kowalski?"}
    body = {"input": [item, {"role": "user", "content": [image, text]}]}
    masked = mask_responses_request(turn(), body, pass_unsupported=True)
    assert masked["input"][0] == item
    assert masked["input"][1]["content"] == [image, {"type": "input_text", "text": "<OSOBA_1>?"}]
    with pytest.raises(ValueError, match="drop or pass"):
        mask_responses_request(turn(), body, drop_unsupported=True, pass_unsupported=True)


def test_tool_outputs_and_custom_tools_are_masked():
    body = {
        "input": [
            {"type": "custom_tool_call", "call_id": "c2", "name": "sql", "input": "Jan Kowalski"},
            {"type": "custom_tool_call_output", "call_id": "c2", "output": "Kowalski"},
            {
                "type": "function_call_output",
                "call_id": "c3",
                "output": [{"type": "input_text", "text": f"PESEL {PESEL}"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "Nie, Jan Kowalski."}],
            },
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Jan Kowalski"}]},
        ]
    }
    sent = json.dumps(mask_responses_request(turn(), body), ensure_ascii=False)
    assert "Kowalski" not in sent
    assert PESEL not in sent


def test_the_tools_and_the_answer_format_of_a_responses_request_are_masked():
    function = leave_tool()["function"]
    body = {
        "instructions": "Notatki od Zielińskiego czytaj uważnie.",
        "input": [{"role": "user", "content": "Potwierdź urlop."}],
        "tools": [
            {"type": "function", **function},
            {"type": "custom", "name": "sql", "description": "Zapytania Anny Nowak."},
            {"type": "web_search"},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "leave",
                "description": "Dane Jana Kowalskiego.",
                "schema": function["parameters"],
            }
        },
    }
    one = turn()
    masked = mask_responses_request(one, body)
    sent = json.dumps(masked, ensure_ascii=False)
    for value in ("Kowalski", "Nowak", "Zielińskiego", PESEL):
        assert value not in sent
    assert leaked_tokens(masked, one.mapping) == []
    assert [tool.get("name") for tool in masked["tools"]] == ["send_leave_notice", "sql", None]
    assert masked["tools"][2] == {"type": "web_search"}
    assert masked["text"]["format"]["name"] == "leave"
    assert masked["text"]["format"]["schema"]["required"] == ["employee", "description"]


def test_an_input_item_or_part_that_is_not_a_dictionary_is_refused():
    with pytest.raises(TypeError):
        mask_responses_request(turn(), {"input": ["Jan Kowalski"]})
    with pytest.raises(TypeError):
        mask_responses_request(turn(), {"input": [{"role": "user", "content": ["Jan"]}]})


def test_a_responses_answer_is_restored_with_valid_arguments():
    one = turn()
    mask_responses_request(one, {"input": 'Firma "Kot & Syn" Sp. z o.o. i Jan Kowalski.'})
    company, person = list(one.mapping)
    message = {"type": "output_text", "text": f"{person} pracuje w {company}."}
    response = {
        "id": "resp_1",
        "object": "response",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": f"Pyta {person}."}]},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "f",
                "arguments": json.dumps({"firma": company}),
            },
            {"type": "message", "role": "assistant", "content": [message]},
        ],
    }
    restored = restore_responses_response(one, response)
    output = restored["output"]
    assert output[0]["summary"][0]["text"] == "Pyta Jan Kowalski."
    assert json.loads(output[1]["arguments"]) == {"firma": one.mapping[company]}
    assert output[2]["content"][0]["text"] == f"Jan Kowalski pracuje w {one.mapping[company]}."


def _delta(kind, delta, number, **where):
    return {"type": f"response.{kind}.delta", "sequence_number": number, "delta": delta, **where}


def test_a_streamed_responses_answer_is_restored_and_numbered_anew():
    one = turn()
    mask_responses_request(one, {"input": f"Jan Kowalski, PESEL {PESEL}."})
    stream = ResponsesStream(one.mapping)
    where = {"item_id": "m1", "output_index": 0, "content_index": 0}
    text = "Tak, <OSOBA_1> ma PESEL <PESEL_1>"
    created = {"instructions": "<OSOBA_1>"}
    events = [{"type": "response.created", "sequence_number": 0, "response": created}]
    events += [_delta("output_text", piece, n + 1, **where) for n, piece in enumerate(_cut(text))]
    events.append(
        {"type": "response.output_text.done", "sequence_number": 99, "text": text, **where}
    )
    out = [sent for event in events for sent in stream.events(event)]
    deltas = "".join(e["delta"] for e in out if e["type"] == "response.output_text.delta")
    assert deltas == f"Tak, Jan Kowalski ma PESEL {PESEL}"
    assert out[-1]["text"] == f"Tak, Jan Kowalski ma PESEL {PESEL}"
    assert out[0]["response"]["instructions"] == "Jan Kowalski"
    assert [e["sequence_number"] for e in out] == list(range(len(out)))
    assert stream.rest() == []


def test_the_arguments_of_a_streamed_function_call_come_back_as_valid_json():
    one = turn()
    mask_responses_request(one, {"input": 'Firma "Kot & Syn" Sp. z o.o.'})
    token = next(iter(one.mapping))
    arguments = json.dumps({"firma": token})
    stream = ResponsesStream(one.mapping)
    where = {"item_id": "fc1", "output_index": 1}
    out = []
    for piece in _cut(arguments, 4):
        out += stream.events(_delta("function_call_arguments", piece, 0, **where))
    done = {"type": "response.function_call_arguments.done", "arguments": arguments, **where}
    out += stream.events(done)
    joined = "".join(e["delta"] for e in out if e["type"].endswith(".delta"))
    assert json.loads(joined) == {"firma": one.mapping[token]}
    assert json.loads(out[-1]["arguments"]) == {"firma": one.mapping[token]}


def test_a_streamed_responses_answer_comes_back_in_the_case_asked_for():
    one = turn()
    mask_responses_request(one, {"input": "Wniosek pracownika Jana Kowalskiego."})
    stream = ResponsesStream(one.mapping)
    where = {"item_id": "m1", "output_index": 0, "content_index": 0}
    text = "Szanowny Panie <OSOBA_1:wołacz>, rozmawiałem z <OSOBA_1:narzędnik>."
    events = [_delta("output_text", piece, n, **where) for n, piece in enumerate(_cut(text))]
    events.append({"type": "response.output_text.done", "text": text, **where})
    out = [sent for event in events for sent in stream.events(event)]
    expected = "Szanowny Panie Janie Kowalski, rozmawiałem z Janem Kowalskim."
    assert "".join(e["delta"] for e in out if e["type"].endswith(".delta")) == expected
    assert out[-1]["text"] == expected
    restored = restore_responses_response(
        one, {"output": [{"type": "function_call", "arguments": '{"who": "<OSOBA_1:mianownik>"}'}]}
    )
    assert json.loads(restored["output"][0]["arguments"]) == {"who": "Jan Kowalski"}


def test_a_responses_stream_cut_short_gives_back_what_it_held():
    stream = ResponsesStream({"<OSOBA_1>": "Jan"})
    where = {"item_id": "m1", "output_index": 0, "content_index": 0}
    assert stream.events(_delta("output_text", "To <OSOBA", 0, **where))[0]["delta"] == "To "
    assert stream.events("ping") == ["ping"]
    done = {"type": "response.refusal.done", "refusal": "", "item_id": "x", "output_index": 3}
    assert stream.events(done) == [done]
    (rest,) = stream.rest()
    assert rest["delta"] == "<OSOBA"
    assert rest["item_id"] == "m1"
    assert stream.rest() == []


def test_a_system_prompt_of_blocks_is_masked_and_gets_the_note_as_one_more_block():
    from polanon.gateway.payloads import mask_messages_request

    turn = MaskedTurn(ANONYMIZER)
    body = {
        "model": "claude-test",
        "system": [
            {
                "type": "text",
                "text": "Klient: Jan Kowalski.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": "Kto to jest Jan Kowalski?"}],
    }
    masked = mask_messages_request(turn, body, instructions="Tokens stand for values.")
    assert masked["system"][0]["text"] == "Klient: <OSOBA_1>."
    assert masked["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert masked["system"][1] == {"type": "text", "text": "Tokens stand for values."}
    assert masked["messages"][0]["content"] == "Kto to jest <OSOBA_1>?"
    assert body["system"][0]["text"] == "Klient: Jan Kowalski."


def test_a_messages_stream_cut_short_gives_back_what_it_held():
    from polanon.gateway.payloads import MessagesStream

    stream = MessagesStream({"<OSOBA_1>": "Jan Kowalski"})
    delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Dla <OSO"},
    }
    first = stream.events(delta)
    assert first[0]["delta"]["text"] == "Dla "
    rest = stream.rest()
    assert rest[0]["delta"] == {"type": "text_delta", "text": "<OSO"}
    assert stream.rest() == []
