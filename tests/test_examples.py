"""The integration recipes in examples/ work with the real provider SDKs, offline.

Each SDK talks to an httpx2 MockTransport that plays the provider: it records every request and
answers with the tokens it was sent. That checks both halves of what a recipe promises: the
provider sees no raw value, and the reader sees every value put back, also when a token is streamed
in pieces. The SDKs are not dependencies of the library, so these tests run where they are
installed.
"""

import importlib.util
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from polanon import TOKEN_PATTERN

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")
httpx2 = pytest.importorskip("httpx2")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
QUESTION = (
    "Napisz krótką odpowiedź do Jana Kowalskiego (PESEL 50810112354, "
    "jan.kowalski@example.com), że korekta faktury jest gotowa."
)
RAW_VALUES = ("Jana Kowalskiego", "50810112354", "jan.kowalski@example.com")
RESTORED = "Gotowe dla " + ", ".join(RAW_VALUES)
CUT_SHORT = ". Pozdrawiam, <OSO"
TOKEN = re.compile(TOKEN_PATTERN)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pieces(text, size=3):
    """Split a reply the way a provider streams it, cutting through the tokens."""
    return [text[i : i + size] for i in range(0, len(text), size)]


class FakeProvider:
    """Records each request and replies "Gotowe dla" followed by the tokens of the last message.

    The reply ends in a token cut short, as an answer stopped by its length limit does, so a
    streaming recipe that never flushes its restorer loses the end of it.
    """

    def __init__(self, reply: Callable[[str], dict], stream: Callable[[str], str]) -> None:
        self.bodies: list[str] = []
        self._reply = reply
        self._stream = stream

    def handle(self, request):
        body = request.content.decode()
        self.bodies.append(body)
        payload = json.loads(body)
        last = json.dumps(payload["messages"][-1])
        tokens = dict.fromkeys(match.group(0) for match in TOKEN.finditer(last))
        answer = "Gotowe dla " + ", ".join(tokens) + CUT_SHORT
        if payload.get("stream"):
            headers = {"content-type": "text/event-stream"}
            return httpx2.Response(200, headers=headers, content=self._stream(answer).encode())
        return httpx2.Response(200, json=self._reply(answer))

    def assert_saw_no_raw_value(self):
        for body in self.bodies:
            assert "kowalsk" not in body.lower()
            assert "50810112354" not in body


def _claude_message(text):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": "end_turn" if text else None,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _claude_events(text):
    deltas = [
        ("content_block_delta", {"type": "text_delta", "text": piece}) for piece in _pieces(text)
    ]
    events = [
        ("message_start", {"message": _claude_message("")}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        *[(name, {"index": 0, "delta": delta}) for name, delta in deltas],
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        ("message_stop", {}),
    ]
    return "".join(
        f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n" for name, data in events
    )


def _chat_completion(text):
    message = {"role": "assistant", "content": text}
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


def _chat_chunks(text):
    chunks = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        for piece in _pieces(text)
    ]
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


@pytest.fixture
def claude():
    provider = FakeProvider(_claude_message, _claude_events)
    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(provider.handle))
    client = anthropic.Anthropic(api_key="test", http_client=http, max_retries=0)
    return provider, client


@pytest.fixture
def chat_endpoint():
    provider = FakeProvider(_chat_completion, _chat_chunks)
    http = openai.DefaultHttpxClient(transport=httpx2.MockTransport(provider.handle))
    client = openai.OpenAI(
        api_key="test", base_url="https://llm.test/v1", http_client=http, max_retries=0
    )
    return provider, client


def test_claude_recipe_sends_no_raw_value_and_restores_the_answer(claude):
    provider, client = claude
    answer = _load("anthropic_chat").ask(client, QUESTION)
    provider.assert_saw_no_raw_value()
    for value in RAW_VALUES:
        assert value in answer


def test_claude_streaming_recipe_restores_tokens_split_across_events(claude):
    provider, client = claude
    answer = "".join(_load("anthropic_chat").ask_streaming(client, QUESTION))
    provider.assert_saw_no_raw_value()
    assert answer == RESTORED + CUT_SHORT


def test_claude_conversation_keeps_tokens_and_masked_history_across_turns(claude):
    provider, client = claude
    chat = _load("anthropic_chat").PrivateChat(client)
    chat.send(QUESTION)
    second = chat.send("Dopisz, że PESEL 50810112354 zweryfikowano.")
    provider.assert_saw_no_raw_value()
    assert second == "Gotowe dla 50810112354" + CUT_SHORT
    messages = json.loads(provider.bodies[-1])["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert "<PESEL_1>" in messages[1]["content"]
    assert "<PESEL_1>" in messages[2]["content"]


def test_openai_recipe_sends_no_raw_value_and_restores_the_answer(chat_endpoint):
    provider, client = chat_endpoint
    answer = _load("openai_chat").ask(client, "test-model", QUESTION)
    provider.assert_saw_no_raw_value()
    for value in RAW_VALUES:
        assert value in answer


def test_openai_streaming_recipe_restores_tokens_split_across_chunks(chat_endpoint):
    provider, client = chat_endpoint
    answer = "".join(_load("openai_chat").ask_streaming(client, "test-model", QUESTION))
    provider.assert_saw_no_raw_value()
    assert answer == RESTORED + CUT_SHORT
