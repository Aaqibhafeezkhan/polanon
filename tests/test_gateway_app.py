"""The gateway over HTTP, with a provider that answers in the test and records what it got."""

import json
import logging

import pytest

httpx = pytest.importorskip("httpx")

from polanon import Anonymizer, default_guards  # noqa: E402
from polanon.gateway.app import Gateway, GatewayConfig  # noqa: E402

ANONYMIZER = Anonymizer(default_guards(with_ner=False))
PESEL = "50810112354"
RAW = ("Jan Kowalski", PESEL)


class Provider:
    """An OpenAI-compatible API that records every request and echoes the last message back."""

    def __init__(self, *, status=200, tool_call=None, fail=False):
        self.requests = []
        self.status = status
        self.tool_call = tool_call
        self.fail = fail

    def __call__(self, request):
        if self.fail:
            raise httpx.ConnectError("refused", request=request)
        body = json.loads(request.content) if request.content else None
        self.requests.append((request, body))
        path = request.url.path
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "slow down"}})
        if "/models" in path:
            return httpx.Response(200, json={"data": [{"id": "gpt-test"}]})
        if path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
        last = body["messages"][-1]["content"]
        if body.get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(f"Echo: {last}")
            )
        message = {"role": "assistant", "content": f"Echo: {last}"}
        if self.tool_call is not None:
            message["tool_calls"] = [self.tool_call(body)]
        return httpx.Response(200, json={"id": "r1", "choices": [{"index": 0, "message": message}]})

    def sent(self):
        """Everything the provider received, as one text."""
        return " ".join(request.content.decode() for request, _ in self.requests)


def _sse(text, size=3):
    lines = []
    for start in range(0, len(text), size):
        chunk = {
            "id": "s1",
            "choices": [{"index": 0, "delta": {"content": text[start : start + size]}}],
        }
        lines.append(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")
    lines.append(
        'data: {"id": "s1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n'
    )
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def client(provider, **config):
    gateway = Gateway(
        ANONYMIZER,
        GatewayConfig(upstream="https://provider.test/v1", **config),
        transport=httpx.MockTransport(provider),
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gw")


def chat(content, **fields):
    return {"model": "gpt-test", "messages": [{"role": "user", "content": content}], **fields}


async def test_a_chat_reaches_the_provider_masked_and_comes_back_restored():
    provider = Provider()
    async with client(provider) as gw:
        response = await gw.post(
            "/v1/chat/completions",
            json=chat(f"Jan Kowalski, PESEL {PESEL}."),
            headers={"authorization": "Bearer sk-client"},
        )
    assert response.status_code == 200
    assert (
        response.json()["choices"][0]["message"]["content"] == f"Echo: Jan Kowalski, PESEL {PESEL}."
    )
    request, body = provider.requests[0]
    assert not any(value in provider.sent() for value in RAW)
    assert body["messages"][-1]["content"] == "<OSOBA_1>, PESEL <PESEL_1>."
    assert body["messages"][0]["role"] == "system"  # the note about tokens
    assert request.headers["authorization"] == "Bearer sk-client"
    assert str(request.url) == "https://provider.test/v1/chat/completions"


async def test_a_streamed_answer_comes_back_restored():
    provider = Provider()
    async with client(provider) as gw:
        response = await gw.post(
            "/v1/chat/completions", json=chat("Jan Kowalski dzwonił.", stream=True)
        )
    assert response.headers["content-type"].startswith("text/event-stream")
    text = ""
    for line in response.text.splitlines():
        if line.startswith("data: {"):
            text += json.loads(line[6:])["choices"][0]["delta"].get("content", "")
    assert text == "Echo: Jan Kowalski dzwonił."
    assert response.text.rstrip().endswith("data: [DONE]")
    assert "Kowalski" not in provider.sent()


async def test_a_tool_call_comes_back_with_the_original_values_in_valid_json():
    def tool_call(body):
        token = body["messages"][-1]["content"].split()[-1].rstrip(".")
        arguments = json.dumps({"who": token})
        return {
            "id": "c1",
            "type": "function",
            "function": {"name": "find", "arguments": arguments},
        }

    provider = Provider(tool_call=tool_call)
    async with client(provider) as gw:
        response = await gw.post("/v1/chat/completions", json=chat("Znajdź umowę: Jan Kowalski."))
    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"who": "Jan Kowalski"}


async def test_an_image_is_refused_and_nothing_is_sent():
    provider = Provider()
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    async with client(provider) as gw:
        response = await gw.post(
            "/v1/chat/completions", json=chat([{"type": "text", "text": "Co to?"}, image])
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_content"
    assert "--drop-unsupported" in response.json()["error"]["message"]
    assert provider.requests == []


async def test_an_image_is_left_out_when_the_gateway_is_told_to_drop_it():
    provider = Provider()
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    async with client(provider, unsupported="drop") as gw:
        response = await gw.post(
            "/v1/chat/completions", json=chat([{"type": "text", "text": "Jan Kowalski?"}, image])
        )
    assert response.status_code == 200
    _, body = provider.requests[0]
    assert body["messages"][-1]["content"] == [{"type": "text", "text": "<OSOBA_1>?"}]


async def test_embeddings_are_asked_for_masked_text():
    provider = Provider()
    async with client(provider) as gw:
        response = await gw.post("/v1/embeddings", json={"model": "e", "input": ["Jan Kowalski"]})
    assert response.json() == {"data": [{"embedding": [0.1, 0.2]}]}
    assert provider.requests[0][1]["input"] == ["<OSOBA_1>"]


async def test_the_list_of_models_passes_through():
    provider = Provider()
    async with client(provider) as gw:
        listed = await gw.get("/v1/models")
        one = await gw.get("/models/gpt-test")
    assert listed.json() == {"data": [{"id": "gpt-test"}]}
    assert one.status_code == 200
    assert str(provider.requests[1][0].url) == "https://provider.test/v1/models/gpt-test"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/files"),
        ("POST", "/v1/audio/speech"),
        ("GET", "/v1/responses/resp_1"),
        ("GET", "/v1/chat/completions"),
    ],
)
async def test_an_endpoint_the_gateway_cannot_mask_is_refused_and_nothing_is_sent(method, path):
    provider = Provider()
    async with client(provider) as gw:
        response = await gw.request(method, path, json=chat("Jan Kowalski"))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unsupported_endpoint"
    assert provider.requests == []


async def test_with_a_gateway_key_a_client_needs_it_and_its_own_key_never_reaches_the_provider():
    provider = Provider()
    async with client(provider, gateway_key="local-secret", upstream_key="sk-operator") as gw:
        refused = await gw.post("/v1/chat/completions", json=chat("Hej."))
        wrong = await gw.post(
            "/v1/chat/completions", json=chat("Hej."), headers={"authorization": "Bearer nope"}
        )
        allowed = await gw.post(
            "/v1/chat/completions",
            json=chat("Hej."),
            headers={"authorization": "Bearer local-secret"},
        )
    assert refused.status_code == wrong.status_code == 401
    assert allowed.status_code == 200
    assert len(provider.requests) == 1
    assert provider.requests[0][0].headers["authorization"] == "Bearer sk-operator"


async def test_with_a_gateway_key_and_no_operator_key_the_provider_gets_no_key():
    """A local model needs none, and the client's key there is the gateway's."""
    provider = Provider()
    async with client(provider, gateway_key="local-secret") as gw:
        await gw.post(
            "/v1/chat/completions",
            json=chat("Hej."),
            headers={"authorization": "Bearer local-secret"},
        )
    assert "authorization" not in provider.requests[0][0].headers


@pytest.mark.parametrize(
    ("content", "status", "code"),
    [
        (b"{not json", 400, "invalid_json"),
        (b"[1, 2]", 400, "invalid_json"),
        (b"x" * 2048, 413, "too_large"),
    ],
)
async def test_a_request_that_is_not_a_json_object_or_too_large_is_refused(content, status, code):
    provider = Provider()
    async with client(provider, max_body=1024) as gw:
        response = await gw.post("/v1/chat/completions", content=content)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert provider.requests == []


async def test_a_field_of_the_wrong_type_is_refused():
    async with client(Provider()) as gw:
        response = await gw.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": 5}]}
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


async def test_the_provider_s_error_passes_through():
    async with client(Provider(status=429)) as gw:
        plain = await gw.post("/v1/chat/completions", json=chat("Hej."))
        streamed = await gw.post("/v1/chat/completions", json=chat("Hej.", stream=True))
    assert plain.status_code == streamed.status_code == 429
    assert plain.json() == {"error": {"message": "slow down"}}


async def test_a_provider_out_of_reach_gives_a_bad_gateway():
    async with client(Provider(fail=True)) as gw:
        plain = await gw.post("/v1/chat/completions", json=chat("Hej."))
        streamed = await gw.post("/v1/chat/completions", json=chat("Hej.", stream=True))
        models = await gw.get("/v1/models")
    assert plain.status_code == streamed.status_code == models.status_code == 502


async def test_a_payload_that_would_still_carry_a_value_is_not_sent(monkeypatch):
    """The last check stands apart from the masking: a field the masking missed is caught."""
    from polanon.gateway import app

    real = app.mask_chat_request

    def careless(turn, body, **kwargs):
        masked = real(turn, body, **kwargs)
        return {**masked, "notes": body["messages"][-1]["content"]}

    monkeypatch.setattr(app, "mask_chat_request", careless)
    provider = Provider()
    async with client(provider) as gw:
        response = await gw.post("/v1/chat/completions", json=chat("Jan Kowalski."))
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "leak_prevented"
    assert "<OSOBA_1>" in response.json()["error"]["message"]
    assert "Kowalski" not in response.text
    assert provider.requests == []


async def test_the_log_names_counts_and_never_a_value(caplog):
    caplog.set_level(logging.INFO, logger="polanon.gateway")
    async with client(Provider()) as gw:
        await gw.post("/v1/chat/completions", json=chat(f"Jan Kowalski, PESEL {PESEL}."))
    logged = caplog.text
    assert "/chat/completions -> 200" in logged
    assert "PERSON" in logged
    assert not any(value in logged for value in RAW)


async def test_the_health_check_needs_no_key():
    async with client(Provider(), gateway_key="local-secret") as gw:
        response = await gw.get("/health")
    assert response.json() == {"status": "ok"}


async def test_the_gateway_closes_its_client_when_the_server_stops():
    gateway = Gateway(ANONYMIZER, GatewayConfig(upstream="https://provider.test/v1"))
    messages = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message["type"])

    await gateway({"type": "lifespan"}, receive, send)
    assert sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert gateway._client.is_closed


def _stream_provider(lines, *, break_off=False):
    """A provider that streams the given event lines, and may break off after them."""

    async def body():
        for line in lines:
            yield line.encode()
        if break_off:
            raise httpx.ReadError("connection reset")

    def answer(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    return answer


async def test_a_stream_that_ends_without_finishing_still_gives_back_the_held_text():
    lines = [
        'data: {"id": "s1", "choices": [{"index": 0, "delta": {"content": "Tak, <OSOBA"}}]}\n\n',
        "data: {not json\n\n",
        ": keep-alive\n\n",
    ]
    async with client(_stream_provider(lines)) as gw:
        response = await gw.post("/v1/chat/completions", json=chat("Jan Kowalski?", stream=True))
    assert "data: {not json" in response.text
    assert ": keep-alive" in response.text
    contents = [
        json.loads(line[6:])["choices"][0]["delta"]["content"]
        for line in response.text.splitlines()
        if line.startswith('data: {"')
    ]
    assert (
        "".join(contents) == "Tak, <OSOBA"
    )  # a fragment that never became a token stays as it came


async def test_a_provider_that_breaks_off_a_stream_ends_it():
    lines = ['data: {"id": "s1", "choices": [{"index": 0, "delta": {"content": "Tak"}}]}\n\n']
    async with client(_stream_provider(lines, break_off=True)) as gw:
        response = await gw.post("/v1/chat/completions", json=chat("Hej?", stream=True))
    assert response.status_code == 200
    assert '"Tak"' in response.text


async def test_an_answer_that_is_not_json_gives_a_bad_gateway():
    async with client(lambda request: httpx.Response(200, text="oops")) as gw:
        response = await gw.post("/v1/chat/completions", json=chat("Hej."))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "bad_upstream"


async def test_token_ids_to_embed_are_refused():
    provider = Provider()
    async with client(provider) as gw:
        response = await gw.post("/v1/embeddings", json={"input": [[101, 2045]]})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert provider.requests == []


async def test_a_masking_failure_sends_nothing(monkeypatch):
    from polanon import MaskingUnavailableError
    from polanon.turn import MaskedTurn

    def broken(self, text):
        raise MaskingUnavailableError("the detector failed with RuntimeError")

    monkeypatch.setattr(MaskedTurn, "mask", broken)
    provider = Provider()
    async with client(provider) as gw:
        response = await gw.post("/v1/chat/completions", json=chat("Jan Kowalski."))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "masking_failed"
    assert provider.requests == []


async def test_a_client_that_goes_away_before_its_request_is_read_is_answered_and_forgotten():
    gateway = Gateway(ANONYMIZER, GatewayConfig(upstream="https://provider.test/v1"))
    sent = []

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    await gateway(scope, receive, send)
    assert sent[0]["status"] == 400


async def test_a_connection_that_is_not_http_is_ignored():
    gateway = Gateway(ANONYMIZER, GatewayConfig(upstream="https://provider.test/v1"))

    async def never():
        raise AssertionError("nothing should be read")

    await gateway({"type": "websocket"}, never, never)


# --- the Responses API -------------------------------------------------------------------------


def _responses_provider(received):
    """A /responses provider: echoes the input's last text, streamed as named events when asked."""

    def answer(request):
        body = json.loads(request.content)
        received.append(body)
        last = body["input"] if isinstance(body["input"], str) else body["input"][-1]["content"]
        text = f"Echo: {last}"
        message = {"type": "output_text", "text": text}
        output = [{"type": "message", "id": "m1", "role": "assistant", "content": [message]}]
        response = {"id": "resp_1", "object": "response", "output": output}
        if not body.get("stream"):
            return httpx.Response(200, json=response)
        where = {"item_id": "m1", "output_index": 0, "content_index": 0}
        events = [{"type": "response.created", "response": {**response, "output": []}}]
        events += [
            {"type": "response.output_text.delta", "delta": text[start : start + 3], **where}
            for start in range(0, len(text), 3)
        ]
        events.append({"type": "response.output_text.done", "text": text, **where})
        events.append({"type": "response.completed", "response": response})
        lines = "".join(
            f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': n})}\n\n"
            for n, event in enumerate(events)
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=lines)

    return answer


def responses_body(text, **fields):
    return {
        "model": "gpt-test",
        "store": False,
        "input": [{"role": "user", "content": text}],
        **fields,
    }


async def test_a_responses_call_reaches_the_provider_masked_and_comes_back_restored():
    received = []
    async with client(_responses_provider(received)) as gw:
        response = await gw.post(
            "/v1/responses", json=responses_body(f"Jan Kowalski, PESEL {PESEL}.")
        )
    assert response.status_code == 200
    text = response.json()["output"][0]["content"][0]["text"]
    assert text == f"Echo: Jan Kowalski, PESEL {PESEL}."
    sent = json.dumps(received[0], ensure_ascii=False)
    assert "Kowalski" not in sent
    assert PESEL not in sent
    assert received[0]["instructions"]  # the note about tokens


async def test_a_streamed_responses_answer_comes_back_restored_as_named_events():
    received = []
    async with client(_responses_provider(received)) as gw:
        response = await gw.post(
            "/v1/responses", json=responses_body("Jan Kowalski dzwonił.", stream=True)
        )
    lines = response.text.splitlines()
    events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
    names = [line[7:] for line in lines if line.startswith("event: ")]
    assert names == [event["type"] for event in events]
    deltas = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
    assert deltas == "Echo: Jan Kowalski dzwonił."
    done = next(e for e in events if e["type"] == "response.output_text.done")
    assert done["text"] == "Echo: Jan Kowalski dzwonił."
    completed = events[-1]["response"]["output"][0]["content"][0]["text"]
    assert completed == "Echo: Jan Kowalski dzwonił."
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert "Kowalski" not in json.dumps(received[0], ensure_ascii=False)


async def test_a_responses_stream_cut_short_still_gives_back_the_held_text():
    lines = [
        "event: response.output_text.delta\n",
        'data: {"type": "response.output_text.delta", "item_id": "m1", "delta": "Tak, <OSOBA"}\n',
        "\n",
        "event: ping\n",
        "\n",
        "data: {broken\n\n",
    ]
    async with client(_stream_provider(lines)) as gw:
        response = await gw.post("/v1/responses", json=responses_body("Hej?", stream=True))
    assert "event: ping" in response.text
    assert "data: {broken" in response.text
    deltas = [
        json.loads(line[6:])["delta"]
        for line in response.text.splitlines()
        if line.startswith('data: {"type"')
    ]
    assert "".join(deltas) == "Tak, <OSOBA"


async def test_a_responses_call_that_keeps_state_at_the_provider_is_refused():
    received = []
    async with client(_responses_provider(received)) as gw:
        response = await gw.post(
            "/v1/responses", json=responses_body("Hej.", previous_response_id="resp_0")
        )
    assert response.status_code == 400
    assert "previous_response_id" in response.json()["error"]["message"]
    assert received == []


async def test_an_image_in_a_responses_call_is_refused_or_left_out():
    image = {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
    body = {"input": [{"role": "user", "content": [image]}, {"role": "user", "content": "Hej."}]}
    received = []
    async with client(_responses_provider(received)) as gw:
        refused = await gw.post("/v1/responses", json=body)
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "unsupported_content"
    assert received == []
    async with client(_responses_provider(received), unsupported="drop") as gw:
        dropped = await gw.post("/v1/responses", json=body)
    assert dropped.status_code == 200
    assert received[0]["input"][0]["content"] == []


async def test_the_provider_s_error_on_a_responses_call_passes_through():
    async with client(Provider(status=429)) as gw:
        response = await gw.post("/v1/responses", json=responses_body("Hej."))
    assert response.status_code == 429
