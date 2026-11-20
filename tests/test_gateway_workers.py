"""The gateway with ``--workers``: calls masked side by side, in processes of their own.

A process that masks is started as the gateway would start it, so these tests start real ones,
with the rules alone. Everything a process sends back went through a pipe, pickled.
"""

import asyncio
import contextlib
import functools
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

import polanon  # noqa: E402
from polanon import Anonymizer, MaskingUnavailableError, default_guards  # noqa: E402
from polanon.gateway import workers  # noqa: E402
from polanon.gateway.__main__ import _parser  # noqa: E402
from polanon.gateway._build import worker_anonymizer  # noqa: E402
from polanon.gateway.app import Gateway, GatewayConfig  # noqa: E402
from polanon.gateway.workers import MaskingPool  # noqa: E402

PESEL = "50810112354"
KEY = "ab" * 32
UPSTREAM = "https://provider.test/v1"
#: How each process builds its anonymizer: pickled, so a partial of a class, never a lambda.
BUILD = functools.partial(Anonymizer, default_guards(with_ner=False))


class Echo:
    """A provider for every endpoint the gateway serves: it keeps each request and echoes it."""

    def __init__(self):
        self.received = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.received.append(body)
        path = request.url.path
        if path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
        if path.endswith("/responses"):
            text = {"type": "output_text", "text": f"Echo: {body['input'][-1]['content']}"}
            message = {"type": "message", "id": "m1", "role": "assistant", "content": [text]}
            return httpx.Response(200, json={"id": "r1", "object": "response", "output": [message]})
        if path.endswith("/messages"):
            text = {"type": "text", "text": f"Echo: {body['messages'][-1]['content']}"}
            message = {"id": "m1", "type": "message", "role": "assistant", "content": [text]}
            return httpx.Response(200, json={**message, "stop_reason": "end_turn"})
        answer = f"Echo: {body['messages'][-1]['content']}"
        if body.get("stream"):
            chunks = [
                {"choices": [{"index": 0, "delta": {"content": answer[start : start + 3]}}]}
                for start in range(0, len(answer), 3)
            ]
            stream = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=stream + "data: [DONE]\n\n",
            )
        message = {"role": "assistant", "content": answer}
        return httpx.Response(200, json={"choices": [{"index": 0, "message": message}]})

    def sent(self):
        """Everything the provider received, as one text."""
        return json.dumps(self.received, ensure_ascii=False)


def gateway_of(echo, build=BUILD, **config):
    config = {"upstream": UPSTREAM, "workers": 2, **config}
    return Gateway(
        build(), GatewayConfig(**config), transport=httpx.MockTransport(echo), build=build
    )


def client(gateway):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gw")


def chat(content, **fields):
    return {"model": "m", "messages": [{"role": "user", "content": content}], **fields}


def streamed(response):
    """The text of a streamed chat completion."""
    text = ""
    for line in response.text.splitlines():
        if line.startswith("data: {"):
            text += json.loads(line[6:])["choices"][0]["delta"].get("content", "")
    return text


@pytest.fixture(scope="module")
def pooled():
    """A gateway with two workers, ready, that hands back mappings when asked; shared by tests
    that leave its processes running."""
    echo = Echo()
    gateway = gateway_of(echo, pii_map=True)
    assert len(gateway._pool.wait_ready()) == 2
    yield gateway, echo
    gateway._pool.close()


def test_one_call_at_a_time_in_the_gateway_s_own_process_is_the_default():
    gateway = Gateway(BUILD(), GatewayConfig(upstream=UPSTREAM))
    assert gateway._config.workers == 1
    assert gateway._pool is None


@pytest.mark.parametrize("count", [0, -2, True, 1.5])
def test_a_number_of_workers_that_is_not_one_or_more_is_refused(count):
    with pytest.raises(ValueError, match="workers is 1 or more"):
        GatewayConfig(upstream=UPSTREAM, workers=count)


def test_a_pool_has_two_workers_or_more_and_no_more_than_windows_allows(monkeypatch):
    with pytest.raises(ValueError, match="2 or more"):
        MaskingPool(BUILD, 1)
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(ValueError, match="at most 61"):
        MaskingPool(BUILD, 62)
    assert MaskingPool(BUILD, 61).workers == 61


def test_more_than_one_worker_needs_a_way_to_build_each_one_s_anonymizer():
    with pytest.raises(ValueError, match="needs build"):
        Gateway(BUILD(), GatewayConfig(upstream=UPSTREAM, workers=2))


async def test_calls_masked_in_the_workers_go_out_masked_and_come_back_restored(pooled):
    gateway, echo = pooled
    echo.received.clear()
    text = f"Jan Kowalski, PESEL {PESEL}."
    async with client(gateway) as gw:
        plain = await gw.post("/v1/chat/completions", json=chat(text))
        stream = await gw.post("/v1/chat/completions", json=chat(text, stream=True))
        responses = await gw.post(
            "/v1/responses",
            json={"model": "m", "store": False, "input": [{"role": "user", "content": text}]},
        )
        messages = await gw.post(
            "/v1/messages",
            json={"model": "m", "max_tokens": 9, "messages": [{"role": "user", "content": text}]},
        )
        embeddings = await gw.post("/v1/embeddings", json={"model": "e", "input": [text]})
    assert plain.json()["choices"][0]["message"]["content"] == f"Echo: {text}"
    assert streamed(stream) == f"Echo: {text}"
    assert responses.json()["output"][0]["content"][0]["text"] == f"Echo: {text}"
    assert messages.json()["content"][0]["text"] == f"Echo: {text}"
    assert embeddings.json() == {"data": [{"embedding": [0.1, 0.2]}]}
    assert len(echo.received) == 5
    assert "Kowalski" not in echo.sent()
    assert PESEL not in echo.sent()
    assert echo.received[-1]["input"] == ["<OSOBA_1>, PESEL <PESEL_1>."]


async def test_a_refusal_in_a_worker_reaches_the_client_as_the_gateway_s_own_does(pooled):
    gateway, echo = pooled
    echo.received.clear()
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    async with client(gateway) as gw:
        empty = await gw.post("/v1/chat/completions", json={"model": "m", "messages": []})
        wrong = await gw.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": 5}]}
        )
        picture = await gw.post(
            "/v1/chat/completions", json=chat([{"type": "text", "text": "Co to?"}, image])
        )
    assert (empty.status_code, empty.json()["error"]["code"]) == (400, "invalid_request")
    assert (wrong.status_code, wrong.json()["error"]["code"]) == (400, "invalid_request")
    assert (picture.status_code, picture.json()["error"]["code"]) == (400, "unsupported_content")
    assert "--drop-unsupported" in picture.json()["error"]["message"]
    assert echo.received == []


async def test_each_call_gets_back_from_its_worker_a_mapping_of_its_own(pooled):
    """The mapping lives as long as the call: the next call, in whichever process, starts afresh,
    and its tokens are numbered from 1 again."""
    gateway, _ = pooled
    async with client(gateway) as gw:
        first, second = await asyncio.gather(
            gw.post(
                "/v1/chat/completions", json=chat("Jan Kowalski dzwonił.", return_pii_map=True)
            ),
            gw.post("/v1/chat/completions", json=chat(f"PESEL {PESEL}.", return_pii_map=True)),
        )
    assert first.json()["return_pii_map"] == {"<OSOBA_1>": "Jan Kowalski"}
    assert second.json()["return_pii_map"] == {"<PESEL_1>": PESEL}


async def test_stable_tokens_are_the_same_in_every_worker_as_in_the_gateway(monkeypatch):
    """Each process builds its anonymizer from the gateway's options and reads POLANON_KEY from
    the environment it inherits; the key itself never travels to it."""
    monkeypatch.setenv("POLANON_KEY", KEY)
    args = _parser().parse_args(["--upstream", UPSTREAM, "--rules-only"])
    build = functools.partial(worker_anonymizer, args)
    echo = Echo()
    gateway = gateway_of(echo, build=build, instructions=None)
    text = f"Jan Kowalski, PESEL {PESEL}."
    try:
        assert len(await asyncio.to_thread(gateway._pool.wait_ready)) == 2
        async with client(gateway) as gw:
            answers = await asyncio.gather(
                *(
                    gw.post("/v1/chat/completions", json=chat(f"Pismo {n}: {text}"))
                    for n in range(6)
                )
            )
    finally:
        await gateway.aclose()
    expected = build().anonymize_sync(text, {}).anonymized
    assert "<OSOBA_1>" not in expected
    assert {body["messages"][-1]["content"].split(": ", 1)[1] for body in echo.received} == {
        expected
    }
    assert [answer.json()["choices"][0]["message"]["content"] for answer in answers] == [
        f"Echo: Pismo {n}: {text}" for n in range(6)
    ]


async def test_a_worker_that_stops_fails_its_calls_and_new_workers_take_the_next_ones():
    """A process killed for lack of memory fails what the pool was masking. The call is refused
    as masking that failed, which a client may send again, and nothing reaches the provider."""
    echo = Echo()
    gateway = gateway_of(echo)
    pool = gateway._pool
    try:
        await asyncio.to_thread(pool.wait_ready)
        executor = pool._executor
        worker = next(iter(executor._processes.values()))
        worker.kill()
        worker.join()
        deadline = time.monotonic() + 30
        while not executor._broken and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        async with client(gateway) as gw:
            refused = await gw.post("/v1/chat/completions", json=chat("Jan Kowalski dzwonił."))
            served = await gw.post("/v1/chat/completions", json=chat("Jan Kowalski dzwonił."))
    finally:
        await gateway.aclose()
    assert refused.status_code == 503
    assert refused.json()["error"]["code"] == "masking_failed"
    assert served.status_code == 200
    assert served.json()["choices"][0]["message"]["content"] == "Echo: Jan Kowalski dzwonił."
    assert len(echo.received) == 1
    assert pool._executor is None


async def test_workers_that_cannot_start_keep_the_gateway_from_starting():
    """Rather than report itself healthy and refuse every call, the gateway does not start."""
    broken = functools.partial(int, "not an anonymizer")
    gateway = Gateway(BUILD(), GatewayConfig(upstream=UPSTREAM, workers=2), build=broken)
    events = [{"type": "lifespan.startup"}]
    sent = []

    async def receive():
        return events.pop(0)

    async def send(message):
        sent.append(message)

    await gateway({"type": "lifespan"}, receive, send)
    assert sent[0]["type"] == "lifespan.startup.failed"
    assert "the masking processes could not start" in sent[0]["message"]
    assert gateway._pool._executor is None


async def test_the_gateway_s_lifespan_starts_its_workers_and_stops_them():
    gateway = gateway_of(Echo())
    events = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
    sent = []

    async def receive():
        return events.pop(0)

    async def send(message):
        sent.append(message["type"])
        if message["type"] == "lifespan.startup.complete":
            assert len(gateway._pool._executor._processes) == 2

    await gateway({"type": "lifespan"}, receive, send)
    assert sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert gateway._pool._executor is None


def test_a_worker_masks_its_call_and_forgets_the_call_s_values(monkeypatch):
    """What a process runs, run here: it ignores Ctrl+C, which the gateway answers for it, loads
    its model before the first call, and forgets the values of each call once it is masked."""
    ignored, forgotten = [], []
    monkeypatch.setattr(workers.signal, "signal", lambda *args: ignored.append(args))
    monkeypatch.setattr(workers, "forget_prepared_searches", lambda: forgotten.append(True))
    monkeypatch.setattr(workers, "_state", None)
    with pytest.raises(MaskingUnavailableError, match="no anonymizer"):
        workers._mask("chat", chat("Jan Kowalski."), None, "refuse")
    workers._start(BUILD, threading.Barrier(1))
    assert ignored == [(signal.SIGINT, signal.SIG_IGN)]
    assert forgotten == [True]
    masked, mapping = workers._mask("chat", chat(f"Jan Kowalski, PESEL {PESEL}."), None, "refuse")
    assert masked["messages"][-1]["content"] == "<OSOBA_1>, PESEL <PESEL_1>."
    assert mapping == {"<OSOBA_1>": "Jan Kowalski", "<PESEL_1>": PESEL}
    assert forgotten == [True, True]
    assert workers._ready() == os.getpid()


def test_a_worker_watches_the_process_that_started_it(monkeypatch):
    ended = threading.Event()

    class Gone:
        def join(self):
            pass

    monkeypatch.setattr(workers.multiprocessing, "parent_process", lambda: Gone())
    monkeypatch.setattr(workers.os, "_exit", lambda status: ended.set())
    workers._exit_with_parent()
    assert ended.wait(10)


def test_the_workers_end_when_the_gateway_is_killed_outright(tmp_path):
    """A gateway killed outright cannot stop its workers, and each holds a language model of its
    own: none may live on. Every process still alive holds the pipe of the script's output, so it
    closes only when the last of them has ended; without the watch, it never did on Windows."""
    script = tmp_path / "pool.py"
    script.write_text(
        textwrap.dedent(
            """
            import functools
            import time

            from polanon import Anonymizer, default_guards
            from polanon.gateway.workers import MaskingPool

            if __name__ == "__main__":
                build = functools.partial(Anonymizer, default_guards(with_ner=False))
                pool = MaskingPool(build, 2)
                print(*sorted(pool.wait_ready()), flush=True)
                time.sleep(600)
            """
        ),
        encoding="utf-8",
    )
    package = str(Path(polanon.__file__).resolve().parent.parent)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [package, os.environ.get("PYTHONPATH")])),
    }
    gateway = subprocess.Popen(  # noqa: S603 - the interpreter running these tests
        [sys.executable, str(script)], stdout=subprocess.PIPE, env=env, text=True
    )
    pids = [int(pid) for pid in gateway.stdout.readline().split()]
    gateway.kill()
    gateway.wait()
    closed = threading.Event()
    threading.Thread(target=lambda: (gateway.stdout.read(), closed.set()), daemon=True).start()
    try:
        assert len(pids) == 2
        assert closed.wait(60), "a worker outlived the gateway"
    finally:
        if not closed.is_set():
            for pid in pids:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGTERM)
        gateway.stdout.close()
