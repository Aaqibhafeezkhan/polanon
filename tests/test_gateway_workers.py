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
import types
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

import polanon  # noqa: E402
from polanon import Anonymizer, Guard, MaskingUnavailableError, default_guards  # noqa: E402
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
#: The signals that ask every process of a console or of a service to stop.
STOPS = [signal.SIGINT, signal.SIGTERM, *([signal.SIGBREAK] if sys.platform == "win32" else [])]


class Trap(Guard):
    """A guard that finds nothing, and does to its own process what a text asks: ZATRZYMAJ ends
    it at once, as the system ends one that took too much memory, and SYGNAŁY sends it every
    signal that asks a whole console or service to stop."""

    name = "trap"

    def detect(self, text, enabled):
        if "ZATRZYMAJ" in text:
            os._exit(1)
        if "SYGNAŁY" in text:
            for stop in STOPS:
                signal.raise_signal(stop)
        return []


def trapped():
    """The anonymizer of a worker that springs the trap; the worker finds it here by its name."""
    return Anonymizer([*default_guards(with_ner=False), Trap()])


def slow_once(mark):
    """The first worker to build its anonymizer takes 2 s more than the others.

    A pool once let a worker that was ready wait that long for the others, then take another's
    task and report itself twice, so it said ready with a worker still loading; this one waits
    0.2 s, to show it within the test.
    """
    workers._READY_WAIT = 0.2
    try:
        os.close(os.open(mark, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return BUILD()
    time.sleep(2)
    return BUILD()


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
    """A gateway whose workers build their anonymizers with `build`; its own masks with BUILD."""
    config = {"upstream": UPSTREAM, "workers": 2, **config}
    return Gateway(
        BUILD(), GatewayConfig(**config), transport=httpx.MockTransport(echo), build=build
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


def test_a_pool_has_two_workers_or_more():
    with pytest.raises(ValueError, match="2 or more"):
        MaskingPool(BUILD, 1)
    assert MaskingPool(BUILD, 64).workers == 64


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


async def test_a_call_that_stops_its_worker_fails_alone_and_is_not_sent_again(caplog):
    """A worker killed for lack of memory, which the call it masked may have taken, fails that
    call: the client may send it again, and nothing reaches the provider. Sent again by the
    gateway, it could stop the next worker too. The other calls, masked by the other worker or
    waiting for one, are masked as if nothing happened, the waiting ones by a new worker."""
    echo = Echo()
    gateway = gateway_of(echo, build=trapped)
    try:
        await asyncio.to_thread(gateway._pool.wait_ready)
        async with client(gateway) as gw:
            stopping, *others = await asyncio.gather(
                gw.post("/v1/chat/completions", json=chat("ZATRZYMAJ, Jan Kowalski.")),
                *(gw.post("/v1/chat/completions", json=chat(f"Pismo {n}.")) for n in range(6)),
            )
    finally:
        await gateway.aclose()
    assert stopping.status_code == 503
    assert stopping.json()["error"]["code"] == "masking_failed"
    assert [other.status_code for other in others] == [200] * 6
    assert sorted(body["messages"][-1]["content"] for body in echo.received) == [
        f"Pismo {n}." for n in range(6)
    ]
    assert "Kowalski" not in echo.sent()
    assert caplog.text.count("a masking process stopped") == 1


async def test_workers_that_stopped_while_idle_cost_no_call():
    """A worker that stopped between calls never saw the next one: a new worker masks it."""
    echo = Echo()
    gateway = gateway_of(echo)
    try:
        for pid in await asyncio.to_thread(gateway._pool.wait_ready):
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        await asyncio.sleep(1)
        async with client(gateway) as gw:
            answers = await asyncio.gather(
                gw.post("/v1/chat/completions", json=chat("Jan Kowalski dzwonił.")),
                gw.post("/v1/chat/completions", json=chat("Anna Nowak pisała.")),
            )
    finally:
        await gateway.aclose()
    assert [answer.status_code for answer in answers] == [200, 200]
    assert [answer.json()["choices"][0]["message"]["content"] for answer in answers] == [
        "Echo: Jan Kowalski dzwonił.",
        "Echo: Anna Nowak pisała.",
    ]
    assert len(echo.received) == 2


async def test_a_worker_lets_the_gateway_answer_a_request_to_stop():
    """Ctrl+C and Ctrl+Break reach every process of a console, and systemd sends SIGTERM to every
    process of a service: the gateway finishes the calls it masks and stops its workers itself,
    so a worker that gets one masks on."""
    echo = Echo()
    gateway = gateway_of(echo, build=trapped)
    try:
        before = await asyncio.to_thread(gateway._pool.wait_ready)
        async with client(gateway) as gw:
            signalled = await gw.post("/v1/chat/completions", json=chat("SYGNAŁY dla Jana."))
            after = await asyncio.gather(
                *(gw.post("/v1/chat/completions", json=chat(f"Pismo {n}.")) for n in range(4))
            )
        assert await asyncio.to_thread(gateway._pool.wait_ready) == before
    finally:
        await gateway.aclose()
    assert signalled.status_code == 200
    assert signalled.json()["choices"][0]["message"]["content"] == "Echo: SYGNAŁY dla Jana."
    assert [answer.status_code for answer in after] == [200] * 4


def test_the_pool_is_ready_only_once_every_worker_is(tmp_path):
    pool = MaskingPool(functools.partial(slow_once, str(tmp_path / "slow")), 2)
    try:
        assert len(pool.wait_ready()) == 2
    finally:
        pool.close()


async def test_a_call_no_worker_could_begin_is_refused_as_masking_that_failed():
    """Workers that cannot build their anonymizers stop before a call reaches them: the call is
    sent to the worker that takes the place of the first, and refused when that one stops too."""
    pool = MaskingPool(functools.partial(int, "not an anonymizer"), 2)
    try:
        with pytest.raises(MaskingUnavailableError, match="before the call was masked"):
            await pool.mask("chat", chat("Jan Kowalski."), instructions=None, unsupported="refuse")
    finally:
        pool.close()


async def test_a_closed_pool_refuses_calls():
    pool = MaskingPool(BUILD, 2)
    pool.close()
    with pytest.raises(MaskingUnavailableError, match="have stopped"):
        await pool.mask("chat", chat("Jan Kowalski."), instructions=None, unsupported="refuse")
    with pytest.raises(MaskingUnavailableError, match="have stopped"):
        pool.wait_ready()
    assert sorted(pool._free) == [0, 1]


async def test_calls_wait_in_the_gateway_for_a_free_worker_in_the_order_they_came():
    """No process starts here: the pool hands out its places before any process takes them."""
    pool = MaskingPool(BUILD, 2)
    first, second = await pool._take(), await pool._take()
    waiting = [asyncio.create_task(pool._take()) for _ in range(3)]
    await asyncio.sleep(0)
    waiting[0].cancel()
    pool._give(second)
    assert await waiting[1] == second
    assert waiting[0].cancelled()
    pool._give(first)
    waiting[2].cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting[2]
    assert list(pool._free) == [first]
    assert not pool._waiting


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
    assert gateway._pool._slots is None


async def test_the_gateway_s_lifespan_starts_its_workers_and_stops_them():
    gateway = gateway_of(Echo())
    events = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
    sent = []

    async def receive():
        return events.pop(0)

    async def send(message):
        sent.append(message["type"])
        if message["type"] == "lifespan.startup.complete":
            assert len(gateway._pool.wait_ready(0)) == 2

    await gateway({"type": "lifespan"}, receive, send)
    assert sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert gateway._pool._slots is None


def test_a_worker_masks_its_call_and_forgets_the_call_s_values(monkeypatch):
    """What a process runs, run here: it ignores the requests to stop that the gateway answers
    for it, loads its model before the first call, notes the call it masks, and forgets the
    values of each call once it is masked."""
    ignored, forgotten = [], []
    monkeypatch.setattr(workers.signal, "signal", lambda *args: ignored.append(args))
    monkeypatch.setattr(workers, "forget_prepared_searches", lambda: forgotten.append(True))
    monkeypatch.setattr(workers, "_state", None)
    with pytest.raises(MaskingUnavailableError, match="no anonymizer"):
        workers._mask(1, "chat", chat("Jan Kowalski."), None, "refuse")
    began = types.SimpleNamespace(value=0)
    workers._start(BUILD, began)
    assert ignored == [(stop, signal.SIG_IGN) for stop in STOPS]
    assert forgotten == [True]
    text = f"Jan Kowalski, PESEL {PESEL}."
    masked, mapping = workers._mask(7, "chat", chat(text), None, "refuse")
    assert masked["messages"][-1]["content"] == "<OSOBA_1>, PESEL <PESEL_1>."
    assert mapping == {"<OSOBA_1>": "Jan Kowalski", "<PESEL_1>": PESEL}
    assert began.value == 7
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
