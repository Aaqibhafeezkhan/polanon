"""The gateway served by uvicorn, end to end, with a provider on localhost that records its input.

The other gateway tests call the application directly; this one goes through a real server and
real sockets, the way a tool on the operator's machine does, with the calls masked in the
gateway's own process and in processes of their own, as ``--workers`` has them. It needs the
gateway extra.
"""

import functools
import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")

from polanon import Anonymizer, default_guards  # noqa: E402
from polanon.gateway.app import Gateway, GatewayConfig  # noqa: E402

PESEL = "50810112354"
#: How each masking process builds its anonymizer: pickled, so a partial, never a lambda.
BUILD = functools.partial(Anonymizer, default_guards(with_ner=False))


class _Provider(BaseHTTPRequestHandler):
    """Echoes the last message back, streamed in three-character chunks when asked."""

    received: ClassVar[list[bytes]] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["content-length"]))
        _Provider.received.append(raw)
        body = json.loads(raw)
        answer = f"Echo: {body['messages'][-1]['content']}"
        if not body.get("stream"):
            data = json.dumps({"choices": [{"index": 0, "message": {"content": answer}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for start in range(0, len(answer), 3):
            chunk = {"choices": [{"index": 0, "delta": {"content": answer[start : start + 3]}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def upstream():
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=provider.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{provider.server_address[1]}/v1"
    provider.shutdown()


@contextmanager
def _served(app):
    """The application served by uvicorn on a free port, once its startup is complete."""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=30)


@pytest.fixture(params=[1, 2], ids=["in its own process", "in two workers"])
def gateway_url(request, upstream):
    config = GatewayConfig(upstream=upstream, workers=request.param)
    with _served(Gateway(BUILD(), config, build=BUILD)) as url:
        _Provider.received = []
        yield url


def test_a_real_server_masks_on_the_way_out_and_restores_on_the_way_back(gateway_url):
    question = {
        "model": "echo",
        "messages": [{"role": "user", "content": f"Jan Kowalski, PESEL {PESEL}."}],
    }
    plain = httpx.post(f"{gateway_url}/chat/completions", json=question, timeout=10)
    streamed = ""
    with httpx.stream(
        "POST", f"{gateway_url}/chat/completions", json={**question, "stream": True}, timeout=10
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: {"):
                streamed += json.loads(line[6:])["choices"][0]["delta"].get("content", "")
    expected = f"Echo: Jan Kowalski, PESEL {PESEL}."
    assert plain.json()["choices"][0]["message"]["content"] == expected
    assert streamed == expected
    sent = b" ".join(_Provider.received).decode()
    assert "Kowalski" not in sent
    assert PESEL not in sent
    assert "<OSOBA_1>" in sent
    assert httpx.get(gateway_url.removesuffix("/v1") + "/health").json() == {"status": "ok"}


_PEOPLE = ("Jan Kowalski", "Anna Nowak", "Piotr Wiśniewski", "Ewa Zielińska")
_PLACES = ("Gdańska", "Krakowa", "Poznania", "Lublina")


def _letter(n, size=20_000):
    """An invented letter of about `size` characters, a different one for every `n`."""
    lines = []
    for line in range(size // 100):
        lines.append(
            f"Pismo nr {n}/{line}: {_PEOPLE[line % 4]} z {_PLACES[line % 4]} prosi o kontakt "
            f"pod numerem 600 100 {line % 1000:03d} w sprawie umowy. "
        )
    return "".join(lines)


def _ask(url, n):
    letter = _letter(n)
    question = {"model": "echo", "messages": [{"role": "user", "content": letter}]}
    answer = httpx.post(f"{url}/chat/completions", json=question, timeout=120)
    assert answer.json()["choices"][0]["message"]["content"] == f"Echo: {letter}"


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="masking side by side needs two cores")
def test_calls_at_once_are_masked_side_by_side_in_the_workers(upstream):
    """With one worker the gateway masks one call at a time, and four calls at once took as long
    as four in a row. Two workers mask two at a time, so four at once must take clearly less:
    about half as long. A limit in seconds would fail on a slower machine, or under
    `pytest --cov`, which runs both alike, so the two are compared, each the faster of two runs."""
    calls = 4

    def fastest(run):
        times = []
        for _ in range(2):
            began = time.perf_counter()
            run()
            times.append(time.perf_counter() - began)
        return min(times)

    config = GatewayConfig(upstream=upstream, workers=2)
    with _served(Gateway(BUILD(), config, build=BUILD)) as url, ThreadPoolExecutor(calls) as tools:
        _ask(url, -1)

        def in_a_row():
            for n in range(calls):
                _ask(url, n)

        def at_once():
            list(tools.map(functools.partial(_ask, url), range(calls)))

        assert fastest(at_once) < 0.75 * fastest(in_a_row)
    sent = b" ".join(_Provider.received).decode()
    assert "Kowalski" not in sent
    assert "600 100" not in sent
