"""The gateway served by uvicorn, end to end, with a provider on localhost that records its input.

The other gateway tests call the application directly; this one goes through a real server and
real sockets, the way a tool on the operator's machine does. It needs the gateway extra.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")

from polanon import Anonymizer, default_guards  # noqa: E402
from polanon.gateway.app import Gateway, GatewayConfig  # noqa: E402

PESEL = "50810112354"


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


@pytest.fixture
def gateway_url():
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=provider.serve_forever, daemon=True).start()
    upstream = f"http://127.0.0.1:{provider.server_address[1]}/v1"
    app = Gateway(Anonymizer(default_guards(with_ner=False)), GatewayConfig(upstream=upstream))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    _Provider.received = []
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=10)
    provider.shutdown()


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
