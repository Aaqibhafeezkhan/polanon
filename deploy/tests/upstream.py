"""A stand-in for the provider: it echoes the last message and remembers what it was sent.

Run by compose.test.yml in place of the real API. `GET /seen` hands back every request it got, so
probe.py can check that no personal data and no gateway key reached it.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar


class Provider(BaseHTTPRequestHandler):
    """Answers the calls Open WebUI makes: the list of models and a chat completion."""

    seen: ClassVar[list[dict[str, str]]] = []

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the container's log empty; `/seen` is the record."""

    def do_GET(self) -> None:
        """Serve the list of models, and the record of requests at `/seen`."""
        if self.path == "/seen":
            self._json(Provider.seen)
            return
        self._remember("")
        self._json({"object": "list", "data": [{"id": "atrapa", "object": "model"}]})

    def do_POST(self) -> None:
        """Echo the last message of a chat completion, streamed when the call asks for it."""
        raw = self.rfile.read(int(self.headers.get("content-length", "0"))).decode()
        self._remember(raw)
        body = json.loads(raw)
        answer = f"Echo: {body['messages'][-1]['content']}"
        if not body.get("stream"):
            self._json({"choices": [{"index": 0, "message": {"content": answer}}]})
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        # Three characters at a time, so a token arrives cut across chunks.
        for start in range(0, len(answer), 3):
            chunk = {"choices": [{"index": 0, "delta": {"content": answer[start : start + 3]}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def _remember(self, body: str) -> None:
        Provider.seen.append(
            {
                "path": self.path,
                "authorization": self.headers.get("authorization", ""),
                "body": body,
            }
        )

    def _json(self, payload: object) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    # Inside its container, so the gateway's container can reach it; no port is published.
    ThreadingHTTPServer(("0.0.0.0", port), Provider).serve_forever()  # noqa: S104
