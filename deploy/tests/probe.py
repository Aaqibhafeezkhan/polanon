"""Calls the gateway the way a tool in the stack would, and checks what the provider received.

check_stack.py runs it inside the stack's network with the address and the key the stack gives
its tool, so a stack that points the tool anywhere but at the gateway fails here. The person and
the PESEL number are invented.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

PESEL = "50810112354"
QUESTION = f"Jan Kowalski, PESEL {PESEL}."
EXPECTED = f"Echo: {QUESTION}"


def call(url: str, key: str | None, body: dict[str, object] | None = None) -> tuple[int, str]:
    """Send one request and return the status and the body, also for a refusal."""
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    # The addresses come from check_stack.py and are http:// on the stack's network.
    request = urllib.request.Request(url, data=data, headers=headers)  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def streamed_text(body: str) -> str:
    """Join the pieces of a streamed chat completion."""
    text = ""
    for line in body.splitlines():
        if line.startswith("data: {"):
            text += json.loads(line[6:])["choices"][0]["delta"].get("content", "")
    return text


def main() -> None:
    """Check the models, a plain and a streamed call, a call without the key, and the provider."""
    base = os.environ["TOOL_BASE_URL"].rstrip("/")
    key = os.environ["TOOL_API_KEY"]
    upstream_key = os.environ["POLANON_UPSTREAM_KEY"]
    question = {"model": "atrapa", "messages": [{"role": "user", "content": QUESTION}]}

    status, body = call(f"{base}/models", key)
    assert status == 200, (status, body)
    assert json.loads(body)["data"][0]["id"] == "atrapa", body

    status, body = call(f"{base}/chat/completions", key, question)
    assert status == 200, (status, body)
    assert json.loads(body)["choices"][0]["message"]["content"] == EXPECTED, body

    status, body = call(f"{base}/chat/completions", key, {**question, "stream": True})
    assert status == 200, (status, body)
    assert streamed_text(body) == EXPECTED, body

    status, body = call(f"{base}/chat/completions", None, question)
    assert status == 401, (status, body)

    status, body = call(os.environ["UPSTREAM_SEEN_URL"], None)
    assert status == 200, (status, body)
    seen = json.loads(body)
    chats = [request for request in seen if request["path"].endswith("/chat/completions")]
    assert len(chats) == 2, seen
    for request in seen:
        assert PESEL not in request["body"], "the PESEL number reached the provider"
        assert "Kowalski" not in request["body"], "the surname reached the provider"
        assert request["authorization"] == f"Bearer {upstream_key}", "the wrong key was sent on"
    print(f"ok: {len(seen)} requests reached the provider, none with the person or the number")


if __name__ == "__main__":
    main()
