"""The gateway served over HTTP: a plain ASGI application in front of an OpenAI-compatible API.

It serves ``/v1/chat/completions``, ``/v1/responses``, ``/v1/embeddings`` and ``/v1/models``,
and refuses every other endpoint: a request the gateway cannot mask is never forwarded as it came.
Each call is masked by its own :class:`~polanon.MaskedTurn`, checked once more for any original
value, sent to the provider and restored on the way back, streamed or not.

The application keeps nothing between calls and writes nothing to disk. It logs the path, the
status, the time taken and how many values of each type were masked, never a value.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from polanon import Anonymizer, MaskedTurn, MaskingUnavailableError, token_instructions
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

logger = logging.getLogger("polanon.gateway")

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

#: Headers of the client's request that are never passed on: the connection's own, the body's,
#: cookies, and the client's credentials, which the gateway handles itself.
_NOT_FORWARDED = frozenset(
    {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "upgrade"}
    | {"accept-encoding", "authorization", "proxy-authorization", "cookie"}
)
#: Headers of the provider's response that describe a body the gateway rewrites.
_NOT_RETURNED = frozenset(
    {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}
)


@dataclass(frozen=True)
class GatewayConfig:
    """Where the gateway sends calls, and who may use it.

    Attributes:
        upstream: the provider's API root, such as ``https://api.openai.com/v1``.
        upstream_key: a key sent to the provider in place of the client's own; None passes the
            client's key on, unless `gateway_key` is set.
        gateway_key: a key every client must send; the client's key then never reaches the
            provider, which gets `upstream_key` or no key at all, as a local model needs.
        instructions: the language of the note that tells the model what the tokens are, "pl" or
            "en", or None to leave the messages without it.
        unsupported: what to do with content that cannot be masked, such as an image: "refuse" the
            call or "drop" the content and send the rest.
        max_body: the largest request accepted, in bytes.
        timeout: seconds to wait for the provider.
    """

    upstream: str
    upstream_key: str | None = field(default=None, repr=False)
    gateway_key: str | None = field(default=None, repr=False)
    instructions: str | None = "pl"
    unsupported: Literal["refuse", "drop"] = "refuse"
    max_body: int = 10 * 1024 * 1024
    timeout: float = 600.0


class _RefusalError(Exception):
    """A call the gateway answers itself, with an error in the shape the OpenAI API uses."""

    def __init__(self, status: int, message: str, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


class Gateway:
    """An ASGI application that masks calls to an OpenAI-compatible API and restores the answers.

    Build it once per process, with the anonymizer that masks every call; serve it with any ASGI
    server, as ``polanon-gateway`` does with uvicorn.
    """

    def __init__(
        self,
        anonymizer: Anonymizer,
        config: GatewayConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Set the gateway up.

        Args:
            anonymizer: the guards and the policy every call is masked with. It must have no
                stage-2 judge, which a turn cannot run.
            config: where calls go and who may make them.
            transport: what carries calls to the provider; tests pass one that answers itself.
        """
        self._anonymizer = anonymizer
        self._config = config
        self._instructions = (
            token_instructions(config.instructions) if config.instructions else None
        )
        self._client = httpx.AsyncClient(
            base_url=config.upstream, timeout=config.timeout, transport=transport
        )
        # One call is masked at a time: the language model is not shared safely between threads,
        # and masking runs in a thread so the event loop keeps serving streams meanwhile.
        self._masking = asyncio.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve one ASGI connection."""
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        started = time.perf_counter()
        method, path = scope["method"], _route(scope["path"])
        status = 200
        counts: dict[str, int] = {}
        try:
            if path == "/health" and method == "GET":
                await _send_json(send, 200, {"status": "ok"})
            elif not self._authorised(scope):
                raise _RefusalError(401, "the gateway key is missing or wrong", "invalid_api_key")
            elif path == "/chat/completions" and method == "POST":
                status, counts = await self._chat(scope, receive, send)
            elif path == "/responses" and method == "POST":
                status, counts = await self._responses(scope, receive, send)
            elif path == "/embeddings" and method == "POST":
                status, counts = await self._embeddings(scope, receive, send)
            elif method == "GET" and (path == "/models" or path.startswith("/models/")):
                status = await self._models(scope, send, path)
            else:
                raise _RefusalError(
                    404,
                    f"the polanon gateway does not serve {method} {path}: it masks "
                    "/v1/chat/completions, /v1/responses and /v1/embeddings only",
                    "unsupported_endpoint",
                )
        except _RefusalError as refusal:
            status = refusal.status
            await _send_error(send, refusal.status, refusal.message, refusal.code)
        logger.info(
            "%s %s -> %d in %.2fs, masked %s",
            method,
            path,
            status,
            time.perf_counter() - started,
            counts or "nothing",
        )

    async def _chat(self, scope: Scope, receive: Receive, send: Send) -> tuple[int, dict[str, int]]:
        body = await self._body(receive)
        turn = self._turn()
        masked = await self._mask(
            lambda: mask_chat_request(turn, body, instructions=self._instructions)
        )
        self._check(masked, turn)
        headers = self._upstream_headers(scope)
        if body.get("stream"):
            lines = _ChatLines(turn.mapping)
            status = await self._stream("chat/completions", masked, headers, send, lines)
            return status, turn.counts()
        response = await self._post("chat/completions", masked, headers)
        if response.is_success:
            restored = restore_chat_response(turn, _json_of(response))
            await _send_json(send, response.status_code, restored, response.headers)
        else:
            await _send_raw(send, response)
        return response.status_code, turn.counts()

    async def _responses(
        self, scope: Scope, receive: Receive, send: Send
    ) -> tuple[int, dict[str, int]]:
        body = await self._body(receive)
        turn = self._turn()
        drop = self._config.unsupported == "drop"
        masked = await self._mask(
            lambda: mask_responses_request(
                turn, body, instructions=self._instructions, drop_unsupported=drop
            )
        )
        self._check(masked, turn)
        headers = self._upstream_headers(scope)
        if body.get("stream"):
            lines = _ResponsesLines(turn.mapping)
            status = await self._stream("responses", masked, headers, send, lines)
            return status, turn.counts()
        response = await self._post("responses", masked, headers)
        if response.is_success:
            restored = restore_responses_response(turn, _json_of(response))
            await _send_json(send, response.status_code, restored, response.headers)
        else:
            await _send_raw(send, response)
        return response.status_code, turn.counts()

    async def _embeddings(
        self, scope: Scope, receive: Receive, send: Send
    ) -> tuple[int, dict[str, int]]:
        body = await self._body(receive)
        turn = self._turn()
        masked = await self._mask(lambda: mask_embeddings_request(turn, body))
        self._check(masked, turn)
        response = await self._post("embeddings", masked, self._upstream_headers(scope))
        await _send_raw(send, response)
        return response.status_code, turn.counts()

    async def _models(self, scope: Scope, send: Send, path: str) -> int:
        try:
            response = await self._client.get(
                path.lstrip("/"), headers=self._upstream_headers(scope)
            )
        except httpx.HTTPError as exc:
            raise _unreachable(exc) from exc
        await _send_raw(send, response)
        return response.status_code

    async def _stream(
        self,
        path: str,
        masked: dict[str, Any],
        headers: dict[str, str],
        send: Send,
        lines: _ChatLines | _ResponsesLines,
    ) -> int:
        """Stream the answer back, restored chunk by chunk.

        Once the answer has started, a provider that breaks off can no longer be reported with an
        error status: the stream simply ends, as the provider's own would.
        """
        started = False
        try:
            async with self._client.stream("POST", path, json=masked, headers=headers) as response:
                if not response.is_success:
                    await response.aread()
                    await _send_raw(send, response)
                    return response.status_code
                started = True
                await send(
                    {
                        "type": "http.response.start",
                        "status": response.status_code,
                        "headers": [
                            (b"content-type", b"text/event-stream; charset=utf-8"),
                            (b"cache-control", b"no-cache"),
                        ],
                    }
                )
                async for line in response.aiter_lines():
                    for out in lines.line(line):
                        await send({"type": "http.response.body", "body": out, "more_body": True})
                for out in lines.end():
                    await send({"type": "http.response.body", "body": out, "more_body": True})
                await send({"type": "http.response.body", "body": b""})
                return response.status_code
        except httpx.HTTPError as exc:
            if not started:
                raise _unreachable(exc) from exc
            logger.warning("the provider broke off a streamed answer: %s", type(exc).__name__)
            await send({"type": "http.response.body", "body": b""})
            return 502

    async def _post(
        self, path: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        try:
            return await self._client.post(path, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise _unreachable(exc) from exc

    async def _body(self, receive: Receive) -> dict[str, Any]:
        """Read the request's JSON object, refusing one larger than the limit."""
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise _RefusalError(
                    400, "the client went away before sending the request", "aborted"
                )
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self._config.max_body:
                raise _RefusalError(
                    413, f"the request is larger than {self._config.max_body} bytes", "too_large"
                )
            chunks.append(chunk)
            if not message.get("more_body"):
                break
        try:
            body = json.loads(b"".join(chunks))
        except ValueError:
            raise _RefusalError(400, "the request is not valid JSON", "invalid_json") from None
        if not isinstance(body, dict):
            raise _RefusalError(400, "the request is a JSON object", "invalid_json")
        return body

    def _turn(self) -> MaskedTurn:
        unknown: Literal["raise", "drop"] = (
            "drop" if self._config.unsupported == "drop" else "raise"
        )
        return MaskedTurn(self._anonymizer, unknown_blocks=unknown)

    async def _mask(self, masking: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Mask in a thread, one call at a time, and turn a refusal into an error for the client."""
        try:
            async with self._masking:
                return await asyncio.to_thread(masking)
        except GatewayRequestError as exc:
            raise _RefusalError(400, str(exc), "invalid_request") from exc
        except MaskingUnavailableError as exc:
            if "cannot be masked" in str(exc):
                what = str(exc).split(";", 1)[0]
                raise _RefusalError(
                    400,
                    f"{what}, so the call was not sent; start the gateway with "
                    "--drop-unsupported to leave such content out",
                    "unsupported_content",
                ) from exc
            raise _RefusalError(
                503, "masking failed, so nothing was sent", "masking_failed"
            ) from exc
        except TypeError as exc:
            raise _RefusalError(
                400, "the request has a field of the wrong type", "invalid_request"
            ) from exc

    def _check(self, masked: dict[str, Any], turn: MaskedTurn) -> None:
        """The last check: refuse to send a payload that still holds an original value."""
        leaked = leaked_tokens(masked, turn.mapping)
        if leaked:
            logger.error("refused to send: original values of %s still in the payload", leaked)
            raise _RefusalError(
                500,
                "the gateway found original values it would have sent, for "
                f"{', '.join(leaked)}; nothing was sent",
                "leak_prevented",
            )

    def _authorised(self, scope: Scope) -> bool:
        key = self._config.gateway_key
        if key is None:
            return True
        given = _header(scope, "authorization") or ""
        return hmac.compare_digest(given.encode(), f"Bearer {key}".encode())

    def _upstream_headers(self, scope: Scope) -> dict[str, str]:
        """The client's headers to pass on, with the key the provider should get."""
        headers = {
            name.decode("latin-1"): value.decode("latin-1")
            for name, value in scope.get("headers", [])
            if name.decode("latin-1").lower() not in _NOT_FORWARDED
        }
        if self._config.upstream_key:
            headers["authorization"] = f"Bearer {self._config.upstream_key}"
        elif self._config.gateway_key is None:
            client_key = _header(scope, "authorization")
            if client_key:
                headers["authorization"] = client_key
        return headers

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self._client.aclose()
                await send({"type": "lifespan.shutdown.complete"})
                return


def _route(path: str) -> str:
    """The path with the API's version taken off: clients set the base URL with or without it."""
    path = path.rstrip("/") or "/"
    return path[3:] if path.startswith("/v1/") else path


def _header(scope: Scope, name: str) -> str | None:
    for key, value in scope.get("headers", []):
        if key.decode("latin-1").lower() == name:
            return str(value.decode("latin-1"))
    return None


def _json_of(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        raise _RefusalError(
            502, "the provider answered with something that is not JSON", "bad_upstream"
        ) from None


def _unreachable(exc: httpx.HTTPError) -> _RefusalError:
    logger.warning("the provider could not be reached: %s", type(exc).__name__)
    return _RefusalError(
        502, f"the provider could not be reached ({type(exc).__name__})", "bad_upstream"
    )


class _ChatLines:
    """The event stream of a chat completion, restored line by line."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._chat = ChatStream(mapping)

    def line(self, line: str) -> list[bytes]:
        """One line of the provider's stream, as the lines to send for it."""
        if not line.startswith("data:"):
            return [f"{line}\n".encode()]
        data = line[5:].strip()
        if data == "[DONE]":
            return [*self.end(), b"data: [DONE]\n"]
        try:
            chunk = json.loads(data)
        except ValueError:
            return [f"{line}\n".encode()]
        return [f"data: {_dumps(self._chat.chunk(chunk))}\n".encode()]

    def end(self) -> list[bytes]:
        """What the stream still holds back, as one more event."""
        rest = self._chat.rest()
        return [] if rest is None else [f"data: {_dumps(rest)}\n\n".encode()]


class _ResponsesLines:
    """The event stream of a Responses API answer, restored event by event.

    An ``event:`` line waits for the ``data:`` line under it, since the restored event may have
    to be preceded by a delta of what was held back, which needs lines of its own.
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._stream = ResponsesStream(mapping)
        self._name: str | None = None
        self._named = False

    def line(self, line: str) -> list[bytes]:
        """One line of the provider's stream, as the lines to send for it."""
        if line.startswith("event:"):
            self._name, self._named = line[6:].strip(), True
            return []
        event: Any = None
        if line.startswith("data:"):
            try:
                event = json.loads(line[5:].strip())
            except ValueError:
                event = None
        if event is None:
            pending, self._name = self._name, None
            head = [] if pending is None else [f"event: {pending}\n".encode()]
            return [*head, f"{line}\n".encode()]
        events = self._stream.events(event)
        self._name = None
        return [*(self._event(item) + b"\n" for item in events[:-1]), self._event(events[-1])]

    def end(self) -> list[bytes]:
        """What the stream still holds back, as one more delta for each text cut short."""
        return [self._event(item) + b"\n" for item in self._stream.rest()]

    def _event(self, event: Any) -> bytes:
        name = event.get("type") if isinstance(event, dict) else None
        head = f"event: {name}\n" if self._named and name else ""
        return f"{head}data: {_dumps(event)}\n".encode()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


async def _send_json(
    send: Send, status: int, payload: Any, upstream: httpx.Headers | None = None
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode()
    headers = [(b"content-type", b"application/json")]
    if upstream is not None:
        headers += [
            (name.encode("latin-1"), value.encode("latin-1"))
            for name, value in upstream.items()
            if name.lower() not in _NOT_RETURNED and name.lower() != "content-type"
        ]
    headers.append((b"content-length", str(len(body)).encode()))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _send_raw(send: Send, response: httpx.Response) -> None:
    """Pass the provider's answer on as it came: an error, a list of models or embeddings."""
    body = response.content
    headers = [
        (name.encode("latin-1"), value.encode("latin-1"))
        for name, value in response.headers.items()
        if name.lower() not in _NOT_RETURNED
    ]
    headers.append((b"content-length", str(len(body)).encode()))
    await send({"type": "http.response.start", "status": response.status_code, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _send_error(send: Send, status: int, message: str, code: str) -> None:
    error = {"error": {"message": message, "type": "polanon_gateway_error", "code": code}}
    await _send_json(send, status, error)
