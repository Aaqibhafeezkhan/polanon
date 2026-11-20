"""The gateway served over HTTP: a plain ASGI application in front of an OpenAI-compatible API.

It serves ``/v1/chat/completions``, ``/v1/responses``, ``/v1/embeddings`` and ``/v1/models``,
and refuses every other endpoint: a request the gateway cannot mask is never forwarded as it came.
Each call is masked by its own :class:`~polanon.MaskedTurn`, checked once more for any original
value, sent to the provider and restored on the way back, streamed or not.

The application keeps nothing between calls and writes nothing to disk. It logs the path, the
status, the time taken and how many values of each type were masked, never a value. It masks one
call at a time in its own process, or, with ``workers`` above one, N calls at once in the
processes of :mod:`polanon.gateway.workers`.
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
from polanon._matching import forget as forget_prepared_searches
from polanon.gateway.payloads import (
    ChatStream,
    GatewayRequestError,
    MessagesStream,
    ResponsesStream,
    leaked_tokens,
    restore_chat_response,
    restore_messages_response,
    restore_responses_response,
)
from polanon.gateway.workers import Build, Kind, MaskingPool, masked_request, new_turn

logger = logging.getLogger("polanon.gateway")

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

#: Headers of the client's request that are never passed on: the connection's own, the body's,
#: cookies, and the client's credentials, which the gateway handles itself.
_NOT_FORWARDED = frozenset(
    {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "upgrade"}
    | {"accept-encoding", "authorization", "proxy-authorization", "cookie", "x-api-key"}
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
            call, "drop" the content and send the rest, or "pass" it to the provider as it is,
            unmasked.
        max_body: the largest request accepted, in bytes.
        timeout: seconds to wait for the provider.
        pii_map: whether a client that sends ``"return_pii_map": true`` gets the call's mapping,
            token to original, with the answer. The mapping holds what the client sent, and with
            stable tokens it tells which token stands for which value, so it is off by default.
        workers: how many calls are masked at once. 1 masks one at a time in the gateway's own
            process; more mask in as many processes of their own, each with its own anonymizer
            and language model, which the gateway's `build` makes.
    """

    upstream: str
    upstream_key: str | None = field(default=None, repr=False)
    gateway_key: str | None = field(default=None, repr=False)
    instructions: str | None = "pl"
    unsupported: Literal["refuse", "drop", "pass"] = "refuse"
    max_body: int = 10 * 1024 * 1024
    timeout: float = 600.0
    pii_map: bool = False
    workers: int = 1

    def __post_init__(self) -> None:
        """Refuse a number of workers that is not one or more."""
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError(f"workers is 1 or more, not {self.workers!r}")


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
    server, as ``polanon-gateway`` does with uvicorn. With ``config.workers`` above one, the
    server's lifespan events start the masking processes and stop them; without them the first
    call starts them, and :meth:`aclose` stops them.
    """

    def __init__(
        self,
        anonymizer: Anonymizer,
        config: GatewayConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        build: Build | None = None,
    ) -> None:
        """Set the gateway up.

        Args:
            anonymizer: the guards and the policy every call is masked with. It must have no
                stage-2 judge, which a turn cannot run. With more than one worker it masks
                nothing: the workers' anonymizers do, and the language model of this one is never
                loaded.
            config: where calls go and who may make them.
            transport: what carries calls to the provider; tests pass one that answers itself.
            build: with more than one worker, makes the anonymizer of each masking process, in
                that process, as `anonymizer` was made. It is pickled, so it is a function
                importable by its name or a :func:`functools.partial` of one, and it reads
                ``POLANON_KEY`` from the environment, never from its arguments: a
                :class:`~polanon.Pseudonyms` cannot be pickled, so the key never travels.

        Raises:
            ValueError: for more than one worker without `build`.
        """
        self._anonymizer = anonymizer
        self._config = config
        self._instructions = (
            token_instructions(config.instructions) if config.instructions else None
        )
        self._client = httpx.AsyncClient(
            base_url=config.upstream, timeout=config.timeout, transport=transport
        )
        # With one worker, one call is masked at a time: the language model is not shared safely
        # between threads, and masking runs in a thread so the event loop keeps serving streams
        # meanwhile. With more, each process masks one call at a time.
        self._masking = asyncio.Lock()
        self._pool: MaskingPool | None = None
        if config.workers > 1:
            if build is None:
                raise ValueError(
                    "a gateway with more than one worker needs build, which makes the "
                    "anonymizer of each masking process"
                )
            self._pool = MaskingPool(build, config.workers)

    async def aclose(self) -> None:
        """Stop the masking processes and close the connections to the provider."""
        if self._pool is not None:
            await asyncio.to_thread(self._pool.close)
        await self._client.aclose()

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
            elif path == "/messages" and method == "POST":
                status, counts = await self._messages(scope, receive, send, "messages")
            elif path == "/messages/count_tokens" and method == "POST":
                status, counts = await self._messages(scope, receive, send, "messages/count_tokens")
            elif method == "GET" and (path == "/models" or path.startswith("/models/")):
                status = await self._models(scope, send, path)
            else:
                raise _RefusalError(
                    404,
                    f"the polanon gateway does not serve {method} {scope['path']}: it masks "
                    "/v1/chat/completions, /v1/responses, /v1/embeddings and /v1/messages only",
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
        # The search for known values is prepared once per call; it holds the call's values.
        forget_prepared_searches()

    async def _chat(self, scope: Scope, receive: Receive, send: Send) -> tuple[int, dict[str, int]]:
        body, wants_map = self._map_asked(await self._body(receive))
        turn, masked = await self._mask("chat", body)
        headers = self._upstream_headers(scope)
        if body.get("stream"):
            lines = _ChatLines(turn.mapping, _returned(turn) if wants_map else None)
            status = await self._stream("chat/completions", masked, headers, send, lines)
            return status, turn.counts()
        response = await self._post("chat/completions", masked, headers)
        if response.is_success:
            restored = restore_chat_response(turn, _json_of(response))
            restored = _with_mapping(restored, turn) if wants_map else restored
            await _send_json(send, response.status_code, restored, response.headers)
        else:
            await _send_raw(send, response)
        return response.status_code, turn.counts()

    async def _responses(
        self, scope: Scope, receive: Receive, send: Send
    ) -> tuple[int, dict[str, int]]:
        body, wants_map = self._map_asked(await self._body(receive))
        turn, masked = await self._mask("responses", body)
        headers = self._upstream_headers(scope)
        if body.get("stream"):
            lines = _ResponsesLines(turn.mapping, _returned(turn) if wants_map else None)
            status = await self._stream("responses", masked, headers, send, lines)
            return status, turn.counts()
        response = await self._post("responses", masked, headers)
        if response.is_success:
            restored = restore_responses_response(turn, _json_of(response))
            restored = _with_mapping(restored, turn) if wants_map else restored
            await _send_json(send, response.status_code, restored, response.headers)
        else:
            await _send_raw(send, response)
        return response.status_code, turn.counts()

    async def _embeddings(
        self, scope: Scope, receive: Receive, send: Send
    ) -> tuple[int, dict[str, int]]:
        body, wants_map = self._map_asked(await self._body(receive))
        turn, masked = await self._mask("embeddings", body)
        response = await self._post("embeddings", masked, self._upstream_headers(scope))
        if response.is_success and wants_map:
            answer = _with_mapping(_json_of(response), turn)
            await _send_json(send, response.status_code, answer, response.headers)
        else:
            await _send_raw(send, response)
        return response.status_code, turn.counts()

    async def _messages(
        self, scope: Scope, receive: Receive, send: Send, endpoint: str
    ) -> tuple[int, dict[str, int]]:
        """Anthropic's Messages API, and the count of its tokens, which is asked masked too."""
        body, wants_map = self._map_asked(await self._body(receive))
        turn, masked = await self._mask("messages", body)
        headers = self._upstream_headers(scope, anthropic=True)
        if endpoint == "messages" and body.get("stream"):
            lines = _MessagesLines(turn.mapping, _returned(turn) if wants_map else None)
            status = await self._stream(endpoint, masked, headers, send, lines)
            return status, turn.counts()
        response = await self._post(endpoint, masked, headers)
        if response.is_success and endpoint == "messages":
            restored = restore_messages_response(turn, _json_of(response))
            restored = _with_mapping(restored, turn) if wants_map else restored
            await _send_json(send, response.status_code, restored, response.headers)
        else:
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
        error status: the stream simply ends, as the provider's own would, after the text it
        still held back.
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
            for out in lines.end():
                await send({"type": "http.response.body", "body": out, "more_body": True})
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

    def _map_asked(self, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Take ``return_pii_map`` off the request, and say whether the answer carries the mapping.

        The provider never sees the field. A value that is not true or false is refused before
        anything is sent, and so is true where the operator did not start the gateway with
        ``--pii-map``.
        """
        if "return_pii_map" not in body:
            return body, False
        wanted = body["return_pii_map"]
        if not isinstance(wanted, bool):
            raise _RefusalError(400, "return_pii_map is true or false", "invalid_request")
        if wanted and not self._config.pii_map:
            raise _RefusalError(
                400,
                "this gateway does not return mappings: its operator can start it with --pii-map",
                "pii_map_off",
            )
        return {key: value for key, value in body.items() if key != "return_pii_map"}, wanted

    async def _mask(self, kind: Kind, body: dict[str, Any]) -> tuple[MaskedTurn, dict[str, Any]]:
        """Mask one request, check it, and turn a refusal into an error for the client.

        With one worker the request is masked in a thread of this process, one call at a time;
        with more, in the next free masking process, which sends back the call's mapping, and the
        turn that restores the answer is made here from it.
        """
        unsupported = self._config.unsupported
        try:
            if self._pool is None:
                turn = new_turn(self._anonymizer, unsupported)
                async with self._masking:
                    masked = await asyncio.to_thread(
                        masked_request,
                        turn,
                        kind,
                        body,
                        instructions=self._instructions,
                        unsupported=unsupported,
                    )
            else:
                masked, mapping = await self._pool.mask(
                    kind, body, instructions=self._instructions, unsupported=unsupported
                )
                turn = new_turn(self._anonymizer, unsupported, mapping)
        except GatewayRequestError as exc:
            raise _RefusalError(400, str(exc), "invalid_request") from exc
        except MaskingUnavailableError as exc:
            if "cannot be masked" in str(exc):
                what = str(exc).split(";", 1)[0]
                raise _RefusalError(
                    400,
                    f"{what}, so the call was not sent; start the gateway with "
                    "--drop-unsupported to leave such content out, or with --pass-unsupported "
                    "to send it unmasked",
                    "unsupported_content",
                ) from exc
            raise _RefusalError(
                503, "masking failed, so nothing was sent", "masking_failed"
            ) from exc
        except TypeError as exc:
            raise _RefusalError(
                400, "the request has a field of the wrong type", "invalid_request"
            ) from exc
        self._check(masked, turn)
        return turn, masked

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
        """Whether the client sent the gateway key, as a bearer token or as an x-api-key."""
        key = self._config.gateway_key
        if key is None:
            return True
        bearer = _header(scope, "authorization") or ""
        api_key = _header(scope, "x-api-key") or ""
        return hmac.compare_digest(bearer.encode(), f"Bearer {key}".encode()) or (
            bool(api_key) and hmac.compare_digest(api_key.encode(), key.encode())
        )

    def _upstream_headers(self, scope: Scope, *, anthropic: bool = False) -> dict[str, str]:
        """The client's headers to pass on, with the key the provider should get.

        Anthropic's API takes its key as ``x-api-key``, the OpenAI API as a bearer token. With a
        gateway key the client's own key is the gateway's, and never goes on.
        """
        headers = {
            name.decode("latin-1"): value.decode("latin-1")
            for name, value in scope.get("headers", [])
            if name.decode("latin-1").lower() not in _NOT_FORWARDED
        }
        if self._config.upstream_key:
            if anthropic:
                headers["x-api-key"] = self._config.upstream_key
            else:
                headers["authorization"] = f"Bearer {self._config.upstream_key}"
        elif self._config.gateway_key is None:
            for name in ("authorization", "x-api-key"):
                client_key = _header(scope, name)
                if client_key:
                    headers[name] = client_key
        return headers

    async def _start_workers(self) -> str | None:
        """Start the masking processes and wait until each has loaded its model.

        The server opens its port only after this, so a call never waits for a model to load,
        and a gateway whose workers cannot start does not start either, rather than refuse every
        call while it reports itself healthy.

        Returns:
            None once the workers are ready, or why they could not start.
        """
        if self._pool is None:
            return None
        try:
            processes = await asyncio.to_thread(self._pool.wait_ready)
        except Exception as exc:
            logger.error("the masking processes could not start: %s", type(exc).__name__)
            await asyncio.to_thread(self._pool.close)
            return f"the masking processes could not start ({type(exc).__name__})"
        logger.info("%d masking processes ready", len(processes))
        return None

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                failed = await self._start_workers()
                if failed:
                    await send({"type": "lifespan.startup.failed", "message": failed})
                    return
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.aclose()
                await send({"type": "lifespan.shutdown.complete"})
                return


def _route(path: str) -> str:
    """The path with the API's version taken off: clients set the base URL with or without it.

    Anthropic's SDKs add /v1 themselves, so one given the base URL an OpenAI client takes, ending
    in /v1, asks for /v1/v1/messages: the version comes off as many times as it is there.
    """
    path = path.rstrip("/") or "/"
    while path.startswith("/v1/"):
        path = path[3:]
    return path


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


def _returned(turn: MaskedTurn) -> dict[str, str]:
    """The call's mapping as a client gets it back: a token the client wrote itself is left out."""
    return {token: value for token, value in turn.mapping.items() if value != token}


def _with_mapping(answer: Any, turn: MaskedTurn) -> Any:
    """The provider's answer with the call's mapping under ``return_pii_map``."""
    return {**answer, "return_pii_map": _returned(turn)} if isinstance(answer, dict) else answer


class _ChatLines:
    """The event stream of a chat completion, restored line by line.

    A mapping asked for goes last, before ``[DONE]``, in a chunk of its own with no choices, as
    the usage of a stream does: every event stays one JSON object.
    """

    def __init__(self, mapping: Mapping[str, str], returned: dict[str, str] | None = None) -> None:
        self._chat = ChatStream(mapping)
        self._returned = returned
        self._last: dict[str, Any] = {}

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
        if isinstance(chunk, dict):
            self._last = {key: chunk[key] for key in _CHUNK_HEAD if key in chunk}
        return [f"data: {_dumps(self._chat.chunk(chunk))}\n".encode()]

    def end(self) -> list[bytes]:
        """What the stream still holds back, as one more event, and the mapping if asked for."""
        rest = self._chat.rest()
        out = [] if rest is None else [f"data: {_dumps(rest)}\n\n".encode()]
        if self._returned is not None:
            chunk = {**self._last, "choices": [], "return_pii_map": self._returned}
            out.append(f"data: {_dumps(chunk)}\n\n".encode())
            self._returned = None
        return out


#: What a chunk of the mapping repeats of the provider's chunks, so a client that reads each as a
#: chat completion chunk finds its fields.
_CHUNK_HEAD = ("id", "object", "created", "model")


class _ResponsesLines:
    """The event stream of a Responses API answer, restored event by event.

    An ``event:`` line waits for the ``data:`` line under it, since the restored event may have
    to be preceded by a delta of what was held back, which needs lines of its own.
    """

    #: The event that ends an answer, where a mapping asked for goes.
    _LAST = frozenset({"response.completed", "response.incomplete"})

    def __init__(self, mapping: Mapping[str, str], returned: dict[str, str] | None = None) -> None:
        self._stream = ResponsesStream(mapping)
        self._returned = returned
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
        last = events[-1]
        if self._returned is not None and isinstance(last, dict) and last.get("type") in self._LAST:
            events[-1] = {**last, "return_pii_map": self._returned}
            self._returned = None
        return [*(self._event(item) + b"\n" for item in events[:-1]), self._event(events[-1])]

    def end(self) -> list[bytes]:
        """What the stream still holds back, as one more delta for each text cut short."""
        return [self._event(item) + b"\n" for item in self._stream.rest()]

    def _event(self, event: Any) -> bytes:
        name = event.get("type") if isinstance(event, dict) else None
        head = f"event: {name}\n" if self._named and name else ""
        return f"{head}data: {_dumps(event)}\n".encode()


class _MessagesLines(_ResponsesLines):
    """The event stream of an Anthropic Messages answer, restored event by event."""

    _LAST = frozenset({"message_stop"})

    def __init__(self, mapping: Mapping[str, str], returned: dict[str, str] | None = None) -> None:
        super().__init__(mapping, returned)
        self._stream = MessagesStream(mapping)  # type: ignore[assignment]


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
