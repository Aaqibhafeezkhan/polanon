"""The processes that mask calls side by side, for a gateway started with ``--workers N``.

One process masks one call at a time: the spaCy model is not shared safely between threads, and
Python runs one thread of a process at a time anyway. So a gateway with more than one worker
keeps serving HTTP, streams and restoring in its own process and hands the masking of each call
to one of N processes of its own, each with an anonymizer and a model of its own. A process that
is free takes the next call, so N calls at once are masked on N cores.

A process gets the request and sends back the masked request with the call's mapping, through
a pipe on the same machine; nothing is written to disk, and the process forgets the values when
the call is masked, as the gateway does when it is answered. Each process builds its anonymizer
itself, from the same settings as the gateway's own, and reads ``POLANON_KEY`` from the
environment it inherits: :class:`~polanon.Pseudonyms` are never pickled, so the key never travels,
and the same key gives the same stable tokens in every process.

Each process is a pool of one of its own, and the calls that wait for a process wait in the
gateway: so a process that stops takes with it at most the call it was masking, never another
process's call or one still waiting.

The module needs nothing beyond the standard library, so a process imports neither httpx nor
uvicorn.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import multiprocessing
import os
import signal
import sys
import threading
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Literal

from polanon import Anonymizer, MaskedTurn, MaskingUnavailableError
from polanon._matching import forget as forget_prepared_searches
from polanon.gateway.payloads import (
    mask_chat_request,
    mask_embeddings_request,
    mask_messages_request,
    mask_responses_request,
)

logger = logging.getLogger("polanon.gateway")

#: Builds the anonymizer of one process. It is pickled, so it is a function importable by its
#: name, or a :func:`functools.partial` of one, never a lambda.
Build = Callable[[], Anonymizer]
#: The endpoint a request was sent to: what decides how it is masked.
Kind = Literal["chat", "responses", "embeddings", "messages"]
#: What to do with content that cannot be masked, as :class:`~polanon.gateway.app.GatewayConfig`
#: says.
Unsupported = Literal["refuse", "drop", "pass"]
Masked = tuple[dict[str, Any], dict[str, str]]

#: An invented text each process masks before its first call, so the call does not wait for the
#: language model to load.
_WARM_UP = "Jan Kowalski mieszka w Gdańsku."
#: How many processes a call may be sent to: one that stopped before it began the call, and the
#: one that took its place.
_TRIES = 2
#: What asks every process of a console or of a service to stop: Ctrl+C, Ctrl+Break on Windows,
#: and the SIGTERM systemd sends each process of a service. The gateway answers it, finishes the
#: calls it is masking and then stops its workers itself, so a worker ignores it.
_STOPS: list[signal.Signals] = [signal.SIGINT, signal.SIGTERM]
if sys.platform == "win32":
    _STOPS.append(signal.SIGBREAK)

_BLOCKS: dict[str, Literal["raise", "drop", "keep"]] = {"drop": "drop", "pass": "keep"}


def new_turn(
    anonymizer: Anonymizer, unsupported: Unsupported, mapping: Mapping[str, str] | None = None
) -> MaskedTurn:
    """The turn of one call, which refuses, drops or keeps what cannot be masked as configured."""
    return MaskedTurn(anonymizer, mapping=mapping, unknown_blocks=_BLOCKS.get(unsupported, "raise"))


def masked_request(
    turn: MaskedTurn,
    kind: Kind,
    body: Mapping[str, Any],
    *,
    instructions: str | None,
    unsupported: Unsupported,
) -> dict[str, Any]:
    """Mask one request as its endpoint needs, in the gateway's process or in one of its workers.

    Raises:
        GatewayRequestError: for a request the gateway refuses to forward.
        MaskingUnavailableError: when masking fails, or for content that cannot be masked.
    """
    if kind == "chat":
        return mask_chat_request(turn, body, instructions=instructions)
    if kind == "responses":
        return mask_responses_request(
            turn,
            body,
            instructions=instructions,
            drop_unsupported=unsupported == "drop",
            pass_unsupported=unsupported == "pass",
        )
    if kind == "embeddings":
        return mask_embeddings_request(turn, body)
    return mask_messages_request(turn, body, instructions=instructions)


class MaskingPool:
    """N processes that mask calls side by side, each with an anonymizer of its own.

    Nothing starts when the pool is made: :meth:`wait_ready` starts every process and waits
    until each has loaded its model, and the first call starts them if nothing did. A call goes
    to a process that is free, or waits in the gateway until one is. A process that stops, killed
    for lack of memory say, fails the call it was masking, which is refused as masking that
    failed, and a new process takes its place; a call it had not begun goes to the new one.
    """

    def __init__(self, build: Build, workers: int) -> None:
        """Set the pool up.

        Args:
            build: makes the anonymizer of one process, in that process.
            workers: how many processes mask at once, 2 or more.

        Raises:
            ValueError: for fewer than two workers: one masks in the gateway's own process.
        """
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 2:
            raise ValueError(f"a masking pool has 2 or more workers, not {workers!r}")
        self._build = build
        self._workers = workers
        self._slots: list[_Worker] | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._free: deque[int] = deque(range(workers))
        self._waiting: deque[Future[int]] = deque()
        self._numbers = itertools.count(1)

    @property
    def workers(self) -> int:
        """How many processes mask at once."""
        return self._workers

    def wait_ready(self, timeout: float | None = None) -> set[int]:
        """Start every process, wait until each has built its anonymizer, and return their ids.

        Raises:
            TimeoutError: when one is not ready in `timeout` seconds.
            BrokenProcessPool: when a process could not start or build its anonymizer.
            MaskingUnavailableError: once the pool is closed.
        """
        return {worker.ready.result(timeout) for worker in self._running()}

    async def mask(
        self,
        kind: Kind,
        body: Mapping[str, Any],
        *,
        instructions: str | None,
        unsupported: Unsupported,
    ) -> Masked:
        """Mask one request in the next free process: the masked request and the call's mapping.

        Raises:
            GatewayRequestError: for a request the gateway refuses to forward.
            MaskingUnavailableError: when masking fails, when the request holds content that
                cannot be masked, when the process stopped while it masked the call, and once
                the pool is closed.
        """
        number = next(self._numbers)
        slot = await self._take()
        try:
            return await self._mask_in(slot, number, kind, body, instructions, unsupported)
        finally:
            self._give(slot)

    def close(self) -> None:
        """Stop every process once it has masked its call, and refuse the calls still waiting."""
        with self._lock:
            self._closed = True
            slots, self._slots = self._slots or [], None
        for worker in slots:
            worker.executor.shutdown(wait=True, cancel_futures=True)

    async def _mask_in(
        self,
        slot: int,
        number: int,
        kind: Kind,
        body: Mapping[str, Any],
        instructions: str | None,
        unsupported: Unsupported,
    ) -> Masked:
        """Mask the call numbered `number` in the process of `slot`, or in the one after it."""
        call = (number, kind, body, instructions, unsupported)
        stopped: BrokenProcessPool | None = None
        for _ in range(_TRIES):
            worker = self._running()[slot]
            try:
                return await asyncio.wrap_future(worker.executor.submit(_mask, *call))
            except BrokenProcessPool as exc:
                self._replace(slot, worker)
                if worker.began.value == number:
                    # The call itself may have stopped it, by the memory it took: sent again, it
                    # would stop the next process too.
                    raise MaskingUnavailableError(
                        "the masking process stopped while it masked the call"
                    ) from exc
                stopped = exc
        raise MaskingUnavailableError(
            "a masking process stopped before the call was masked"
        ) from stopped

    def _running(self) -> list[_Worker]:
        with self._lock:
            if self._closed:
                raise MaskingUnavailableError("the masking processes have stopped")
            if self._slots is None:
                self._slots = [_Worker(self._build) for _ in range(self._workers)]
            return self._slots

    def _replace(self, slot: int, stopped: _Worker) -> None:
        """Start a new process in the place of one that stopped."""
        with self._lock:
            if not self._closed and self._slots is not None and self._slots[slot] is stopped:
                logger.error("a masking process stopped: starting another in its place")
                self._slots[slot] = _Worker(self._build)
        stopped.executor.shutdown(wait=False, cancel_futures=True)

    async def _take(self) -> int:
        """A process that is free, once there is one; calls get them in the order they came."""
        with self._lock:
            if self._free:
                return self._free.popleft()
            waiter: Future[int] = Future()
            self._waiting.append(waiter)
        try:
            return await asyncio.wrap_future(waiter)
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                self._give(waiter.result())
            raise

    def _give(self, slot: int) -> None:
        """Hand a process that is free to the call that has waited longest, if one waits."""
        with self._lock:
            while self._waiting:
                waiter = self._waiting.popleft()
                if waiter.set_running_or_notify_cancel():
                    waiter.set_result(slot)
                    return
            self._free.append(slot)


class _Worker:
    """One masking process, in a pool of its own, and the number of the last call it began."""

    def __init__(self, build: Build) -> None:
        context = multiprocessing.get_context("spawn")
        self.began = context.RawValue("q", 0)
        self.executor = ProcessPoolExecutor(
            1, mp_context=context, initializer=_start, initargs=(build, self.began)
        )
        # Starts the process now, so it loads its model before a call needs it.
        self.ready: Future[int] = self.executor.submit(_ready)


#: The anonymizer of this process and where it notes the call it masks, once :func:`_start` ran.
_state: tuple[Anonymizer, Any] | None = None


def _start(build: Build, began: Any) -> None:
    """Make a worker ready: its anonymizer, its model loaded, and an end with the gateway.

    Ctrl+C and Ctrl+Break reach every process of a console, and systemd stops a service by
    sending SIGTERM to each of its processes; the gateway stops its workers itself, after the
    calls they mask. A gateway killed outright cannot, so each worker ends when it does.
    """
    global _state
    for stop in _STOPS:
        signal.signal(stop, signal.SIG_IGN)
    _exit_with_parent()
    anonymizer = build()
    anonymizer.anonymize_sync(_WARM_UP, {})
    forget_prepared_searches()
    _state = (anonymizer, began)


def _exit_with_parent() -> None:
    """End this process when the one that started it ends, however it ended."""
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        os._exit(0)

    threading.Thread(target=watch, name="polanon-parent-watch", daemon=True).start()


def _ready() -> int:
    """This worker's process id: its pool runs nothing before the worker has its anonymizer."""
    return os.getpid()


def _mask(
    number: int,
    kind: Kind,
    body: Mapping[str, Any],
    instructions: str | None,
    unsupported: Unsupported,
) -> Masked:
    """Mask one request in a worker, and forget the call's values once it is masked."""
    if _state is None:
        raise MaskingUnavailableError("the masking process has no anonymizer")
    anonymizer, began = _state
    began.value = number
    turn = new_turn(anonymizer, unsupported)
    try:
        masked = masked_request(
            turn, kind, body, instructions=instructions, unsupported=unsupported
        )
        return masked, turn.mapping
    finally:
        forget_prepared_searches()
