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

The module needs nothing beyond the standard library, so a process imports neither httpx nor
uvicorn.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import os
import signal
import sys
import threading
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

#: An invented text each process masks before its first call, so the call does not wait for the
#: language model to load.
_WARM_UP = "Jan Kowalski mieszka w Gdańsku."
#: How long a process that is ready waits for the others, in seconds, before it says so alone.
_READY_WAIT = 120.0
#: The most processes a pool may have on Windows, which waits on at most 63 handles at once.
_MOST_ON_WINDOWS = 61

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
    until each has loaded its model, and the first call starts them if nothing did. A process
    that stops, killed for lack of memory say, fails the calls it and the others were masking,
    which are refused as masking that failed, and the next call starts a new set of processes.
    """

    def __init__(self, build: Build, workers: int) -> None:
        """Set the pool up.

        Args:
            build: makes the anonymizer of one process, in that process.
            workers: how many processes mask at once, 2 or more.

        Raises:
            ValueError: for fewer than two workers, one masks in the gateway's own process, and
                for more than Windows lets a pool of processes have.
        """
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 2:
            raise ValueError(f"a masking pool has 2 or more workers, not {workers!r}")
        if sys.platform == "win32" and workers > _MOST_ON_WINDOWS:
            raise ValueError(f"on Windows a masking pool has at most {_MOST_ON_WINDOWS} workers")
        self._build = build
        self._workers = workers
        self._executor: ProcessPoolExecutor | None = None
        self._ready: list[Future[int]] = []

    @property
    def workers(self) -> int:
        """How many processes mask at once."""
        return self._workers

    def wait_ready(self, timeout: float | None = None) -> set[int]:
        """Start every process, wait until each has built its anonymizer, and return their ids.

        Raises:
            TimeoutError: when they are not ready in `timeout` seconds.
            BrokenProcessPool: when a process could not start or build its anonymizer.
        """
        self._running()
        return {future.result(timeout) for future in self._ready}

    async def mask(
        self,
        kind: Kind,
        body: Mapping[str, Any],
        *,
        instructions: str | None,
        unsupported: Unsupported,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Mask one request in the next free process: the masked request and the call's mapping.

        Raises:
            GatewayRequestError: for a request the gateway refuses to forward.
            MaskingUnavailableError: when masking fails, when the request holds content that
                cannot be masked, and when a process stopped before the call was masked.
        """
        executor = self._running()
        try:
            future = executor.submit(_mask, kind, body, instructions, unsupported)
            return await asyncio.wrap_future(future)
        except BrokenProcessPool as exc:
            self._replace(executor)
            raise MaskingUnavailableError(
                "a masking process stopped before the call was masked"
            ) from exc

    def close(self) -> None:
        """Stop every process, and drop the calls that wait for one."""
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _running(self) -> ProcessPoolExecutor:
        if self._executor is None:
            context = multiprocessing.get_context("spawn")
            self._executor = ProcessPoolExecutor(
                self._workers,
                mp_context=context,
                initializer=_start,
                initargs=(self._build, context.Barrier(self._workers)),
            )
            # One task a process: each waits at the barrier until all are ready, so a process
            # cannot take two of them, and every one of the workers is started now.
            self._ready = [self._executor.submit(_ready) for _ in range(self._workers)]
        return self._executor

    def _replace(self, broken: ProcessPoolExecutor) -> None:
        """Let the next call start new processes; the calls of the broken ones are refused."""
        if self._executor is broken:
            logger.error("a masking process stopped: starting a new set of %d", self._workers)
            self._executor = None
            broken.shutdown(wait=False, cancel_futures=True)


#: The anonymizer of this process and the barrier of its pool, once :func:`_start` has run.
_state: tuple[Anonymizer, Any] | None = None


def _start(build: Build, barrier: Any) -> None:
    """Make a worker ready: its anonymizer, its model loaded, and an end with the gateway.

    Ctrl+C reaches every process of a console, and the gateway stops its workers itself, after
    the calls they mask; a gateway killed outright cannot, so each worker ends when it does.
    """
    global _state
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _exit_with_parent()
    anonymizer = build()
    anonymizer.anonymize_sync(_WARM_UP, {})
    forget_prepared_searches()
    _state = (anonymizer, barrier)


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
    """Wait until every worker of the pool is ready, and return this one's process id."""
    if _state is not None:
        with contextlib.suppress(threading.BrokenBarrierError):
            _state[1].wait(_READY_WAIT)
    return os.getpid()


def _mask(
    kind: Kind, body: Mapping[str, Any], instructions: str | None, unsupported: Unsupported
) -> tuple[dict[str, Any], dict[str, str]]:
    """Mask one request in a worker, and forget the call's values once it is masked."""
    if _state is None:
        raise MaskingUnavailableError("the masking process has no anonymizer")
    turn = new_turn(_state[0], unsupported)
    try:
        masked = masked_request(
            turn, kind, body, instructions=instructions, unsupported=unsupported
        )
        return masked, turn.mapping
    finally:
        forget_prepared_searches()
