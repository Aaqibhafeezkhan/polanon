"""Stage-2 judge running on a local Ollama server.

This is the production configuration: the review never leaves the company network, so the judge can
be shown text that stage 1 might have missed something in.
"""

from __future__ import annotations

import logging

import httpx

from polanon.judge.base import Judge, JudgeError, JudgeFinding
from polanon.judge.parsing import parse_findings, reply_content
from polanon.judge.prompt import JUDGE_SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class OllamaJudge(Judge):
    """Reviews masked text with a small local model served by Ollama."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b",
        timeout: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the endpoint; pass `client` to inject a transport in tests."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client

    async def review(self, masked_text: str) -> list[JudgeFinding]:
        """Ask the local model which fragments still identify someone.

        Raises:
            JudgeError: on transport failure, timeout or unparseable output.
        """
        payload = {
            "model": self.model,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(masked_text)},
            ],
        }
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            content = reply_content(response.json(), "message", "content")
        # The status or the type, never the httpx error, not even as the cause: an error can quote
        # the request it failed on, credentials in the URL included, and logger.exception prints
        # the cause as well.
        except httpx.HTTPStatusError as exc:
            raise JudgeError(f"ollama judge failed: HTTP {exc.response.status_code}") from None
        except (httpx.HTTPError, ValueError, RecursionError) as exc:
            raise JudgeError(f"ollama judge failed: {type(exc).__name__}") from None
        finally:
            if self._client is None:
                await client.aclose()
        return parse_findings(content, masked_text)
