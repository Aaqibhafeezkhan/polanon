"""Stage-2 judge running on a cloud, OpenAI-compatible endpoint.

Use only when no local model is available. The review text is stage-1 masked, but it still leaves
the network, so this judge is a testing aid rather than a privacy control.
"""

from __future__ import annotations

import logging

import httpx

from polanon.judge.base import Judge, JudgeError, JudgeFinding
from polanon.judge.parsing import parse_findings, reply_content
from polanon.judge.prompt import JUDGE_SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class RemoteJudge(Judge):
    """Reviews masked text with a hosted model behind an OpenAI-compatible API."""

    name = "remote"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the endpoint; pass `client` to inject a transport in tests.

        The key is stripped: one read from a file or a Docker secret ends in a newline, which made
        every request fail with an error message that quoted the whole key.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model
        self.timeout = timeout
        self._client = client

    async def review(self, masked_text: str) -> list[JudgeFinding]:
        """Ask the hosted model which fragments still identify someone.

        Raises:
            JudgeError: on transport failure, timeout or unparseable output.
        """
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(masked_text)},
            ],
        }
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            content = reply_content(response.json(), "choices", 0, "message", "content")
        # The status or the type, never the httpx error, not even as the cause: an invalid header
        # error quotes the header, key included, and logger.exception prints the cause as well.
        except httpx.HTTPStatusError as exc:
            raise JudgeError(f"remote judge failed: HTTP {exc.response.status_code}") from None
        except (httpx.HTTPError, ValueError, RecursionError) as exc:
            raise JudgeError(f"remote judge failed: {type(exc).__name__}") from None
        finally:
            if self._client is None:
                await client.aclose()
        return parse_findings(content, masked_text)
