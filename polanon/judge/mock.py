"""A scripted judge for tests and demos.

It flags configured phrases and can be told to fail, which exercises the fallback policy.
"""

from polanon.judge.base import Judge, JudgeError, JudgeFinding


class MockJudge(Judge):
    """A judge with scripted answers, for tests and for the default demo configuration."""

    name = "mock"

    def __init__(self, phrases: dict[str, str] | None = None, fail: bool = False) -> None:
        """phrases: {"substring to flag": "PERSON|ORG|LOCATION|OTHER"}."""
        self.phrases = phrases or {}
        self.fail = fail

    async def review(self, masked_text: str) -> list[JudgeFinding]:
        """Return the configured findings, or fail when the mock is set to fail."""
        if self.fail:
            raise JudgeError("mock judge configured to fail")
        return [
            JudgeFinding(text=p, entity_type=t, reason="mock")
            for p, t in self.phrases.items()
            if p in masked_text
        ]
