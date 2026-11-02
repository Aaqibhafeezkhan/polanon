"""Stage 2: an LLM 'judge' that reads ALREADY MASKED text and flags remaining identifying spans.

Contract:
- Input: text after stage 1 (contains <OSOBA_1> style tokens, never raw PII that stage 1 caught).
- Output: list of JudgeFinding(text=exact substring of the input, entity_type, reason).
The pipeline verifies each `text` occurs verbatim in the input; anything else is dropped.
- Must raise JudgeError on transport/timeouts/unparseable output; the pipeline applies the
Anonymizer's judge_fallback.

The prompt every LLM-backed judge shares lives in `judge/prompt.py`. The model answers ONLY with
JSON: {"findings":[{"text":"...","type":"PERSON|ORG|LOCATION|OTHER","reason":"..."}]}, and
`judge/parsing.py` drops anything that is not a verbatim quote of the reviewed text.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class JudgeError(RuntimeError):
    """Raised when a judge cannot produce a usable answer."""


@dataclass(frozen=True)
class JudgeFinding:
    """One fragment the judge believes still identifies a person or an organisation."""

    text: str
    entity_type: str  # PERSON | ORG | LOCATION | OTHER (OTHER maps to token label INNE)
    reason: str = ""


class Judge(ABC):
    """Reviews already-masked text and reports what still identifies someone."""

    name: str = "judge"

    @abstractmethod
    async def review(self, masked_text: str) -> list[JudgeFinding]:
        """Return additional identifying spans. Raise JudgeError on failure."""
