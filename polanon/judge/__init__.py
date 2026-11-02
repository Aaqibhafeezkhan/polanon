"""Stage-2 judges: a small model reviews the masked text for what still identifies someone.

``OllamaJudge`` (a local model) and ``RemoteJudge`` (an OpenAI-compatible endpoint) need httpx,
installed by ``pip install "polanon[judge]"``. They are imported on first use, so the rest of the
library keeps running on the standard library alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polanon.judge.ollama import OllamaJudge
    from polanon.judge.remote import RemoteJudge

__all__ = ["OllamaJudge", "RemoteJudge"]


def __getattr__(name: str) -> Any:
    """Import a judge on first use, naming the extra to install when httpx is missing."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import httpx  # noqa: F401
    except ImportError as exc:
        raise ImportError(f'{name} needs httpx: pip install "polanon[judge]"') from exc
    from polanon.judge.ollama import OllamaJudge
    from polanon.judge.remote import RemoteJudge

    return {"OllamaJudge": OllamaJudge, "RemoteJudge": RemoteJudge}[name]
