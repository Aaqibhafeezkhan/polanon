"""Guard interface.

A guard is a deterministic, local detector that returns spans in original text coordinates.
"""

from abc import ABC, abstractmethod

from polanon.types import Span


class Guard(ABC):
    """A deterministic, local detector returning spans in original text coordinates."""

    name: str = "guard"
    #: Entity types this guard produces beyond the built-in ones in `polanon.policy.ENTITY_TYPES`.
    #: An Anonymizer enables them by default and accepts them in `enabled`. A custom type becomes
    #: the label of its tokens, so it is spelled in uppercase letters and underscores.
    entity_types: frozenset[str] = frozenset()

    @abstractmethod
    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return spans for entity types in `enabled` only. Must be side-effect free and fast."""
