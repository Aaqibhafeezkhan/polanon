"""polanon: reversible PII anonymization for LLM calls, Polish-first. Zero web/DB dependencies.

This package is deliberately self-contained, so it can be dropped into any project: a CLI tool, a
LangChain callback, an MCP server, an Open WebUI filter. It imports no web or database framework,
and nothing from the hub built on it (enforced by tests/test_layering.py).

Public API (stable):

    from polanon import Anonymizer, default_guards

    anonymizer = Anonymizer(default_guards(with_ner=False))
    result = anonymizer.anonymize_sync("Jan Kowalski, PESEL 50810112354")  # or: await .anonymize()
    result.anonymized        # "<OSOBA_1>, PESEL <PESEL_1>"
    result.mapping           # {"<OSOBA_1>": "Jan Kowalski", "<PESEL_1>": "50810112354"}
    Anonymizer.restore("Dzień dobry <OSOBA_1>", result.mapping)   # original back in the answer
    restorer = Anonymizer.stream_restorer(result.mapping)         # for token-by-token streams

Runtime dependencies of the core: the standard library only. `NerGuard` needs spaCy and a language
model, the remote judges need httpx, and both are imported lazily so neither is required to mask
identifiers.
"""

import importlib.util
import re
from collections.abc import Iterable

from polanon._detection import custom_type_problem
from polanon.allowlist import WELL_KNOWN_NAMES, Allowlist
from polanon.guards.base import Guard
from polanon.guards.company_pl import CompanyGuard
from polanon.guards.denylist import DenylistGuard
from polanon.guards.international import InternationalGuard
from polanon.guards.names_pl import PersonNameGuard
from polanon.guards.recognizers_pl import RegexGuard
from polanon.guards.secrets import SecretsGuard
from polanon.judge.base import Judge, JudgeError, JudgeFinding
from polanon.judge.mock import MockJudge
from polanon.output_sweep import SweepReport, sweep
from polanon.pipeline import _anonymize_stage1, anonymize, mask_known_values
from polanon.policy import ENTITY_TYPES, TOKEN_PATTERN, token_for
from polanon.stream_restore import StreamRestorer
from polanon.types import AnonymizationResult, JudgeFallback, Span, merge_spans

__all__ = [
    "DEFAULT_ENABLED",
    "ENTITY_TYPES",
    "TOKEN_PATTERN",
    "WELL_KNOWN_NAMES",
    "Allowlist",
    "AnonymizationResult",
    "Anonymizer",
    "CompanyGuard",
    "DenylistGuard",
    "Guard",
    "InternationalGuard",
    "Judge",
    "JudgeError",
    "JudgeFallback",
    "JudgeFinding",
    "MockJudge",
    "PersonNameGuard",
    "RegexGuard",
    "SecretsGuard",
    "Span",
    "StreamRestorer",
    "SweepReport",
    "default_guards",
    "mask_known_values",
    "merge_spans",
    "sweep",
    "token_for",
]

__version__ = "0.2.2"

#: The entity types an Anonymizer masks unless told otherwise: every built-in type except DATE.
#: A date rarely identifies anyone on its own, and masking every deadline and invoice date leaves
#: the model unable to answer the question the text was sent with. Add it back with
#: ``DEFAULT_ENABLED | {"DATE"}``.
DEFAULT_ENABLED = frozenset(ENTITY_TYPES) - {"DATE"}

_NER_INSTALL = (
    'pip install "polanon[ner]", then python -m spacy download {model}; '
    "or call default_guards(with_ner=False) to mask with the rules alone"
)


def default_guards(
    *, denylist: Iterable[str] = (), with_ner: bool = True, spacy_model: str = "pl_core_news_md"
) -> list[Guard]:
    """Guards in recommended order. NER guard is imported lazily because spaCy is heavy/optional.

    The Polish rules come first, so where a Polish and a foreign rule read the same characters —
    a "PL" VAT number is also a NIP — the checksum-validated Polish reading keeps its label.

    Args:
        denylist: terms that must always be masked, as a list.
        with_ner: add the spaCy guard for places and the names no rule can see.
        spacy_model: the spaCy model that guard loads on first use.

    Raises:
        TypeError: when `denylist` is a single string, which would be read letter by letter.
        ModuleNotFoundError: when `with_ner` is on and spaCy or the model is not installed. It is
            raised here rather than on the first text to mask, with the command that fixes it.
    """
    if isinstance(denylist, str):
        # The term itself stays out of the message: it is what the caller wants kept hidden.
        raise TypeError("denylist takes a list of terms, not a single string")
    guards: list[Guard] = [
        RegexGuard(),
        SecretsGuard(),
        CompanyGuard(),
        PersonNameGuard(),
        InternationalGuard(),
        DenylistGuard(list(denylist)),
    ]
    if with_ner:
        _require_spacy(spacy_model)
        from polanon.guards.ner import NerGuard

        guards.append(NerGuard(spacy_model=spacy_model))
    return guards


def _require_spacy(spacy_model: str) -> None:
    """Fail now, with the command that fixes it, instead of on the first text to mask."""
    hint = _NER_INSTALL.format(model=spacy_model)
    if importlib.util.find_spec("spacy") is None:
        raise ModuleNotFoundError(f"the language model guard needs spaCy: {hint}")
    # A model can also be loaded from a path, which find_spec cannot check; spaCy reports that one.
    if spacy_model.isidentifier() and importlib.util.find_spec(spacy_model) is None:
        raise ModuleNotFoundError(f"the spaCy model {spacy_model!r} is not installed: {hint}")


class Anonymizer:
    """Facade over the anonymization pipeline.

    Create one instance per process and pass the per-conversation `mapping` on every call. An
    instance keeps no per-call state, so threads and coroutines can share it.
    """

    def __init__(
        self,
        guards: list[Guard],
        judge: Judge | None = None,
        *,
        enabled: Iterable[str] | None = None,
        judge_fallback: JudgeFallback | str = JudgeFallback.SEND_STAGE1,
        allowlist: Allowlist | None = None,
    ) -> None:
        """Configure the guards, the optional judge and the policy this facade applies.

        Args:
            guards: stage-1 detectors, run in order.
            judge: optional stage-2 reviewer.
            enabled: entity types to mask. When omitted, :data:`DEFAULT_ENABLED` plus the custom
                types the guards declare in ``Guard.entity_types``.
            judge_fallback: what to do when the judge fails: a :class:`JudgeFallback`, or its
                value as text ("send_stage1", "block").
            allowlist: public names left unmasked, such as :data:`WELL_KNOWN_NAMES`. Nothing is
                exempt when omitted.

        Raises:
            ValueError: for an entity type that is neither built in nor declared in a guard's
                ``entity_types``, such as a typo, which would otherwise switch masking off
                without a word; for a declared type that cannot label a token; and for an
                unknown fallback.
        """
        for guard in guards:
            for entity_type in sorted(guard.entity_types):
                problem = custom_type_problem(entity_type)
                if problem is not None:
                    raise ValueError(f"guard {guard.name!r} declares the entity type {problem}")
        declared = frozenset().union(*(guard.entity_types for guard in guards))
        self.guards = guards
        self.judge = judge
        self.enabled = set(DEFAULT_ENABLED | declared) if enabled is None else set(enabled)
        unknown = self.enabled - set(ENTITY_TYPES) - declared
        if unknown:
            raise ValueError(
                f"unknown entity types {sorted(unknown)}: nothing would mask them. The built-in "
                f"types are {', '.join(ENTITY_TYPES)}; a custom guard lists its own in "
                "Guard.entity_types."
            )
        self.judge_fallback = JudgeFallback(judge_fallback)
        self.allowlist = allowlist

    async def anonymize(
        self, text: str, mapping: dict[str, str] | None = None
    ) -> AnonymizationResult:
        """Mask `text`, reusing any tokens already allocated in `mapping`.

        The detectors run in a worker thread, so the event loop stays free. Without an event loop
        to await on, use :meth:`anonymize_sync`.
        """
        return await anonymize(
            text,
            enabled=self.enabled,
            guards=self.guards,
            judge=self.judge,
            existing_mapping=dict(mapping or {}),
            judge_fallback=self.judge_fallback,
            allowlist=self.allowlist,
        )

    def anonymize_sync(
        self, text: str, mapping: dict[str, str] | None = None
    ) -> AnonymizationResult:
        """Mask `text` without an event loop, reusing any tokens already allocated in `mapping`.

        The blocking twin of :meth:`anonymize`, for scripts, Django and Flask views, Celery tasks
        and notebooks. It returns the same result for the same input and never touches asyncio, so
        it works whether or not an event loop is running in the calling thread, and from several
        threads at once. Inside a coroutine prefer ``await anonymize()``, which keeps the work off
        the event loop.

        Args:
            text: the text to mask.
            mapping: token -> original value from earlier turns of the same conversation. It is
                read, never modified; the returned mapping holds it plus any new tokens.

        Returns:
            The masked text with its spans, entity counts and the conversation's full mapping.

        Raises:
            RuntimeError: when this anonymizer has a stage-2 judge. Judges are asynchronous, and
                running one behind a blocking call would hide its network I/O and its failures.
        """
        if self.judge is not None:
            raise RuntimeError(
                f"anonymize_sync() cannot run the stage-2 judge {self.judge.name!r}, because "
                "judges are asynchronous: use 'await anonymizer.anonymize(...)', or an Anonymizer "
                "without a judge."
            )
        return _anonymize_stage1(
            text,
            enabled=self.enabled,
            guards=self.guards,
            existing_mapping=dict(mapping or {}),
            allowlist=self.allowlist,
        )

    @staticmethod
    def restore(text: str, mapping: dict[str, str]) -> str:
        """Replace every known token with its original value. Unknown tokens are left untouched."""
        return re.sub(TOKEN_PATTERN, lambda m: mapping.get(m.group(0), m.group(0)), text)

    @staticmethod
    def stream_restorer(mapping: dict[str, str]) -> StreamRestorer:
        """Return a restorer for a streamed provider response."""
        return StreamRestorer(mapping)

    def sweep(self, restored_text: str, mapping: dict[str, str]) -> SweepReport:
        """Check a restored answer for invented tokens and unexpected personal data."""
        return sweep(restored_text, mapping, self.guards, self.enabled, self.allowlist)
