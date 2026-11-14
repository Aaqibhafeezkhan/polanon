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

    turn = anonymizer.turn(known=["Klub Orlik"])   # one call: history, question, tool results
    messages = turn.mask_messages(history)         # Anthropic or OpenAI messages, one mapping
    turn.mask_data(tool_result)                    # MaskingUnavailableError: send nothing
    turn.restore_data(tool_input), turn.restore(answer), turn.stream_restorer()

Runtime dependencies of the core: the standard library only. `NerGuard` needs spaCy and a language
model, the remote judges need httpx, and both are imported lazily so neither is required to mask
identifiers.
"""

import importlib.util
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from polanon._detection import custom_type_problem
from polanon.allowlist import WELL_KNOWN_NAMES, Allowlist
from polanon.guards.base import Guard
from polanon.guards.company_pl import CompanyGuard
from polanon.guards.denylist import DenylistGuard
from polanon.guards.international import InternationalGuard
from polanon.guards.names_pl import PersonNameGuard
from polanon.guards.recognizers_pl import RegexGuard
from polanon.guards.secrets import SecretsGuard
from polanon.index import (
    InMemoryMappingStore,
    MappingStore,
    MaskedChunks,
    StoredValue,
    index_chunks,
    mask_chunks,
    mask_query,
    stored_mapping,
    tokens_in,
)
from polanon.judge.base import Judge, JudgeError, JudgeFinding
from polanon.judge.mock import MockJudge
from polanon.output_sweep import SweepReport, sweep
from polanon.pipeline import _anonymize_stage1, anonymize, mask_known_values
from polanon.policy import ENTITY_TYPES, TOKEN_PATTERN, token_for
from polanon.pseudonyms import PseudonymCollisionError, Pseudonyms
from polanon.stream_restore import StreamRestorer
from polanon.turn import MaskedTurn, UnknownBlocks, token_instructions
from polanon.types import (
    AnonymizationResult,
    JudgeFallback,
    MaskingUnavailableError,
    NerModelUnavailableError,
    Span,
    merge_spans,
)

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
    "InMemoryMappingStore",
    "InternationalGuard",
    "Judge",
    "JudgeError",
    "JudgeFallback",
    "JudgeFinding",
    "MappingStore",
    "MaskedChunks",
    "MaskedTurn",
    "MaskingUnavailableError",
    "MockJudge",
    "NerModelUnavailableError",
    "PersonNameGuard",
    "PseudonymCollisionError",
    "Pseudonyms",
    "RegexGuard",
    "SecretsGuard",
    "Span",
    "StoredValue",
    "StreamRestorer",
    "SweepReport",
    "default_guards",
    "mask_known_values",
    "merge_spans",
    "ner_available",
    "stored_mapping",
    "sweep",
    "token_for",
    "token_instructions",
    "tokens_in",
]

__version__ = "0.3.4"

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
        spacy_model: the installed spaCy model's package name (underscores or hyphens), or a
            model directory. The directory must contain meta.json and config.cfg; the pipeline
            itself loads on first use.

    Raises:
        TypeError: when `denylist` is a single string, which would be read letter by letter.
        NerModelUnavailableError: when `with_ner` is on and spaCy or the model is not installed.
            It is raised here rather than on the first text to mask, with the command that fixes
            it, and it is both a ModuleNotFoundError and a MaskingUnavailableError.
            Also raised for a model directory without `meta.json` or `config.cfg`.
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
        try:
            _require_spacy(spacy_model)
        except ModuleNotFoundError as exc:
            raise NerModelUnavailableError(str(exc)) from None
        from polanon.guards.ner import NerGuard

        guards.append(NerGuard(spacy_model=spacy_model))
    return guards


def _require_spacy(spacy_model: str) -> None:
    """Fail now, with the command that fixes it, instead of on the first text to mask."""
    from polanon.guards.ner import _model_reference

    model = _model_reference(spacy_model)
    hint = (
        'pip install "polanon[ner]"; or call default_guards(with_ner=False)'
        if isinstance(model, Path)
        else _NER_INSTALL.format(model=model)
    )
    if importlib.util.find_spec("spacy") is None:
        raise ModuleNotFoundError(f"the language model guard needs spaCy: {hint}")
    if isinstance(model, Path):
        if not (model / "meta.json").is_file() or not (model / "config.cfg").is_file():
            raise ModuleNotFoundError(f"no spaCy model at {model}")
    elif importlib.util.find_spec(model) is None:
        raise ModuleNotFoundError(f"the spaCy model {model!r} is not installed: {hint}")


def ner_available(spacy_model: str = "pl_core_news_md") -> bool:
    """Return whether ``default_guards(with_ner=True)`` would find spaCy and the model.

    It runs the same check as :func:`default_guards` without loading anything, so a settings page
    can show whether the language model can be switched on.

    Args:
        spacy_model: the spaCy model the guard would load.
    """
    try:
        _require_spacy(spacy_model)
    except ModuleNotFoundError:
        return False
    return True


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
        pseudonyms: Pseudonyms | None = None,
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
            pseudonyms: computes every new token from a secret key, so a value has the same
                token in every document, process and question, as a search index needs. Without
                it, tokens are numbered per conversation: ``<OSOBA_1>``, ``<OSOBA_2>``.

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
        self.pseudonyms = pseudonyms

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
            pseudonyms=self.pseudonyms,
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
            pseudonyms=self.pseudonyms,
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

    def mask_chunks(
        self, chunks: str | Sequence[str], mapping: dict[str, str] | None = None
    ) -> MaskedChunks:
        """Mask the passages of one document with one mapping.

        A value recognised in one passage is masked in every other, including the ones before it,
        so a surname that stands alone in the first passage is masked once the full name turns up
        in the third.

        Args:
            chunks: the passages, in order, or a whole document as one str.
            mapping: token -> original value from earlier, as in :meth:`anonymize_sync`.

        Returns:
            The masked passages, the mapping of each as written there, and the whole mapping.

        Raises:
            MaskingUnavailableError: when masking fails; nothing of the document may be indexed.
            ValueError: when this anonymizer has a stage-2 judge.
        """
        return mask_chunks(self, chunks, mapping)

    def index_chunks(
        self, chunks: str | Sequence[str], store: MappingStore, *, lookup: bool = True
    ) -> MaskedChunks:
        """Mask a document for a search index, and put what its tokens hide into `store`.

        It needs stable tokens (``pseudonyms=``), so a question masked later finds the same ones.
        The store gets each token's value, a name in the nominative, with its fingerprint; a
        token it already holds for another value is a collision, which is recorded, and the
        document is masked again so that value gets a longer token.

        Args:
            chunks: the passages of the document, in order, or the document as one str.
            store: the application's store of what tokens hide, kept apart from the index.
            lookup: also mask the names the store already holds a token for, such as a surname
                that stands alone here and was masked in full in another document. It asks the
                store once for every run of up to four words; pass False for a store too slow
                for that.

        Returns:
            The masked passages for the embedder and the index, and the mapping of each as written
            there, to restore a passage exactly. Neither holds anything the index may keep.

        Raises:
            MaskingUnavailableError: when masking fails; nothing of the document may be indexed.
            ValueError: without pseudonyms, or with a stage-2 judge.
            PseudonymCollisionError: when collisions keep coming back, which a working key and a
                store shared by every process do not cause.
        """
        return index_chunks(self, chunks, store, lookup=lookup)

    def mask_query(
        self,
        text: str,
        store: MappingStore,
        mapping: dict[str, str] | None = None,
        *,
        lookup: bool = True,
    ) -> AnonymizationResult:
        """Mask a question to a search index with the tokens the documents got.

        A token the store holds for another value is a collision: it is recorded, and the
        question is masked again, so it finds nothing rather than another person's passages.

        Args:
            text: the question.
            store: the store the documents were indexed into.
            mapping: token -> original value from earlier turns of the conversation.
            lookup: also mask the names the store holds a token for, such as "Co podpisał
                Kowalczyk?", where no rule reads the surname alone.

        Raises:
            MaskingUnavailableError: when masking fails; nothing of the question may be sent.
            ValueError: without pseudonyms, or with a stage-2 judge.
        """
        return mask_query(self, text, store, mapping, lookup=lookup)

    def turn(
        self,
        *,
        known: Iterable[str] = (),
        mapping: dict[str, str] | None = None,
        unknown_blocks: UnknownBlocks = "raise",
    ) -> MaskedTurn:
        """Start masking one call to a model: its history, question and tool results.

        ``anonymizer.turn(...)`` is ``MaskedTurn(anonymizer, ...)``; see :class:`MaskedTurn`.

        Args:
            known: values already known to be sensitive, masked wherever they appear.
            mapping: token -> original value from earlier turns of the same conversation.
            unknown_blocks: ``"raise"`` or ``"drop"`` a message block that cannot be masked.
        """
        return MaskedTurn(self, known=known, mapping=mapping, unknown_blocks=unknown_blocks)
