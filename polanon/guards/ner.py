"""Named-entity guard for people, places and organisations, backed by spaCy.

spaCy is used directly rather than through Presidio: the Polish models are trained on the NKJP tag
set and emit ``persName`` / ``placeName`` / ``orgName``, which Presidio's recognizer does not map,
so it silently returns nothing. Talking to spaCy ourselves keeps the mapping explicit and testable.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from typing import Any

from polanon.guards.base import Guard
from polanon.guards.names_pl import is_common_adjective
from polanon.types import Span

logger = logging.getLogger(__name__)

# : spaCy entity label -> our canonical entity type. Covers Polish (NKJP) and English (OntoNotes)
# models.
LABEL_MAP: dict[str, str] = {
    "persName": "PERSON",
    "PERSON": "PERSON",
    "PER": "PERSON",
    "placeName": "LOCATION",
    "geogName": "LOCATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "orgName": "ORG",
    "ORG": "ORG",
}

#: Entity types this guard can produce.
SUPPORTED = frozenset({"PERSON", "LOCATION", "ORG"})

#: Confidence of a statistical detection, deliberately below the 1.0 of checksum-validated ones.
NER_SCORE = 0.85

#: Longest piece of text handed to spaCy at once. spaCy refuses anything over one million
#: characters, and a long export is cut at a line break below this, where no name continues.
CHUNK_CHARS = 100_000

#: Loading a model takes seconds and hundreds of megabytes. Threads that need it at the same time
#: wait for the first one instead of each loading their own copy. One lock for the module rather
#: than one per guard, so a guard can still be pickled.
_LOAD_LOCK = threading.Lock()

#: Form and document labels the Polish models sometimes tag as names. Masking them only makes the
#: text harder to read, and they identify nobody. A span is dropped when *every* word in it is on
#: this list, which is what catches a heading like "NOTATKA WEWNĘTRZNA" without needing the phrase.
LABEL_STOPWORDS = frozenset(
    {
        "pesel",
        "nip",
        "regon",
        "krs",
        "iban",
        "e-mail",
        "email",
        "tel",
        "telefon",
        "faks",
        "fax",
        "ulica",
        "ul",
        "adres",
        "imię",
        "nazwisko",
        "firma",
        "sprzedawca",
        "nabywca",
        "faktura",
        "data",
        "razem",
        "sp",
        "z o.o.",
        "s.a.",
        "login",
        "hasło",
        "haslo",
        "użytkownik",
        "uzytkownik",
        "klucz",
        "token",
        "konto",
        "dowód",
        "dowod",
        "paszport",
        "vin",
        "mac",
        "ip",
        "www",
        "http",
        "https",
        "hr",
        "it",
        "dział",
        "dzial",
        "temat",
        "od",
        "do",
        "kwota",
        "netto",
        "brutto",
        "vat",
        "al",
        "aleja",
        "plac",
        "osiedle",
        "bearer",
        "authorization",
        "notatka",
        "wewnętrzna",
        "wewnetrzna",
        "zastępstwo",
        "zastepstwo",
        "prowadzi",
        "stan",
        "obecny",
        "kontakt",
        "dostawca",
        "odbiorca",
        "zgłaszający",
        "zglaszajacy",
        "zgłoszenie",
        "zgloszenie",
        "opis",
        "uwagi",
        "pozycje",
        "referencje",
        "doświadczenie",
        "doswiadczenie",
        "wykształcenie",
        "wyksztalcenie",
        "stanowisko",
        "umowa",
        "kadry",
        "podpis",
        "podpisy",
        "załącznik",
        "zalacznik",
        "termin",
        # Business and technical abbreviations the model takes for organisations: "co to jest
        # SEO?" went out as "co to jest <FIRMA_1>?", which leaves nothing to answer.
        "seo",
        "sem",
        "ppc",
        "cpc",
        "cpm",
        "cpa",
        "ctr",
        "roi",
        "roas",
        "kpi",
        "crm",
        "erp",
        "cms",
        "b2b",
        "b2c",
        "ux",
        "ui",
        "api",
        "saas",
        "sla",
        "nda",
        "faq",
        "usp",
        "mvp",
        "ceo",
        "cfo",
        "cto",
        "coo",
        "cmo",
        "pr",
        "rodo",
        "gdpr",
        "ai",
        "llm",
        "pdf",
        "csv",
        "url",
        "html",
        "css",
        "sql",
        # Invoice and letter labels in English, Swedish and German. Run over a foreign invoice,
        # the Polish model masked "Subtotal", "Invoice Date" and "Beneficiary Account Number" as
        # people and companies, which buried the real values under tokens that mean nothing.
        "invoice",
        "invoiced",
        "subtotal",
        "sub",
        "total",
        "net",
        "gross",
        "tax",
        "sales",
        "date",
        "due",
        "bill",
        "billing",
        "delivery",
        "shipping",
        "handling",
        "ship",
        "order",
        "number",
        "no",
        "account",
        "accounts",
        "payable",
        "receivable",
        "beneficiary",
        "name",
        "address",
        "registered",
        "office",
        "company",
        "terms",
        "conditions",
        "payment",
        "description",
        "quantity",
        "qty",
        "unit",
        "price",
        "balance",
        "transaction",
        "transactions",
        "amount",
        "reference",
        "ref",
        "phone",
        "sort",
        "code",
        "swift",
        "bic",
        "and",
        "of",
        "the",
        "to",
        "from",
        "for",
        "your",
        "our",
        "item",
        "items",
        "rate",
        "standard",
        "sample",
        "bankgiro",
        "plusgiro",
        "org",
        "nr",
        "cost",
        "centre",
        "center",
        "object",
        "project",
        "point",
        "customer",
        "po",
        "box",
        "page",
        "web",
        "website",
        "zip",
        "city",
        "country",
        "street",
        "mobile",
        "ocr",
        "moms",
        "mwst",
        "ust",
        "uid",
        "rechnung",
        "betrag",
        "kund",
        "referens",
        "summa",
        "vår",
        "er",
        "tfn",
        "attn",
        "momsreg.nr",
        "org.nr",
        "geschäftsführer",
    }
)


def _is_only_labels(fragment: str) -> bool:
    """Return whether a detected span is made entirely of form and heading words."""
    words = [word.strip(".:,()").lower() for word in fragment.split()]
    return all(word in LABEL_STOPWORDS for word in words if word)


def _is_lowercase_adjective(fragment: str) -> bool:
    """Return whether a detected span is one lowercase adjective such as "polskiej".

    Typed in lowercase, "nie ma ani polskiej, ani angielskiej wersji" came back from the Polish
    model with both adjectives tagged as places. With a capital letter, "Polska" is the country, so
    only the lowercase spelling is dropped.
    """
    return " " not in fragment and fragment.islower() and is_common_adjective(fragment)


def _join_hyphenated(text: str, spans: list[Span]) -> list[Span]:
    """Join two spans of the same type separated by nothing but a hyphen.

    Polish double surnames are frequently tagged as two names, so "Maria Nowak-Kowalska" came back
    masked as ``<OSOBA_1>-<OSOBA_2>``: not a leak, but one person shown as two, and two mapping
    entries where one belongs.
    """
    joined: list[Span] = []
    for span in sorted(spans):
        previous = joined[-1] if joined else None
        bridges = (
            previous is not None
            and previous.entity_type == span.entity_type
            and text[previous.end : span.start] in ("-", "\u2011")
        )
        if bridges and previous is not None:
            joined[-1] = Span(
                previous.start, span.end, span.entity_type, previous.score, previous.source
            )
            continue
        joined.append(span)
    return joined


def _chunks(text: str, limit: int) -> Iterator[tuple[int, str]]:
    """Cut `text` into pieces of at most `limit` characters, at line breaks where there is one.

    Yields:
        Each piece with the offset at which it starts in `text`.
    """
    start = 0
    while len(text) - start > limit:
        cut = text.rfind("\n", start, start + limit)
        end = cut + 1 if cut > start else start + limit
        yield start, text[start:end]
        start = end
    yield start, text[start:]


class NerGuard(Guard):
    """Detects people, places and organisations with a spaCy pipeline."""

    name = "spacy_ner"

    def __init__(self, spacy_model: str = "pl_core_news_md") -> None:
        """Store the model name; the pipeline itself loads on first use."""
        self.spacy_model = spacy_model
        self._nlp: Any | None = None

    def _get_pipeline(self) -> Any:
        """Load and cache the spaCy pipeline, once even when several threads ask at the same time.

        Raises:
            RuntimeError: when spaCy or the language model is not installed.
        """
        if self._nlp is not None:
            return self._nlp
        with _LOAD_LOCK:
            if self._nlp is None:
                self._nlp = self._load()
        return self._nlp

    def _load(self) -> Any:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - only without the dependency
            raise RuntimeError('spaCy is not installed. Run: pip install "polanon[ner]"') from exc
        try:
            nlp = spacy.load(self.spacy_model, exclude=["lemmatizer", "tagger"])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model {self.spacy_model!r} is missing. "
                f"Run: python -m spacy download {self.spacy_model}"
            ) from exc
        logger.info("spaCy NER ready (model=%s)", self.spacy_model)
        return nlp

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return spans for the enabled subset of PERSON, LOCATION and ORG."""
        wanted = SUPPORTED & enabled
        if not wanted or not text.strip():
            return []
        nlp = self._get_pipeline()
        spans: list[Span] = []
        for offset, piece in _chunks(text, CHUNK_CHARS):
            for entity in nlp(piece).ents:
                entity_type = LABEL_MAP.get(entity.label_)
                if entity_type not in wanted:
                    continue
                start, end = offset + entity.start_char, offset + entity.end_char
                spans.extend(self._line_spans(text, start, end, str(entity_type)))
        return _join_hyphenated(text, spans)

    def _line_spans(self, text: str, start: int, end: int, entity_type: str) -> list[Span]:
        """Split an entity at line breaks and trim surrounding whitespace.

        Models occasionally run an entity across a newline, which then swallows whatever sits on the
        next line. A name never spans two lines in the documents this library is written for.
        """
        spans: list[Span] = []
        cursor = start
        for piece in text[start:end].split("\n"):
            piece_start = cursor
            cursor += len(piece) + 1
            stripped = piece.strip()
            if not stripped or _is_only_labels(stripped) or _is_lowercase_adjective(stripped):
                continue
            offset = piece.index(stripped)
            spans.append(
                Span(
                    piece_start + offset,
                    piece_start + offset + len(stripped),
                    entity_type,
                    NER_SCORE,
                    self.name,
                )
            )
        return spans
