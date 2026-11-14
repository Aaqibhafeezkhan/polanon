"""Named-entity guard for people, places and organisations, backed by spaCy.

spaCy is used directly rather than through Presidio: the Polish models are trained on the NKJP tag
set and emit ``persName`` / ``placeName`` / ``orgName``, which Presidio's recognizer does not map,
so it silently returns nothing. Talking to spaCy ourselves keeps the mapping explicit and testable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from polanon.guards.base import Guard
from polanon.guards.names_pl import NAME_FORMS, fold, is_common_adjective
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
#: characters, and a long export is cut below this where no name continues: see :func:`_chunks`.
CHUNK_CHARS = 100_000

#: The end of a sentence, and any whitespace: where a piece of a text with no line break is cut.
_SENTENCE_END = re.compile(r"[.!?…][\"'”»)\]]*\s+")
_WHITESPACE = re.compile(r"\s+")

#: Loading a model takes seconds and hundreds of megabytes. Threads that need it at the same time
#: wait for the first one instead of each loading their own copy. One lock for the module rather
#: than one per guard, so a guard can still be pickled.
_LOAD_LOCK = threading.Lock()

#: How many texts a guard remembers what it found in. A chat sends its whole history with every
#: call, and an agent its system prompt and the descriptions of its tools too: some two thousand
#: texts a call, nearly all of them read in the call before.
CACHE_TEXTS = 20_000
_CACHE_LOCK = threading.Lock()

#: Form and document labels the Polish models sometimes tag as names. Masking them only makes the
#: text harder to read, and they identify nobody. A span is dropped when *every* word in it is on
#: this list, which is what catches a heading like "NOTATKA WEWNĘTRZNA" without needing the phrase.
LABEL_STOPWORDS = frozenset(
    {
        # Academic and professional titles: the model tagged the "n." of "dr n. med." as a person.
        "dr",
        "n",
        "med",
        "hab",
        "prof",
        "mgr",
        "inż",
        "inz",
        "lek",
        "dent",
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


#: Short English words Polish has no word for: the evidence that a line is English. Words the two
#: languages share, such as "to", "a", "i", "on", "no", "do", "we", "by" and "most", are left out.
_ENGLISH_WORDS = frozenset(
    {
        "the", "and", "of", "is", "are", "was", "were", "be", "been", "for", "with", "that",
        "this", "these", "those", "you", "your", "it", "its", "it's", "they", "their", "them",
        "there", "not", "if", "when", "then", "than", "from", "at", "an", "or", "can", "cannot",
        "will", "would", "should", "must", "may", "might", "could", "have", "has", "had", "what",
        "which", "who", "how", "why", "where", "all", "any", "each", "every", "only", "into",
        "about", "after", "before", "over", "under", "between", "without", "within", "use",
        "don't", "doesn't", "isn't", "here", "such", "some", "more", "other", "also", "just",
        "very",
    }
)  # fmt: skip

#: Short Polish words, with and without their diacritics: the evidence against. Those English
#: has too, such as "ten", "pan", "pod" and "ale", are left out.
_POLISH_WORDS = frozenset(
    {
        "się", "sie", "nie", "jest", "są", "sa", "że", "ze", "oraz", "lub", "czy", "jak", "dla",
        "przez", "jego", "jej", "ich", "który", "ktory", "która", "ktora", "które", "ktore",
        "tak", "też", "tez", "już", "juz", "przy", "przed", "po", "od", "na", "w", "z", "o",
        "pani", "panem", "panu", "proszę", "prosze", "tylko", "jeszcze", "tego", "tej", "jako",
        "nad", "za", "bez", "gdy", "kiedy", "gdzie", "co", "kto", "mnie", "mi", "nam", "sprawie",
    }
)  # fmt: skip

#: The literals of Python, JSON and YAML, which tool results print where a value is missing.
_CODE_LITERALS = frozenset({"None", "True", "False", "null", "true", "false", "nil", "NaN"})

_POLISH_LETTERS = re.compile("[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
_NAME_PARTS = re.compile(r"[\W\d_]+")

#: Fewest English words that make a line English, and how many times the Polish ones they must
#: outnumber.
_ENGLISH_EVIDENCE = 3
_ENGLISH_MARGIN = 3

#: What may stand between the start of a sentence and its first word: spaces, quotes, brackets and
#: the marks of a Markdown list or heading. A colon is no end: "Contact: John Smith" names him.
_SENTENCE_OPENERS = frozenset(" \t\"'“”„«»([*_`#>-•")
_SENTENCE_ENDS = frozenset("\n.!?")

_ENGLISH, _POLISH, _UNKNOWN = "en", "pl", ""


def _language(line: str) -> str:
    """Return whether a line is plainly English, Polish, or shows no language at all.

    A line with one Polish letter is Polish, and so is one with more Polish words than English
    ones, so a Polish sentence typed without diacritics stays Polish. English needs three English
    words and three times as many as Polish ones. Any other line, such as a short heading, a
    parameter's description or a name from code, shows none.
    """
    if _POLISH_LETTERS.search(line):
        return _POLISH
    words = [word.lower() for word in _WORD.findall(line)]
    english = sum(word in _ENGLISH_WORDS for word in words)
    polish = sum(word in _POLISH_WORDS for word in words)
    if english >= _ENGLISH_EVIDENCE and english > _ENGLISH_MARGIN * polish:
        return _ENGLISH
    return _POLISH if polish > english else _UNKNOWN


def _capitalised(word: str) -> bool:
    """Return whether a word is written as a name is: "Lindqvist", not "result", "NOT" or "APIs"."""
    word = word.strip(".,;:!?()[]\"'“”„«»")
    return bool(word) and word[0].isupper() and (len(word) == 1 or not word[1].isupper())


def _opens_a_sentence(text: str, start: int) -> bool:
    """Return whether the word at `start` begins its line or its sentence."""
    index = start - 1
    while index >= 0 and text[index] in _SENTENCE_OPENERS:
        index -= 1
    return index < 0 or text[index] in _SENTENCE_ENDS


def _named_in_english(text: str, span: Span) -> bool:
    """Return whether a span the Polish model found on an English line reads as a name.

    Reading English, the Polish model tags ordinary words: "result", "NOT", "Safety", "the user's
    own". Kept is a capitalised span in mid-sentence, such as "Maria Lindqvist" in "ask Maria
    Lindqvist for it"; a name with a Polish letter never gets here, as its line is not English. A
    name that opens a sentence is lost to this model; the rules still find a Polish first name
    wherever it stands.
    """
    words = text[span.start : span.end].split()
    return (
        bool(words)
        and words[0].lower() not in _ENGLISH_WORDS
        and _capitalised(words[0])
        and _capitalised(words[-1])
        and not _opens_a_sentence(text, span.start)
    )


def _holds_a_first_name(fragment: str) -> bool:
    """Return whether a word of `fragment`, split also at "_" and digits, is a Polish first name."""
    return any(fold(part.lower()) in NAME_FORMS for part in _NAME_PARTS.split(fragment) if part)


def _is_code(fragment: str) -> bool:
    """Return whether a span is no name on any line: code, a number, or an English small word.

    A backtick, a double quote and a "_" with no Polish first name mark code: "replace_all",
    "volume_level", `File "` of a Python traceback; "jan_kowalski" is somebody's login. A span of
    digits alone, such as "2026", is a number, and the rules find numbers that matter; "None",
    "True" or "null" is a literal of code. One English function word in lowercase, "has" or
    "before", is no name even on a Polish line.
    """
    if "`" in fragment or '"' in fragment:
        return True
    if "_" in fragment and not _holds_a_first_name(fragment):
        return True
    if fragment.replace(" ", "").isdigit() or fragment in _CODE_LITERALS:
        return True
    return fragment.islower() and fragment in _ENGLISH_WORDS


def _named_without_language(fragment: str) -> bool:
    """Return whether a span on a line that shows no language reads as a name.

    A parameter described as "Max results" or a heading such as "Skills for" shows neither
    language, and the Polish model tagged such words too. A span with a word in lowercase is kept
    only when it holds a Polish first name, as "anna nowak" does.
    """
    words = fragment.split()
    return all(not word[0].islower() for word in words) or _holds_a_first_name(fragment)


def _without_english_misreadings(text: str, spans: list[Span]) -> list[Span]:
    """Drop what the Polish model read as a name where the text is not Polish, unless it is one."""
    languages: dict[int, str] = {}
    kept = []
    for span in spans:
        fragment = text[span.start : span.end]
        if _is_code(fragment):
            continue
        line_start = text.rfind("\n", 0, span.start) + 1
        if line_start not in languages:
            line_end = text.find("\n", span.start)
            languages[line_start] = _language(text[line_start : None if line_end < 0 else line_end])
        language = languages[line_start]
        if (
            language == _POLISH
            or (language == _ENGLISH and _named_in_english(text, span))
            or (language == _UNKNOWN and _named_without_language(fragment))
        ):
            kept.append(span)
    return kept


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
    """Cut `text` into pieces of at most `limit` characters where no name is likely to continue.

    A piece ends after its last line break, else after its last sentence, else after its last
    space, and only a text with no whitespace at all is cut mid-word. Cutting a single-line text
    at exactly `limit` characters split a name across two pieces, and spaCy, seeing half of it in
    each, missed it.

    Yields:
        Each piece with the offset at which it starts in `text`.
    """
    start = 0
    while len(text) - start > limit:
        end = _cut(text, start, start + limit)
        yield start, text[start:end]
        start = end
    yield start, text[start:]


def _cut(text: str, start: int, stop: int) -> int:
    """Return where the piece of `text` from `start` ends, at most at `stop`."""
    newline = text.rfind("\n", start, stop)
    if newline > start:
        return newline + 1
    for boundary in (_SENTENCE_END, _WHITESPACE):
        ends = [match.end() for match in boundary.finditer(text, start, stop)]
        if ends:
            return ends[-1]
    return stop


def _model_reference(spacy_model: str) -> str | Path:
    """Normalize a package's pip spelling without rewriting a model directory's name."""
    package = spacy_model.replace("-", "_")
    path = Path(spacy_model)
    if package.isidentifier() and (
        importlib.util.find_spec(package) is not None or not path.exists()
    ):
        return package
    return path


class NerGuard(Guard):
    """Detects people, places and organisations with a spaCy pipeline."""

    name = "spacy_ner"

    def __init__(
        self, spacy_model: str = "pl_core_news_md", cache_texts: int = CACHE_TEXTS
    ) -> None:
        """Store the model name; the pipeline itself loads on first use.

        Args:
            spacy_model: the spaCy package or directory to load.
            cache_texts: how many texts to remember what the model found in, so a text sent again,
                as a chat's history is, is not read again; 0 reads every text every time. A text
                is remembered by its SHA-256 digest, never as written.
        """
        self.spacy_model = spacy_model
        self.cache_texts = cache_texts
        self._nlp: Any | None = None
        self._found: OrderedDict[tuple[str, frozenset[str]], tuple[Span, ...]] = OrderedDict()

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
        model = _model_reference(self.spacy_model)
        try:
            nlp = spacy.load(model, exclude=["lemmatizer", "tagger"])
        except OSError as exc:
            if isinstance(model, Path):
                raise RuntimeError(f"no spaCy model at {model}") from exc
            raise RuntimeError(
                f"spaCy model {model!r} is missing. Run: python -m spacy download {model}"
            ) from exc
        logger.info("spaCy NER ready (model=%s)", self.spacy_model)
        return nlp

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return spans for the enabled subset of PERSON, LOCATION and ORG."""
        wanted = SUPPORTED & enabled
        if not wanted or not text.strip():
            return []
        digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
        key = (digest, frozenset(wanted))
        with _CACHE_LOCK:
            found = self._found.get(key)
            if found is not None:
                self._found.move_to_end(key)
                return list(found)
        spans = self._read(text, wanted)
        if self.cache_texts > 0:
            with _CACHE_LOCK:
                self._found[key] = tuple(spans)
                while len(self._found) > self.cache_texts:
                    self._found.popitem(last=False)
        return spans

    def _read(self, text: str, wanted: frozenset[str] | set[str]) -> list[Span]:
        """Run the model over `text` and keep what reads as a name."""
        nlp = self._get_pipeline()
        spans: list[Span] = []
        for offset, piece in _chunks(text, CHUNK_CHARS):
            for entity in nlp(piece).ents:
                entity_type = LABEL_MAP.get(entity.label_)
                if entity_type not in wanted:
                    continue
                start, end = offset + entity.start_char, offset + entity.end_char
                spans.extend(self._line_spans(text, start, end, str(entity_type)))
        spans = _join_hyphenated(text, spans)
        if getattr(nlp, "lang", "pl") == "pl":
            spans = _without_english_misreadings(text, spans)
        return spans

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
