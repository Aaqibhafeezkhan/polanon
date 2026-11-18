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
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from collections.abc import Iterator
from itertools import accumulate, pairwise
from pathlib import Path
from typing import Any

from polanon.guards._software_english import SOFTWARE_WORDS
from polanon.guards.base import Guard
from polanon.guards.names_pl import NAME_FORMS, fold, is_common_adjective
from polanon.policy import Mode, check_mode
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

#: Texts whose words a guard remembers for the ML filter when it remembers no spans.
_WORDS_KEPT = 64

#: The spaCy model the ML filter was fitted on, as its pipeline names itself.
_ML_FILTER_MODEL = "core_news_md"

#: The parties of a Polish contract in their cases, which it writes with a capital letter.
_CONTRACT_PARTIES = """
    najemca najemcy najemcę najemcą najemców najemcom najemcami
    wynajmujący wynajmującego wynajmującemu wynajmującym wynajmująca wynajmującej wynajmującą
    zleceniodawca zleceniodawcy zleceniodawcę zleceniodawcą
    zleceniobiorca zleceniobiorcy zleceniobiorcę zleceniobiorcą
    zamawiający zamawiającego zamawiającemu zamawiającym zamawiająca zamawiającej zamawiającą
    wykonawca wykonawcy wykonawcę wykonawcą
    sprzedający sprzedającego sprzedającemu sprzedającym sprzedawca sprzedawcy sprzedawcę
    sprzedawcą kupujący kupującego kupującemu kupującym nabywca nabywcy nabywcę nabywcą
    pracodawca pracodawcy pracodawcę pracodawcą pracownik pracownika pracownikowi pracownikiem
    dzierżawca dzierżawcy dzierżawcę dzierżawcą
    wydzierżawiający wydzierżawiającego wydzierżawiającemu wydzierżawiającym
    pożyczkodawca pożyczkodawcy pożyczkodawcę pożyczkodawcą
    pożyczkobiorca pożyczkobiorcy pożyczkobiorcę pożyczkobiorcą
    strona strony stronie stronę stroną stron stronom stronami
"""

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
        # The parties of a Polish contract, which it writes with a capital letter as defined
        # terms: in a lease read by an agent the model took "Najemcy" for a company.
        *_CONTRACT_PARTIES.split(),
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
        "very", "while", "because", "although", "however", "though", "since", "until", "unless",
        "whether", "both", "either", "neither", "instead", "already",
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

#: The English names of languages and nations, which the Polish model reads as places.
_LANGUAGES_IN_ENGLISH = frozenset(
    {
        "polish", "english", "german", "french", "spanish", "italian", "portuguese", "dutch",
        "russian", "ukrainian", "belarusian", "czech", "slovak", "hungarian", "romanian",
        "bulgarian", "croatian", "serbian", "slovenian", "lithuanian", "latvian", "estonian",
        "finnish", "swedish", "norwegian", "danish", "icelandic", "greek", "turkish", "arabic",
        "hebrew", "persian", "hindi", "chinese", "japanese", "korean", "vietnamese", "thai",
        "indonesian", "latin", "european", "american", "british", "canadian", "australian",
    }
)  # fmt: skip

#: A model number, letters and digits in one word, then the spaces before a span: "M4 ".
_MODEL_NUMBER_BEFORE = re.compile(r"(?<![^\W_])((?=[^\W_]*[^\W\d_])[^\W_]+)[ \t]+$")

#: The punctuation of Chinese, Japanese and Korean, trimmed off a span's ends as a space is.
_EAST_ASIAN_MARKS = "。，、；：！？「」『』（）《》【】・"  # noqa: RUF001

#: A letter of Chinese, Japanese or Korean.
_EAST_ASIAN_LETTER = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯]")

#: What ends a line or a cell of a table: no name runs across it.
_CELL_BREAKS = re.compile(r"[\n;|\t]")

#: The tiers of a product's model, "M4 Max" or "S8 Pro Ultra", which alone name nobody.
_PRODUCT_TIERS = frozenset({"max", "pro", "ultra", "plus", "mini", "lite", "air", "prime"})

#: The ends of the names of errors in code: `JSONDecodeError`, `HTTPError`, `ValueError`.
_CODE_ENDINGS = ("Error", "Exception", "Warning")

#: A capital after a small letter, as code writes `MediaPlayerEntity`, an abbreviation run into
#: a word, as in `HTTPServer`, the parts of such a word, and the starts of names that have a
#: capital inside too: McDonald, MacLeod, O'Brien, DeVito, LaToya, VanDyke.
_INNER_CAPITAL = re.compile(r"[a-ząćęłńóśźż][A-ZĄĆĘŁŃÓŚŹŻ]")
_RUN_IN_ABBREVIATION = re.compile(r"[A-ZĄĆĘŁŃÓŚŹŻ]{2}[a-ząćęłńóśźż]")
_CAMEL_PARTS = re.compile(r"[A-ZĄĆĘŁŃÓŚŹŻ]?[a-ząćęłńóśźż]+|[A-ZĄĆĘŁŃÓŚŹŻ]+(?![a-ząćęłńóśźż])")
_NAME_PREFIX = re.compile(r"(?:Mc|Mac|O'|D'|De|Di|Da|Du|La|Le|Van|Von)[A-Z]")

#: English small words that are first names too: "Will Nowak" opens with one, and is a person.
_FIRST_NAMES_TOO = frozenset({"will", "may"})

#: The literals of Python, JSON and YAML, which tool results print where a value is missing.
_CODE_LITERALS = frozenset({"None", "True", "False", "null", "true", "false", "nil", "NaN"})

_POLISH_LETTERS = re.compile("[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
_NAME_PARTS = re.compile(r"[\W\d_]+")

#: Fewest English words that make a line English, and how many times the Polish ones they must
#: outnumber.
_ENGLISH_EVIDENCE = 3
_ENGLISH_MARGIN = 3
#: The same for a whole text, whose lines that show no language are then read as English.
_TEXT_ENGLISH_EVIDENCE = 12
_TEXT_ENGLISH_MARGIN = 4
#: A quotation: the words of somebody else, maybe in another language.
_QUOTED = re.compile(r'„[^”\n]*”|“[^”\n]*”|«[^»\n]*»|»[^«\n]*«|"[^"\n]*"')

#: What may stand between the start of a sentence and its first word: spaces, quotes, brackets and
#: the marks of a Markdown list or heading. A colon is no end: "Contact: John Smith" names him.
_SENTENCE_OPENERS = frozenset(" \t\"'“”„«»([*_`#>-•")
_SENTENCE_ENDS = frozenset("\n.!?")

_ENGLISH, _POLISH, _UNKNOWN, _CODE = "en", "pl", "", "code"

#: A line break or a tab written out as an escaped JSON string prints it.
_ESCAPED_BREAKS = frozenset({"\\n", "\\t", "\\r"})

#: A line of a Python traceback, where the model took "File" for a place.
_TRACEBACK = re.compile(r'\s*File "[^"\n]*", line \d+')


def _evidence(text: str) -> tuple[int, int]:
    """Count the English words and the Polish ones, a word with a Polish letter among them.

    What stands in quotation marks is left out: a model reasoning in English quotes the user in
    Polish, and the quote is the user's language, not the line's.
    """
    english = polish = 0
    for word in _WORD.findall(_QUOTED.sub(" ", text)):
        folded = word.lower()
        english += folded in _ENGLISH_WORDS
        polish += folded in _POLISH_WORDS or _POLISH_LETTERS.search(word) is not None
    return english, polish


def _language(line: str) -> str:
    """Return whether a line is plainly English, Polish, or shows no language at all.

    A line with more Polish words than English ones is Polish, a word with a Polish letter
    counting as Polish, so a Polish sentence typed without diacritics stays Polish. English needs
    three English words and three times as many as Polish ones. Any other line, such as a short
    heading, a parameter's description or a name from code, shows none.
    """
    english, polish = _evidence(line)
    if english >= _ENGLISH_EVIDENCE and english > _ENGLISH_MARGIN * polish:
        return _ENGLISH
    return _POLISH if polish > english else _UNKNOWN


def _is_english_text(text: str) -> bool:
    """Return whether a whole text is plainly English, as a model's reasoning or a manual is."""
    english, polish = _evidence(text)
    return english >= _TEXT_ENGLISH_EVIDENCE and english > _TEXT_ENGLISH_MARGIN * polish


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
        and _capitalised(words[0])
        and _capitalised(words[-1])
        and not _opens_a_sentence(text, span.start)
    )


def _holds_a_first_name(fragment: str) -> bool:
    """Return whether a word of `fragment`, split also at "_" and digits, is a Polish first name."""
    return any(fold(part.lower()) in NAME_FORMS for part in _NAME_PARTS.split(fragment) if part)


def _is_code(fragment: str) -> bool:
    """Return whether a span is no name on any line: code, a number, or no word at all.

    A backtick, a double quote and a "_" with no Polish first name mark code: "replace_all",
    "volume_level", `File "` of a Python traceback; "jan_kowalski" is somebody's login. A span
    without a letter is no name: "2026" is a number, and the rules find numbers that matter, the
    carets under a traceback or a heading's "####" are marks. "None", "True" or "null" is a
    literal of code.
    """
    if "`" in fragment or '"' in fragment:
        return True
    if "_" in fragment and not _holds_a_first_name(fragment):
        return True
    return not _has_a_letter(fragment) or fragment in _CODE_LITERALS


def _product_or_code(fragment: str) -> bool:
    """Return whether a span of one word is a product's tier, an identifier of code, or a number.

    In an agent's calls the Polish model took a vacuum's fan speed "Max" for a person, masked 57
    times once known, Python's `JSONDecodeError` and `HTTPError` too, an invoice number
    "FV/2026/10/014" for a place and a line of a CSV file for a company. A tier alone names
    nobody, and a name holds no digit, but for a login with a Polish first name in it. See
    :func:`_code_identifier` for a capital inside a word.
    """
    if " " in fragment:
        return False
    if fragment.lower().rstrip("+") in _PRODUCT_TIERS:
        return True
    if fragment.endswith(_CODE_ENDINGS):
        return True
    if any(char.isdigit() for char in fragment):
        return not _holds_a_first_name(fragment)
    return _code_identifier(fragment)


def _code_identifier(fragment: str) -> bool:
    """Return whether a word is written as code writes a name: `MediaPlayerEntity`, `HTTPServer`.

    One capital inside a word is not enough: a person's handle or a small company's name is
    written so too, "AnnaLis" or "ZielonyDom", and dropping it sent it in the clear. Two
    capitals after small letters, or an abbreviation run into a word, mark code, unless a part
    of the word is a Polish first name, as in "OlaNowakDesign", or it starts as McDonald,
    MacLeod or DeVito do.
    """
    if _NAME_PREFIX.match(fragment):
        return False
    if any(fold(part.lower()) in NAME_FORMS for part in _CAMEL_PARTS.findall(fragment)):
        return False
    return (
        len(_INNER_CAPITAL.findall(fragment)) >= 2
        or _RUN_IN_ABBREVIATION.search(fragment) is not None
    )


def _east_asian(fragment: str) -> bool:
    """Return whether every letter of a span is Chinese, Japanese or Korean.

    The Polish model reads such text as a string of names: a page in Chinese in an agent's
    results gave it a "person" of six characters. A name in those scripts is beyond it anyway.
    """
    letters = [char for char in fragment if char.isalpha()]
    return bool(letters) and all(_EAST_ASIAN_LETTER.match(char) for char in letters)


def _software_english(fragment: str) -> bool:
    """Return whether a span is made of the English of software: "Uncheck", "Privacy Policy".

    An agent's answers name buttons and settings in English inside Polish sentences, and the
    Polish model took "Uncheck" for a person, masked 48 times in one call once known, and
    "Authorization" too. A span whose every word is on :data:`SOFTWARE_WORDS` or is an English
    small word is dropped, unless it holds a Polish first name. The list leaves out words that
    are also first names or surnames, so Hugh Grant is still a person.
    """
    words = [_bare(word).lower() for word in fragment.split()]
    return (
        any(word in SOFTWARE_WORDS for word in words)
        and all(word in SOFTWARE_WORDS or word in _ENGLISH_WORDS for word in words)
        and not _holds_a_first_name(fragment)
    )


def _without_software_words_in_front(text: str, span: Span) -> Span:
    """Cut the English of software off the front of a person, as "Login" of "Login Kowalski".

    The Polish model reads the label of a field before a surname as a first name: "Login
    Kowalski", "Email Nowaka" and "Admin: Ola Kowalska" came back as one person each, and the
    label went to the model inside the token. :data:`SOFTWARE_WORDS` leaves out every word that is
    also a first name or a surname, so such a word in front names nobody, and the rest of the
    span is the person. The last word is never cut, so the span keeps something to mask.
    """
    if span.entity_type != "PERSON":
        return span
    fragment = text[span.start : span.end]
    words = list(re.finditer(r"\S+", fragment))
    cut = 0
    for word, following in pairwise(words):
        if _bare(word[0]).lower() not in SOFTWARE_WORDS or _holds_a_first_name(word[0]):
            break
        cut = following.start()
    if cut == 0:
        return span
    return Span(span.start + cut, span.end, span.entity_type, span.score, span.source)


def _has_a_letter(fragment: str) -> bool:
    return any(char.isalpha() for char in fragment)


def _bare(word: str) -> str:
    return word.strip(".,;:!?()[]")


def _opens_with_an_english_word(fragment: str) -> bool:
    """Return whether a span opens with an English small word, and so is no name on any line.

    In an agent's call "That's" opened a span in mid-sentence on an English line, and was masked
    76 times once known; a Polish manual quoting English gave "Before" and "the model". "Will" and
    "May" are first names too, so a span they open with a capital letter is left to its line.
    """
    first = _bare(fragment.split()[0]).split("'")[0]
    folded = first.lower()
    return folded in _ENGLISH_WORDS and (first.islower() or folded not in _FIRST_NAMES_TOO)


def _reads_as_a_name(fragment: str) -> bool:
    """Return whether a piece of a span cut at an English word is a name on its own.

    Next to English words one capitalised word is mostly English too, "Skills" of "Skills for";
    a piece is kept with two capitalised words or more, or with a Polish first name.
    """
    words = fragment.split()
    return _holds_a_first_name(fragment) or (
        len(words) >= 2 and all(_capitalised(word) for word in words)
    )


def _cut_at_english_words(text: str, span: Span) -> list[Span]:
    """Cut a span at the English small words it holds in lowercase, keeping what reads as a name.

    No name holds "and", "at" or "the" in lowercase, so such a word ends one. The Polish model
    read "Maria Lindqvist at Northwind Traders" as one span, and "tokens and passwords" of a
    Polish manual as a company: the first gives two names, the second none.
    """
    fragment = text[span.start : span.end]
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", fragment)]
    if not any(_bare(fragment[a:b]) in _ENGLISH_WORDS for a, b in words[1:]):
        return [span]
    pieces: list[list[tuple[int, int]]] = [[]]
    for start, end in words:
        if _bare(fragment[start:end]) in _ENGLISH_WORDS:
            pieces.append([])
        else:
            pieces[-1].append((start, end))
    kept = []
    for piece in pieces:
        if not piece:
            continue
        first, last = piece[0][0], piece[-1][1]
        while last > first and fragment[last - 1] in ".,;:!?":
            last -= 1
        if last > first and _reads_as_a_name(fragment[first:last]):
            kept.append(
                Span(
                    span.start + first, span.start + last, span.entity_type, span.score, span.source
                )
            )
    return kept


def _in_inline_code(text: str, span: Span) -> bool:
    r"""Return whether a span stands between two backticks of its line, as `Uncheck` does.

    Markdown sets code, a command or a button's label in backticks, and in an agent's Polish
    answers the Polish model took `Uncheck` for a person on line after line: masked 85 times in
    one call once known. A line break written out as "\n" ends a line too. A login with a Polish
    first name in it, `anna.nowak`, is still a person's.
    """
    real, escaped = text.rfind("\n", 0, span.start), text.rfind("\\n", 0, span.start)
    line_start = max(real + 1, escaped + 2 if escaped >= 0 else 0)
    ends = [end for end in (text.find("\n", span.end), text.find("\\n", span.end)) if end >= 0]
    line_end = min(ends, default=len(text))
    return (
        text.count("`", line_start, span.start) % 2 == 1
        and text.count("`", span.end, line_end) > 0
        and not _holds_a_first_name(text[span.start : span.end])
    )


def _inside_a_word(text: str, span: Span) -> bool:
    r"""Return whether a span is a piece of a longer word, "service" of "ha_call_service".

    A line break written as the two characters "\n", as an escaped JSON string prints it, ends
    the word before it: "Kontakt:\nJan Kowalski" names a person.
    """
    before = text[span.start - 1] if span.start > 0 else " "
    if text[max(0, span.start - 2) : span.start] in _ESCAPED_BREAKS:
        before = " "
    after = text[span.end] if span.end < len(text) else " "
    return before == "_" or after == "_" or before.isalnum() or after.isalnum()


def _lowercase_place(span: Span, fragment: str) -> bool:
    """Return whether a span is a place or a company of one word written in lowercase.

    Polish writes both with a capital letter. In lowercase the Polish model found English words of
    manuals instead, "text", "context", "service", masked dozens of times in one agent's call once
    known. The price is a place typed in lowercase in a chat, "jadę do gdyni", which goes out as
    written; a person in lowercase is kept, and the rules find a Polish first name too.
    """
    return span.entity_type in ("LOCATION", "ORG") and " " not in fragment and fragment.islower()


def _after_a_model_number(text: str, span: Span, fragment: str) -> bool:
    """Return whether a person follows a word of letters and digits, as "Max" of "M4 Max" does.

    An agent's tool results listed a laptop's "M4 Max" and a vacuum's "S8 Pro", and the Polish
    model took "Max" for a person, masked 44 times in one call once known. A person with a Polish
    first name is kept: "12A Maja Nowak" is a house number and a woman.
    """
    if span.entity_type != "PERSON" or _holds_a_first_name(fragment):
        return False
    before = _MODEL_NUMBER_BEFORE.search(text, max(0, span.start - 40), span.start)
    return before is not None and any(char.isdigit() for char in before[1])


def _language_named_in_english(span: Span, fragment: str) -> bool:
    """Return whether a place or a company opens with the English name of a language or nation.

    A Polish manual in an agent's tool results spoke of "Polish" and "English" texts, and the
    Polish model took both for places, "Polish names" and "Polish-first" for companies. A real
    one, "British Airways" or "Czech Republic", is no secret either. A person is left alone:
    "German" is a Polish first name too.
    """
    first = fragment.split()[0].split("-")[0]
    return span.entity_type in ("LOCATION", "ORG") and first.lower() in _LANGUAGES_IN_ENGLISH


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
    english_text: bool | None = None
    kept = []
    pieces = (
        piece
        for span in spans
        if not _opens_with_an_english_word(text[span.start : span.end])
        for piece in _cut_at_english_words(text, _without_software_words_in_front(text, span))
    )
    for span in pieces:
        fragment = text[span.start : span.end]
        if (
            _is_code(fragment)
            or _in_inline_code(text, span)
            or _inside_a_word(text, span)
            or _lowercase_place(span, fragment)
            or _language_named_in_english(span, fragment)
            or _after_a_model_number(text, span, fragment)
            or _product_or_code(fragment)
            or _software_english(fragment)
            or _east_asian(fragment)
        ):
            continue
        line_start = text.rfind("\n", 0, span.start) + 1
        if line_start not in languages:
            line_end = text.find("\n", span.start)
            line = text[line_start : None if line_end < 0 else line_end]
            languages[line_start] = _CODE if _TRACEBACK.match(line) else _language(line)
        language = languages[line_start]
        if language == _CODE:
            continue
        if language == _UNKNOWN:
            if english_text is None:
                english_text = _is_english_text(text)
            if english_text:
                language = _ENGLISH
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


def _words_of(
    document: Any, offset: int, ranges: list[tuple[int, int]]
) -> Iterator[tuple[int, int, Any]]:
    """Yield the words of a piece inside the model's entities, with what the model read in them.

    `ranges` are the entities' characters in the piece. Only their words are kept: the ML filter
    reads no other, and a guard remembers thousands of texts, an agent's long tool results among
    them.
    """
    from polanon.guards._span_features import Word

    ranges = sorted(ranges)
    at = 0
    for token in document:
        if token.is_space:
            continue
        start, end = token.idx, token.idx + len(token.text)
        while at < len(ranges) and ranges[at][1] <= start:
            at += 1
        if at < len(ranges) and ranges[at][0] < end:
            yield offset + start, offset + end, Word.of(token)


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
        self,
        spacy_model: str = "pl_core_news_md",
        cache_texts: int = CACHE_TEXTS,
        mode: Mode = "normal",
        ml_filter: bool = True,
    ) -> None:
        """Store the model name; the pipeline itself loads on first use.

        Args:
            spacy_model: the spaCy package or directory to load.
            cache_texts: how many texts to remember what the model found in, so a text sent again,
                as a chat's history is, is not read again; 0 reads every text every time. A text
                is remembered by its SHA-256 digest, never as written.
            mode: "normal" leaves out what the Polish model reads as a name in English prose,
                in code, in identifiers and in a place written in lowercase, where nearly all of
                it is ordinary words; a name opening an English sentence and "jadę do gdyni" go
                out as written. "paranoid" keeps every span with a letter in it. In the calls of
                an AI agent in ``benchmarks/run_agent.py``, "normal" lets 3 of 243 values
                through and 2% of the token uses hide nothing, "paranoid" 1 and more than half.
            ml_filter: in the normal mode, let the ML filter of
                :mod:`polanon.guards._span_features`, a logistic regression, drop more of what the
                hand-written filters keep, weighing the span, its words, the words around it,
                their vectors and what the other guards found there. On OpenPII's test part it
                leaves 306 spans outside the annotations instead of 434 and masks the same 3,733
                values whole; on a part held out until it was done, 400 instead of 539, no value
                lost. It adds about half a millisecond to a text the model takes 12 to read. On
                by default; False leaves the decision to the hand-written filters alone.

        Raises:
            ValueError: for a mode that is neither "normal" nor "paranoid".
            TypeError: for an `ml_filter` that is not True or False.
        """
        check_mode(mode)
        if not isinstance(ml_filter, bool):
            raise TypeError(f"ml_filter is True or False, not {ml_filter!r}")
        self.spacy_model = spacy_model
        self.cache_texts = cache_texts
        self.mode = mode
        self.ml_filter = ml_filter
        self._nlp: Any | None = None
        self._found: OrderedDict[tuple[str, frozenset[str]], tuple[Span, ...]] = OrderedDict()
        #: What the model read in every word of the texts it read lately, for the ML filter.
        self._words: OrderedDict[str, tuple[tuple[int, int, Any], ...]] = OrderedDict()

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
        spans = self._read(text, wanted, digest)
        if self.cache_texts > 0:
            with _CACHE_LOCK:
                self._found[key] = tuple(spans)
                while len(self._found) > self.cache_texts:
                    self._found.popitem(last=False)
        return spans

    def _filters_by_ml(self) -> bool:
        """Whether the ML filter reviews the spans: on, in the normal mode, with its own model.

        Its weights read the word vectors of the model it was fitted on, `pl_core_news_md`.
        Another model's vectors are another space, and the small model has none, so with any
        other model the hand-written filters decide alone, as with `ml_filter=False`.
        """
        pipeline = self._get_pipeline()
        meta = getattr(pipeline, "meta", None) or {}
        return (
            self.ml_filter
            and self.mode == "normal"
            and getattr(pipeline, "lang", "pl") == "pl"
            and meta.get("name", _ML_FILTER_MODEL) == _ML_FILTER_MODEL
        )

    def _read(self, text: str, wanted: frozenset[str] | set[str], digest: str = "") -> list[Span]:
        """Run the model over `text` and keep what reads as a name."""
        nlp = self._get_pipeline()
        keeps_words = self._filters_by_ml()
        spans: list[Span] = []
        words: list[tuple[int, int, Any]] = []
        for offset, piece in _chunks(text, CHUNK_CHARS):
            document = nlp(piece)
            ranges = []
            for entity in document.ents:
                entity_type = LABEL_MAP.get(entity.label_)
                if entity_type not in wanted:
                    continue
                ranges.append((entity.start_char, entity.end_char))
                start, end = offset + entity.start_char, offset + entity.end_char
                spans.extend(self._line_spans(text, start, end, str(entity_type)))
            if keeps_words and ranges:
                words.extend(_words_of(document, offset, ranges))
        if keeps_words:
            with _CACHE_LOCK:
                self._words[digest] = tuple(words)
                while len(self._words) > max(self.cache_texts, _WORDS_KEPT):
                    self._words.popitem(last=False)
        spans = _join_hyphenated(text, spans)
        if self.mode == "paranoid":
            return [span for span in spans if _has_a_letter(text[span.start : span.end])]
        if getattr(nlp, "lang", "pl") == "pl":
            spans = _without_english_misreadings(text, spans)
        return spans

    def _review(self, text: str, spans: list[Span], others: list[Span]) -> list[Span]:
        """Keep the spans the ML filter keeps, knowing what the other guards found.

        Called by the pipeline once every guard has read `text`. A text whose words are no longer
        remembered keeps every span: more masked, nothing let through.
        """
        if not spans or not self._filters_by_ml():
            return spans
        from polanon.guards._span_features import (
            line_bounds,
            ml_filter_keeps,
            span_context,
            span_features,
            span_vector,
        )

        digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
        with _CACHE_LOCK:
            words = self._words.get(digest)
        if words is None:
            return spans
        vocab = getattr(self._get_pipeline(), "vocab", None)
        # What every span would read from the whole text, or from a line as long as the text, is
        # read once: per span, a text of 100 KB took 13 s to mask instead of 2.5 s.
        text_english = _is_english_text(text)
        repeated: dict[str, bool] = {}
        languages: dict[tuple[int, int], str] = {}
        word_starts = [start for start, _, _ in words]
        word_ends = [end for _, end, _ in words]
        rule_spans = sorted(others, key=lambda other: other.start)
        rule_starts = [other.start for other in rule_spans]
        rule_reach = list(accumulate((other.end for other in rule_spans), max))
        kept = []
        for span in spans:
            fragment = text[span.start : span.end]
            if fragment not in repeated:
                repeated[fragment] = text.count(fragment) > 1
            line = line_bounds(text, span.start)
            if line not in languages:
                languages[line] = _language(text[slice(*line)])
            before = bisect_left(rule_starts, span.end)
            features = span_features(
                text,
                span,
                rules_found_it=before > 0 and rule_reach[before - 1] > span.start,
                words=[
                    word
                    for _, _, word in words[
                        bisect_right(word_ends, span.start) : bisect_left(word_starts, span.end)
                    ]
                ],
                vocab=vocab,
                text_english=text_english,
                repeated=repeated[fragment],
                line_language=languages[line],
            )
            vector = span_vector(text[span.start : span.end], vocab) if vocab is not None else []
            context = span_context(text, span)
            if ml_filter_keeps(features, span.entity_type, vector, context):
                kept.append(span)
        return kept

    def _line_spans(self, text: str, start: int, end: int, entity_type: str) -> list[Span]:
        """Split an entity at line breaks and the columns of a table, and trim the whitespace.

        Models occasionally run an entity across a newline, which then swallows whatever sits on the
        next line. A name never spans two lines in the documents this library is written for, nor
        two cells of a CSV file or a Markdown table: across "FV/2026/10/014;Studio Lumen" the model
        read one company, and the invoice number was masked with it.
        """
        spans: list[Span] = []
        cursor = start
        for piece in _CELL_BREAKS.split(text[start:end]):
            piece_start = cursor
            cursor += len(piece) + 1
            stripped = piece.strip().strip(_EAST_ASIAN_MARKS).strip()
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
