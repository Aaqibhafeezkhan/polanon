r"""Polish company names, found by their legal form rather than by a model.

Statistical NER treats a Polish company name as what it looks like word by word, so "Malinowski
Handel S.A." comes back as a surname, a common noun and an abbreviation, and the mask ends up
covering only part of the name. A legal form is unambiguous, though: nothing but a company is called
``Sp. z o.o.``, ``S.A.`` or ``Sp. j.``, and the name is the run of capitalised words in front of it,
or the quoted name the way contracts and court registers write it: „ABC” Sp. z o.o.

How far that run reaches is the judgement call, and it is kept inside one sentence and one line.
A signature block puts the person, the job title and the company on separate lines, and a heading
or the previous sentence ends right before the name; letting the run cross either would hide a
person inside a company token and give one company two tokens. So name words are joined by spaces,
an ordinary word never ends in a full stop, a document heading ("Faktura", "Umowa") or a
preposition is never part of a name, and another company's legal form ("Muster GmbH i ...") ends
it. Walking back from a legal form, a line break is crossed only where a long line was wrapped, as
in a PDF or a hard-wrapped e-mail, since a signature's lines are short; the legal form alone may
always stand on the next line. Walking forward from "Fundacja" or "Kancelaria" never leaves the
line, because what follows a line break there is as often the signatory as the rest of the name.

Scoring sits at 0.95: above the NER guard, so this span replaces the fragments it produced, and
below the checksum-validated identifiers, so a NIP sitting inside a company block still wins.
"""

from __future__ import annotations

import re

from polanon.guards.base import Guard
from polanon.types import Span

_UPPER = "A-ZĄĆĘŁŃÓŚŹŻ"
_LETTERS = r"[\w&'\u2019-]"

#: Spaces between the words of one name, including the no-break space Polish typesetting puts
#: after short words.
_GAP = r"[ \t\u00a0]+"

#: Before a legal form, also a line break that wraps a long line: the line it ends holds at least
#: fifty characters, which a signature's "Jan Kowalski" or "Kierownik Sprzedaży" never does.
_JOIN = rf"(?:{_GAP}|[ \t\u00a0]*(?<=[^\n]{{50}})\r?\n[ \t\u00a0]*)"

#: Where a name may start: not inside a word, an abbreviation or a web address. Anchoring the start
#: also keeps a long token such as a base64 blob from being tried at each of its capitals.
_START = r"(?<![\w&'\u2019.-])"

#: Legal forms written after the name. Spelled out with explicit case classes rather than with
#: IGNORECASE, because the flag would also drop the capitalisation requirement on the name.
_SUFFIX_FORMS = "|".join(
    (
        # Sp. z o.o. in every spelling people use: sp. z o.o, Sp.z.o.o., sp z oo, SP. Z O.O.,
        # Spółka z o.o., and the limited partnership built on it, sp. z o.o. sp. k.
        r"[Ss][Pp](?:[ÓóOo][ŁłLl](?:[Kk]\w{1,2}|[Cc][Ee]))?\.?[ \t\u00a0]*[Zz]\.?[ \t\u00a0]*"
        r"[Oo]\.?[ \t\u00a0]*[Oo]\.?(?:[ \t\u00a0]*[Ss][Pp]\.?[ \t\u00a0]*[Kk]\.?)?",
        r"[Ss]p[óo][łl](?:k[aiąęe]|ce)[ \t\u00a0]+z[ \t\u00a0]+ograniczon\w+[ \t\u00a0]+"
        r"odpowiedzialno\w+",
        r"[Ss]p[óo][łl](?:k[aiąęe]|ce)[ \t\u00a0]+(?:[Jj]awn|[Kk]omandytow(?:o-[Aa]kcyjn)?"
        r"|[Pp]artnersk|[Cc]ywiln|[Aa]kcyjn)\w*",
        r"S\.?[ \t\u00a0]?K\.?[ \t\u00a0]?A\.?",
        r"S\.[ \t\u00a0]?A\.?",
        r"[Ss][Pp]\.[ \t\u00a0]*[JjKkPp]\.?|[Ss][Pp][ \t\u00a0]+[JjKkPp]\.",
        r"[Ss]\.[ \t\u00a0]*[Cc]\.",
        r"P\.[ \t\u00a0]*P\.[ \t\u00a0]*H\.[ \t\u00a0]*U\.?",
    )
)

#: Foreign legal forms. They are masked by the international guard; here they only end the name of
#: a Polish company that follows them, so "Muster GmbH i Kowalski Handel Sp. z o.o." is two names.
_FOREIGN_FORMS = r"GmbH|AG|KGaA|OHG|Ltd|Inc|Corp|LLC|LLP|PLC|B\.V|N\.V|S\.p\.A|S\.r\.l|SARL|Oyj"

#: Trade abbreviations written before a sole trader's name: PPHU Kowalczyk, P.H.U. Kowalczyk.
_TRADE_ABBREVIATIONS = (
    r"(?:P\.?P\.?H\.?U|P\.?P\.?U\.?H|[FZ]\.?P\.?H\.?U|[PF]\.?H\.?U|[PZF]\.?U\.?H)\.?"
)

#: Legal forms written before the name.
_PREFIX_FORMS = "|".join(
    (
        r"Fundacja",
        r"Stowarzyszenie",
        r"Spółdzielnia",
        rf"Kancelaria(?:{_GAP}(?:Adwokacka|Radcy{_GAP}Prawnego|Notarialna|Prawna))?",
        rf"Przedsiębiorstwo(?:{_GAP}\w+)?",
        _TRADE_ABBREVIATIONS,
        rf"Zakłady?(?:{_GAP}\w+)?",
    )
)

#: Where a legal form ends: not inside a longer word or abbreviation ("SAMSUNG", "S.A.R.L.").
_END = r"(?!\.?[\w'\u2019-])"

#: Words that are never part of a company's name, whatever their case: "Faktura VAT Budex Sp. z
#: o.o." names Budex, and "Szanowni Państwo" opens a letter.
_NEVER_NAME_WORDS = r"VAT|PILNE|Re|Fwd?|Dotyczy|Temat|Pan|Pani|Państwo|Szanown[iya]"

#: Document headings and prepositions. Written like the words around them they belong to a name
#: ("Restauracja Przy Moście", "Fundacja Dla Polski", "Raport Consulting"), so they are refused
#: only in a shouted line, where they are capitals followed by capitals: "PROSZĘ O FAKTURĘ DLA BUDEX
#: SP. Z O.O." names Budex alone.
_HEADINGS_AND_PREPOSITIONS = "|".join(
    (
        r"Faktur[aęy]|Umow[aęy]|Ofert[aęy]|Zamówienie|Zlecenie|Rachunek|Nota|Korekta|Paragon",
        r"Aneks|Protokół|Raport|Pismo|Wezwanie|Reklamacja|Załącznik|Przelew|Wyrok|Postanowienie",
        r"Uchwała|Dostawca|Odbiorca|Nabywca|Sprzedawca|Kupujący|Klient|Kontrahent|Zleceniodawca",
        r"Zleceniobiorca|Wykonawca|Zamawiający|Wystawca|Płatnik|Pracodawca|Najemca|Wynajmujący",
        r"W|We|Z|Ze|Do|Od|Dla|Na|Po|Przez|Przy|U|O|Za",
    )
)
_SHOUTED = rf"(?=[{_UPPER}]+{_END})(?i:{_HEADINGS_AND_PREPOSITIONS}){_END}{_GAP}[{_UPPER}]{{2}}"

#: A word of a name. It starts with a capital, or with one lowercase letter before it as in
#: "eBudex"; dots may join its parts ("Kowalczyk.pl") but never end it, since a full stop after an
#: ordinary word ends a sentence. Initials and dotted abbreviations ("A.", "P.H.U.") keep theirs.
#: The capital is checked first, so the refusals behind it run only where a word can start; asked
#: at every position of a long line, they made this guard thirty times slower.
_WORD = (
    rf"(?=[a-z]?[{_UPPER}])"
    rf"(?!(?:{_SUFFIX_FORMS}|(?i:{_NEVER_NAME_WORDS})){_END})"
    # Another company's legal form ends a name only where a second name is joined to it.
    rf"(?!(?:{_FOREIGN_FORMS}){_END}{_GAP}[i&]{_GAP})"
    rf"(?!{_SHOUTED})"
    rf"(?:[a-z]?[{_UPPER}]{_LETTERS}*(?:\.{_LETTERS}+)*|(?:[{_UPPER}]\.)+)"
)

#: Inside a name, after its first word, a bracketed qualifier: "Kowalczyk (Polska)".
_INNER_WORD = rf"(?:{_WORD}|\([{_UPPER}][^()\n]{{0,40}}\))"

#: A number ends a name ("Budownictwo 2000 Sp. z o.o."); anywhere else it is more likely an amount
#: or a date standing between a sentence's first word and the company.
_NUMBER = rf"\d{_LETTERS}*"

#: The first word of a name, never inside another word.
_FIRST_WORD = rf"{_START}{_WORD}"

#: Words that join two halves of one name ("Nowak i Wspólnicy", "Kowalski & Synowie").
_CONNECTOR = r"(?:i|&)"

#: At most eight words: enough for "Kowalczyk Biuro Rachunkowe Doradztwo Podatkowe Kadry", and a
#: bound that keeps a long line of capitalised words linear to scan.
_RUN = rf"{_FIRST_WORD}(?:{_JOIN}(?:{_CONNECTOR}{_GAP})?{_INNER_WORD}){{0,7}}(?:{_GAP}{_NUMBER})?"

#: A quoted name, the way contracts and registers write it. The quotes hold any words, lowercase
#: ones included ("„Dom i ogród”"), up to a line's worth.
_QUOTE_CHARS = '"„“”«»'
_QUOTED = (
    rf"{_START}[{_QUOTE_CHARS}][^\s{_QUOTE_CHARS}]"
    rf"(?:[^\n{_QUOTE_CHARS}]{{0,58}}[^\s{_QUOTE_CHARS}])?[{_QUOTE_CHARS}]"
)

#: A name in front of a legal form: a run of words, or a quoted name with up to three words before
#: it ("Przedsiębiorstwo Handlowe „ABC”").
_NAME = rf"(?:(?:{_FIRST_WORD}(?:{_GAP}{_INNER_WORD}){{0,2}}{_GAP})?{_QUOTED}|{_RUN})"

#: Before the legal form, the name may wrap onto the next line once.
_BEFORE_FORM = r"(?:[ \t\u00a0]+|[ \t\u00a0]*\r?\n[ \t\u00a0]*)"

_SUFFIX_RE = re.compile(rf"(?P<value>{_NAME}{_BEFORE_FORM}(?:{_SUFFIX_FORMS})){_END}")
_PREFIX_RE = re.compile(
    rf"{_START}(?P<value>(?:{_PREFIX_FORMS}){_END}{_GAP}"
    rf"(?:{_QUOTED}|{_INNER_WORD}(?:{_GAP}(?:{_CONNECTOR}{_GAP})?{_INNER_WORD}){{0,6}}))"
)

#: Words a court's or an office's name follows in a citation: "zgodnie z wyrokiem SA w Krakowie".
_CITING_WORDS = r"(?i:wyrok\w*|postanowieni\w*|uzasadnieni\w*|orzeczeni\w*|apelacj\w*|opłat\w*)"

#: "SA" without dots is a word as well as a legal form: "Wyrokiem SA w Krakowie" cites a court of
#: appeal and "Opłata SA" is a charge. Without dots it is taken after two words of a name ("Bank
#: Pekao SA"), or after one that does not open a sentence ("zamówienie od Orlen SA").
_SA_AFTER_NAME_RE = re.compile(
    rf"{_START}(?P<value>{_WORD}(?:{_JOIN}(?:{_CONNECTOR}{_GAP})?{_INNER_WORD}){{1,7}}{_GAP}SA)"
    rf"{_END}"
)
_SA_MID_SENTENCE_RE = re.compile(
    rf"(?<=[a-ząćęłńóśźż0-9,:;][ \t])(?!{_CITING_WORDS}){_START}(?P<value>{_WORD}{_GAP}SA){_END}"
)

#: Confidence: a legal form is certain, the extent of the name in front of it is a judgement call.
COMPANY_SCORE = 0.95


class CompanyGuard(Guard):
    """Masks company names anchored on a Polish legal form as ``ORG``."""

    name = "company_pl"

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return one span per company name, covering the name and its legal form."""
        if "ORG" not in enabled:
            return []
        spans = [
            Span(m.start("value"), m.end("value"), "ORG", COMPANY_SCORE, self.name)
            for pattern in (_SUFFIX_RE, _PREFIX_RE, _SA_AFTER_NAME_RE, _SA_MID_SENTENCE_RE)
            for m in pattern.finditer(text)
        ]
        return _coalesce(spans)


def _coalesce(spans: list[Span]) -> list[Span]:
    """Join overlapping spans into their union.

    The patterns often describe the same organisation from opposite ends, as in "Kancelaria
    Adwokacka Nowak i Wspólnicy s.c.". The pipeline's merge would join them as well, but callers
    look at a guard's spans before that merge — the allowlist decides on each detection — so the
    guard promises one span per organisation itself.
    """
    merged: list[Span] = []
    for span in sorted(spans):
        if merged and span.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = Span(
                previous.start, max(previous.end, span.end), "ORG", COMPANY_SCORE, previous.source
            )
            continue
        merged.append(span)
    return merged
