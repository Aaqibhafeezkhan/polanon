r"""Foreign invoices and letters: the formats the Polish guards do not know.

Sent through a real provider, English, UK, Swedish and Swiss sample invoices lost company names
with Inc/Ltd suffixes, street addresses, phone numbers, VAT and company numbers, and Bankgiro and
sort-code accounts, because every detector here was written for Polish paperwork.

Most of these values have no checksum anyone publishes, and a bare eight-digit number could be
anything, so detection leans on what an invoice always prints next to them: the label ("VAT No",
"Sort Code", "Org nr"), the country prefix of an EU VAT number, a street suffix, a legal form.
Values are matched within one line; a label never reaches into the next line except where a
header such as "Invoiced To" or a closing formula is followed by the name below it.

The text these rules read is mostly Polish, and Polish business prose is full of shapes a foreign
rule would take for a value: a year opening a heading ("2024 Podsumowanie"), a signed amount
("+12 345 678 zł"), an invoice number after "Faktura VAT", a "sterownik PLC" or a "klasa AB".
So a shape that is also ordinary text has to be vouched for by its surroundings: a bare postcode
by the street line above it, a short legal form by a name that is not merely the first word of a
sentence, a tax number by a label that names an identifier rather than the tax.

Scores follow the rest of the library: identifiers a label or a format vouches for score 1.0, like
the Polish phone and address rules; a company found by its legal form 0.95; a name found by a form
of address or a label 0.9.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from polanon.guards.base import Guard
from polanon.types import Span

#: A test of one match: a validator of the value, or a look at the text around it.
Check = Callable[[re.Match[str]], bool]

# ---------- validators ----------


def _digit_count(value: str) -> int:
    return sum(character.isdigit() for character in value)


def _on_value(validator: Callable[[str], bool]) -> Check:
    """Apply a validator of the matched value to the whole match."""
    return lambda match: validator(match.group("value"))


def _has_digits(low: int, high: int) -> Check:
    """Return a check accepting values with `low` to `high` digits."""
    return _on_value(lambda value: low <= _digit_count(value) <= high)


def _all_of(*checks: Check) -> Check:
    """Return a check that passes when every one of `checks` does."""
    return lambda match: all(check(match) for check in checks)


def is_valid_aba_routing(value: str) -> bool:
    """Return True for a nine-digit US routing number whose 3-7-1 checksum holds."""
    digits = [int(character) for character in value if character.isdigit()]
    if len(digits) != 9:
        return False
    weights = (3, 7, 1) * 3
    return sum(d * w for d, w in zip(digits, weights, strict=True)) % 10 == 0


def is_valid_uk_postcode_inward(value: str) -> bool:
    """Return True when the last three characters are a UK inward code.

    The two letters of an inward code never include C, I, K, M, O or V, which keeps "A4 1CM"-shaped
    product sizes from passing as postcodes.
    """
    inward = value.replace(" ", "")[-3:]
    return inward[0].isdigit() and not set(inward[1:]) & set("CIKMOV")


def _words(block: str) -> frozenset[str]:
    """Split a whitespace-separated block, which keeps long word lists readable."""
    return frozenset(block.split())


# ---------- building blocks ----------

#: Between a label and its value: spaces or tabs and at most one punctuation mark, never a line
#: break, because values in extracted documents are laid out in columns.
_SEP = r"[ \t]*[:#.]?[ \t]*"

#: Capital letters of the Latin alphabets foreign names, firms and towns are written in.
_UPPER = "A-ZÀ-ÖØ-ÞĀĂĄĆĈĊČĎĐĒĔĖĘĚĜĞĠĢĤĦĨĪĬĮİĶĹĻĽĿŁŃŅŇŊŌŎŐŒŔŖŘŚŜŞŠŢŤŦŨŪŬŮŰŲŴŶŸŹŻŽ"

#: Any letter that is not one of those capitals.
_LOWER = rf"[^\W\d_{_UPPER}]"

#: Country-prefixed VAT numbers in their official shapes, compact or with the separators each
#: country prints ("BE 0123.456.789", "DK 12 34 56 78", "FR 12 345 678 901"). Polish "PL" numbers
#: are left to the NIP rule, which validates the checksum.
_EU_VAT = (
    r"ATU[ ]?\d{8}|BE[ ]?[01]?\d{3}[ .]?\d{3}[ .]?\d{3}|BG\d{9,10}|CY\d{8}[A-Z]|CZ[ ]?\d{8,10}"
    r"|DE\d{9}|DK[ ]?\d{2}(?:[ ]?\d{2}){3}|EE\d{9}|EL\d{9}|ES[A-Z0-9]\d{7}[A-Z0-9]|FI\d{8}"
    r"|FR[ ]?[A-HJ-NP-Z0-9]{2}(?:[ ]?\d{3}){3}|HR\d{11}|HU\d{8}|IE\d[A-Z0-9+*]\d{5}[A-Z]{1,2}"
    r"|IT\d{11}|LT(?:\d{9}|\d{12})|LU\d{8}|LV\d{11}|MT\d{8}|NL[ ]?\d{4}\.?\d{2}\.?\d{3}\.?B\.?\d{2}"
    r"|PT\d{9}|RO\d{6,10}|SE[ ]?\d{6}-?\d{4}[ ]?01|SI\d{8}|SK[ ]?\d{10}|XI\d{9}|EU\d{9}"
    r"|GB(?:\d{9}|\d{12}|GD\d{3}|HA\d{3})|GB[ ]?\d{3}[ ]?\d{4}[ ]?\d{2}"
    r"|CHE[-. ]?\d{3}\.?\d{3}\.?\d{3}(?:[ ]?(?:MWST|TVA|IVA))?|NO\d{9}(?:[ ]?MVA)?"
)

#: Currencies and magnitudes that turn a number into an amount: "+12 345 678 zł".
_AMOUNT_UNIT = r"zł|zl\b|PLN|EUR|USD|GBP|CHF|SEK|NOK|DKK|CZK|HUF|€|\$|£|%|tys\b|mln\b|mld\b"

#: Names of the tax itself. On their own they label an amount or the kind of an invoice
#: ("Kwota VAT 1 234,56", "Faktura VAT 12/2024") as often as a number.
_TAX_NAME = r"VAT|MwSt|USt|TVA|IVA|BTW|Moms|MVA|Tax"

#: Words that turn a tax name into the name of an identifier: "VAT No", "USt-IdNr", "Moms reg.nr".
_TAX_QUALIFIER = (
    r"Reg(?:istration)?|IdNr|ID|Nr|No|Number|Nummer|Num|Code|reg\.?nr|intracom(?:munautaire)?"
)

#: A label that names a tax identifier, in the languages the samples came in.
_TAX_ID_LABEL = (
    rf"(?i:\b(?:UID|EIN|TIN|UTR|(?:{_TAX_NAME})(?:[ \t]*[/-]?[ \t]*(?:{_TAX_NAME})){{0,2}}"
    rf"(?:[ \t]*[/-]?[ \t]*(?:{_TAX_QUALIFIER})\.?){{1,3}})\.?)"
)

#: A bare tax name, which labels a number only when a colon or a hash follows it ("VAT: ...").
_TAX_NAME_LABEL = rf"(?i:\b(?:{_TAX_NAME})(?:[ \t]*[/-]?[ \t]*(?:{_TAX_NAME})){{0,2}})"

#: A tax number after its label: an optional country prefix, then digits in groups. A value that
#: runs on into a decimal part, a slash, more digits or a currency is an amount or an invoice
#: number.
_TAX_VALUE = (
    r"(?P<value>(?:[A-Z]{2,3}[ -]?)?\d+(?:[ .-]\d+)*"
    r"(?:[A-Z]{1,2}\b|[ ]?(?:MWST|TVA|IVA|MVA)\b|\.?B\.?\d{2})?)"
    rf"(?![ .,/-]?\d|[ \t]*(?:{_AMOUNT_UNIT}))"
)

_COMPANY_LABEL = (
    r"(?i:\b(?:Org(?:anisation|anization)?\.?[ \t]*(?:nr|no|number)|Organisationsnummer"
    r"|(?<!VAT )Company[ \t]+(?:Reg(?:istration)?\.?[ \t]+)?(?:No|Number|Nr)"
    r"|(?:Company[ \t]+)?Registered[ \t]+in[ \t]+England(?:[ \t]+(?:and|&)[ \t]+Wales)?"
    r"(?:[ \t]+No)?|(?<!VAT )Reg(?:istration)?\.?[ \t]+(?:No|Nr|Number)|CRN"
    r"|Companies[ \t]+House[ \t]+(?:No|Number)|KvK(?:-nummer)?|SIRE[NT]|CVR(?:-nr)?"
    r"|Y-tunnus|ABN|ACN|I[ČC]O|IČ(?![ \t]*DPH)|Ondernemingsnummer"
    r"|Num[ée]ro[ \t]+d['\u2019][ \t]?entreprise|KBO(?:-nr)?|BCE)\b\.?)"
)

_ACCOUNT_LABEL = (
    r"(?i:\b(?:Acc(?:ount|t)?|A/C|Kontonummer|Kontonr|Compte|Conto|Rekeningnummer)\.?"
    r"(?:[ \t]*(?:No|Nr|Number|Num|#))?\.?)"
)

_PHONE_LABEL = (
    r"(?i:\b(?:(?:Telefon|Mobil|Ruf|Handy|Fax|Tel)(?:nummer|nr)|T[eé]l(?:ephone|efon)?|Phone|Fax"
    r"|Mobile?|Mob|Cell|Handy|Tfn|Puh|T[ée]l[ée]phone)(?:\.?[ \t-]*(?:No|Nr|Number))?\.?"
    r"|\b[TFM]\.?(?=[ \t]*:))"
)

#: English street suffixes that follow the name of the street.
_STREET_SUFFIX = (
    r"Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way|Walk|Place|Pl|Court|Ct"
    r"|Square|Sq|Terrace|Close|Crescent|Parkway|Pkwy|Highway|Hwy|Row|Mews|Gardens|Grove|Hill"
    r"|Circle|Cir|Trail|Wharf|Quay|Parade"
)

#: Articles inside a French, Italian, Spanish or Portuguese street name: "rue de la Paix".
_ROMANCE_ARTICLE = r"(?:(?:de|du|des|la|le|les|del|della|dei|di|da|dos|das|do)[ \t]+|l'|d')"

#: A capitalised word in a proper name. A match never starts inside a hyphen- or apostrophe-joined
#: token (see the lookbehinds below): every "-X" of a long base64url blob would otherwise be a new
#: start that rescans the rest of the token, which made the rules quadratic.
_CAP = rf"[{_UPPER}][\w&'\u2019.-]*"

#: US states and territories, for the "KS 66102" line of an American address.
_US_STATES = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV"
    "|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY"
)

#: "Limited" is a legal form, but "Casio Limited Edition" is a watch and "Limited Warranty" a term.
_NOT_AFTER_LIMITED = (
    r"(?![ \t]+(?i:edition|series|warranty|offer|time|access|availability|stock|run|release"
    r"|version)\b)"
)

#: Legal forms written after a company name outside Poland.
_LEGAL_FORMS = (
    rf"Inc\.?|Incorporated|Corp\.?|Corporation|Co\.|Ltd\.?|Limited{_NOT_AFTER_LIMITED}"
    r"|L\.?L\.?C\.?|LLP|PLC|plc|GmbH(?:[ \t]*&[ \t]*Co\.?[ \t]*KG)?|AG|KGaA|KG|OHG|e\.K\.|SE"
    r"|S\.?A\.?R\.?L\.?|SA|S\.à[ \t]?r\.l\.|SAS|S\.p\.A\.|S\.r\.l\.|S\.L\.|B\.V\.|BV|N\.V\.|NV|AB"
    r"|ASA|A/S|ApS|Oyj?|Pty\.?[ \t]+Ltd\.?|s\.r\.o\.|a\.s\.|Kft\.|Zrt\.|d\.o\.o\."
)

#: The mixed-case legal forms as a letterhead or a bank line prints them in capitals.
_CAPS_LEGAL_FORMS = (
    rf"LTD\.?|LIMITED{_NOT_AFTER_LIMITED}|INC\.?|INCORPORATED|CORP\.?|CORPORATION"
    r"|GMBH(?:[ \t]*&[ \t]*CO\.?[ \t]*KG)?|SARL|PTY\.?[ \t]+LTD\.?|S\.P\.A\.|S\.R\.L\."
)

#: Short legal forms that are also everyday abbreviations in Polish text: a "sterownik PLC" is a
#: controller, "klasa AB" a building class, "KG" a kilogram, "SAS" a disk interface, "SA" a court.
_AMBIGUOUS_FORMS = frozenset({"AB", "AG", "SE", "KG", "SAS", "PLC", "NV", "BV", "SA"})

#: Table headers a unit follows: "Masa netto KG" is a column, not a limited partnership.
_MEASURE_WORDS = _words("netto brutto masa waga ciężar ilość cena net gross weight")

#: Polish nouns an ambiguous abbreviation qualifies in prose: "Sterownik PLC", "Klasa AB budynku",
#: "Raport SAS z analizy". Listed, because refusing every one-word name that opens a sentence also
#: refused "Siemens AG potwierdził zamówienie", the commonest sentence of a business e-mail.
_PROSE_NOUNS = _words(
    """
    sterownik sterownika sterowniki sterownikiem kontroler kontrolera moduł modułu klasa klasy
    klasie klasę raport raportu raporcie dysk dysku interfejs interfejsu protokół protokołu port
    portu karta karty norma normy standard standardu tryb trybu typ typu wersja wersji model modelu
    seria serii system systemu masa waga zestaw zestawu pakiet pakietu wariant wariantu edycja
    edycji
    """
)

#: Spelled-out forms that also end ordinary capitalised prose: "QUANTITIES ARE LIMITED".
_SPELLED_OUT_FORMS = frozenset({"LIMITED", "CORPORATION", "INCORPORATED"})

#: Words of capitalised prose that never sit inside a company name.
_PROSE_WORDS = _words("is are was were be been not no to by at in on it this that all any our we")

#: Lowercase words that join the parts of one company name: "Bank of the West", "K & S",
#: "Banco de Sabadell". Not "for": "Invoice for Acme Ltd" would take the document into the name.
_ORG_CONNECTOR = r"&|and|und|et|och|og|of(?:[ \t]+the)?|de|du|des|del|la|van|von|der"

#: Words that open a Polish sentence but never begin a company name: "Dziękujemy Acme Ltd",
#: "Wczoraj Siemens AG", "Firma Siemens AG". Polish capitalises them only because they come first,
#: and taking them into the name both masks the verb and gives one company two tokens.
_POLISH_LEADING_WORDS = _words(
    """
    w we z ze do od dla na po przez przed za pod nad przy bez u o ku między wobec według zgodnie
    wraz oprócz poza dzięki mimo wśród
    oraz ale lecz jednak natomiast także również też ponadto dodatkowo czy gdy kiedy jeśli jeżeli
    ponieważ bo że aby żeby więc zatem dlatego albo lub ani nawet tylko jedynie właśnie
    oczywiście niestety szczególnie zwłaszcza
    wczoraj dzisiaj dziś jutro teraz obecnie wtedy potem następnie najpierw wkrótce ostatnio
    niedawno dotychczas nadal wciąż już jeszcze zawsze często
    ten ta to te tę tego tej temu tym tych nasz nasza nasze naszego naszej naszym nasi wasz wasza
    wasze waszej mój moja moje który która które którego której każdy każda każde kto gdzie jak
    jest są był była było były byli będzie będą został została zostało zostały zostali
    przesyłam załączam potwierdzam zapraszam polecam pozdrawiam wysyłam zamawiam czekam mam
    przypominam
    firma firmy firmie firmą firmę spółka spółki spółce spółką spółkę klient klienta klientowi
    klientem kontrahent kontrahenta kontrahentem dostawca dostawcy dostawcą odbiorca odbiorcy
    nabywca nabywcy sprzedawca sprzedawcy producent producenta zleceniodawca zleceniobiorca
    zarząd zarządu
    """
)

#: A Polish leading word, listed or recognised by a first-person verb ending that no foreign name
#: ends in: "Informujemy", "Prosimy", "Zamówiliśmy", "Proszę".
_LEADING_WORD = (
    rf"(?i:(?:{'|'.join(sorted(_POLISH_LEADING_WORDS))})"
    r"|[^\W\d_]{2,}(?:jemy|amy|imy|ymy|śmy)|[^\W\d_]{3,}ę)(?![\w'\u2019.&-])"
)

#: Up to five capitalised words of a company name, never starting on a Polish leading word.
_ORG_NAME = rf"(?!{_LEADING_WORD}){_CAP}(?:[ \t]+(?:{_CAP}|{_ORG_CONNECTOR})){{0,4}}"

#: The same for a name before a legal form in capitals, which also never takes in a word of
#: capitalised prose: "PAY TO ACME LTD" names "ACME LTD", "THIS WARRANTY IS LIMITED" nobody.
_NOT_PROSE = rf"(?!(?i:{'|'.join(sorted(_PROSE_WORDS))})(?![\w'\u2019.&-]))"
_CAPS_ORG_NAME = (
    rf"(?!{_LEADING_WORD}){_NOT_PROSE}{_CAP}"
    rf"(?:[ \t]+(?:{_NOT_PROSE}{_CAP}|{_ORG_CONNECTOR})){{0,4}}"
)

#: Words that end a name found after a label: "contact Customer Service" names a desk, not a
#: person, and "Bill to: Acme Widgets Ltd" a company the ORG rule takes whole.
_NOT_PERSON_WORDS = _words(
    """
    Customer Service Services Support Sales Team Department Dept Office Accounts Payable Receivable
    Billing Credit Control Finance Help Desk Center Centre Us Our Your The Same Above Below As
    Please Warehouse Manager Management Purchasing Procurement Reception Bank Company Ltd Limited
    Inc Corp GmbH Plc Llc
    """
)
_NOT_A_NAME = rf"(?i:(?:{'|'.join(sorted(_NOT_PERSON_WORDS))})(?![^\W\d_]))"

#: One capitalised word of a personal name in any Latin alphabet, with the prefixes that carry a
#: second capital ("McDonald", "MacLeod", "O'Neill", "DeVito") and double-barrelled parts
#: ("Schmidt-Weber", "Jean-Pierre"). The closing lookahead means a name the rule cannot read in
#: full is not matched at all: a fragment would be masked alone and leave the rest of it in clear.
_NAME_PART = rf"[{_UPPER}]{_LOWER}+"
_MIXED_NAME = rf"(?:Ma?c|Fitz|De|Di|Le|La|Van|[OD]['\u2019])?{_NAME_PART}(?:-{_NAME_PART})*"
_CAPS_NAME = rf"[{_UPPER}]{{2,}}(?:-[{_UPPER}]{{2,}})*"
_WORD_END = r"(?![^\W\d_]|['\u2019-][^\W\d_])"

_HONORIFIC = r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Sir|Dame|Herr|Frau|Mme|Mlle|Monsieur|Madame)\.?"
#: A form of address is never a word of the name: "Frau Prof. Schmidt" read "Prof" as her name,
#: and every "Prof" further down was masked as that person.
_NOT_A_TITLE = rf"(?!{_HONORIFIC}(?![^\W\d_]))"

_ADDRESSED_WORD = rf"(?!{_NOT_A_NAME}){_NOT_A_TITLE}{_MIXED_NAME}{_WORD_END}"
_NAME_WORD = rf"(?!{_NOT_A_NAME}){_NOT_A_TITLE}(?:{_MIXED_NAME}|{_CAPS_NAME}){_WORD_END}"
_INITIAL = rf"[{_UPPER}]\.[ \t]?"

#: Lowercase particles inside a surname: "van Dijk", "von der Leyen", "de la Cruz".
_PARTICLE = (
    r"(?:van|von|de|del|della|di|da|du|dos|das|des|ten|ter|zu|zum|zur|vom)"
    r"(?:[ \t]+(?:der|den|de|la|las|los|le))?[ \t]+"
)

#: The forms of address in front of a name after a label, kept out of the name: "z. Hd. Frau
#: Wiśniewska" read whole made "Frau" the first name, so every "Frau" further down was masked as
#: that person, and "Attn: Mr Oliver Bennett" got a token apart from "Oliver Bennett".
_TITLE_BEFORE = rf"(?:{_HONORIFIC}[ \t]+)*{_NOT_A_TITLE}"

#: A personal name: first and last name, or initials and a surname ("F.Coggle", "J. R. Smith").
#: Names in capitals count only with at least two words, as "OLIVER BENNETT" under a sign-off.
_PERSON_NAME = (
    rf"{_TITLE_BEFORE}(?P<value>(?:{_INITIAL})+{_NAME_WORD}"
    rf"|{_NAME_WORD}(?:[ \t]+(?:{_INITIAL})?(?:{_PARTICLE})?{_NAME_WORD}){{1,3}})"
)

#: After a form of address a surname alone is a name: "Dear Ms Carter", "Frau Wiśniewska",
#: "Mr van Dijk".
_ADDRESSED_NAME = (
    rf"(?P<value>(?:{_INITIAL})*(?:{_PARTICLE})?{_ADDRESSED_WORD}"
    rf"(?:[ \t]+(?:{_PARTICLE})?{_ADDRESSED_WORD}){{0,2}})"
)

_PERSON_LABEL = (
    r"(?i:\b(?:Attn|Attention|Contact(?:[ \t]+person)?|Kontaktperson|Ansprechpartner(?:in)?"
    r"|Your[ \t]+contact|Signed(?:[ \t]+by)?|Prepared[ \t]+by|Approved[ \t]+by"
    r"|Account[ \t]+manager|CEO|CFO|COO|CTO|Managing[ \t]+Director|Geschäftsführer(?:in)?"
    r"|Inhaber(?:in)?|Invoiced[ \t]+to|Bill[ \t]+to|Ship[ \t]+to|Sold[ \t]+to|c/o"
    r"|(?:Your|Our)[ \t]+ref(?:erence)?\.?|(?:Er|Vår)[ \t]+referens|z\.[ \t]?Hd\.)"
    r"|(?<!\w[ \t])(?<!\w)(?:(?:Full|Contact|Customer|Client|Account(?:[ \t]+holder)?)[ \t]+)?"
    r"name(?=[ \t]*:)|\bBeneficiary(?=[ \t]*:))"
)

#: Closing formulas; the name signed under them follows on the next line, after a blank line as
#: in a DIN 5008 letter, or after a comma on the same line. A bare "Thanks" or "Best" counts only
#: when it opens a line and nothing but a comma follows it.
_SIGN_OFF = (
    r"(?i:\b(?:Kind[ \t]+regards|Best[ \t]+regards|Warm[ \t]+regards|Regards"
    r"|Yours[ \t]+(?:sincerely|faithfully|truly)|Sincerely|Best[ \t]+wishes|Many[ \t]+thanks"
    r"|Mit[ \t]+(?:freundlichen|besten|herzlichen)[ \t]+Gr(?:ü|ue)(?:ß|ss)en"
    r"|(?:Viele|Beste|Freundliche|Liebe|Herzliche)[ \t]+Gr(?:ü|ue)(?:ß|ss)e"
    r"|Cordialement|Med[ \t]+vänliga[ \t]+hälsningar)"
    r"|(?<![^\n])[ \t]*(?:Thanks|Best)(?=,|[ \t]*\r?\n))"
)

# ---------- streets and postcodes ----------

#: "27 Kingfisher Road", "350 5th Avenue", "20 W 34th Street".
_STREET_EN = (
    rf"(?<![\w.,/-])\d{{1,5}}[A-Za-z]?[ \t]+(?:(?:{_CAP}|\d{{1,3}}(?:st|nd|rd|th))[ \t]+){{1,3}}"
    rf"(?:{_STREET_SUFFIX})\b\.?"
)

#: The suffix glued to the street name, as German, Nordic and Dutch streets are written, with the
#: adjective some of them carry in front and the hyphenated names of people: "Bahnhofstrasse 21",
#: "Kungsgatan 12", "Wiedner Hauptstraße 63", "Konrad-Adenauer-Straße 1".
_STREET_GLUED = (
    r"(?<![\w'\u2019-])(?:[A-ZÄÖÜ][a-zäöüß]+er[ \t]+)?[A-ZÄÖÜÅÆØ][\wäöüßåæøéè'-]*"
    r"(?:straße|strasse|str\.|gasse|weg|platz|allee|gatan|vägen|gränd|torget|laan|straat"
    r"|gracht|plein|kade|vej|gade|veien|gata"
    r"|-(?:Straße|Strasse|Str\.|Gasse|Weg|Platz|Allee|Ring|Damm|Ufer))[ \t]+\d{1,4}[a-zA-Z]?\b"
)

#: The German suffix written as a word of its own after an adjective: "Berliner Straße 12",
#: "Kölner Str. 5", "Frankfurter Allee 1", "Lange Gasse 3".
_STREET_DE_SPLIT = (
    r"(?<![\w'\u2019-])[A-ZÄÖÜ][a-zäöüß]+(?:er|e|en)[ \t]+"
    r"(?:Straße|Strasse|Str\.|Allee|Gasse|Weg|Platz|Ring|Damm|Ufer|Chaussee|Steig|Pfad)"
    r"[ \t]+\d{1,4}[a-zA-Z]?\b"
)

_STREET_ROMANCE = (
    r"\b(?:Rue|Avenue|Boulevard|Chemin|Via|Viale|Piazza|Corso|Calle|Avenida|Plaza|Paseo|Rua"
    rf"|Travessa|Largo)[ \t]+{_ROMANCE_ARTICLE}*{_CAP}(?:[ \t]+{_CAP}){{0,3}},?[ \t]+"
    r"\d{1,4}[a-zA-Z]?\b"
)

_STREET_ROMANCE_NUMBERED = (
    r"(?<![\w.,/-])\d{1,4}(?:[ \t]?(?:bis|ter))?,?[ \t]+"
    r"(?i:rue|avenue|boulevard|bd|chemin|place|impasse|allée|quai|route)[ \t]+"
    rf"{_ROMANCE_ARTICLE}*{_CAP}(?:[ \t]+{_CAP}){{0,3}}"
)

_PO_BOX = (
    r"\b(?:P\.?[ \t]?O\.?[ \t]?Box|Postfach|Postbus|Postboks|Postbox|Apartado"
    r"|Case[ \t]+postale|Casella[ \t]+postale)[ \t]+\d{1,6}\b"
)

_STREETS = (_STREET_EN, _STREET_GLUED, _STREET_DE_SPLIT, _STREET_ROMANCE, _STREET_ROMANCE_NUMBERED)

#: A line an address block puts above the postcode: a street the rules above know, a name ending
#: in a house number ("Damrak 1", "ul. Długa 145"), or a PO box.
_STREET_LINE = re.compile(
    "|".join(
        (
            *_STREETS,
            _PO_BOX,
            # Starting only where a word starts: retried at every letter of a long line with no
            # space, the run of letters was rescanned each time, quadratically.
            r"(?<![^\W\d_])[^\W\d_]{2,}\.?,?[ \t]+\d{1,3}[a-zA-Z]?"
            r"(?:[ \t]*[-/][ \t]*\d{1,4}[a-zA-Z]?)?[ \t]*,?$",
            r"^[ \t]*(?:Postfach|Postbus|Postboks|Postbox|Apartado|P\.?[ \t]?O\.?[ \t]?Box|Boks"
            r"|Box|Case[ \t]+postale|Casella[ \t]+postale|Boîte[ \t]+postale)\b",
        )
    )
)

#: Postcodes without a country prefix: Dutch "1012 LG", Portuguese "1100-053", the four or five
#: digits of most of Europe, and the Swedish, Czech and Greek "111 22".
_POSTCODE = r"\d{4}[ ]?(?!SA|SD|SS)[A-Z]{2}(?=[ \t])|\d{4}-\d{3}|\d{4,5}|\d{3}[ ]\d{2}"
_SPLIT_POSTCODE = re.compile(r"\d{3} \d{2}")

#: A town after its postcode: "Zürich", "PARIS", "Frankfurt am Main", "Halle (Saale)".
_CITY = rf"[{_UPPER}][^\W\d_]*\.?(?:[ \t/-]+\(?[^\W\d_]+\)?\.?)*"


def _previous_line(text: str, position: int) -> str:
    """Return the last non-blank line above the line holding `position`, or an empty string."""
    end = text.rfind("\n", 0, position)
    while end > 0:
        start = text.rfind("\n", 0, end) + 1
        line = text[start:end].strip()
        if line:
            return line
        end = start - 1
    return ""


#: Words that make a state code and five digits a ticket or an order rather than an address.
_NOT_A_TOWN = _words("id ok nr no order ticket status ref kod code numer number zamówienie")


def _after_a_town(match: re.Match[str]) -> bool:
    """Accept a state and ZIP code written without a comma only after a capitalised town name."""
    before = match.string[max(0, match.start("value") - 40) : match.start("value")].split()
    if not before:
        return False
    word = before[-1].strip(".:#")
    return word[:1].isupper() and word.lower() not in _NOT_A_TOWN


def _is_name_line(line: str) -> bool:
    """Return whether a line is a recipient's or a company's name: a few capitalised words.

    "Alpenblick Treuhand GmbH", "Frau Anna Muster", "Firma XYZ" — but not "Order 12345", and
    not a heading that ends in a colon ("Wyniki:").
    """
    words = line.split()
    return (
        1 <= len(words) <= 6
        and not line.endswith(":")
        and all(
            word[0].isupper() and not any(character.isdigit() for character in word)
            for word in words
        )
    )


def _below_a_street(match: re.Match[str]) -> bool:
    """Accept a postcode-and-town line only where an address block puts it.

    A line opening with four or five digits and a capitalised word is as often a year heading
    ("2024 Podsumowanie roku") or a quantity ("1500 Sztuk") as a postcode, and once masked the
    year would be replaced everywhere else in the conversation too. Under a street line it is a
    postcode, and so it is under a recipient's or a company's name, unless it reads as a year. The
    split "111 22" shape is never a year or an amount, so it needs no street.
    """
    value = match.group("value")
    if _SPLIT_POSTCODE.fullmatch(value):
        return True
    previous = _previous_line(match.string, match.start("value"))
    if _STREET_LINE.search(previous) is not None:
        return True
    looks_like_a_year = len(value) == 4 and value.isdigit() and 1900 <= int(value) <= 2099
    return _is_name_line(previous) and not looks_like_a_year


# ---------- companies ----------


#: Connectors the capitalised rule spells in capitals too: "SMITH AND SONS", "BANK OF SCOTLAND".
_CONNECTOR_WORDS = _words("& and und et of the")


def _name_words(name: str) -> list[str]:
    """Return the capitalised words of a company name, without its connectors."""
    return [
        word for word in name.split() if word[0].isupper() and word.lower() not in _CONNECTOR_WORDS
    ]


def _plausible_company(match: re.Match[str]) -> bool:
    """Reject a short, ambiguous legal form where nothing but the form says "company".

    "Sterownik PLC Siemens", "Klasa AB budynku" and "Raport SAS z analizy" are prose: a Polish noun
    an abbreviation qualifies. Any other single word before the form is a name, whether it opens a
    sentence ("Siemens AG potwierdził zamówienie") or not. A bare "SA" needs two words, since the
    Polish court of appeal is written "SA" too.
    """
    form = match.group("form")
    if form not in _AMBIGUOUS_FORMS:
        return True
    words = _name_words(match.group("name"))
    if form == "KG" and words[-1].lower() in _MEASURE_WORDS:
        return False
    if len(words) > 1:
        return True
    return form != "SA" and words[0].lower() not in _PROSE_NOUNS


def _plausible_caps_company(match: re.Match[str]) -> bool:
    """Reject one capitalised word before a spelled-out form: "TIME LIMITED" is an offer."""
    words = _name_words(match.group("name"))
    return match.group("form") not in _SPELLED_OUT_FORMS or len(words) > 1


# ---------- labels in front of the value ----------

_INVOICE_KIND = re.compile(r"(?i:faktur\w*)[ \t]+\Z")


def _not_an_invoice_kind(match: re.Match[str]) -> bool:
    """Reject "Faktura VAT nr 20240001": after "faktura" the tax names the kind of the invoice."""
    start = match.start()
    return _INVOICE_KIND.search(match.string, max(0, start - 16), start) is None


#: One rule: entity type, pattern, optional check of the match and the confidence of a match.
Rule = tuple[str, re.Pattern[str], Check | None, float]

RULES: list[Rule] = [
    # --- tax and company numbers ---
    (
        "TAX_ID",
        re.compile(_TAX_ID_LABEL + _SEP + _TAX_VALUE),
        _all_of(_has_digits(8, 14), _not_an_invoice_kind),
        1.0,
    ),
    (
        "TAX_ID",
        re.compile(_TAX_NAME_LABEL + r"[ \t]*[:#][ \t]*" + _TAX_VALUE),
        _all_of(_has_digits(8, 14), _not_an_invoice_kind),
        1.0,
    ),
    ("TAX_ID", re.compile(rf"(?<![\w-])(?P<value>{_EU_VAT})(?![\w-])"), None, 1.0),
    # The German Steuernummer, which §14 UStG puts on every invoice: "12/345/67890".
    (
        "TAX_ID",
        re.compile(
            r"(?i:\b(?:Steuernummer|Steuer-?Nr|St\.?[ \t]*-?[ \t]*Nr)\.?)"
            + _SEP
            + r"(?P<value>\d{2,3}/\d{3,4}/\d{4,5}|\d{10,13})(?![\d/])"
        ),
        None,
        1.0,
    ),
    (
        "COMPANY_ID",
        re.compile(_COMPANY_LABEL + _SEP + r"(?P<value>[A-Z]{0,2}\d+(?:[ .-]\d+)*)\b"),
        _has_digits(6, 14),
        1.0,
    ),
    # A German commercial register number, usually one to six digits, with the letters some courts
    # add: "HRB 12345", "HRB 98765 B". A legal form after it is not one of those letters.
    (
        "COMPANY_ID",
        re.compile(
            r"\bHR[AB][ \t]*(?:Nr\.?[ \t]*)?"
            r"(?P<value>\d{1,7}(?:[ \t](?!AG|SE|KG)[A-Z]{1,2}(?!\w))?)(?!\d)"
        ),
        None,
        1.0,
    ),
    # --- bank accounts ---
    (
        "BANK_ACCOUNT",
        re.compile(
            r"(?i:\b(?:Bankgiro(?:nummer|nr|t)?|BG)(?:[ \t]*-?[ \t]*(?:nr|no))?\.?)"
            + _SEP
            + r"(?P<value>\d{3,4}-\d{4})\b"
        ),
        None,
        1.0,
    ),
    (
        "BANK_ACCOUNT",
        re.compile(
            r"(?i:\b(?:Plusgiro(?:nummer|nr|t)?|Postgiro|PG)(?:[ \t]*-?[ \t]*(?:nr|no))?\.?)"
            r"[ \t]*[:.]?[ \t]*(?P<value>\d{1,2}(?:[ ]?\d{2}){0,3}-\d)\b"
        ),
        None,
        1.0,
    ),
    (
        "BANK_ACCOUNT",
        re.compile(r"(?i:\bSort[ \t-]*Code)" + _SEP + r"(?P<value>\d{2}[- ]?\d{2}[- ]?\d{2})\b"),
        None,
        1.0,
    ),
    (
        "BANK_ACCOUNT",
        # Up to 34 digits, the longest account an IBAN can carry: after an account label, a long
        # run of groups is the account, and refusing it left all of it in the clear.
        re.compile(_ACCOUNT_LABEL + _SEP + r"(?P<value>\d+(?:[ .-]\d+)*)\b"),
        _has_digits(6, 34),
        1.0,
    ),
    (
        "BANK_ACCOUNT",
        re.compile(
            r"(?i:\b(?:Routing|ABA|RTN)(?:[ \t]*(?:No|Number))?\.?)" + _SEP + r"(?P<value>\d{9})\b"
        ),
        _on_value(is_valid_aba_routing),
        1.0,
    ),
    (
        "BANK_ACCOUNT",
        re.compile(r"(?i:\b(?:Postkonto|PC-Konto|CCP))" + _SEP + r"(?P<value>\d{2}-\d{1,6}-\d)\b"),
        None,
        1.0,
    ),
    # --- phone numbers ---
    # A "+" and grouped digits is also how a Polish report writes a change: "+12 345 678 zł". A
    # currency, a decimal part or more digits after the groups make it an amount. A slash is not
    # one of them: "+44 20 7946 0958/9" is a number with its alternative extension.
    (
        "PHONE",
        re.compile(
            r"(?<![\w+])(?P<value>(?:\(\+\d{1,3}\)|\+\d{1,3}|00\d{1,3}(?=[ \t(]))[ \t]?"
            r"(?:\(0\)[ \t]?)?(?:\(?\d{1,4}\)?[ .-]?){1,5}\d{1,4})(?!\d)"
            rf"(?![ \t]*(?:{_AMOUNT_UNIT})|[ .]?\d|,\d)"
        ),
        _has_digits(8, 15),
        1.0,
    ),
    (
        "PHONE",
        re.compile(_PHONE_LABEL + _SEP + r"(?P<value>\(?\+?\d[\d ()./-]{4,20}\d)(?!\d)"),
        _has_digits(6, 15),
        1.0,
    ),
    (
        "PHONE",
        re.compile(r"(?<![\w.,])(?P<value>0\d{2,4}[ ]\d{3,4}[ ]\d{3,4})(?!\w|[.,]\d)"),
        _has_digits(10, 11),
        1.0,
    ),
    # --- addresses ---
    *[("ADDRESS", re.compile(rf"(?P<value>{street})"), None, 1.0) for street in _STREETS],
    ("ADDRESS", re.compile(rf"(?P<value>{_PO_BOX})"), None, 1.0),
    # --- postal codes ---
    (
        "POSTAL_CODE",
        re.compile(r"(?P<value>\b[A-Z]{1,2}\d[A-Z\d]?[ ]\d[A-Z]{2}\b)"),
        _on_value(is_valid_uk_postcode_inward),
        1.0,
    ),
    # Canada Post prints a space in the middle; without one the shape is a hex colour ("C0C0C0"),
    # unless a province code stands in front of it ("ON K1A0B1": the lookbehind spans the code,
    # the space and the three characters already read).
    (
        "POSTAL_CODE",
        re.compile(
            r"(?<![#\w])(?P<value>[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]"
            r"(?:[ ]|(?<=\b(?:AB|BC|MB|NB|NL|NS|NT|NU|ON|PE|QC|SK|YT)[ ]...))"
            r"\d[ABCEGHJ-NPRSTV-Z]\d)\b"
        ),
        None,
        1.0,
    ),
    # A US state with its ZIP code follows the town, with a comma ("Kansas City, KS 66102") or, as
    # the postal service recommends, without one ("NEW YORK NY 10118"). Without a town before it
    # "ID 12345" and "OK 54321" are ticket and order numbers.
    (
        "POSTAL_CODE",
        re.compile(
            r"(?:(?<=[^\W\d_],[ ])|(?<=[^\W\d_],[ ]{2})|(?<=[^\W\d_],\t))"
            rf"(?P<value>\b(?:{_US_STATES})[ \t]+\d{{5}}(?:-\d{{4}})?)\b"
        ),
        None,
        1.0,
    ),
    (
        "POSTAL_CODE",
        re.compile(rf"(?<=[^\W\d_][ \t])(?P<value>\b(?:{_US_STATES})[ \t]+\d{{5}}(?:-\d{{4}})?)\b"),
        _after_a_town,
        1.0,
    ),
    # Compact Canadian postcode after its label: "Postcode K1A0B1".
    (
        "POSTAL_CODE",
        re.compile(
            r"(?i:\b(?:post(?:al)?[ \t]*code|code[ \t]+postal))[ \t]*[:#]?[ \t]*"
            r"(?P<value>[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ ]?\d[ABCEGHJ-NPRSTV-Z]\d)\b"
        ),
        None,
        1.0,
    ),
    # A country prefix with the number of digits that country's postcodes have, so a model name
    # such as "A-320 Neo" or "E-2500 Kompakt" is not read as Austria or Spain.
    (
        "POSTAL_CODE",
        re.compile(
            r"(?<![\w-])(?P<value>(?:CH|AT|A|BE|B|LU|L|DK|NO|N|HU|H)-\d{4}"
            r"|(?:DE|D|FR|F|IT|I|ES|E|FI|FIN)-\d{5}|(?:SE|S|CZ|SK|GR)-\d{3}[ ]?\d{2}"
            r"|(?:PT|P)-\d{4}(?:-\d{3})?|NL-\d{4}(?:[ ]?[A-Z]{2})?)"
            rf"(?![\w-])(?=[ \t]+[{_UPPER}])"
        ),
        None,
        1.0,
    ),
    # A postcode and town on a line of their own, with an optional CEDEX and country.
    (
        "POSTAL_CODE",
        re.compile(
            rf"(?m)^[ \t]*(?P<value>{_POSTCODE})[ \t]+{_CITY}"
            rf"(?:[ \t]+CEDEX(?:[ \t]+\d{{1,2}})?)?(?:,[ \t]*{_CITY})?[ \t]*\r?$"
        ),
        _below_a_street,
        1.0,
    ),
    # A postcode after a street on the same line: "Hauptstraße 5, 10115 Berlin".
    (
        "POSTAL_CODE",
        re.compile(rf"(?:{'|'.join(_STREETS)}),[ \t]*(?P<value>{_POSTCODE})(?=[ \t]+[{_UPPER}])"),
        None,
        1.0,
    ),
    # --- organisations and people ---
    (
        "ORG",
        re.compile(
            rf"(?<![\w.'\u2019&-])(?P<value>(?P<name>{_ORG_NAME}),?[ \t]+"
            rf"(?P<form>{_LEGAL_FORMS}))(?!\w)"
        ),
        _plausible_company,
        0.95,
    ),
    (
        "ORG",
        re.compile(
            rf"(?<![\w.'\u2019&-])(?P<value>(?P<name>{_CAPS_ORG_NAME}),?[ \t]+"
            rf"(?P<form>{_CAPS_LEGAL_FORMS}))(?!\w)"
        ),
        _plausible_caps_company,
        0.95,
    ),
    (
        "PERSON",
        re.compile(rf"{_HONORIFIC}(?:[ \t]+{_HONORIFIC})*[ \t]+{_ADDRESSED_NAME}"),
        None,
        0.9,
    ),
    (
        "PERSON",
        re.compile(rf"{_PERSON_LABEL}[ \t]*[:-]?[ \t]*(?:\r?\n[ \t]*)?{_PERSON_NAME}"),
        None,
        0.9,
    ),
    (
        "PERSON",
        re.compile(rf"{_SIGN_OFF}(?:,?[ \t]*(?:\r?\n[ \t]*){{1,3}}|,[ \t]+){_PERSON_NAME}"),
        None,
        0.9,
    ),
]


class InternationalGuard(Guard):
    """Finds foreign tax, company and bank numbers, phones, addresses, companies and names."""

    name = "intl"

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return spans for every enabled rule whose match also passes its check."""
        spans: list[Span] = []
        for entity_type, pattern, check, score in RULES:
            if entity_type not in enabled:
                continue
            spans.extend(
                Span(match.start("value"), match.end("value"), entity_type, score, self.name)
                for match in pattern.finditer(text)
                if check is None or check(match)
            )
        return spans
