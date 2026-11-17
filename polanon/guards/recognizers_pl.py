"""Polish identifier validators + regex guard (checksums implemented, no ML needed).

Each `is_valid_*` returns True only when the checksum passes -> very low false-positive rate.
RegexGuard finds candidates with regex and keeps only checksum-valid ones (where a checksum exists).
"""

import ipaddress
import re
from collections.abc import Callable, Iterator
from itertools import takewhile

from polanon.guards.base import Guard
from polanon.types import Span


def _words(block: str) -> list[str]:
    """Split a whitespace-separated block, which keeps long word lists readable."""
    return block.split()


# ---------- validators ----------


def is_valid_pesel(s: str) -> bool:
    """Return True for an 11-digit PESEL with a valid checksum and a plausible birth date.

    Spaces and dashes are ignored: a PESEL is also written as the birth date and the serial number
    apart ("508101 12354"), and a form with one box per digit comes out of text extraction spaced.
    """
    s = re.sub(r"[\s-]", "", s)
    if not re.fullmatch(r"\d{11}", s):
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    total = sum(int(c) * w for c, w in zip(s[:10], weights, strict=True))
    if (10 - total % 10) % 10 != int(s[10]):
        return False
    month = int(s[2:4]) % 20  # century encoded by adding 20/40/60/80 to month
    day = int(s[4:6])
    return 1 <= month <= 12 and 1 <= day <= 31


def is_valid_nip(s: str) -> bool:
    """Return True for a 10-digit Polish tax identifier with a valid checksum.

    The "PL" of an EU VAT number is accepted in front: the pattern matches it, and a validator that
    saw the letters rejected every "PL8992611767" on an invoice, which then left unmasked.
    """
    s = re.sub(r"[\s-]", "", s).removeprefix("PL")
    if not re.fullmatch(r"\d{10}", s):
        return False
    weights = (6, 5, 7, 2, 3, 4, 5, 6, 7)
    total = sum(int(c) * w for c, w in zip(s[:9], weights, strict=True)) % 11
    return total != 10 and total == int(s[9])


def is_valid_regon(s: str) -> bool:
    """Return True for a 9- or 14-digit REGON business number with a valid checksum."""
    s = re.sub(r"[\s-]", "", s)
    if re.fullmatch(r"\d{9}", s):
        short_weights = (8, 9, 2, 3, 4, 5, 6, 7)
        total = sum(int(c) * w for c, w in zip(s[:8], short_weights, strict=True)) % 11
        return (0 if total == 10 else total) == int(s[8])
    if re.fullmatch(r"\d{14}", s):
        if not is_valid_regon(s[:9]):
            return False
        long_weights = (2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8)
        total = sum(int(c) * w for c, w in zip(s[:13], long_weights, strict=True)) % 11
        return (0 if total == 10 else total) == int(s[13])
    return False


def is_valid_pl_id(s: str) -> bool:
    """Return True for a Polish ID card number: three letters, a check digit and five digits."""
    s = s.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{3}\d{6}", s):
        return False
    values = [ord(c) - 55 for c in s[:3]] + [int(c) for c in s[3:]]
    weights = (7, 3, 1, 0, 7, 3, 1, 7, 3)
    return sum(v * w for v, w in zip(values, weights, strict=True)) % 10 == values[3]


def is_valid_iban(s: str) -> bool:
    """Return True for an IBAN whose mod-97 checksum holds; spaces and hyphens are ignored."""
    s = re.sub(r"[\s-]", "", s).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    return int(numeric) % 97 == 1


def is_valid_luhn(s: str) -> bool:
    """Return True for a payment card number that satisfies the Luhn checksum."""
    digits = [int(c) for c in re.sub(r"[\s-]", "", s)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        checksum += d
    return checksum % 10 == 0


def is_valid_ipv4(s: str) -> bool:
    """Return True for a dotted-quad the standard library recognises."""
    try:
        ipaddress.IPv4Address(s)
    except ValueError:
        return False
    return True


def is_valid_ipv6(s: str) -> bool:
    """Return True for an IPv6 address.

    Written IPv6 has too many shapes to check with a regex — "::1", the "::" abbreviation anywhere,
    an IPv4 tail — so the pattern finds candidates and `ipaddress` decides. A MAC address looks
    similar and is rejected here, which is why it keeps its own rule.
    """
    try:
        ipaddress.IPv6Address(s)
    except ValueError:
        return False
    return True


def is_valid_vin(s: str) -> bool:
    """Return True for a 17-character VIN that mixes letters and digits.

    VINs never use I, O or Q, and a run of 17 digits is far more likely to be some other number,
    so both are rejected.
    """
    s = s.upper()
    if not re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", s):
        return False
    return any(c.isalpha() for c in s) and any(c.isdigit() for c in s)


def is_valid_plate(s: str) -> bool:
    """Return True for a Polish vehicle registration: a 2-3 letter area code then 4-5 characters."""
    compact = s.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2,3}[A-Z0-9]{4,5}", compact):
        return False
    tail = compact[2:] if len(compact) == 7 else compact[3:]
    return any(c.isdigit() for c in tail)


#: Short Polish words that stand between a registration label and a year or an amount:
#: "Rejestracja od 2025 roku", "Tablica na 2024 rok". "po" is left out, because PO is Poznań.
_NOT_A_PLATE_AREA = frozenset({"na", "do", "od", "za", "we", "ze", "ku"})


def _is_plate_value(value: str) -> bool:
    """Return whether the word after a registration label is a plate rather than the sentence.

    The label is case-insensitive and so is the value, because a plate typed in a chat is often
    lowercase ("nr rej. kr 12345"). Two shapes are not plates: a preposition followed by a plain
    number, which is a year or an amount, and a document symbol, since "data rejestracji FV 12345"
    is the day an invoice was booked.
    """
    if not is_valid_plate(value):
        return False
    letters = "".join(takewhile(str.isalpha, value))
    if letters.upper() in _DOCUMENT_SERIES:
        return False
    return not (letters in _NOT_A_PLATE_AREA and value[len(letters) :].strip().isdigit())


#: Street prefixes that introduce a Polish address. In prose the word is mostly in an oblique case,
#: "przy ulicy Długiej 5", "na placu Wolności 2", which the nominative alone let through whole.
_STREET = (
    r"(?:ul|al|pl|os|ulic[aąęy]|ale(?:ja|je|i|ją|ję|jach)|plac(?:u|em)?|osiedl(?:e|a|u|em)"
    r"|rond(?:o|a|em|zie)|skwer(?:u|em|ze)?"
    # Streets the TERYT register files under other kinds: "Wybrzeże Kościuszkowskie 20",
    # "Nabrzeże Wieleckie 5", "Trakt Lubelski 120", "Pasaż Grunwaldzki 3", "bulw. Filadelfijski".
    r"|bulw|bulwar\w*|wybrze[żz]\w*|nabrze[żz]\w*|trak(?:t\w*|cie)|pasa[żz]\w*|zau[łl]\w*)"
)
#: A name component: a capitalised word (Roman numerals such as "II" included), in quotes as in
#: skwer im. "Solidarności", a leading day number as in "3 Maja", or a number the name counts
#: by: "1000-lecia", "10-go Pułku", "83. Pułku". A separate Roman-numeral branch matched "I" a
#: second way, and on OCR output with a run of "I" tokens the failed match backtracked through
#: 2^n combinations.
_NAME_PART = (
    r"(?:[\"„“]?[A-ZŁŚŻŹĆÓĄĘŃ][\w'\u2019-]*[\"”]?|\d{1,2}(?=\s+[A-ZŁŚŻŹĆÓĄĘŃ])"
    r"|\d{1,4}-(?:lecia|lecie|go|ej|ego|tej|ty|ci|ym)\b|\d{1,3}\.(?=\s+[A-ZŁŚŻŹĆÓĄĘŃ]))"
)
#: An abbreviated title in front of the patron's name, or an initial: "ul. gen. Andersa", "al.
#: Kard. Wyszyńskiego", "ul. inż. Bohdana Łozińskiego", "ul. A. Słonimskiego".
_TITLE = (
    r"(?:(?i:gen|ks|św|sw|prof|kard|marsz|płk|plk|ppłk|pplk|abp|bp|bł|dr|mjr|kpt|ppor|por|hetm"
    r"|inż|inz|mgr|adw|asp|kmdr|hm|phm|kpr|plut|sierż|sierz|adm)|[A-ZŁŚŻŹĆÓĄĘŃ])\."
)
#: A house number, optionally with a flat number: "43A", "12/4", "15 m. 3", "5 lok. 12".
_HOUSE = r"\d+[A-Za-z]?(?:\s*/\s*\d+[A-Za-z]?)?(?:\s*(?:m|lok(?:al)?)\.?\s*\d+[A-Za-z]?)?"
#: The most words a street's name takes, and the most titles in front of it: "Jana Pawła II",
#: "Jana III Sobieskiego" and "gen. Józefa Bema" have three. The bound keeps every address rule
#: linear: without it, each capitalised word of a long run started a street that ran to the end of
#: the run, so "Ulica Ulica ..." or "Z Z ..." of 5,000 words took seconds.
_STREET_WORDS = 8
_TITLES = 3
#: The small words and years a street's name holds after its first word: "Pawła II", "im. Jana",
#: "pod Lipami nad Wisłą", "Bohaterów Wojny 1920 r.", "Andrzeja Szwajkerta ps. Jędruś".
_STREET_LINKS = (
    r"i|w|we|z|ze|na|nad|pod|przy|za|od|do|o|u|im\.|imienia|ps\.|vel|von|van|der|de|la|di|del"
    r"|(?:1[0-9]|20)\d\d(?:\s+r\.)?"
)
#: The words of a street's name after its first one.
_MORE_STREET_WORDS = rf"(?:\s+(?:{_NAME_PART}|{_TITLE}|{_STREET_LINKS})){{0,{_STREET_WORDS - 1}}}"
#: "nr" or "numer" written between a street and its house number: "przy ulicy Długiej nr 36".
#: It ended the street, and the street with its number went out in clear.
_HOUSE_NR = r"(?i:nr|numer)\.?"

ADDRESS_PATTERN = re.compile(
    # The word boundary matters: without it the "al" at the end of "TOTAL" reads as "al." (aleja),
    # and an invoice line such as "TOTAL M0061 Catalyst 1" became an address cut mid-word.
    # PDF output and quick typing drop the space after the prefix's dot: "ul.Długa 5".
    rf"\b(?i:{_STREET})(?:\.\s*|\s+)(?:(?:{_TITLE}|im\.|imienia)\s+){{0,{_TITLES}}}{_NAME_PART}"
    rf"{_MORE_STREET_WORDS}\s+(?:{_HOUSE_NR}\s*)?{_HOUSE}"
)

#: A short house number of a street written without its prefix, with a flat number if any.
_SHORT_HOUSE = (
    r"\d{1,4}[A-Za-z]?(?:[ \t]*/[ \t]*\d{1,4}[A-Za-z]?)?"
    r"(?:[ \t]*(?:m|lok(?:al)?)\.?[ \t]*\d{1,4}[A-Za-z]?)?(?!\d)"
)

#: A street written without its prefix, as forms and CVs often do: "Długa 5 m. 4, 00-238
#: Warszawa". The postcode right after it is what makes it an address; without one, a capitalised
#: word and a number are a product and its version ("Windows 11") far more often than a street. The
#: numbers are short, so "NIP 1234567890, 00-950" is not read as a street.
ADDRESS_BEFORE_POSTCODE_PATTERN = re.compile(
    rf"(?<![\w.]){_NAME_PART}"
    rf"(?:[ \t]+(?:{_NAME_PART}|{_TITLE}|{_STREET_LINKS})){{0,{_STREET_WORDS - 1}}}[ \t]+"
    rf"(?:{_HOUSE_NR}[ \t]*)?{_SHORT_HOUSE}(?=[ \t]*,?[ \t]*\d{{2}}-\d{{3}}\b)"
)

#: A street without its prefix that a sentence names: "Adres: Japczaka 17", "zam. Długa 5",
#: "mieszka przy Długiej 5", or after a preposition with the town right after the number, "z
#: Długiej 17 w Białymstoku". Either the label or the town is what makes it an address, as the
#: postcode does above; "Windows 11 w Polsce" has neither. The number must not run on into a
#: dotted number, so "Adres IP 10.0.0.1" is no street. See :func:`_locate_address`.
_STREET_WITHOUT_PREFIX = (
    rf"(?:{_TITLE}[ \t]+){{0,{_TITLES}}}{_NAME_PART}"
    rf"(?:[ \t]+(?:{_NAME_PART}|{_TITLE})){{0,{_STREET_WORDS - 1}}}?"
    rf"[ \t]+(?:(?P<nr>{_HOUSE_NR})[ \t]*)?(?P<house>{_SHORT_HOUSE})(?![\w]|[.,:/-]\d)"
)
ADDRESS_AFTER_LABEL_PATTERN = re.compile(
    r"(?<![\w.])(?i:adres|zam\.|zamieszka[łl][aey]|mieszka(?:[ \t]+(?:przy|na))?)(?!\w)"
    rf"[ \t]*:?[ \t]*(?P<value>{_STREET_WITHOUT_PREFIX})"
)
ADDRESS_BEFORE_TOWN_PATTERN = re.compile(
    rf"(?<![\w.])(?i:z|ze|na|przy|do)[ \t]+(?P<value>{_STREET_WITHOUT_PREFIX})"
    r"(?=[ \t]+w[ \t]+(?P<town>[A-ZŁŚŻŹĆÓĄĘŃ][\w-]*))"
)
#: A street and its house number in fields of their own, as a form or a CRM export writes them:
#: "Ulica: Kwiatowa", "Numer domu: 8A", "house number: 5". The colon makes it a field; without
#: one, "ulica handlowa nr 1" or "numer domu 5 osób" is a sentence.
ADDRESS_STREET_FIELD_PATTERN = re.compile(
    r"(?<![\w.])(?i:ulica|street)[ \t]*:[ \t]*"
    rf"(?P<value>{_NAME_PART}"
    rf"(?:[ \t]+(?:{_NAME_PART}|{_TITLE}|{_STREET_LINKS})){{0,{_STREET_WORDS - 1}}})"
)
ADDRESS_NUMBER_FIELD_PATTERN = re.compile(
    r"(?<![\w.])(?i:(?:nr|numer)\.?[ \t]+(?:domu|budynku|lokalu|mieszkania)"
    r"|(?:house|building|flat|apartment)[ \t]+(?:no|number)\.?)[ \t]*:[ \t]*"
    rf"(?P<value>{_SHORT_HOUSE})(?![\w]|[.,:/-]\d)"
)
#: The endings of a street after "z", "na", "przy" or "do": "Długiej", "Mickiewicza",
#: "Piłsudskiego", "Kościuszki", "Rynku", "Kwiatów". A product's name has none: "Windows".
_STREET_IN_AN_OBLIQUE_CASE = re.compile(r"(?:ej|a|ego|i|y|u|ów)$")
#: The endings of a town after "w": "Krakowie", "Gdańsku", "Łodzi", "Katowicach", "Zielonej Górze",
#: "Nowym Sączu". "Polsce" and "Teams" have none; countries in -ii or -ji are left out as well.
_TOWN_IN_THE_LOCATIVE = re.compile(r"(?:ie|u|i|y|ach|ej|ym|im)$")
_NOT_A_TOWN = frozenset(
    {"europie", "litwie", "łotwie", "ukrainie", "kanadzie", "chinach", "indiach", "czechach"}
    | {"stanach", "emiratach", "niemczech", "włoszech", "węgrzech"}
)

#: What an address may hold, in a character class: letters of any alphabet, as an address with a
#: French, German or Czech name in it has them, digits, "_" and ".%+-". Only Polish letters were
#: allowed once, and "françois.dupont@…" was masked from its "ç" on, with "franç" left in front.
_EMAIL_LOCAL = r"\w.%+\-"
_EMAIL_LABEL = r"[\w\-]+"
#: A top-level domain: letters only, of any alphabet.
_EMAIL_TLD = r"[^\W\d_]{2,}"
_EMAIL_AT = r"\s*[\[({]\s*(?i:at|małpa|malpa)\s*[\])}]\s*"
_EMAIL_DOT = r"(?:\.|\s*[\[({]\s*(?i:dot|kropka)\s*[\])}]\s*)"

#: An e-mail address with its "@" written out in brackets, and optionally its dots too.
OBFUSCATED_EMAIL_PATTERN = re.compile(
    rf"(?<![{_EMAIL_LOCAL}])[{_EMAIL_LOCAL}]+{_EMAIL_AT}"
    rf"{_EMAIL_LABEL}(?:{_EMAIL_DOT}{_EMAIL_LABEL})*{_EMAIL_DOT}{_EMAIL_TLD}"
)

#: Sites whose path starts with the handle of the person or the organisation the page belongs to.
_HANDLE_SITES = (
    r"github\.com|gitlab\.com|bitbucket\.org|codeberg\.org|behance\.net|dribbble\.com|kaggle\.com"
    r"|codepen\.io|dev\.to|huggingface\.co|instagram\.com|facebook\.com|fb\.com|twitter\.com"
    r"|x\.com|threads\.net|medium\.com|tiktok\.com|youtube\.com(?:/(?:c|user|channel))?"
)
#: A handle as it stands in a path. A full stop ends it unless more of the handle follows, so the
#: one that closes a sentence is left out.
_HANDLE = r"[\w%-]+(?:\.[\w%-]+)*"

#: A link to a person's page: a LinkedIn profile, a GitHub or GitLab account and anything under it,
#: a portfolio on GitHub Pages, a social media profile. The match ends with the handle.
PROFILE_LINK_PATTERN = re.compile(
    r"(?<![\w@./-])(?i:(?:https?://)?(?:"
    r"(?:(?:www|m|mobile|gist|[a-z]{2})\.)?"
    rf"(?:linkedin\.com/(?:in|pub)/|stackoverflow\.com/users/\d+/|leetcode\.com/u/"
    rf"|(?:{_HANDLE_SITES})/@?)(?P<handle>{_HANDLE})"
    r"|(?P<page>[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)\.(?:github|gitlab)\.io(?![\w-])))"
)

#: Paths on those sites that are the site's own pages rather than somebody's: github.com/features,
#: facebook.com/groups, youtube.com/watch, x.com/search.
_SITE_PAGES = frozenset(
    _words(
        """
        about accounts apps blog business challenges code codespaces collections competitions
        compose contact copilot customer-stories dashboard datasets designers discussions docs
        embed enterprise events explore features feed gaming groups hashtag help home i in intent
        issues jobs join learn login marketplace messages models new notifications organizations
        orgs p pages papers pen permalink.php photo photo.php playlist policies pricing profile.php
        projects pulls readme reel reels results search security settings share sharer sharer.php
        shorts shots signup site spaces sponsors stories story.php tag tags team topic topics
        trending tv users watch
        """
    )
)

#: An IPv6 address in any of its written forms, including the "::" abbreviation and the
#: IPv4-mapped tail. Validation is left to :func:`is_valid_ipv6`, which simply asks the standard
#: library, so the pattern only has to be generous enough to find candidates.
IPV6_PATTERN = re.compile(
    r"(?<![0-9A-Za-z:.])(?:[0-9A-Fa-f]{0,4}:){2,7}"
    r"(?:\d{1,3}(?:\.\d{1,3}){3}|[0-9A-Fa-f]{0,4})(?![0-9A-Za-z:.])"
)

#: Polish month names in the genitive, as they appear in written dates.
_MONTHS = (
    "stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|"
    "wrzesnia|października|pazdziernika|listopada|grudnia"
)

#: English month names and their abbreviations, for dates in a CV or a form written in English.
_EN_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)

#: A written date: 15.03.1985, 15-03-1985, 15/03/1985, 15 marca 1985, 1985-03-15, and in English
#: 15 March 1985 or March 15, 1985. Without the English forms, "date of birth: 15 March 1985" went
#: out whole.
_DATE_VALUE = (
    r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4}"
    rf"|\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}"
    r"|\d{4}-\d{2}-\d{2}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?i:{_EN_MONTHS})\.?,?\s+\d{{4}}"
    rf"|(?i:{_EN_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
)
#: Words that introduce a birth date: "ur.", "urodzona", "data urodzenia:", "DOB", "geb.", and
#: the names code gives them, "birthday", "birth_date" or "dateOfBirth".
_BIRTH_LABEL = (
    r"(?i:\b(?:data[\s_]+urodzenia|urodzon[aey]?|urodz\.|ur\.|urodziny|date[\s_]+of[\s_]+birth"
    r"|dateofbirth|birth[\s_-]?(?:date|day)|born|dob|geburtsdatum|geboren|geb\.))"
)

#: Registered IBAN length per country (ISO 13616 / the SWIFT IBAN registry). An IBAN is often
#: followed on the same line by a currency, a BIC or an amount ("PL61 ... 2874 PLN"), which the
#: pattern cannot tell from one more group, so the candidate is cut to its country's length
#: before the checksum is asked.
IBAN_LENGTHS: dict[str, int] = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22, "BH": 22,
    "BI": 27, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DJ": 27,
    "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24, "FI": 18, "FK": 18, "FO": 18, "FR": 27,
    "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28, "HN": 28, "HR": 21, "HU": 28,
    "IE": 22, "IL": 23, "IQ": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28,
    "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "LY": 25, "MC": 27, "MD": 24, "ME": 22,
    "MK": 19, "MN": 20, "MR": 27, "MT": 31, "MU": 30, "NI": 28, "NL": 18, "NO": 15, "OM": 23,
    "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "RU": 33, "SA": 24,
    "SC": 31, "SD": 18, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "SO": 23, "ST": 25, "SV": 28,
    "TL": 23, "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20, "YE": 30,
}  # fmt: skip

#: Three capitals that open an amount or a register number far more often than an ID card:
#: "PLN 100000" passes the ID card's mod-10 check one time in ten.
_NOT_AN_ID_SERIES = frozenset(
    {"PLN", "EUR", "USD", "GBP", "CHF", "CZK", "SEK", "NOK", "DKK", "HUF", "JPY", "CAD", "AUD"}
    | {"CNY", "UAH", "RUB", "RON", "BGN", "KRS", "NIP", "VAT"}
)

#: Symbols of Polish accounting and warehouse documents (faktura VAT, wydanie zewnętrzne,
#: zamówienie od klienta, ...). "FV 2024123" has a passport's shape and is not one.
_DOCUMENT_SERIES = frozenset(
    {"FV", "FA", "FS", "FZ", "FK", "FP", "PF", "PA", "KP", "KW", "WZ", "PZ", "RW", "PW", "MM"}
    | {"ZK", "ZD", "ZW", "ZO", "RK", "WB", "NK"}
)

#: Words that end right before an ID card number: "dowód osobisty nr", "seria i numer:".
_ID_CARD_LABEL = re.compile(
    r"(?i)(?:dow[oó]d\w*|seri[aięy]\w*|identity\s+card|id\s+card)"
    r"(?:\s+(?:osobist\w*|i\s+numer\w*|nr|no|number|numer\w*|to))*\.?\s*[:-]?\s*$"
)
#: Words that end right before a passport number: "nr paszportu:", "passport no.".
_PASSPORT_LABEL = re.compile(
    r"(?i)(?:paszport\w*|passport)(?:\s+(?:nr|no|numer|number|to))?\.?\s*[:-]?\s*$"
)
#: Words that end right before a PESEL: "PESEL:", "nr PESEL", "numer PESEL to", and the social
#: insurance number a translated form asks for, which in Poland is the PESEL.
_PESEL_LABEL = re.compile(
    r"(?i)(?:pesel\w*|(?:nr|numer\w*)\.?\s+ubezpieczenia(?:\s+społeczn\w*)?)"
    r"(?:\s+(?:nr|numer|to))?\.?\s*[:(-]?\s*$"
)
#: Words that end right before a NIP: "NIP:", "nr NIP", "NIP UE". A label starts a word.
_NIP_LABEL = re.compile(r"(?i)(?<![^\W\d_])nip(?:\s+(?:ue|nr|numer|to))?\.?\s*[:(-]?\s*$")
#: Words that end right before a payment card's number: "numer karty kredytowej", "nr karty:",
#: "karta płatnicza nr", "card number".
_CARD_LABEL = re.compile(
    r"(?i)(?:(?:nr|numer\w*)\.?\s+kart\w*(?:\s+(?:kredytow|płatnicz|debetow)\w*)?"
    r"|kart[aą]\s+(?:kredytow|płatnicz|debetow)\w*(?:\s+(?:nr|numer))?"
    r"|(?:credit\s+|debit\s+)?card\s+(?:no|number))\.?\s*[:(-]?\s*$"
)
#: Phone labels: a nine-digit REGON-looking number after these is a phone (#8). A label starts a
#: word, so the "tel" that ends "hotel" is none.
_PHONE_LABEL = re.compile(
    r"(?i)(?<![^\W\d_])(?:tel(?:\.\s*kom\.?|\.)?|kom\.?|telefon\w*|komórk\w*|fax|phone)"
    r"\s*[:-]?\s*$"
)
#: How far back a label is looked for: one short phrase, never the previous sentence.
_LABEL_REACH = 40

#: Words that introduce a vehicle registration, in the inflected forms Polish actually uses:
#: "nr rej.", "o numerze rejestracyjnym", "tablica rejestracyjna:", optionally "... pojazdu".
_PLATE_LABEL = (
    r"(?:(?:nr|numer\w*)\.?\s*rej(?:\.|estracyjn\w*)?"
    r"|tablic\w*(?:\s+rejestracyjn\w*)?"
    r"|rejestracj\w*)"
    r"(?:\s+(?:pojazdu|samochodu|auta|motocykla|przyczepy))?"
)


def _boxed(digits: int) -> str:
    """Return a pattern for `digits` digits written one per box, as text extraction spaces them.

    Neither neighbour may be another spaced digit, so a longer boxed number is not read in parts.
    """
    return rf"(?<!\d)(?<!\d )\d(?: \d){{{digits - 1}}}(?! ?\d)"


# ---------- regex guard ----------

#: One detection rule: entity type, pattern, and an optional checksum validator.
Rule = tuple[str, re.Pattern[str], Callable[[str], bool] | None]

#: Finds where in a match the identifier really is, if anywhere: given the text and a candidate,
#: returns the extents to mask. For rules whose answer is more than yes or no on the matched text.
Locator = Callable[[str, re.Match[str]], list[tuple[int, int]]]

PATTERNS: list[Rule] = [
    ("PESEL", re.compile(r"(?<!\d)\d{11}(?!\d)"), is_valid_pesel),
    # Split as the birth date and the serial number, or one digit per box: only after a PESEL
    # label, because a date-prefixed order number ("202410-00106") has the same shape and passes
    # the checksum one time in ten. See _LOCATORS.
    ("PESEL", re.compile(rf"(?<!\d)\d{{6}}[ -]\d{{5}}(?!\d)|{_boxed(11)}"), None),
    (
        "NIP",
        re.compile(
            r"(?<!\d)(?:PL)?\d{3}[- ]?\d{3}[- ]?\d{2}[- ]?\d{2}(?!\d)"
            r"|(?<!\d)\d{3}[- ]?\d{2}[- ]?\d{2}[- ]?\d{3}(?!\d)"
        ),
        is_valid_nip,
    ),
    ("NIP", re.compile(_boxed(10)), is_valid_nip),
    ("REGON", re.compile(r"(?<!\d)\d{9}(?:\d{5})?(?!\d)"), is_valid_regon),
    # A branch REGON written as the company's nine digits and the five-digit branch code apart.
    # A separate rule, so a nine-digit REGON followed by some other number is still found.
    ("REGON", re.compile(r"(?<!\d)\d{9}[ -]\d{5}(?!\d)"), is_valid_regon),
    # No public checksum, so the label is what makes the match trustworthy. A KRS number is also
    # written without its leading zeros.
    # "KRS 0000123456", "KRS nr 123456", "wpisana do KRS pod numerem 0000123456", "KRS No. 123456".
    (
        "KRS",
        re.compile(
            r"(?i)\bKRS(?:\s*(?:pod\s+)?(?:nr|numer\w*|no|number)\.?)?[\s:#-]*"
            r"(?P<value>\d{5,10})\b"
        ),
        None,
    ),
    # A lowercase ID card is only taken written together, "abc412345": with a space, a short
    # Polish word and a six-figure number ("dla 120000 osób") would pass the check one time in ten.
    ("PL_ID", re.compile(r"\b(?:[A-Z]{3} ?|[a-z]{3})\d{6}\b"), None),  # see _LOCATORS
    ("PL_PASSPORT", re.compile(r"\b(?:[A-Z]{2} ?|[a-z]{2})\d{7}\b"), None),  # see _LOCATORS
    # A driving licence number has a different shape in every country, "01234/15/1234" in Poland,
    # so only its label makes a run of letters and digits one, and so does the label of a
    # tachograph's driver card. It must hold a digit: "prawo jazdy kategorii B" or "prawo jazdy:
    # tak" names no number. Between the label and the number may stand the categories ("kat.
    # B"), a remark in brackets ("(jeśli dotyczy)"), a colon, a dash or an opening bracket.
    (
        "DRIVING_LICENCE",
        re.compile(
            r"(?i:\b(?:praw(?:o|a|em)\s+jazdy|driv(?:ing|er'?s|er\u2019s)\s+licen[cs]e"
            r"|kart(?:a|y|ę|ą|cie)\s+kierowcy)"
            r"(?:\s+kat(?:\.|egorii)\s*[A-Z][A-Z0-9+]*(?:\s*,\s*[A-Z][A-Z0-9+]*)*)?"
            r"(?:\s+(?:nr|numer|no|number))?\.?)"
            r"(?:\s*\([^()\n]{1,30}\))?\s*[:#(\u2013\u2014-]?\s*"
            r"(?P<value>(?=[A-Za-z0-9/-]*\d)[A-Za-z0-9][A-Za-z0-9/-]{3,18}[A-Za-z0-9])(?![\w/-])"
        ),
        None,
    ),
    (
        "IBAN",
        re.compile(
            # Groups apart by a space, or by a hyphen as some banks and Word's dashes write them,
            # or wrapped onto the next line, as a narrow column in a PDF or an e-mail does.
            r"\b(?i:[A-Z]{2}\d{2}(?:(?:[ -]|\r?\n)?[A-Z0-9]{4}){2,7}"
            r"(?:(?:[ -]|\r?\n)?[A-Z0-9]{1,4})?)\b"
            r"|\b\d{2}(?:[ -]?\d{4}){6}\b"
        ),
        None,
    ),  # see _LOCATORS
    ("IBAN", re.compile(_boxed(26)), None),
    # Every run of digit groups; the locator picks out what is shaped like a card inside it.
    ("CREDIT_CARD", re.compile(r"(?<!\d)\d+(?:[ -]\d+)*"), None),
    # The lookbehind starts a match only where a run of address characters starts. Without it a
    # long run with no "@" — a hex dump, a base64 blob — was rescanned from every position.
    # Letters of any alphabet are allowed in the local part and in every domain label (#7).
    (
        "EMAIL",
        re.compile(rf"(?<![{_EMAIL_LOCAL}])[{_EMAIL_LOCAL}]+@[\w.\-]+\.{_EMAIL_TLD}"),
        None,
    ),
    # An address written to slip past harvesters: "jan.kowalski [at] example.com", "(małpa)",
    # "[dot]". Only the bracketed forms: a bare "at" between two words is ordinary English.
    ("EMAIL", OBFUSCATED_EMAIL_PATTERN, None),
    # A link to a person's own page: LinkedIn puts the name in it, and a GitHub login reappears in
    # every repository and in the user's pages. Only the part up to the handle is masked, so
    # "<LINK_1>/weather-app" still tells the model which project it is. See _LOCATORS.
    ("LINK", PROFILE_LINK_PATTERN, None),
    (
        "PHONE",
        re.compile(
            r"\(\d{2,3}\)[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}(?!\d)"
            # "0048" is how a Polish number is dialled from abroad; bare "48" is also used in
            # SMS gateway output and CRM exports (#6). The lookbehind avoids taking a time's ":48".
            r"|(?<!\d)(?:\+48[ -]?|0048[ -]?|(?<!\d:)48[ -]?(?=[1-9]))?"
            r"(?:\d{3}[ -]?\d{3}[ -]?\d{3}|\d{2}[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}"
            r"|\d{2}[ -]?\d{7})(?!\d)"
        ),
        None,
    ),
    ("POSTAL_CODE", re.compile(r"\b\d{2}-\d{3}\b"), None),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), is_valid_ipv4),
    ("IP_ADDRESS", IPV6_PATTERN, is_valid_ipv6),
    ("MAC_ADDRESS", re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"), None),
    # Cisco's dotted notation, the default in "show mac address-table" output. Digits alone in
    # that shape are a build or version number ("2024.1015.0001") far more often than a MAC.
    (
        "MAC_ADDRESS",
        re.compile(r"(?<![\w.])[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}(?!\.?\w)"),
        lambda value: any(c.isalpha() for c in value),
    ),
    ("ADDRESS", ADDRESS_PATTERN, None),
    ("ADDRESS", ADDRESS_BEFORE_POSTCODE_PATTERN, None),
    ("ADDRESS", ADDRESS_AFTER_LABEL_PATTERN, None),
    ("ADDRESS", ADDRESS_BEFORE_TOWN_PATTERN, None),
    ("ADDRESS", ADDRESS_STREET_FIELD_PATTERN, None),
    ("ADDRESS", ADDRESS_NUMBER_FIELD_PATTERN, None),
    ("VIN", re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"), is_valid_vin),
    # A lowercase VIN only after its label: seventeen lowercase letters and digits are a request
    # id, a session key or a hash far more often, and a European VIN carries no check digit.
    (
        "VIN",
        re.compile(
            r"(?i:\bVIN\b(?:\s+(?:auta|samochodu|pojazdu|nr|no|numer|number))*)\.?\s*[:#-]?\s*"
            r"(?P<value>(?i:[A-HJ-NPR-Z0-9]{17}))\b"
        ),
        is_valid_vin,
    ),
    # A bare plate is shaped like a product code or a standard number ("ISO 27001"), so the
    # surrounding label is what separates a real registration from noise.
    (
        "PLATE",
        re.compile(
            rf"(?i:\b{_PLATE_LABEL})[:\s]\s*(?P<value>[A-Za-z]{{2,3}} ?[A-Za-z0-9]{{4,5}})\b"
        ),
        _is_plate_value,
    ),
    # Before the DATE rule, so a birth date read by both keeps this type. A deadline identifies
    # nobody, but a birth date next to a name does, so this type stays on where dates are off.
    # A date ends where its digits do: Polish glues the year's abbreviation on, "1985r.", and a
    # word boundary there let a labelled birth date go out whole.
    (
        "BIRTH_DATE",
        re.compile(
            rf"{_BIRTH_LABEL}[\"']?\s*[:=-]?\s*[\"']?(?:(?:w\s+)?dniu\s+|dnia\s+|on\s+|am\s+)?"
            rf"(?P<value>{_DATE_VALUE})(?!\d)"
        ),
        None,
    ),
    ("DATE", re.compile(rf"\b(?:{_DATE_VALUE})(?!\d)"), None),
]


def _locate_iban(text: str, match: re.Match[str]) -> list[tuple[int, int]]:
    """Return the IBAN at the start of the match, cut to its registered length.

    The checksum decides, but not after an account label: "konto PL61 1090 1014 0000 0712 1921
    2974", a digit or two off a real account, is still somebody's account, as eleven digits
    after "PESEL" are somebody's number. Such a number is masked whole when it has the length its
    country registers, or the 26 digits of a Polish account.
    """
    candidate = match.group()
    labelled = _follows(_ACCOUNT_LABEL, text, match)
    if candidate[0].isdigit():  # a Polish account number written without the PL prefix
        compact = re.sub(r"[\s-]", "", candidate)
        return [match.span()] if labelled or is_valid_iban("PL" + compact) else []
    end = _iban_end(candidate)
    if end is None and labelled:
        end = _registered_end(candidate)
    return [] if end is None else [(match.start(), match.start() + end)]


def _registered_end(candidate: str) -> int | None:
    """Return where an IBAN of its country's registered length ends in `candidate`, valid or not."""
    length = IBAN_LENGTHS.get(candidate[:2].upper())
    if length is None:
        return None
    end = _end_of_nth_character(candidate, length)
    if end is None or candidate[end : end + 1] not in ("", " ", "-", "\r", "\n"):
        return None
    return end


def _iban_end(candidate: str) -> int | None:
    """Return where the IBAN that opens `candidate` ends, or None when there is no valid one.

    The cut must fall between groups: an IBAN glued to the next word is not taken apart. Whatever
    the registry says, a candidate written in capitals that validates whole is still accepted,
    as it was before the registry, so a country added to it later is not lost.
    """
    length = IBAN_LENGTHS.get(candidate[:2].upper())
    if length is not None:
        end = _end_of_nth_character(candidate, length)
        boundary = end is not None and candidate[end : end + 1] in ("", " ", "-", "\r", "\n")
        if end is not None and boundary and is_valid_iban(candidate[:end]):
            return end
    return len(candidate) if candidate.isupper() and is_valid_iban(candidate) else None


def _end_of_nth_character(candidate: str, count: int) -> int | None:
    """Return the index just past the `count`-th character of `candidate` not a group separator."""
    seen = 0
    for index, character in enumerate(candidate):
        if character not in " -\r\n":
            seen += 1
            if seen == count:
                return index + 1
    return None


def _locate_cards(text: str, match: re.Match[str]) -> list[tuple[int, int]]:
    """Return every card number inside a run of digit groups.

    The run takes in the numbers written next to the card — an item number before it, the expiry
    "12/27" after it — so testing the whole run, as a single pattern did, failed the checksum and
    sent the card in clear. Instead a card is looked for at the start of each chain of groups.
    """
    run = match.group()
    groups = [m.span() for m in re.finditer(r"\d+", run)]
    if _account_shaped(text, match.start(), groups):
        return []
    # Right after "numer karty" the run is the card, its checksum broken by a typo or not.
    if _follows(_CARD_LABEL, text, match) and 12 <= sum(_width(g) for g in groups) <= 19:
        return [match.span()]
    found: list[tuple[int, int]] = []
    first = 0
    while first < len(groups):
        last = _longest_card(run, groups, first)
        if last is not None:
            found.append((match.start() + groups[first][0], match.start() + groups[last][1]))
            first = last + 1
            continue
        first += 1
        # Inside a chain of four-digit groups a card would only be a slice of a longer number,
        # an account number for one, so the next try starts where such a chain starts. A group
        # that is not four digits wide ends the chain: skipping it as well lost a card written
        # whole after a year, "2024 4111111111111111".
        while (
            first < len(groups)
            and _width(groups[first - 1]) == _width(groups[first]) == 4
            and not _card_after_one_number(groups, first)
        ):
            first += 1
    return found


def _account_shaped(text: str, start: int, groups: list[tuple[int, int]]) -> bool:
    """Return whether a run of digit groups is an account number, which holds no card.

    "PL61 1090 1014 0000 0712 1921 2974" failed the IBAN checksum by a typo, and sixteen digits
    from its middle passed Luhn: masked as a card, they left the rest of the account in the
    clear. Two check digits right after a country's letters, or the 2 and six times 4 digits of
    a Polish account, make the run an account number, valid or not.
    """
    widths = [end - begin for begin, end in groups]
    if widths == [2, 4, 4, 4, 4, 4, 4]:
        return True
    after_letters = start >= 2 and text[start - 2 : start].isalpha()
    return after_letters and len(widths) >= 4 and widths[0] == 2 and set(widths[1:-1]) == {4}


def _card_after_one_number(groups: list[tuple[int, int]], index: int) -> bool:
    """Return whether four groups of four digits from `index` follow exactly one such group.

    A statement line puts the year of a date right before the card, "15.03.2024 4111 1111 1111
    1111", as does "ważna do 2027 ...". Only that shape is tried: four groups that end the chain
    after a single number. Any other slice of a chain is part of a longer number.
    """
    if index >= 2 and _width(groups[index - 2]) == 4:
        return False
    end = index + 4
    if end > len(groups) or any(_width(group) != 4 for group in groups[index:end]):
        return False
    return end == len(groups) or _width(groups[end]) != 4


def _longest_card(run: str, groups: list[tuple[int, int]], first: int) -> int | None:
    """Return the last group of the longest card that starts at group `first`, if there is one.

    A card is one compact group of 13-19 digits, or a four-digit group followed by groups of four
    to six digits, the last of which may be shorter. No issuer number starts with 0, which is
    what keeps a phone number dialled as "0048 ..." from passing for a card.
    """
    if run[groups[first][0]] == "0":
        return None
    width = _width(groups[first])
    if width != 4:
        compact = 13 <= width <= 19 and is_valid_luhn(run[slice(*groups[first])])
        return first if compact else None
    candidates: list[int] = []
    total = width
    for last in range(first + 1, len(groups)):
        total += _width(groups[last])
        if total > 19 or _width(groups[last]) > 6:
            break
        if total >= 13:
            candidates.append(last)
        if _width(groups[last]) < 4:  # a short group can only end a card
            break
    for last in reversed(candidates):
        if is_valid_luhn(run[groups[first][0] : groups[last][1]]):
            return last
    return None


def _width(group: tuple[int, int]) -> int:
    return group[1] - group[0]


def _locate_pl_id(text: str, match: re.Match[str]) -> list[tuple[int, int]]:
    """Return the ID card number unless its letters are a currency or a register's name.

    "PLN 100000" is an amount however the checksum falls, unless an ID card label stands before it.
    """
    value = match.group()
    if _follows(_ID_CARD_LABEL, text, match):
        return [match.span()]  # its label names it, typo or not, as with a PESEL
    if not is_valid_pl_id(value) or value[:3].upper() in _NOT_AN_ID_SERIES:
        return []
    return [match.span()]


def _locate_nip(text: str, match: re.Match[str]) -> list[tuple[int, int]]:
    """Return a NIP: right after a NIP label always, elsewhere only with its checksum.

    A NIP mistyped by one digit after "NIP:" is still the company's or the person's, and went out
    in clear; a number without the label is a NIP only when its checksum says so.
    """
    if _follows(_NIP_LABEL, text, match) or is_valid_nip(match.group()):
        return [match.span()]
    return []


def _locate_passport(text: str, match: re.Match[str]) -> list[tuple[int, int]]:
    """Return the passport number unless it is the number of an invoice or another document.

    A Polish passport number is two letters and seven digits, the shape of "FV 2024123" as well.
    A passport label settles it either way; without one, a known document symbol means a document,
    and lowercase letters are taken only after a label.
    """
    if _follows(_PASSPORT_LABEL, text, match):
        return [match.span()]
    series = match.group()[:2]
    if series.islower() or series in _DOCUMENT_SERIES:
        return []
    return [match.span()]


def _locate_pesel(text: str, match: re.Match[str]) -> list[tuple[int, int]]:
    """Return a PESEL: right after a PESEL label always, elsewhere only whole and checksum-valid.

    Eleven digits right after "PESEL" are someone's number even with a typo in them, a wrong check
    digit or day 99, and such a number went out in clear. Without the label the checksum decides,
    and a PESEL in parts needs the label, since a date-prefixed order number has that shape.
    """
    if _follows(_PESEL_LABEL, text, match):
        return [match.span()]
    return [match.span()] if match.group().isdigit() and is_valid_pesel(match.group()) else []


#: Words that introduce a bank account: "konto", "nr konta", "rachunek bankowy", "IBAN".
_ACCOUNT_LABEL = re.compile(
    r"(?i)\b(?:kont[aou]?|koncie|kontem|rachun\w*|iban|account|acct)"
    r"(?:\s+(?:bankow\w*|bank|number|no\.?|nr\.?|numer))*\s*[:-]?\s*$"
)


def _follows(label: re.Pattern[str], text: str, match: re.Match[str]) -> bool:
    """Return whether `label` ends right before the match."""
    return label.search(text, max(0, match.start() - _LABEL_REACH), match.start()) is not None


def _locate_regon(text: str, match: re.Match[str]) -> list[tuple[int, int]]:
    """Return a REGON unless a phone label stands right before it (#8).

    One nine-digit number in ten passes the REGON checksum, so an unspaced phone
    after "tel." would otherwise be labelled REGON. With "REGON" or no label the
    checksum still wins.
    """
    if not is_valid_regon(match.group()):
        return []
    if _follows(_PHONE_LABEL, text, match):
        return []
    return [match.span()]


def _locate_address(text: str, match: re.Match[str]) -> list[tuple[int, int]]:
    """Return an address once: a street the rule with "ul." reads is left to that rule.

    "ul. gen. Andersa 15, 00-159" is also "Andersa 15" before a postcode, and two spans for one
    address made it two values. A street without its prefix after an address label is the part
    after the label, and one before a town counts only when the street and the town are in the
    cases a sentence puts them in, "z Długiej 17 w Białymstoku", unlike "na Windows 11 w Polsce".
    """
    if match.re in (ADDRESS_STREET_FIELD_PATTERN, ADDRESS_NUMBER_FIELD_PATTERN):
        return [match.span("value")]
    if match.re in (ADDRESS_AFTER_LABEL_PATTERN, ADDRESS_BEFORE_TOWN_PATTERN):
        before_postcode = ADDRESS_BEFORE_POSTCODE_PATTERN.match(text, match.start("value"))
        if before_postcode is not None and before_postcode.end() == match.end("value"):
            return []  # the street before a postcode is that rule's
    if match.re is ADDRESS_AFTER_LABEL_PATTERN:
        return [match.span("value")]
    if match.re is ADDRESS_BEFORE_TOWN_PATTERN:
        street_end = match.start("nr") if match.group("nr") else match.start("house")
        street = text[match.start("value") : street_end].split()[-1].lower()
        town = match.group("town").lower()
        if (
            _STREET_IN_AN_OBLIQUE_CASE.search(street) is None
            or _TOWN_IN_THE_LOCATIVE.search(town) is None
            or town.endswith(("ii", "ji"))
            or town in _NOT_A_TOWN
        ):
            return []
        return [match.span("value")]
    if match.re is not ADDRESS_BEFORE_POSTCODE_PATTERN:
        return [match.span()]
    window = max(0, match.start() - 2 * _LABEL_REACH)
    for prefixed in ADDRESS_PATTERN.finditer(text, window, match.end()):
        if prefixed.start() <= match.start() < prefixed.end():
            return []
    return [match.span()]


def _locate_profile_link(text: str, match: re.Match[str]) -> list[tuple[int, int]]:
    """Return a profile link up to its handle, unless the path is one of the site's own pages."""
    return [match.span()] if _handle_of(match) is not None else []


def _handle_of(match: re.Match[str]) -> str | None:
    handle = match.group("handle") or match.group("page")
    if len(handle) < 2 or handle.lower() in _SITE_PAGES:
        return None
    return handle


def profile_handle(link: str) -> str | None:
    """Return the handle a profile link names, or None when it is not a link to a person's page.

    "https://github.com/jkowalski/weather-app" and "jkowalski.github.io" name "jkowalski", and
    "linkedin.com/in/jan-kowalski-1a2b" names "jan-kowalski-1a2b". The pipeline masks such a handle
    wherever it appears on its own: in a JSON field, after "@", in a repository's name.
    """
    match = PROFILE_LINK_PATTERN.match(link.strip())
    return None if match is None else _handle_of(match)


_LOCATORS: dict[str, Locator] = {
    "ADDRESS": _locate_address,
    "LINK": _locate_profile_link,
    "PESEL": _locate_pesel,
    "NIP": _locate_nip,
    "REGON": _locate_regon,
    "IBAN": _locate_iban,
    "CREDIT_CARD": _locate_cards,
    "PL_ID": _locate_pl_id,
    "PL_PASSPORT": _locate_passport,
}


def _validated(
    match: re.Match[str], validator: Callable[[str], bool] | None
) -> list[tuple[int, int]]:
    """Return the extent of a match its validator accepts.

    A pattern that needs surrounding context captures the value itself in a `value` group, so only
    the identifier is masked and the label stays readable.
    """
    group = "value" if "value" in match.re.groupindex else 0
    if validator is not None and not validator(match.group(group)):
        return []
    return [match.span(group)]


class RegexGuard(Guard):
    """Finds Polish identifiers and contact details, keeping only checksum-valid matches."""

    name = "regex_pl"

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return spans for every enabled identifier pattern that also passes its validator."""
        spans: list[Span] = []
        for entity_type, pattern, validator in PATTERNS:
            if entity_type not in enabled:
                continue
            spans.extend(
                Span(start, end, entity_type, 1.0, self.name)
                for start, end in _extents(text, entity_type, pattern, validator)
            )
        return spans


def _extents(
    text: str,
    entity_type: str,
    pattern: re.Pattern[str],
    validator: Callable[[str], bool] | None,
) -> Iterator[tuple[int, int]]:
    """Yield the extent of every match of `pattern` that its locator or validator accepts.

    An IBAN candidate can run on into the next IBAN, in a list written one per line or in
    "X oraz Y", and is then cut back to its registered length. The scan resumes where that IBAN
    ended, not where the candidate did: resuming after the candidate skipped the start of the next
    IBAN, which then went out unmasked.
    """
    locate = _LOCATORS.get(entity_type)
    position = 0
    while (m := pattern.search(text, position)) is not None:
        extents = locate(text, m) if locate is not None else _validated(m, validator)
        yield from extents
        if entity_type == "IBAN":
            position = extents[-1][1] if extents else m.start() + 1
        else:
            position = max(m.end(), m.start() + 1)
