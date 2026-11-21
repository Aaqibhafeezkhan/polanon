"""Polish personal names in lowercase and casual text, found without a model.

spaCy's Polish model recognises a person mostly by the capital letter. "Z panem Kamilem Bednarkiem"
is masked; the same sentence typed the way people type into a chat window, "z panem kamilem
bednarkiem", reached the provider untouched. This guard ignores letter case. It knows the common
Polish first names in every grammatical case, typed with or without Polish letters, and reads the
two things around a name that give a person away: a form of address in front of it ("pan", "pani",
"dr") and a surname next to it that agrees with the first name in case ("kamilem bednarkiem", "ani
kowalskiej"). An initial counts as a first name in a signature: "J. Kowalski".

Some forms of first names are also ordinary words: "marka" is Marek's genitive and a brand, "mają"
is Maja's instrumental and "they have", "ani" is Ania's genitive and "nor", "kasko" is how Kaśko
comes out without Polish letters. Those count as a name only after a form of address, or next to a
surname with a telling ending or on a short list of common surnames. The declension rules were
validated against the hunspell pl_PL word list, and the list of such collisions was drawn from it.

Standard library only, so the guard also runs in the configuration without a language model.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from polanon.guards import _pesel_names
from polanon.guards.base import Guard
from polanon.types import Span

#: A grammatical reading of a word: gender (``m`` or ``f``) and case.
Tag = tuple[str, str]

CASES = ("nom", "gen", "dat", "acc", "ins", "loc", "voc")

#: Every reading at once, for abbreviations such as "dr" that do not inflect.
ANY_TAG: frozenset[Tag] = frozenset((gender, case) for gender in "mf" for case in CASES)

#: Confidence: above the NER guard (0.85), so a name read together with its surname labels the
#: overlap, and below validated identifiers and companies anchored on a legal form.
NAME_SCORE = 0.9
#: A first name alone, with no surname, initial or form of address next to it, scores below the
#: NER guard: "Wakacje na Kubie" is Cuba as likely as a man, and where the model reads a place or a
#: company its label wins. Without the model the word is still masked, as a person.
LONE_NAME_SCORE = 0.8


def _words(block: str) -> list[str]:
    """Split a whitespace-separated block, which keeps long word lists readable."""
    return block.split()


_MALE_NAMES = _words(
    """
    Adam Adrian Alan Albert Aleks Aleksander Alfred Andrzej Antoni Arkadiusz Artur August
    Bartłomiej Bartosz Benedykt Bernard Błażej Bogdan Bogumił Bogusław Bolesław Borys Bronisław
    Cezary Cyprian Czesław Damian Daniel Dariusz Dawid Dominik Edmund Edward Emanuel Emil Ernest
    Eryk Eugeniusz Fabian Feliks Filip Florian Franciszek Fryderyk Gabriel Gerard Gracjan Grzegorz
    Gustaw Henryk Hieronim Hubert Ignacy Igor Ireneusz Jacek Jakub Jan Janusz Jarosław Jędrzej
    Jerzy Joachim Józef Julian Juliusz Kacper Kajetan Kamil Karol Kazimierz Klaudiusz Konrad
    Konstanty Kornel Krystian Krzysztof Ksawery Lech Leon Leonard Leszek Longin Lucjan Ludwik
    Łukasz Maciej Maks Maksymilian Marcel Marcin Marek Marian Mariusz Mateusz Maurycy Michał
    Mieczysław Mikołaj Miłosz Mirosław Natan Nataniel Nikodem Norbert Olaf Olgierd Oliwer Oliwier
    Oskar Patryk Paweł Piotr Przemysław Radosław Rafał Remigiusz Robert Roman Ryszard Sebastian
    Seweryn Sławomir Stanisław Stefan Szczepan Szymon Tadeusz Teodor Tobiasz Tomasz Tymon Tymoteusz
    Wacław Waldemar Walenty Wiesław Wiktor Wincenty Witold Władysław Włodzimierz Wojciech Zbigniew
    Zdzisław Zenon Zygmunt
    Adaś Antek Bartek Bolek Czesiek Darek Franek Grzesiek Heniek Janek Jarek Jasiek Jaś Józek
    Julek Jurek Kazek Krzysiek Kuba Kubuś Maciek Mietek Mirek Olek Piotrek Przemek Radek Romek
    Rysiek Sławek Staś Staszek Szymek Tadek Tadzio Tomek Tymek Waldek Wiesiek Witek Władek Wojtek
    Zbyszek Zdzisiek Zenek
    """
)

_FEMALE_NAMES = _words(
    """
    Ada Adrianna Agata Agnieszka Aldona Aleksandra Alicja Alina Amelia Anastazja Aneta Angelika
    Aniela Anita Anna Antonina Apolonia Aurelia Barbara Beata Bianka Blanka Bogumiła Bogusława
    Bożena Bronisława Cecylia Celina Czesława Dagmara Danuta Daria Diana Dominika Dorota Edyta
    Eliza Elwira Elżbieta Emilia Emma Ewa Ewelina Felicja Gabriela Genowefa Grażyna Halina Hanna
    Helena Honorata Iga Ilona Irena Iwona Izabela Jadwiga Jagoda Janina Joanna Jolanta Józefa
    Judyta Julia Julita Justyna Kaja Kalina Kamila Karina Karolina Katarzyna Kinga Klara Klaudia
    Kornelia Krystyna Ksenia Laura Lena Leokadia Lidia Liliana Lilianna Liwia Lucyna Ludmiła Łucja
    Magdalena Maja Malwina Małgorzata Marcelina Maria Marianna Mariola Marta Martyna Marzena
    Matylda Melania Michalina Milena Mirosława Monika Nadia Natalia Nela Nikola Nina Olga Oliwia
    Otylia Patrycja Paula Paulina Pola Regina Renata Roksana Rozalia Róża Sabina Sandra Sara Sonia
    Stanisława Stefania Sylwia Tamara Teresa Urszula Wanda Weronika Wiesława Wiktoria Wioletta
    Władysława Zdzisława Zofia Zuzanna Żaneta
    Aga Ania Asia Basia Baśka Danusia Ela Gosia Gośka Hania Iza Jadzia Joasia Jola Julka Kasia
    Kaśka Krysia Madzia Magda Majka Małgosia Marysia Ola Oliwka Tosia Ula Wiola Zosia Zuzia
    """
)

#: Common surnames without a telling ending, declined like nouns. Recognised only next to a first
#: name or a form of address, never on their own: most of them are also animals or trades.
_NOUN_SURNAMES = _words(
    """
    Baran Bąk Dziedzic Gil Grzyb Kaczor Kołodziej Kot Kowal Król Kruk Lis Mazur Mróz Stępień Szulc
    Wilk Wróbel Zając
    """
)

#: Common surnames ending in -a, declined like feminine nouns for men and women alike.
_A_SURNAMES = _words("Duda Gajda Kania Kępa Krupa Kula Mucha Sikora Skiba Sowa Wrona Zięba")

#: Oblique stems that are not the nominative minus its ending: a mobile "e" (Paweł → Pawła) or a
#: vowel that alternates (Mróz → Mroza).
_OBLIQUE_STEMS = {
    "aleksander": "aleksandr",
    "kacper": "kacpr",
    "magister": "magistr",
    "minister": "ministr",
    "mróz": "mroz",
    "paweł": "pawł",
    "poseł": "posł",
    "stępień": "stępni",
    "wróbel": "wróbl",
}

#: Diminutives whose vocative ends in -u (Olu, Ulu) rather than -o.
_VOCATIVE_IN_U = frozenset({"ela", "jola", "ola", "ula"})

_VOWELS = frozenset("aąeęioóuy")
_SOFT_ENDINGS = ("i", "j", "l", "c", "sz", "cz", "rz", "dz", "ż", "ś", "ź", "ć", "ń")
_MASCULINE_LOCATIVE = (("st", "ście"), ("t", "cie"), ("d", "dzie"), ("r", "rze"), ("ł", "le"))
_FEMININE_DATIVE = (
    ("k", "ce"),
    ("g", "dze"),
    ("ch", "sze"),
    ("st", "ście"),
    ("tt", "tcie"),
    ("t", "cie"),
    ("d", "dzie"),
    ("r", "rze"),
    ("ł", "le"),
)
_SOFTENED = {"s": "ś", "z": "ź", "c": "ć", "n": "ń"}

_FOLD = str.maketrans("ąćęłńóśźż", "acelnoszz")


def fold(word: str) -> str:
    """Lowercase a word and drop Polish diacritics, the way people type in a hurry."""
    return word.lower().translate(_FOLD)


def _merge(forms: list[tuple[str, set[str]]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for form, cases in forms:
        merged.setdefault(form, set()).update(cases)
    return merged


def _oblique_stem(name: str) -> str:
    """Return the stem the oblique cases of a masculine noun are built on."""
    if name in _OBLIQUE_STEMS:
        return _OBLIQUE_STEMS[name]
    if name.endswith("ś"):
        return name[:-1] + "si"  # Jaś → Jasia
    if name.endswith("io"):
        return name[:-1]  # Tadzio → Tadzia
    if name.endswith("iek") and len(name) > 4 and name[-4] in _SOFTENED:
        return name[:-4] + _SOFTENED[name[-4]] + "k"  # Maciek → Maćka
    if name.endswith("ek"):
        return name[:-2] + "k"  # Marek → Marka
    return name


def _masculine_locative(stem: str) -> str:
    if stem.endswith(("k", "g", "ch", *_SOFT_ENDINGS)):
        return stem + "u"
    for hard, softened in _MASCULINE_LOCATIVE:
        if stem.endswith(hard):
            return stem[: -len(hard)] + softened
    return stem + "ie"


def decline_masculine(name: str) -> dict[str, set[str]]:
    """Return every singular form of a masculine name or noun, mapped to the cases it expresses.

    Args:
        name: the nominative, lowercase.
    """
    if name.endswith("a"):
        return decline_feminine(name)  # Kuba declines like Anna
    if name.endswith(("y", "i")) and name[-2] not in _VOWELS:
        base = name[:-1] if name.endswith("y") else name  # Jerzy → Jerzego, Antoni → Antoniego
        return _merge(
            [
                (name, {"nom", "voc"}),
                (base + "ego", {"gen", "acc"}),
                (base + "emu", {"dat"}),
                (base + ("m" if name.endswith("i") else "ym"), {"ins", "loc"}),
            ]
        )
    stem = _oblique_stem(name)
    return _merge(
        [
            (name, {"nom"}),
            (stem + "a", {"gen", "acc"}),
            (stem + "owi", {"dat"}),
            (stem + ("iem" if stem.endswith(("k", "g")) else "em"), {"ins"}),
            (_masculine_locative(stem), {"loc", "voc"}),
        ]
    )


def decline_feminine(name: str) -> dict[str, set[str]]:
    """Return every singular form of a noun ending in -a, mapped to the cases it expresses.

    Args:
        name: the nominative, lowercase.
    """
    stem = name[:-1]
    if stem.endswith("i"):
        # Maria → Marii, but Kasia → Kasi: the spelling does not say which, so both are kept.
        oblique = {stem + "i"}
        vocative = {stem + "o"}
        if stem[-2:-1] in _SOFTENED:
            oblique.add(stem)
            vocative.add(stem + "u")
    elif stem.endswith("j"):
        oblique = {stem[:-1] + "i" if stem[-2:-1] in _VOWELS else stem + "i"}  # Mai, Patrycji
        vocative = {stem + "o"}
    elif stem.endswith("l"):
        oblique = {stem + "i"}
        vocative = {stem + ("u" if name in _VOCATIVE_IN_U else "o")}
    elif stem.endswith(("sz", "cz", "rz", "ż", "c", "dz")):
        oblique = {stem + "y"}  # Róża → Róży
        vocative = {stem + "o"}
    else:
        genitive = stem + ("i" if stem.endswith(("k", "g")) else "y")
        dative = next(
            (stem[: -len(hard)] + soft for hard, soft in _FEMININE_DATIVE if stem.endswith(hard)),
            stem + "ie",
        )
        return _merge(
            [
                (name, {"nom"}),
                (genitive, {"gen"}),
                (dative, {"dat", "loc"}),
                (stem + "ę", {"acc"}),
                (stem + "ą", {"ins"}),
                (stem + "o", {"voc"}),
            ]
        )
    return _merge(
        [
            (name, {"nom"}),
            *((form, {"gen", "dat", "loc"}) for form in oblique),
            (stem + "ę", {"acc"}),
            (stem + "ą", {"ins"}),
            *((form, {"voc"}) for form in vocative),
        ]
    )


def _index(entries: list[tuple[str, dict[str, set[str]]]]) -> dict[str, frozenset[Tag]]:
    """Build a lookup from folded form to every reading it has."""
    table: dict[str, set[Tag]] = {}
    for gender, forms in entries:
        for form, cases in forms.items():
            table.setdefault(fold(form), set()).update((gender, case) for case in cases)
    return {form: frozenset(tags) for form, tags in table.items()}


#: Folded form of a first name -> its readings.
NAME_FORMS: dict[str, frozenset[Tag]] = _index(
    [("m", decline_masculine(name.lower())) for name in _MALE_NAMES]
    + [("f", decline_feminine(name.lower())) for name in _FEMALE_NAMES]
    # The rest of the names a thousand people or more bear in the PESEL register.
    + [("m", decline_masculine(name.lower())) for name in _words(_pesel_names.MALE)]
    + [("f", decline_feminine(name.lower())) for name in _words(_pesel_names.FEMALE)]
)

#: Forms of first names that are also ordinary Polish words or very common English ones, folded.
AMBIGUOUS_FORMS = frozenset(
    _words(
        """
        age ago ani antonim asia asie august basie czeska czesku franka frankiem franku jagoda
        jagode jagody jagodzie kaja kalina kaline kalinie kaliny kasce kasie kaska kaske kaski kasko
        konstanty laury laurze maja majce majka majke majki maks maksa maksem maksie marek marka
        marzenie mikolaj mikolaja nine oliwce oliwie oliwka oliwke oliwki pola pole polo roza roze
        rozy ryska rysku ula ule ulu wiktoria witka
        """
    )
) | frozenset(_words(_pesel_names.AMBIGUOUS))

_SURNAME_FORMS: dict[str, frozenset[Tag]] = _index(
    [("m", decline_masculine(name.lower())) for name in _NOUN_SURNAMES]
    + [("f", {name.lower(): set(CASES)}) for name in _NOUN_SURNAMES]
    + [(gender, decline_feminine(name.lower())) for name in _A_SURNAMES for gender in "mf"]
)

#: Forms of common surnames that settle an ambiguous first name even in lowercase.
_COMMON_SURNAME_FORMS = frozenset(
    fold(form)
    for name in _words("Nowak Wójcik Woźniak Kowalczyk Kaczmarek Krawczyk Mazur")
    for form in decline_masculine(name.lower())
)

#: Abbreviated titles, which may be followed by a full stop.
_ABBREVIATIONS = frozenset(_words("adw dr dyr hab inz lek mec mgr prof red"))

#: Abbreviated titles that are also ordinary words: "lek" is a medicine and "red" a colour, so "brać
#: lek Apap" or "Colour: red. Size: M" name nobody. They count before a qualifier ("lek. med."), or
#: with their full stop before a word that reads as a surname ("red. Nowak").
_WORDLIKE_ABBREVIATIONS = frozenset({"lek", "red"})

#: Abbreviated qualifiers between a title and the name, written with a full stop: "lek. med.",
#: "lek. dent.", "dr n. med.".
_ABBREVIATED_ROLES = frozenset(_words("dent farm med n wet"))

#: Forms of address on foreign documents, folded. In front of a name they are no part of it, and
#: alone they point to nobody.
FOREIGN_TITLES = frozenset(_words("dame frau herr madame miss mlle mme monsieur mr mrs ms mx sir"))


def _honorifics() -> dict[str, frozenset[Tag]]:
    """Forms of address after which the next word is read as a name."""
    table: dict[str, set[Tag]] = {
        "pan": {("m", "nom")},
        "pana": {("m", "gen"), ("m", "acc")},
        "panu": {("m", "dat")},
        "panem": {("m", "ins")},
        "panie": {("m", "loc"), ("m", "voc")},
        "pani": {("f", case) for case in ("nom", "gen", "dat", "loc", "voc")},
        "pania": {("f", "acc"), ("f", "ins")},
    }
    for abbreviation in _ABBREVIATIONS:
        table[abbreviation] = set(ANY_TAG)
    for title in ("doktor", "profesor", "mecenas"):
        for form, cases in decline_masculine(title).items():
            table.setdefault(fold(form), set()).update(("m", case) for case in cases)
        table[title].update(("f", case) for case in CASES)  # "pani doktor" does not inflect
    for gender, word in (("m", "kolega"), ("f", "koleżanka")):
        for form, cases in decline_feminine(word).items():
            table.setdefault(fold(form), set()).update((gender, case) for case in cases)
    return {form: frozenset(tags) for form, tags in table.items()}


#: Folded form of address -> the readings a name after it has to agree with.
HONORIFICS = _honorifics()

#: Job titles that sit between a form of address and the name ("pani dyrektor Nowak", "pan
#: hydraulik Wróbel"), and the words that complete one ("Pan Prezes Zarządu"). After an address
#: they are never taken for the surname: "z panem hydraulikiem" names a trade, not a person.
_ROLES = (
    frozenset(
        fold(form)
        for title in _words(
            """
            prezes dyrektor kierownik inżynier magister redaktor prokurator komornik kapitan
            burmistrz wójt minister poseł notariusz sędzia radca naczelnik rzecznik hydraulik
            mechanik elektryk kierowca sekretarka księgowy księgowa
            """
        )
        for form in (decline_feminine(title) if title.endswith("a") else decline_masculine(title))
    )
    | {"sedziego", "sedziemu", "ksiegowej", "zarzadu", "rady", "dzialu"}
    | _ABBREVIATED_ROLES
)

#: Capitalised words that never start or continue a person's name.
_NOT_NAMES = frozenset(
    _words(
        """
        al bog boga bogiem bogu boze ceo cfo cto dzial e-mail email fax firma gmbh hr iban inc it
        krs ltd nip nr os pan pani panstwo pesel pl pm qa regon sa sp spolka tel ul vat zespol
        """
    )
)

#: Ordinary nouns that look like a surname ending in -ek, -ak, -ik, -yk or -uk. The second block
#: holds the -ek forms of common feminine nouns in -ka, which read as the genitive of an -ek
#: surname (Bednarek, Bednarka): "Pana przesyłka została nadana" masked the parcel as a person.
_NOT_SURNAMES = frozenset(
    _words(
        """
        analityk budynek chemik chlopak czlowiek czwartek dodatek dzieciak elektronik elektryk
        fizyk grafik hydraulik informatyk jezyk kierownik logistyk lotnik majatek mechanik muzyk
        naczelnik obowiazek ogrodnik pelnomocnik piatek poczatek podatek polityk pomocnik porzadek
        posrednik pracownik prawnik przewodnik przypadek rachunek ratownik rolnik rynek rzecznik
        skutek spadek strazak technik uczestnik urzednik wiek wniosek wspolnik wtorek wydatek
        wyjatek wynik wypadek zakatek

        corek ciotek drukarek dzialek kartek kolezanek komorek ksiazek kurtek ladowarek malzonek
        matek myszek naklejek paczek partnerek podwyzek pozyczek pralek przeprowadzek przesylek
        skladek skrzynek sluchawek suszarek szafek teczek torebek ulotek usterek walizek wizytowek
        wnuczek wycieczek wysylek zabawek zaliczek znizek zmywarek
        """
    )
)

#: Stems of common adjectives in -ski, -cki and -dzki: "ani polskiej, ani angielskiej wersji".
_COMMON_ADJECTIVES = frozenset(
    _words(
        """
        akademick amerykansk angielsk arabsk australijsk austriack baltyck belgijsk bialorusk blisk
        bosk brytyjsk bulgarsk chinsk chorwack czesk damsk dolnoslask dunsk estonsk europejsk finsk
        francusk gdansk greck gruzinsk hiszpansk holendersk indyjsk irlandzk japonsk kanadyjsk
        karpack katowick krakowsk krolewsk kujawsk lekarsk litewsk lodzk lotewsk lubelsk lubusk
        ludzk malopolsk mazowieck mesk miejsk morsk niemieck nisk norwesk opolsk pansk plask
        podkarpack podlask polsk pomorsk portugalsk poznansk rosyjsk rumunsk sasiedzk serbsk slask
        slisk slowack slowensk szkock swietokrzysk szwajcarsk szwedzk tureck ukrainsk wask
        warszawsk wegiersk wielkopolsk wiejsk wlosk wroclawsk zachodniopomorsk zensk zydowsk
        """
    )
)

_ADJECTIVAL = re.compile(r"(?:s|c|dz)k(?P<ending>i|iego|iemu|im|a|iej)$")
_ADJECTIVE_FORM = re.compile(r"(?:s|c|dz)k(?P<ending>i|iego|iemu|im|a|iej|ie|ich|imi)$")
_ADJECTIVAL_TAGS: dict[str, frozenset[Tag]] = {
    "i": frozenset({("m", "nom"), ("m", "voc")}),
    "iego": frozenset({("m", "gen"), ("m", "acc")}),
    "iemu": frozenset({("m", "dat")}),
    "im": frozenset({("m", "ins"), ("m", "loc")}),
    "a": frozenset({("f", "nom"), ("f", "voc"), ("f", "acc"), ("f", "ins")}),
    "iej": frozenset({("f", "gen"), ("f", "dat"), ("f", "loc")}),
}
_NOUN_OBLIQUE = re.compile(r"(?:(?<![eo])k(?P<k>a|owi|iem|u)|wicz(?P<wicz>a|owi|em|u))$")
_NOUN_NOMINATIVE = re.compile(r"(?:[aiuy]k|ek|wicz)$")
_OBLIQUE_TAGS: dict[str, frozenset[Tag]] = {
    "a": frozenset({("m", "gen"), ("m", "acc")}),
    "owi": frozenset({("m", "dat")}),
    "iem": frozenset({("m", "ins")}),
    "em": frozenset({("m", "ins")}),
    "u": frozenset({("m", "loc"), ("m", "voc")}),
}
#: A bare noun surname: a man in the nominative, or a woman in any case, since hers does not
#: inflect ("z Anną Nowak").
_NOMINATIVE_TAGS = frozenset({("m", "nom")} | {("f", case) for case in CASES})

#: The readings a first name has in a list or a form, where it stands in the nominative.
_NOMINATIVE = frozenset({("m", "nom"), ("f", "nom")})

_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_SPACE = re.compile(r"[ \t\u00a0]+")
_SPACE_AFTER_ABBREVIATION = re.compile(r"\.?[ \t\u00a0]+")
_AFTER_INITIAL = re.compile(r"\.[ \t\u00a0]*")
#: One line break, as a hard-wrapped e-mail puts between a first name and its surname.
_LINE_BREAK = re.compile(r"[ \t\u00a0]*\r?\n[ \t\u00a0]*")
#: One ";", "," or "|", as a CSV line or a Markdown table puts between two cells.
_CELL_BREAK = re.compile(r"[ \t\u00a0]*[;,|][ \t\u00a0]*")

#: A login written from a name: a first name or an initial, one dot, "_" or "-", then a word,
#: maybe with digits, standing on its own: not inside an e-mail address, a domain or a longer
#: dotted or hyphenated handle. :func:`_login_names_a_person` decides whether it is a person.
_NAME_LOGIN = re.compile(
    r"(?<![\w.@-])(?P<first>[^\W\d_]+)[._-](?P<surname>[^\W\d_]+)\d{0,4}(?![\w@-]|\.[^\W\d_])"
)

#: A Roman numeral, which follows a name in capitals without being one: "Jana Pawła II".
_ROMAN = re.compile(r"[IVXLCDM]+")

#: What ends a field in a form or a list: a comma, a semicolon or the end of the line.
_FIELD_END = re.compile(r"[ \t\u00a0]*(?:[,;]|\r?\n|$)")

#: A field label's colon, and the words before a colon that open an e-mail subject instead.
_LABEL_GAP = re.compile(r":[ \t\u00a0]*")
_SUBJECT_PREFIXES = frozenset(_words("dotyczy fw fwd odp pd re temat"))

#: Space-based dates retain their original rules: "15 Jan 2024", "Jan 15, 2024".
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"
#: What stands before "maja" when it is May: a day, or "od", "do", "końca", "początku", "połowy".
_MAY_BEFORE = re.compile(
    rf"(?:(?<![\d.,:]){_DAY}|\b(?:od|do|końca|początku|połowy|połowie))[ \t\u00a0]+$",
    re.IGNORECASE,
)
_DAY_BEFORE = re.compile(rf"(?<![\d.,:]){_DAY}\.?[ \t\u00a0]+$")
_YEAR_AFTER = re.compile(r"\.?,?[ \t\u00a0]+(?:19|20)\d\d(?!\d)")
_DAY_AND_YEAR_AFTER = re.compile(
    rf"\.?[ \t\u00a0]+{_DAY}(?:st|nd|rd|th)?,?[ \t\u00a0]+(?:19|20)\d\d(?!\d)"
)

#: Punctuated dates require the same single separator on both sides of the month.
_SEPARATED_DAY_BEFORE = re.compile(rf"(?<![\d.,:]){_DAY}(?P<separator>[-/.])$")
_SEPARATED_YEAR_AFTER = re.compile(r"(?P<separator>[-/.])(?:19|20)\d\d(?!\d)")
_SEPARATED_YEAR_BEFORE = re.compile(r"(?<![\d.,:])(?:19|20)\d\d(?P<separator>[-/.])$")
_SEPARATED_DAY_AFTER = re.compile(rf"(?P<separator>[-/.]){_DAY}(?!\d)")
_SEPARATED_DAY_AND_YEAR_AFTER = re.compile(
    rf"(?P<separator>[-/.]){_DAY}(?:st|nd|rd|th)?(?P=separator)(?:19|20)\d\d(?!\d)"
)


def is_common_adjective(word: str) -> bool:
    """Return whether a word is a form of a common adjective in -ski, -cki or -dzki ("polskiej")."""
    folded = fold(word)
    match = _ADJECTIVE_FORM.search(folded)
    return match is not None and folded[: match.start("ending")] in _COMMON_ADJECTIVES


#: A feminine surname that declines as an adjective: "Bielska", "Nowacka", "Zawadzka".
_FEMININE_ADJECTIVAL_SURNAME = re.compile(r"(?:s|c|dz)ka$")


def _feminine_oblique_forms(surname: str) -> list[str]:
    """Return "Bielskiej" and "Bielską" for "Bielska", and nothing for a surname of another kind.

    The masculine "Kowalskiego" begins with "Kowalski", so an occurrence of the surname inside it
    takes the whole word; no case of "Bielska" but the nominative begins with it.
    """
    if _FEMININE_ADJECTIVAL_SURNAME.search(fold(surname)) is None or is_common_adjective(surname):
        return []
    endings = ("IEJ", "Ą") if surname.isupper() else ("iej", "ą")
    return [surname[:-1] + ending for ending in endings]


#: A vowel of the last syllable that the other cases of a noun surname drop: "Wróbel" and
#: "Wróbla", "Kozieł" and "Kozła", "Krawiec" and "Krawca", "Dudek" and "Dudka", "Kozioł" and
#: "Kozła".
_FLEETING_VOWEL = re.compile(r"(?<=[^aąeęioóuy])(?:i?e(?=[cklł]$)|io(?=ł$))")
#: An "e" after a soft "i" that the other cases may write "o" before a hard consonant, or the
#: other way round: "Kozieł" and "Koziołowi", "Kozioł" and "Kozieła".
_E_OR_O = re.compile(r"(?<=[^aąeęioóuy]i)[eo](?=[tdsznłr]$)")
#: An "ó" before the last consonant, which the other cases write "o": "Mróz" and "Mroza". Not
#: before a "w": "Lisów" is no nominative but the genitive plural of "Lis".
_O_ACUTE = re.compile(r"ó(?=[^aąeęioóuyw]$)")
#: An "o" in that place, which the nominative may write "ó".
_PLAIN_O = re.compile(r"o(?=[^aąeęioóuy]$)")
#: What a noun surname takes after its stem: the genitive, the dative and, for the family, the
#: plural. The instrumental and the locative depend on the stem's last consonant.
_NOUN_SURNAME_ENDINGS = ("a", "owi", "owie", "ów", "om", "ami", "ach")


def _other_vowel(match: re.Match[str]) -> str:
    return "o" if match[0] == "e" else "e"


@lru_cache(maxsize=4096)
def _noun_surname_forms(nominative: str) -> frozenset[str]:
    """Return the other cases of a lowercase surname declined like a masculine noun.

    The stem may keep its last vowel, drop it or change it, since surnames are declined each of
    these ways: "kozieł" gives "kozieła", "kozła" and "koziołowi". Every stem gives every case,
    so some forms are spellings nobody writes, such as "kisla" for "kisiel"; they are no words
    either, and masking them hides nothing anybody needed to read.
    """
    stems = {
        nominative,
        _oblique_stem(nominative),
        _FLEETING_VOWEL.sub("", nominative),
        _E_OR_O.sub(_other_vowel, nominative),
        _O_ACUTE.sub("o", nominative),
    }
    forms: set[str] = set()
    for stem in stems:
        forms.update(stem + ending for ending in _NOUN_SURNAME_ENDINGS)
        forms.add(stem + ("iem" if stem.endswith(("k", "g")) else "em"))
        forms.add(_masculine_locative(stem))
    forms.discard(nominative)
    return frozenset(forms)


def _declines_as_a_noun(surname: str) -> bool:
    """Whether a surname may decline like a masculine noun: one word that ends in a consonant."""
    return len(surname) >= 3 and surname.isalpha() and surname[-1].lower() not in _VOWELS


def _written_like(word: str, forms: Iterable[str]) -> list[str]:
    """Write lowercase forms of `word` in its letter case: in capitals, or with its capital."""
    if word.isupper():
        return [form.upper() for form in forms]
    return [word[0] + form[1:] for form in forms]


def _masculine_oblique_forms(surname: str) -> list[str]:
    """Return the cases of a noun surname that do not begin with it: "Wróbla", "Kozła", "Kocie".

    The other cases of "Nowak" begin with it, so an occurrence of the surname inside them takes
    the whole word; some cases of "Wróbel", "Kozieł" or "Kot" do not. A surname that ends in a
    vowel has none here: "Kowalski" declines from its own spelling, "Bielska" and "Duda" have
    their own forms.
    """
    if not _declines_as_a_noun(surname):
        return []
    key = surname.lower()
    return _written_like(
        surname, sorted(form for form in _noun_surname_forms(key) if not form.startswith(key))
    )


#: A feminine surname made of another with -owa or -ewa, which declines as an adjective:
#: "Nowakowa", "Nowakowej". "Sowa" or "Głowa" is a noun.
_ADJECTIVAL_WIFE_SURNAME = re.compile(r"^.{3,}[oe]wa$")


def _declines_like_duda(surname: str) -> bool:
    """Whether a surname ends in -a and declines like a feminine noun, for a man and a woman."""
    key = surname.lower()
    return (
        len(key) >= 3
        and key.isalpha()
        and key.endswith("a")
        and _FEMININE_ADJECTIVAL_SURNAME.search(fold(key)) is None
        and _ADJECTIVAL_WIFE_SURNAME.search(key) is None
    )


def _a_surname_oblique_forms(surname: str) -> list[str]:
    """Return "Dudy", "Dudzie", "Dudę", "Dudą" and "Dudo" for "Duda": the cases of an -a surname.

    A man's and a woman's alike, as for Duda, Sikora or Sowa, and none of them begins with the
    surname. A surname that declines as an adjective has its own forms: "Bielska" gives
    "Bielskiej", and one in -owa, "Nowakowa", is left out.
    """
    if not _declines_like_duda(surname):
        return []
    key = surname.lower()
    return _written_like(surname, sorted(form for form in decline_feminine(key) if form != key))


def surname_forms(surname: str) -> list[str]:
    """Return the cases of a surname that do not begin with it, whatever kind of surname it is."""
    return [
        *_feminine_oblique_forms(surname),
        *_masculine_oblique_forms(surname),
        *_a_surname_oblique_forms(surname),
    ]


_LOCATIVE = frozenset({"loc", "voc"})
#: The case endings a noun surname may stand in, the last consonant of its stem that the locative
#: softens, and the cases of one person each ending gives: "Kocie" is "Kot", "Mazurze" is
#: "Mazur", "Koziele" is "Kozieł". The plural is the family's.
_CASE_TAILS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("owie", "", frozenset()),
    ("owi", "", frozenset({"dat"})),
    ("ach", "", frozenset()),
    ("ami", "", frozenset()),
    ("iem", "", frozenset({"ins"})),
    ("em", "", frozenset({"ins"})),
    ("om", "", frozenset()),
    ("ów", "", frozenset()),
    ("a", "", frozenset({"gen", "acc"})),
    ("u", "", _LOCATIVE),
    ("ie", "", _LOCATIVE),
    ("ście", "st", _LOCATIVE),
    ("cie", "t", _LOCATIVE),
    ("dzie", "d", _LOCATIVE),
    ("rze", "r", _LOCATIVE),
    ("le", "ł", _LOCATIVE),
)

#: The endings of the other cases of a surname in -a, and what its nominative ends in instead:
#: "Dudy" and "Dudzie" are "Duda", "Musze" is "Mucha", "Kani" is "Kania".
_A_CASE_TAILS = (
    ("y", "a"),
    ("i", "a"),
    ("i", "ia"),
    ("ie", "a"),
    ("ście", "sta"),
    ("dzie", "da"),
    ("rze", "ra"),
    ("cie", "ta"),
    ("le", "ła"),
    ("ce", "ka"),
    ("dze", "ga"),
    ("sze", "cha"),
    ("ę", "a"),
    ("ą", "a"),
    ("o", "a"),
    ("u", "a"),
)


def noun_surname_nominatives(word: str, cases: Iterable[str] | None = None) -> list[str]:
    """Return the surnames declined like a masculine noun of which `word` is another case.

    "Koziołowi" gives "Kozieł", "Kozioł" and "Koziół", since a stem may change its vowel; one of
    them may be a surname somebody stored. Each is checked by declining it again, so only a
    surname that has `word` among its cases is returned, written with the capital `word` has.

    Args:
        word: a capitalised word.
        cases: the cases `word` stands in, when something else tells, such as a first name next
            to it; any case when omitted.
    """
    if not word[:1].isupper():
        return []
    wanted = None if cases is None else frozenset(cases)
    return _written_like(word, _nominatives(word.lower(), wanted))


@lru_cache(maxsize=65_536)
def _nominatives(key: str, cases: frozenset[str] | None) -> tuple[str, ...]:
    if len(key) < 4 or not key.isalpha():
        return ()
    found: set[str] = set()
    for tail, consonant, tail_cases in _CASE_TAILS:
        if not key.endswith(tail) or len(key) - len(tail) < 2:
            continue
        if cases is not None and not cases & tail_cases:
            continue
        stem = key[: len(key) - len(tail)] + consonant
        for nominative in _stems_restored(stem):
            if _declines_as_a_noun(nominative) and key in _noun_surname_forms(nominative):
                found.add(nominative)
    found.discard(key)
    return tuple(sorted(found))


def a_surname_nominatives(word: str, cases: Iterable[str] | None = None) -> list[str]:
    """Return the surnames in -a of which `word` is another case: "Dudzie" gives "Duda".

    Each is checked by declining it again, in one of `cases` when they are given.
    """
    if not word[:1].isupper():
        return []
    wanted = None if cases is None else frozenset(cases)
    return _written_like(word, _a_nominatives(word.lower(), wanted))


@lru_cache(maxsize=65_536)
def _a_nominatives(key: str, cases: frozenset[str] | None) -> tuple[str, ...]:
    if len(key) < 3 or not key.isalpha() or key.endswith("a"):
        return ()
    found: set[str] = set()
    for tail, ending in _A_CASE_TAILS:
        if not key.endswith(tail) or len(key) - len(tail) < 2:
            continue
        nominative = key[: len(key) - len(tail)] + ending
        if not _declines_like_duda(nominative):
            continue
        forms = decline_feminine(nominative)
        if key in forms and (cases is None or cases & forms[key]):
            found.add(nominative)
    return tuple(sorted(found))


def declined_surname_nominatives(name: str) -> list[str]:
    """Return the nominatives the surname of a name in an oblique case may have.

    "Adama Kozieła": "Adama" is only a genitive, so "Kozieła" is one too, of "Kozieł" or of
    "Kozioł", and the rules, which list neither, read the name as written. Given a first name
    this guard knows in no nominative and one other word, the other word is read as a surname in
    the cases the first name stands in: declined like a masculine noun after a man's first name,
    and like Duda after any. Nothing for a name whose first name may be a nominative too, such
    as "Kamila", Kamil's genitive and a woman's name.
    """
    words = [word.strip(".,;:()") for word in name.split()]
    if len(words) != 2:
        return []
    for first, other in (words, words[::-1]):
        tags = NAME_FORMS.get(fold(first))
        if not tags or any(case == "nom" for _, case in tags) or fold(other) in NAME_FORMS:
            continue
        if not _WORD.fullmatch(other) or not _name_shaped(other):
            return []
        cases = {case for _, case in tags}
        found = a_surname_nominatives(other, cases)
        if any(gender == "m" for gender, _ in tags):
            found += noun_surname_nominatives(other, cases)
        return found
    return []


def _stems_restored(stem: str) -> set[str]:
    """Return the nominatives a stem may come from: itself, with a vowel put back or changed."""
    restored = {stem, _E_OR_O.sub(_other_vowel, stem), _PLAIN_O.sub("ó", stem)}
    if len(stem) >= 3 and stem[-1] not in _VOWELS and stem[-2] not in _VOWELS:
        restored.update(stem[:-1] + vowel + stem[-1] for vowel in ("e", "ie", "io"))
    return restored


def forget_forms() -> None:
    """Drop the cases worked out for the latest surnames and words, and the words with them."""
    _noun_surname_forms.cache_clear()
    _nominatives.cache_clear()
    _a_nominatives.cache_clear()


def person_name_parts(
    name: str, *, needs_first_name: bool = False, cases: bool = True
) -> list[str]:
    """Return the words of a person's name that point to the person on their own: the surname.

    "Jan Kowalski" gives "Kowalski", "Anna Nowak-Kępa" gives "Nowak-Kępa", "Nowak" and "Kępa".
    "Natalia Bielska" gives "Bielska" with "Bielskiej" and "Bielską", the cases that do not begin
    with it, and "Paweł Wróbel" gives "Wróbel" with "Wróbla", "Wróblowi" and the other cases
    whose stem drops or changes a vowel, as "Kozieł" gives "Kozła" and "Koziołowi"; "Anna Duda"
    gives "Duda" with "Dudy", "Dudzie", "Dudę", "Dudą" and "Dudo". A first name
    this guard knows is left out, and so are titles, initials and common adjectives: "Jan" or
    "Krakowska" alone point to nobody. A first name the guard does not know is kept, since
    nothing tells it from the surname.

    Args:
        name: a detected name of two to four words.
        needs_first_name: give parts only when the name has a first name this guard knows, for a
            name whose detection cannot be trusted on its own.
        cases: give the cases that do not begin with the surname too; without them, the surname
            alone, as a store of tokens keeps it.
    """
    words = name.split()
    if not 2 <= len(words) <= 4:
        return []
    surnames: list[str] = []
    has_first_name = False
    #: A first name in no nominative, "Adama", puts the surname next to it in another case too,
    #: and the cases of "Bednarka" are no cases of the surname.
    in_another_case = False
    for word in words:
        bare = word.strip(".,;:()")
        key = fold(bare)
        if all(part in NAME_FORMS for part in key.split("-")):  # "Jan", "Anna-Maria"
            has_first_name = True
            tags = NAME_FORMS.get(key, ())
            in_another_case |= bool(tags) and all(case != "nom" for _, case in tags)
            continue
        if len(bare) < 3 or not _WORD.fullmatch(bare) or not _name_shaped(bare):
            continue
        if key in HONORIFICS or key in _ABBREVIATIONS or key in _NOT_NAMES or key in _ROLES:
            continue
        if key in FOREIGN_TITLES:
            continue
        if is_common_adjective(bare):
            continue
        surnames.append(bare)
        if "-" in bare:
            surnames += [p for p in bare.split("-") if len(p) >= 3 and fold(p) not in NAME_FORMS]
    if needs_first_name and not has_first_name:
        return []
    parts: list[str] = []
    for surname in surnames:
        parts.append(surname)
        if cases and not in_another_case:
            parts.extend(surname_forms(surname))
    return parts


def _login_names_a_person(login: re.Match[str]) -> bool:
    """Whether a login written from a name, "jan.nowak" or "t.malinowski", names a person.

    The whole login is in lowercase, the way a login is written, and the word after the separator
    reads as a surname. After a first name this guard knows, any surname reading will do, unless
    the first name is also an ordinary word ("inna"); after a lone initial it must be a strong
    one, a telling ending or a common surname, since "e.kot" names nobody for sure.
    """
    if not login[0].islower():
        return False
    first, surname = fold(login["first"]), login["surname"]
    if fold(surname) in NAME_FORMS:
        return False
    reading = surname_reading(surname)
    if reading is None:
        return False
    if len(first) == 1:
        return reading.strong
    if first not in NAME_FORMS:
        return False
    return reading.strong or first not in AMBIGUOUS_FORMS


def login_name_parts(login: str) -> list[str]:
    """Return the surname in a login written from a name, written as a name: "Malinowski".

    "t.malinowski" or "anna.kowalczyk" is one person, and the surname in it points to the same
    person wherever it is written as a name, and in the other logins made of it ("tmalinowski").
    The surname is given capitalised, so a lowercase word that is also a surname elsewhere in the
    text is left as the ordinary word it may be.
    """
    match = _NAME_LOGIN.fullmatch(login)
    if match is None or not _login_names_a_person(match):
        return []
    surname = match["surname"]
    capitalised = surname[:1].upper() + surname[1:]
    return [capitalised, *_feminine_oblique_forms(capitalised)]


@dataclass(frozen=True, slots=True)
class _Word:
    start: int
    end: int
    text: str
    key: str


@dataclass(frozen=True, slots=True)
class SurnameReading:
    """What a word could be as a surname, and whether it strongly suggests a person's name.

    `common` says the word is strong only for being a common surname written in lowercase, as a
    chat writes "maja nowak": strong enough to settle the first name right before it, not to
    claim a word in the next cell of a table.
    """

    tags: frozenset[Tag]
    strong: bool
    common: bool = False


def surname_reading(word: str) -> SurnameReading | None:
    """Return the grammatical readings a word has as a Polish surname, or None.

    An adjectival ending (-ski, -cka, -dzkiego), -wicz, or a declined noun ending (Nowakiem,
    Bednarka) is a strong sign, as is a form of a surname on the short common-surname list.
    Other bare -ak, -ek, -ik and listed noun surnames are weak ones, because plenty of ordinary
    nouns look the same.

    A double surname, "Kowalska-Nowak", is as strong as its stronger part and takes the cases its
    parts agree on. Read by its last part alone, a strong first part was ignored, and "mają
    kowalską-nowak" went out whole where "mają kowalską" was masked.
    """
    parts = fold(word).split("-")
    if parts[-1] in _NOT_NAMES:
        return None
    lowercase = word.islower()
    readings = [
        reading
        for reading in (_part_reading(part, lowercase) for part in parts if part not in _NOT_NAMES)
        if reading is not None
    ]
    if not readings:
        return None
    agreed = frozenset.intersection(*(reading.tags for reading in readings))
    strong = any(reading.strong for reading in readings)
    return SurnameReading(
        agreed or readings[-1].tags,
        strong=strong,
        common=strong and all(reading.common for reading in readings if reading.strong),
    )


def _part_reading(part: str, lowercase: bool = False) -> SurnameReading | None:
    """Return the readings of one folded word, or one part of a double surname, as a surname.

    A common surname is strong only in lowercase, the chat of #13: in text that writes names with
    capitals, a lowercase word before "Nowak" is the ordinary word, "od 1 maja Nowak".
    """
    common = lowercase and part in _COMMON_SURNAME_FORMS
    listed = _SURNAME_FORMS.get(part)
    if listed is not None:
        return SurnameReading(listed, strong=common, common=common)
    if len(part) < 4:
        return None
    adjectival = _ADJECTIVAL.search(part)
    if adjectival is not None:
        if part[: adjectival.start("ending")] in _COMMON_ADJECTIVES:
            return None
        return SurnameReading(_ADJECTIVAL_TAGS[adjectival.group("ending")], strong=True)
    oblique = _NOUN_OBLIQUE.search(part)
    if oblique is not None:
        ending = oblique.group("k") or oblique.group("wicz")
        nominative = part[: -len(ending)]
        if oblique.group("k") and nominative[-2:-1] not in _VOWELS:
            nominative = nominative[:-1] + "ek"  # Bednarka → Bednarek
        if nominative in _NOT_SURNAMES:
            return None
        return SurnameReading(_OBLIQUE_TAGS[ending], strong=True)
    if _NOUN_NOMINATIVE.search(part) and part not in _NOT_SURNAMES:
        wicz = part.endswith("wicz")
        return SurnameReading(_NOMINATIVE_TAGS, strong=common or wicz, common=common and not wicz)
    return None


class PersonNameGuard(Guard):
    """Masks Polish personal names regardless of letter case, as ``PERSON``."""

    name = "names_pl"

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return one span per person: first names with their surname, or an addressed surname."""
        if "PERSON" not in enabled:
            return []
        # A login written from a name is one person: the dot used to end the first name, and the
        # surname after it went out next to its token.
        logins = [
            (match.start(), match.end())
            for match in _NAME_LOGIN.finditer(text)
            if _login_names_a_person(match)
        ]
        words = [
            _Word(match.start(), match.end(), match.group(), fold(match.group()))
            for match in _WORD.finditer(text)
            if not any(start <= match.start() < end for start, end in logins)
        ]
        spans: list[Span] = [
            Span(start, end, "PERSON", NAME_SCORE, self.name) for start, end in logins
        ]
        index = 0
        taken = 0
        while index < len(words):
            found = _person_at(text, words, index, taken)
            if found is None:
                index += 1
                continue
            first, last, cell, alone = found
            score = LONE_NAME_SCORE if alone else NAME_SCORE
            spans.append(Span(words[first].start, words[last].end, "PERSON", score, self.name))
            if cell is not None:
                # A surname in the next cell is masked on its own, so the row keeps its columns.
                spans.append(
                    Span(words[cell].start, words[cell].end, "PERSON", NAME_SCORE, self.name)
                )
                last = cell
            index = taken = last + 1
        return sorted(spans, key=lambda span: span.start)


def _capitalised(word: str) -> bool:
    """Title case, not all capitals: in a shouted line the capital letter says nothing."""
    return word[0].isupper() and any(character.islower() for character in word[1:])


def _shouted(word: str) -> bool:
    """All capitals and longer than an initial: a form field, a header or a shouted line."""
    return len(word) > 1 and word.isupper()


def _name_shaped(word: str) -> bool:
    """Title case or all capitals, the two ways a name is written apart from lowercase."""
    return _capitalised(word) or _shouted(word)


def _is_title(word: _Word) -> bool:
    return word.key in _NOT_NAMES or word.key in _ROLES or word.key in HONORIFICS


def _is_initial(text: str, word: _Word) -> bool:
    """A single capital letter followed by a full stop: the "J." of "J. Kowalski"."""
    return len(word.text) == 1 and word.text.isupper() and text.startswith(".", word.end)


def _agrees(surname: frozenset[Tag], wanted: frozenset[Tag]) -> bool:
    """Whether a surname agrees in case with the first name or the address in front of it.

    A man's surname declined like a noun keeps its nominative when he is addressed ("Panie Nowak",
    "Pawle Wróbel"), so the nominative also serves as his vocative.
    """
    return bool(surname & wanted) or (("m", "voc") in wanted and ("m", "nom") in surname)


def _written_as_a_name(
    text: str, words: list[_Word], index: int, reference: _Word, nominative: bool
) -> bool:
    """Whether a word without a surname ending is capitalised the way a name after `reference` is.

    Title case says so. Capitals do in the nominative, the case of forms and signatures, after a
    name in Title case ("Jan KONIECZNY") but not after "wyślij Kasi PDF". In a line written in
    capitals every word has them, so there the word counts only where the capitals stop after it,
    as at the end of a form field ("JAN KONIECZNY, PESEL ..."): "ANNA PROSI O FAKTURĘ" asks for an
    invoice.

    Args:
        text: the whole text.
        words: its words, in order.
        index: the word to judge.
        reference: the first name or the form of address in front of it.
        nominative: whether that name or address can be read in the nominative.
    """
    word = words[index].text
    if _capitalised(word):
        return True
    if not (nominative and _shouted(word) and _name_shaped(reference.text)):
        return False
    if _ROMAN.fullmatch(word):
        return False
    if _capitalised(reference.text):
        return True
    following = _neighbour(text, words, index, 1)
    return (
        following is None
        or not words[following].text.isupper()
        or words[following].key in _NOT_NAMES
    )


def _neighbour(text: str, words: list[_Word], index: int, step: int) -> int | None:
    """Return the next (step 1) or previous (step -1) word when only spaces separate them.

    A full stop is allowed too after an abbreviated title, as in "dr. Nowak" or "lek. med. Nowak".
    """
    other = index + step
    if not 0 <= other < len(words):
        return None
    left, right = (words[index], words[other]) if step > 0 else (words[other], words[index])
    gap = text[left.end : right.start]
    abbreviated = left.key in _ABBREVIATIONS or left.key in _ABBREVIATED_ROLES
    pattern = _SPACE_AFTER_ABBREVIATION if abbreviated else _SPACE
    return other if pattern.fullmatch(gap) else None


def _address_before(text: str, words: list[_Word], index: int) -> frozenset[Tag] | None:
    """Return the readings of a form of address in front of a word, skipping job titles."""
    position = index
    for _ in range(3):
        previous = _neighbour(text, words, position, -1)
        if previous is None:
            return None
        key = words[previous].key
        if key in HONORIFICS:
            if key in _WORDLIKE_ABBREVIATIONS and not _abbreviates(text, words, previous, index):
                return None
            return HONORIFICS[key]
        if key not in _ROLES:
            return None
        position = previous
    return None


def _abbreviates(text: str, words: list[_Word], title: int, index: int) -> bool:
    """Whether "lek" or "red" abbreviates a title here rather than being the ordinary word.

    It does before a qualifier ("lek. med. Nowak", typed without the full stops too), or with its
    full stop before a word that reads as a surname ("red. Nowak"); "nowy lek. Jutro idę do
    apteki" ends a sentence instead.
    """
    if title + 1 < index and words[title + 1].key in _ABBREVIATED_ROLES:
        return True
    return text.startswith(".", words[title].end) and surname_reading(words[index].text) is not None


def _in_a_date(text: str, word: _Word) -> bool:
    """Whether a first name occupies the month in a space-based or punctuated date.

    English, German and French months share spellings with Jan and Maja's genitive.
    Punctuated dates use matching separators: "15-Jan-2024", "2024/Jan/15", "Jan.15.2024".
    Spaces keep the original day-first/month-first rules, since "2023 Anna 2" is not a date.
    """
    if _DAY_AND_YEAR_AFTER.match(text, word.end) is not None:
        return True
    if text[word.start : word.end] == "maja" and _MAY_BEFORE.search(
        text, max(0, word.start - 12), word.start
    ):
        return True  # "od 1 maja", "do końca maja": the month, since Maja's genitive is "Mai"
    before = max(0, word.start - 8)
    if (
        _YEAR_AFTER.match(text, word.end) is not None
        and _DAY_BEFORE.search(text, before, word.start) is not None
    ):
        return True
    if _SEPARATED_DAY_AND_YEAR_AFTER.match(text, word.end) is not None:
        return True
    for before_pattern, after_pattern in (
        (_SEPARATED_DAY_BEFORE, _SEPARATED_YEAR_AFTER),
        (_SEPARATED_YEAR_BEFORE, _SEPARATED_DAY_AFTER),
    ):
        preceding = before_pattern.search(text, before, word.start)
        following = after_pattern.match(text, word.end)
        if (
            preceding is not None
            and following is not None
            and preceding["separator"] == following["separator"]
        ):
            return True
    return False


def _labelled(text: str, words: list[_Word], index: int) -> bool:
    """Whether a word fills a labelled field ("Uczestnik: Nowak"), not an e-mail subject ("Re:")."""
    label = index - 1
    return (
        label >= 0
        and words[label].key not in _SUBJECT_PREFIXES
        and _LABEL_GAP.fullmatch(text, words[label].end, words[index].start) is not None
    )


def _after_initial(text: str, words: list[_Word], index: int) -> int | None:
    """Return the word after the initial at `index`, joined to it by the full stop."""
    following = index + 1
    if following >= len(words):
        return None
    gap = _AFTER_INITIAL.fullmatch(text, words[index].end, words[following].start)
    return following if gap is not None else None


def _initials_and_surname(text: str, words: list[_Word], index: int) -> tuple[int, int] | None:
    """A signature such as "J. Kowalski" or "A.M. Nowak": one or two initials, then a surname.

    The initial vouches for a person the way a first name does, so the surname only needs to read
    as one: a telling ending, or a weak one written as a name. A longer run of capitals with full
    stops is an abbreviation such as "P.P.H.U.", and its letters are not initials.
    """
    previous = index - 1
    if (
        previous >= 0
        and _is_initial(text, words[previous])
        and _after_initial(text, words, previous) is not None
    ):
        return None  # inside a run of initials that started earlier
    candidate = _after_initial(text, words, index)
    if candidate is not None and _is_initial(text, words[candidate]):
        candidate = _after_initial(text, words, candidate)
    if candidate is None:
        return None
    word = words[candidate]
    if _is_initial(text, word) or _is_title(word):
        return None
    surname = surname_reading(word.text)
    if surname is None or not (surname.strong or _name_shaped(word.text)):
        return None
    return index, candidate


def _person_at(
    text: str, words: list[_Word], index: int, taken: int
) -> tuple[int, int, int | None, bool] | None:
    """Return the first and last word index of a person found at `index`, or None.

    The third index is a surname in the next cell of a table or a CSV line, masked apart, and the
    flag says whether the person is a first name alone; see :data:`LONE_NAME_SCORE`.

    Args:
        text: the whole text.
        words: its words, in order.
        index: the word to start from.
        taken: the first word not yet inside an earlier span; a surname in front of the first name
            is only taken from there on.
    """
    word = words[index]
    found: tuple[int, int] | None
    if _is_initial(text, word):
        found = _initials_and_surname(text, words, index)
        return None if found is None else (*found, None, False)
    address = _address_before(text, words, index)
    readings = NAME_FORMS.get(word.key) if "-" not in word.key else None
    if readings is not None:
        if _in_a_date(text, word):
            return None
        return _around_first_name(text, words, (index, taken), readings, address)
    if address is None or _is_title(word):
        return None
    found = _addressed_surname(text, words, index, address)
    return None if found is None else (*found, None, False)


def _around_first_name(
    text: str,
    words: list[_Word],
    position: tuple[int, int],
    readings: frozenset[Tag],
    address: frozenset[Tag] | None,
) -> tuple[int, int, int | None, bool] | None:
    """Extend a first name to a second first name and a surname that agree with it.

    The surname may also stand on the next line, or in the next cell of a table or a CSV line,
    returned as the third index; see :func:`_surname_across`. The flag is set for a first name
    with nothing next to it: no second first name, surname or form of address.
    """
    index = position[0]
    ambiguous = words[index].key in AMBIGUOUS_FORMS
    trusted = address is not None or not ambiguous
    tags = (readings & address or readings) if address is not None else readings
    first = last = index
    cell = None

    following = _neighbour(text, words, last, 1)
    if following is not None:
        second = NAME_FORMS.get(words[following].key)
        if second and second & tags and words[following].key not in AMBIGUOUS_FORMS:
            last, tags = following, second & tags  # "Anna Maria", "Jan Paweł"

    strong = False
    following = _neighbour(text, words, last, 1)
    if following is not None:
        candidate = words[following]
        surname = surname_reading(candidate.text)
        if surname is not None and _agrees(surname.tags, tags):
            last = following
            # "Marek Zając": a weak surname still settles it when both words are written as names.
            strong = surname.strong or (
                _name_shaped(words[index].text) and _name_shaped(candidate.text)
            )
        elif (
            trusted
            and not _is_title(candidate)
            and _written_as_a_name(text, words, following, words[last], bool(tags & _NOMINATIVE))
        ):
            last = following
    else:
        across = _surname_across(text, words, last, tags, words[index])
        if across is not None:
            surname_index, on_the_next_line = across
            strong = True
            if on_the_next_line:
                last = surname_index
            else:
                cell = surname_index

    if last == index and cell is None and address is None:
        before = _surname_before(text, words, position, tags, trusted)
        if before is not None:
            first, settled = before
            strong = strong or settled

    if not trusted and not strong:
        return None
    alone = first == last == index and cell is None and address is None
    return first, last, cell, alone


def _surname_across(
    text: str, words: list[_Word], index: int, tags: frozenset[Tag], name: _Word
) -> tuple[int, bool] | None:
    r"""A surname on the next line of a hard-wrapped e-mail, or in the next cell of a table.

    "Janem\nKowalskim", "Jan;Kowalski", "| Jan | Kowalski |". Across a line or a cell the next
    word is as often the next field, a town, a department or a label, so a capitalised word is
    not enough here: the word must read as a surname that agrees with the first name, and a weak
    reading counts only with both words written as names. A first name or a title there is the
    next field: "Jan;Anna", "Imię: Jan\nNazwisko: ...". On the next line the surname must also
    end the name there: under a signature, "Kamil\nNowak Transport" is a company.

    Returns:
        The surname's index and whether a line break, rather than a cell, stands before it.
    """
    following = index + 1
    if following >= len(words):
        return None
    start, end = words[index].end, words[following].start
    on_the_next_line = _LINE_BREAK.fullmatch(text, start, end) is not None
    if not on_the_next_line and _CELL_BREAK.fullmatch(text, start, end) is None:
        return None
    candidate = words[following]
    if _is_title(candidate) or candidate.key in NAME_FORMS:
        return None
    surname = surname_reading(candidate.text)
    if surname is None or not _agrees(surname.tags, tags):
        return None
    strong = surname.strong and not surname.common
    if not (strong or (_name_shaped(name.text) and _name_shaped(candidate.text))):
        return None
    if on_the_next_line:
        after = _neighbour(text, words, following, 1)
        if after is not None and _name_shaped(words[after].text):
            return None
    return following, on_the_next_line


def _surname_before(
    text: str,
    words: list[_Word],
    position: tuple[int, int],
    tags: frozenset[Tag],
    trusted: bool,
) -> tuple[int, bool] | None:
    """A surname written before the first name, the order of a list or a form: "kowalski jan".

    The rules mirror the ones for a surname after the first name ("Wójcik Marek" as "Marek
    Wójcik"). A word with no surname ending is taken only in a labelled field ("Uczestnik: Konieczny
    Jan, ur. 1990"), where the first name is in the nominative and ends the field; at the start of
    a sentence the same word would be an adverb ("Wczoraj Jan dzwonił").

    Returns:
        The index of the surname and whether it settles an ambiguous first name, or None.
    """
    index, taken = position
    preceding = _neighbour(text, words, index, -1)
    if preceding is None or preceding < taken:
        return None
    candidate, name = words[preceding], words[index]
    surname = surname_reading(candidate.text)
    if surname is not None:
        if not surname.tags & tags:
            return None
        strong = surname.strong or (_name_shaped(candidate.text) and _name_shaped(name.text))
        return (preceding, strong) if strong or trusted else None
    in_a_field = (
        trusted
        and bool(tags & _NOMINATIVE)
        and _name_shaped(candidate.text)
        and _name_shaped(name.text)
        and not _is_title(candidate)
        and _labelled(text, words, preceding)
        and _FIELD_END.match(text, name.end) is not None
    )
    return (preceding, False) if in_a_field else None


def _addressed_surname(
    text: str, words: list[_Word], index: int, address: frozenset[Tag]
) -> tuple[int, int] | None:
    """A surname standing alone after a form of address: "z panem bednarkiem", "pani Wróbel"."""
    word = words[index]
    surname = surname_reading(word.text)
    if surname is not None and _agrees(surname.tags, address):
        return index, index
    nominative = bool(address & _NOMINATIVE)
    if not _written_as_a_name(text, words, index, words[index - 1], nominative):
        return None
    last = index
    following = _neighbour(text, words, index, 1)
    if following is not None:
        candidate = words[following]
        if not _is_title(candidate) and _written_as_a_name(
            text, words, following, word, nominative
        ):
            last = following
    return index, last
