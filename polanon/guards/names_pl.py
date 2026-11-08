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
surname with a telling ending. The declension rules were validated against the hunspell pl_PL word
list, and the list of such collisions was drawn from it.

Standard library only, so the guard also runs in the configuration without a language model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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


def _words(block: str) -> list[str]:
    """Split a whitespace-separated block, which keeps long word lists readable."""
    return block.split()


_MALE_NAMES = _words(
    """
    Adam Adrian Alan Albert Aleks Aleksander Alfred Andrzej Antoni Arkadiusz Artur August
    Bartłomiej Bartosz Benedykt Bernard Błażej Bogdan Bogumił Bogusław Bolesław Borys Bronisław
    Cezary Czesław Damian Daniel Dariusz Dawid Dominik Edmund Edward Emanuel Emil Ernest Eryk
    Eugeniusz Fabian Feliks Filip Florian Franciszek Fryderyk Gabriel Gerard Gracjan Grzegorz
    Gustaw Henryk Hieronim Hubert Ignacy Igor Ireneusz Jacek Jakub Jan Janusz Jarosław Jędrzej
    Jerzy Joachim Józef Julian Juliusz Kacper Kajetan Kamil Karol Kazimierz Klaudiusz Konrad
    Konstanty Kornel Krystian Krzysztof Ksawery Lech Leon Leonard Leszek Longin Lucjan Ludwik
    Łukasz Maciej Maks Maksymilian Marcel Marcin Marek Marian Mariusz Mateusz Michał Mieczysław
    Mikołaj Miłosz Mirosław Natan Nataniel Nikodem Norbert Olaf Olgierd Oliwer Oskar Patryk Paweł
    Piotr Przemysław Radosław Rafał Remigiusz Robert Roman Ryszard Sebastian Seweryn Sławomir
    Stanisław Stefan Szczepan Szymon Tadeusz Teodor Tobiasz Tomasz Tymon Tymoteusz Wacław Waldemar
    Walenty Wiesław Wiktor Wincenty Witold Władysław Włodzimierz Wojciech Zbigniew Zdzisław Zenon
    Zygmunt
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
    Bożena Bronisława Cecylia Czesława Dagmara Danuta Daria Diana Dominika Dorota Edyta Eliza
    Elwira Elżbieta Emilia Emma Ewa Ewelina Felicja Gabriela Genowefa Grażyna Halina Hanna Helena
    Honorata Iga Ilona Irena Iwona Izabela Jadwiga Jagoda Janina Joanna Jolanta Józefa Judyta
    Julia Julita Justyna Kaja Kalina Kamila Karina Karolina Katarzyna Kinga Klara Klaudia Kornelia
    Krystyna Laura Lena Leokadia Lidia Liliana Liwia Lucyna Ludmiła Łucja Magdalena Maja Malwina
    Małgorzata Marcelina Maria Marianna Mariola Marta Martyna Marzena Michalina Milena Mirosława
    Monika Nadia Natalia Nela Nikola Nina Olga Oliwia Patrycja Paula Paulina Pola Regina Renata
    Roksana Rozalia Róża Sabina Sandra Sara Sonia Stanisława Stefania Sylwia Tamara Teresa Urszula
    Wanda Weronika Wiesława Wiktoria Wioletta Władysława Zdzisława Zofia Zuzanna Żaneta
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
)

#: Forms of first names that are also ordinary Polish words or very common English ones, folded.
AMBIGUOUS_FORMS = frozenset(
    _words(
        """
        age ago ani antonim asie august basie czeska czesku franka frankiem franku jagoda jagode
        jagody jagodzie kaja kalina kaline kalinie kaliny kasce kasie kaska kaske kaski kasko
        konstanty laury laurze maja majce majka majke majki maks maksa maksem maksie marek marka
        marzenie mikolaj mikolaja nine oliwce oliwie oliwka oliwke oliwki pola pole polo roza roze
        rozy ryska rysku ula ule ulu wiktoria witka
        """
    )
)

_SURNAME_FORMS: dict[str, frozenset[Tag]] = _index(
    [("m", decline_masculine(name.lower())) for name in _NOUN_SURNAMES]
    + [("f", {name.lower(): set(CASES)}) for name in _NOUN_SURNAMES]
    + [(gender, decline_feminine(name.lower())) for name in _A_SURNAMES for gender in "mf"]
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

#: A Roman numeral, which follows a name in capitals without being one: "Jana Pawła II".
_ROMAN = re.compile(r"[IVXLCDM]+")

#: What ends a field in a form or a list: a comma, a semicolon or the end of the line.
_FIELD_END = re.compile(r"[ \t\u00a0]*(?:[,;]|\r?\n|$)")

#: A field label's colon, and the words before a colon that open an e-mail subject instead.
_LABEL_GAP = re.compile(r":[ \t\u00a0]*")
_SUBJECT_PREFIXES = frozenset(_words("dotyczy fw fwd odp pd re temat"))

#: Space-based dates retain their original rules: "15 Jan 2024", "Jan 15, 2024".
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"
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


@dataclass(frozen=True, slots=True)
class _Word:
    start: int
    end: int
    text: str
    key: str


@dataclass(frozen=True, slots=True)
class SurnameReading:
    """What a word could be as a surname, and whether its ending alone makes that likely."""

    tags: frozenset[Tag]
    strong: bool


def surname_reading(word: str) -> SurnameReading | None:
    """Return the grammatical readings a word has as a Polish surname, or None.

    An adjectival ending (-ski, -cka, -dzkiego), -wicz, or a declined noun ending (Nowakiem,
    Bednarka) is a strong sign. A bare -ak, -ek, -ik and the listed surnames are weak ones, because
    plenty of ordinary nouns look the same.

    A double surname, "Kowalska-Nowak", is as strong as its stronger part and takes the cases its
    parts agree on. Read by its last part alone, a strong first part was ignored, and "mają
    kowalską-nowak" went out whole where "mają kowalską" was masked.
    """
    parts = fold(word).split("-")
    if parts[-1] in _NOT_NAMES:
        return None
    readings = [
        reading
        for reading in (_part_reading(part) for part in parts if part not in _NOT_NAMES)
        if reading is not None
    ]
    if not readings:
        return None
    agreed = frozenset.intersection(*(reading.tags for reading in readings))
    return SurnameReading(
        agreed or readings[-1].tags, strong=any(reading.strong for reading in readings)
    )


def _part_reading(part: str) -> SurnameReading | None:
    """Return the readings of one folded word, or one part of a double surname, as a surname."""
    listed = _SURNAME_FORMS.get(part)
    if listed is not None:
        return SurnameReading(listed, strong=False)
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
        return SurnameReading(_NOMINATIVE_TAGS, strong=part.endswith("wicz"))
    return None


class PersonNameGuard(Guard):
    """Masks Polish personal names regardless of letter case, as ``PERSON``."""

    name = "names_pl"

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return one span per person: first names with their surname, or an addressed surname."""
        if "PERSON" not in enabled:
            return []
        words = [
            _Word(match.start(), match.end(), match.group(), fold(match.group()))
            for match in _WORD.finditer(text)
        ]
        spans: list[Span] = []
        index = 0
        taken = 0
        while index < len(words):
            found = _person_at(text, words, index, taken)
            if found is None:
                index += 1
                continue
            first, last = found
            spans.append(Span(words[first].start, words[last].end, "PERSON", NAME_SCORE, self.name))
            index = taken = last + 1
        return spans


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


def _person_at(text: str, words: list[_Word], index: int, taken: int) -> tuple[int, int] | None:
    """Return the first and last word index of a person found at `index`, or None.

    Args:
        text: the whole text.
        words: its words, in order.
        index: the word to start from.
        taken: the first word not yet inside an earlier span; a surname in front of the first name
            is only taken from there on.
    """
    word = words[index]
    if _is_initial(text, word):
        return _initials_and_surname(text, words, index)
    address = _address_before(text, words, index)
    readings = NAME_FORMS.get(word.key) if "-" not in word.key else None
    if readings is not None:
        if _in_a_date(text, word):
            return None
        return _around_first_name(text, words, (index, taken), readings, address)
    if address is None or _is_title(word):
        return None
    return _addressed_surname(text, words, index, address)


def _around_first_name(
    text: str,
    words: list[_Word],
    position: tuple[int, int],
    readings: frozenset[Tag],
    address: frozenset[Tag] | None,
) -> tuple[int, int] | None:
    """Extend a first name to a second first name and a surname that agree with it."""
    index = position[0]
    ambiguous = words[index].key in AMBIGUOUS_FORMS
    trusted = address is not None or not ambiguous
    tags = (readings & address or readings) if address is not None else readings
    first = last = index

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

    if last == index and address is None:
        before = _surname_before(text, words, position, tags, trusted)
        if before is not None:
            first, settled = before
            strong = strong or settled

    if not trusted and not strong:
        return None
    return first, last


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
