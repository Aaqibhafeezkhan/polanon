"""The allowlist leaves public names unmasked and never releases anything else."""

import unicodedata

import pytest

from polanon import (
    WELL_KNOWN_NAMES,
    Allowlist,
    Anonymizer,
    DenylistGuard,
    MockJudge,
    RegexGuard,
    default_guards,
)
from polanon.guards.base import Guard
from polanon.pipeline import anonymize
from polanon.policy import ENTITY_TYPES
from polanon.types import Span

ALL = set(ENTITY_TYPES)
PLATFORMS = Allowlist(WELL_KNOWN_NAMES)


class WordGuard(Guard):
    """Flags fixed words the way the language model flags names."""

    name = "spacy_ner"

    def __init__(self, entity_type: str, *words: str) -> None:
        self.entity_type = entity_type
        self.words = words

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        if self.entity_type not in enabled:
            return []
        return [
            Span(start, start + len(word), self.entity_type, 0.85, self.name)
            for word in self.words
            if (start := text.find(word)) != -1
        ]


@pytest.mark.parametrize(
    "fragment",
    [
        "Facebook",
        "Facebooka",
        "Facebookiem",
        "LINKEDIN",
        "Canvie",
        "YouTubie",
        "YouTube'a",
        "ChatGPT-a",
        "Microsofcie",
        "Pinterestem",
        "Google Ads",
        "Meta Business Suite",
        "Facebooku i Instagramie",
    ],
)
def test_a_platform_is_allowed_in_any_letter_case_and_case_form(fragment):
    assert PLATFORMS.allows(fragment)


@pytest.mark.parametrize(
    "fragment", ["Jan Kowalski z Facebooka", "Facebook Jan", "Metalex", "Teams", "i", ""]
)
def test_a_detection_holding_more_than_allowed_names_is_not_allowed(fragment):
    assert not PLATFORMS.allows(fragment)


def test_blank_and_one_letter_names_allow_nothing():
    allowlist = Allowlist(["", "   ", "X"])
    assert not allowlist
    assert not allowlist.allows("X")


def test_a_short_or_digit_ending_name_matches_only_as_written():
    """Declension is only attempted on words it can apply to; "3M" or "Web3" are taken literally."""
    allowlist = Allowlist(["3M", "Web3"])
    assert allowlist.allows("3M")
    assert allowlist.allows("Web3")
    assert not allowlist.allows("Web3a")


def test_an_empty_allowlist_returns_every_span_untouched():
    spans = [Span(0, 8, "ORG", 0.85, "spacy_ner")]
    assert Allowlist([]).drop_allowed(spans, "Facebook") is spans


def test_an_identifier_is_never_exempt_whatever_text_it_covers():
    allowlist = Allowlist(["kontakt@facebook.example"])
    assert allowlist.exempts("ORG", "kontakt@facebook.example")
    assert not allowlist.exempts("EMAIL", "kontakt@facebook.example")


def test_a_denylisted_word_is_never_exempt():
    assert not PLATFORMS.exempts("ORG", "Facebook", source="denylist")


def test_a_person_the_name_guard_found_is_never_exempt():
    """The name guard needs a first name or a form of address, so its people are people."""
    allowlist = Allowlist(["Ziaja"])
    assert not allowlist.exempts("PERSON", "Ziaja", source="names_pl")
    assert allowlist.exempts("PERSON", "Ziaja", source="spacy_ner")
    assert allowlist.exempts("ORG", "Ziaja", source="names_pl")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Reklamację złożyła pani Ziaja, tel. w systemie.",
            "Reklamację złożyła pani <OSOBA_1>, tel. w systemie.",
        ),
        ("Umów mnie z panem Krukiem na jutro.", "Umów mnie z panem <OSOBA_1> na jutro."),
        ("Dzwonił pan Inglot w sprawie zwrotu.", "Dzwonił pan <OSOBA_1> w sprawie zwrotu."),
    ],
)
async def test_an_allowed_brand_stays_masked_when_a_form_of_address_names_a_person(text, expected):
    shop = Allowlist([*WELL_KNOWN_NAMES, "Ziaja", "Kruk", "Inglot"])
    anonymizer = Anonymizer(default_guards(with_ner=False), allowlist=shop)
    assert (await anonymizer.anonymize(text)).anonymized == expected


async def test_a_platform_the_model_flags_goes_out_as_written():
    text = "Zrób post na Facebooku o Janie Kowalskim, PESEL 50810112354."
    result = await anonymize(
        text,
        enabled=ALL,
        guards=[
            RegexGuard(),
            WordGuard("ORG", "Facebooku"),
            WordGuard("PERSON", "Janie Kowalskim"),
        ],
        judge=None,
        existing_mapping={},
        allowlist=PLATFORMS,
    )
    assert result.anonymized == "Zrób post na Facebooku o <OSOBA_1>, PESEL <PESEL_1>."


async def test_without_an_allowlist_the_platform_is_masked():
    result = await anonymize(
        "post na Facebooku",
        enabled=ALL,
        guards=[WordGuard("ORG", "Facebooku")],
        judge=None,
        existing_mapping={},
    )
    assert result.anonymized == "post na <FIRMA_1>"


async def test_a_word_somebody_asked_to_mask_stays_masked():
    result = await anonymize(
        "post na Facebooku",
        enabled=ALL,
        guards=[DenylistGuard(["Facebooku"]), WordGuard("ORG", "Facebooku")],
        judge=None,
        existing_mapping={},
        allowlist=PLATFORMS,
    )
    assert result.anonymized == "post na <FIRMA_1>"


async def test_a_judge_finding_of_an_allowed_name_is_ignored():
    result = await anonymize(
        "grafiki w Canvie robi szef marketingu",
        enabled=ALL,
        guards=[],
        judge=MockJudge({"Canvie": "ORG", "szef marketingu": "OTHER"}),
        existing_mapping={},
        allowlist=PLATFORMS,
    )
    assert result.anonymized == "grafiki w Canvie robi <INNE_1>"
    assert result.judge_hits == 1


async def test_a_value_already_masked_in_the_conversation_stays_masked():
    """The final check refuses a known value in the clear, so the allowlist must not release it."""
    result = await anonymize(
        "post na Facebook",
        enabled=ALL,
        guards=[],
        judge=None,
        existing_mapping={"<FIRMA_1>": "Facebook"},
        allowlist=PLATFORMS,
    )
    assert result.anonymized == "post na <FIRMA_1>"


async def test_the_facade_hands_the_allowlist_to_the_pipeline_and_the_sweep():
    anonymizer = Anonymizer([WordGuard("ORG", "Canvie")], allowlist=PLATFORMS)
    assert (await anonymizer.anonymize("grafiki w Canvie")).anonymized == "grafiki w Canvie"
    assert anonymizer.sweep("Zrobiłem to w Canvie.", {}).is_clean
    assert not Anonymizer([WordGuard("ORG", "Canvie")]).sweep("To w Canvie.", {}).is_clean


def _model_available() -> bool:
    try:
        import spacy

        spacy.load("pl_core_news_md")
    except (ImportError, OSError):
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not _model_available(), reason="pl_core_news_md is not installed")
async def test_the_polish_model_leaves_a_marketing_teams_tools_alone():
    text = (
        "Zrób post na Facebooka i LinkedIna, a kampanię puść w Google Ads. "
        "Grafiki robimy w Canvie, raport jest z Semrush, rolka wisi na TikToku."
    )
    anonymizer = Anonymizer(default_guards(), allowlist=PLATFORMS)
    assert (await anonymizer.anonymize(text)).anonymized == text


@pytest.mark.parametrize(
    "fragment",
    [
        "ZUS",
        "ZUS-ie",
        "ZUS-u",
        "PUE ZUS",
        "Płatnik",
        "Płatniku",
        "Płatnika",
        "KRUS",
        "NFZ",
        "GUS",
        "KSeF",
        "KSeF-ie",
        "ePUAP",
        "CEIDG",
        "PFRON",
    ],
)
def test_an_institution_or_a_program_people_file_through_is_allowed(fragment):
    """ "Jak to rozliczyć w ZUS?" went out as "<FIRMA_1>", and so did the ZUS program Płatnik."""
    assert PLATFORMS.allows(fragment)


@pytest.mark.slow
@pytest.mark.skipif(not _model_available(), reason="pl_core_news_md is not installed")
@pytest.mark.parametrize(
    "text",
    [
        "Jak to rozliczyć w ZUS?",
        "Księgowa zna program Płatnik.",
        "Wyślij to przez PUE ZUS, a kopię zgłoś do NFZ i KRUS.",
        "Fakturę wystaw w KSeF, dane sprawdź w CEIDG.",
    ],
)
async def test_the_polish_model_leaves_institutions_and_their_programs_alone(text):
    anonymizer = Anonymizer(default_guards(), allowlist=PLATFORMS)
    assert (await anonymizer.anonymize(text)).anonymized == text


async def test_a_person_named_like_an_allowed_program_stays_masked():
    anonymizer = Anonymizer(default_guards(with_ner=False), allowlist=PLATFORMS)
    result = await anonymizer.anonymize("Dzwoniła pani Płatnik w sprawie faktury.")
    assert result.anonymized == "Dzwoniła pani <OSOBA_1> w sprawie faktury."


@pytest.mark.parametrize("fragment", ["Wixie", "Wixem", "OLXie"])
def test_three_letter_platform_is_allowed_in_its_case_forms(fragment):
    """ "strona na Wixie", "ogłoszenie na OLXie": a short brand declines like a long one."""
    assert PLATFORMS.allows(fragment)


def test_allowed_name_made_of_connector_words_is_allowed():
    assert Allowlist(["And Or"]).allows("And Or")


@pytest.mark.parametrize(
    "fragment",
    [
        "Face" + chr(0x200B) + "booku",
        "Face" + chr(0x00AD) + "booku",
        "".join(chr(ord(char) + 0xFEE0) for char in "Facebook"),
    ],
)
def test_platform_written_with_invisible_or_fullwidth_characters_is_allowed(fragment):
    """Copied from a web page, the name still reads "Facebooku", and the guards read it so."""
    assert PLATFORMS.allows(fragment)


def test_name_with_polish_letters_in_decomposed_form_is_allowed():
    assert Allowlist(["Żabka"]).allows(unicodedata.normalize("NFD", "Żabka"))
