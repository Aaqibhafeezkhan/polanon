"""What a text has to survive on its way through the pipeline: odd characters, guards and judges.

These hold the promises the README makes — every occurrence masked, one token per value, tokens
numbered in reading order, restore lossless — against input the guards were not written for.
Positions in the stand-in guard below are counted by hand, so each test also documents its text.
"""

import random
import re
import unicodedata

import pytest

from polanon import (
    Anonymizer,
    DenylistGuard,
    PersonNameGuard,
    RegexGuard,
    SecretsGuard,
)
from polanon._matching import occurrences
from polanon.guards.base import Guard
from polanon.judge.base import Judge, JudgeError, JudgeFinding
from polanon.pipeline import anonymize, mask_known_values
from polanon.policy import ENTITY_TYPES
from polanon.types import JudgeFallback, Span

ALL = set(ENTITY_TYPES)
NBSP = chr(0x00A0)
NARROW_NBSP = chr(0x202F)
ZERO_WIDTH_SPACE = chr(0x200B)
SOFT_HYPHEN = chr(0x00AD)
EN_DASH = chr(0x2013)
IBAN = "PL61 1090 1014 0000 0712 1981 2874"


class FixedGuard(Guard):
    """Returns the spans it was given whatever the text: a stand-in for a third-party guard."""

    name = "fixed"

    def __init__(self, *spans: Span) -> None:
        self.spans = list(spans)

    def detect(self, text, enabled):
        return [span for span in self.spans if span.entity_type in enabled]


class SpyJudge(Judge):
    """Records what it was shown and answers with scripted findings, or fails as told."""

    name = "spy"

    def __init__(self, findings=(), error=None):
        self.findings = list(findings)
        self.error = error
        self.seen = []

    async def review(self, masked_text):
        self.seen.append(masked_text)
        if self.error is not None:
            raise self.error
        return [JudgeFinding(text, entity_type) for text, entity_type in self.findings]


async def mask(
    text, guards=None, *, judge=None, mapping=None, enabled=ALL, fallback=JudgeFallback.SEND_STAGE1
):
    return await anonymize(
        text,
        enabled=enabled,
        guards=[RegexGuard(), PersonNameGuard()] if guards is None else guards,
        judge=judge,
        existing_mapping=mapping or {},
        judge_fallback=fallback,
    )


def is_lossless(result):
    return Anonymizer.restore(result.anonymized, result.mapping) == result.original


@pytest.mark.parametrize("separator", [NBSP, NARROW_NBSP])
async def test_an_iban_with_typographic_spaces_is_masked_whole(separator):
    """Word, PDFs and bank pages keep the groups of an IBAN together with a no-break space."""
    written = IBAN.replace(" ", separator)
    result = await mask(f"Konto {written} do zapłaty.")
    assert result.anonymized == "Konto <IBAN_1> do zapłaty."
    assert result.mapping == {"<IBAN_1>": written}
    assert is_lossless(result)


async def test_a_nip_with_en_dashes_is_masked():
    result = await mask(f"NIP 526{EN_DASH}000{EN_DASH}00{EN_DASH}05 w fakturze")
    assert result.anonymized == "NIP <NIP_1> w fakturze"
    assert is_lossless(result)


async def test_a_phone_with_no_break_spaces_is_masked():
    result = await mask(f"Zadzwoń 601{NBSP}234{NBSP}567 po 17.")
    assert result.anonymized == "Zadzwoń <TELEFON_1> po 17."


@pytest.mark.parametrize("invisible", [ZERO_WIDTH_SPACE, SOFT_HYPHEN])
async def test_an_invisible_character_inside_a_pesel_does_not_hide_it(invisible):
    result = await mask(f"PESEL 900101{invisible}12349")
    assert result.anonymized == "PESEL <PESEL_1>"
    assert is_lossless(result)


async def test_a_name_in_decomposed_unicode_is_masked_whole():
    """Text from macOS and many PDFs spells "ó" as "o" followed by a combining accent."""
    text = unicodedata.normalize("NFD", "Pani Małgorzata Wójcik zgłosiła reklamację.")
    result = await mask(text)
    assert result.anonymized == unicodedata.normalize("NFD", "Pani <OSOBA_1> zgłosiła reklamację.")
    assert result.mapping == {"<OSOBA_1>": unicodedata.normalize("NFD", "Małgorzata Wójcik")}
    assert is_lossless(result)


def fullwidth(text):
    return "".join(chr(ord(character) + 0xFEE0) if character != " " else " " for character in text)


@pytest.mark.parametrize(
    ("written", "token"),
    [("jan.kowalski@example.com", "<EMAIL_1>"), ("PL61109010140000071219812874", "<IBAN_1>")],
)
async def test_fullwidth_letters_are_read_as_letters(written, token):
    """Digits alone would pass anyway: regex and int() already read fullwidth digits."""
    result = await mask(f"Dane: {fullwidth(written)}")
    assert result.anonymized == f"Dane: {token}"
    assert is_lossless(result)


@pytest.mark.parametrize(
    "span", [Span(-3, 5, "PERSON"), Span(2, 999, "PERSON"), Span(5, 2, "PERSON")]
)
async def test_a_guard_span_outside_the_text_is_refused(span):
    with pytest.raises(ValueError, match="'fixed'"):
        await mask("Klient EMP-11111 dzwonił.", [FixedGuard(span)])


async def test_an_empty_guard_span_is_skipped_even_in_a_text_read_through_the_view():
    """Mapped back through the view, an empty span at 0 would have covered the whole first word."""
    text = f"Klient{NBSP}dzwonił."
    result = await mask(text, [FixedGuard(Span(0, 0, "PERSON"))])
    assert result.anonymized == text


async def test_a_custom_type_that_cannot_be_a_token_label_is_refused():
    with pytest.raises(ValueError, match="EMPLOYEE_ID"):
        await mask(
            "Klient EMP-11111.", [FixedGuard(Span(7, 16, "employee_id"))], enabled={"employee_id"}
        )


async def test_a_custom_type_is_masked_at_every_occurrence_and_restored():
    text = "Pracownik EMP-11111 zgłosił. Potwierdzam EMP-11111."
    guard = FixedGuard(Span(10, 19, "EMPLOYEE_ID"))  # the guard only sees the first one
    result = await mask(text, [guard], enabled={"EMPLOYEE_ID"})
    assert result.anonymized == "Pracownik <EMPLOYEE_ID_1> zgłosił. Potwierdzam <EMPLOYEE_ID_1>."
    assert is_lossless(result)


async def test_a_token_already_in_the_text_is_never_handed_out_again():
    """A template's literal placeholder must not become the real PESEL on restore."""
    text = "W szablonie mamy pole <PESEL_1>, a prawdziwy PESEL to 50810112354."
    result = await mask(text)
    assert result.anonymized == "W szablonie mamy pole <PESEL_1>, a prawdziwy PESEL to <PESEL_2>."
    assert result.mapping == {"<PESEL_1>": "<PESEL_1>", "<PESEL_2>": "50810112354"}
    assert is_lossless(result)


async def test_a_token_written_in_an_earlier_turn_is_never_handed_out_later():
    """The placeholder stands for itself in the mapping, so the next turn still steps around it."""
    first = await mask("W szablonie umowy jest pole <PESEL_1> i pole <OSOBA_1>, wypełnij je.")
    second = await mask("Klient Jan Kowalski, PESEL 50810112354.", mapping=first.mapping)
    assert second.anonymized == "Klient <OSOBA_2>, PESEL <PESEL_2>."
    answer = "W polu <PESEL_1> wpisz <PESEL_2>, a pole <OSOBA_1> zostaw puste."
    assert Anonymizer.restore(answer, second.mapping) == (
        "W polu <PESEL_1> wpisz 50810112354, a pole <OSOBA_1> zostaw puste."
    )


async def test_one_value_read_as_two_types_gets_one_token_of_the_surer_type():
    text = "Samochód WA 12345 stoi. Nr rej. WA 12345 zgłoszony."
    guard = FixedGuard(Span(9, 17, "POSTAL_CODE", 0.8), Span(32, 40, "PLATE", 1.0))
    result = await mask(text, [guard])
    assert result.anonymized == "Samochód <REJESTRACJA_1> stoi. Nr rej. <REJESTRACJA_1> zgłoszony."
    assert result.entity_counts == {"PLATE": 2}


async def test_tokens_follow_reading_order_when_a_value_is_recognised_only_later():
    text = "Spółka ABC-1 przejęła DEF-2. Dane: ABC-1."
    guard = FixedGuard(Span(22, 27, "ORG"), Span(35, 40, "ORG"))  # not the first ABC-1
    result = await mask(text, [guard])
    assert result.anonymized == "Spółka <FIRMA_1> przejęła <FIRMA_2>. Dane: <FIRMA_1>."


async def test_a_piece_left_over_from_a_losing_detection_is_not_masked_elsewhere():
    """An address rule once ran from an e-mail domain into the next line and left "NIP" behind;
    every "NIP" in the document then reached the model as an address."""
    text = "Kontakt: jan@example.com\nNIP 526-000-00-05\nProszę podać NIP nabywcy."
    guard = FixedGuard(
        Span(9, 24, "EMAIL"), Span(29, 42, "NIP"), Span(22, 32, "ADDRESS", 0.8, "fixed")
    )
    result = await mask(text, [guard])
    assert result.anonymized == "Kontakt: <EMAIL_1>\n<ADRES_1> <NIP_1>\nProszę podać NIP nabywcy."


async def test_the_judge_reviews_only_the_masked_text():
    judge = SpyJudge()
    result = await mask(f"Jan Kowalski, PESEL 50810112354, konto {IBAN}", judge=judge)
    assert judge.seen == [result.anonymized]
    assert not any(raw in judge.seen[0] for raw in ("Kowalski", "50810112354", "1090"))


async def test_a_short_judge_finding_is_never_replaced_inside_a_longer_word():
    judge = SpyJudge([("Kos", "PERSON")])
    result = await mask("Kosztorys przygotował Kos z działu zakupów.", [], judge=judge)
    assert result.anonymized == "Kosztorys przygotował <OSOBA_1> z działu zakupów."
    assert result.judge_hits == 1


async def test_a_long_judge_finding_inside_a_word_masks_the_whole_word_under_its_own_token():
    judge = SpyJudge([("Nowak", "PERSON")])
    result = await mask("Nowak przyjdzie. Spotkanie z Nowakowską w piątek.", [], judge=judge)
    assert result.anonymized == "<OSOBA_1> przyjdzie. Spotkanie z <OSOBA_2> w piątek."
    assert result.mapping == {"<OSOBA_1>": "Nowak", "<OSOBA_2>": "Nowakowską"}


async def test_a_judge_finding_that_quotes_or_cuts_a_token_allocates_nothing():
    judge = SpyJudge([("PESEL <PESEL_1> z Łodzi", "PERSON"), ("PESEL_1", "OTHER")])
    result = await mask("Klient PESEL 50810112354 z Łodzi.", judge=judge)
    assert result.anonymized == "Klient PESEL <PESEL_1> z Łodzi."
    assert result.mapping == {"<PESEL_1>": "50810112354"}
    assert result.judge_hits == 0


async def test_judge_spans_record_exactly_what_was_replaced():
    # Another surname in the second sentence: a lone "Nowak" is masked by stage 1 already, as the
    # surname of the person it found.
    text = "Anna Nowak pracuje w Radomiu. Kowal pracuje też w Kielcach."
    judge = SpyJudge([("Kowal pracuje", "PERSON")])
    result = await mask(text, [FixedGuard(Span(0, 10, "PERSON"))], judge=judge)
    assert result.anonymized == "<OSOBA_1> pracuje w Radomiu. <OSOBA_2> też w Kielcach."
    token_of = {value: token for token, value in result.mapping.items()}
    rebuilt, cursor = [], 0
    for span in result.spans:
        rebuilt += [text[cursor : span.start], token_of[text[span.start : span.end]]]
        cursor = span.end
    assert "".join([*rebuilt, text[cursor:]]) == result.anonymized


async def test_a_judge_that_breaks_its_contract_still_falls_back_to_stage_one():
    result = await mask("PESEL 50810112354", judge=SpyJudge(error=TypeError("a list, not a dict")))
    assert result.judge_failed
    assert result.anonymized == "PESEL <PESEL_1>"


async def test_a_judge_that_breaks_its_contract_blocks_when_the_policy_says_so():
    with pytest.raises(JudgeError, match="TypeError"):
        await mask(
            "PESEL 50810112354", judge=SpyJudge(error=TypeError("x")), fallback=JudgeFallback.BLOCK
        )


async def test_a_fallback_given_as_plain_text_is_honoured():
    """A string "block" used to compare unequal to the enum and let a failing judge through."""
    with pytest.raises(JudgeError):
        await mask("PESEL 50810112354", judge=SpyJudge(error=JudgeError("down")), fallback="block")


async def test_a_long_export_masks_every_repeated_value():
    rows = [f"Wiersz {index:04d}: Jan Kowalski, konto {IBAN}" for index in range(1000)]
    result = await mask("\n".join(rows))
    assert "Kowalski" not in result.anonymized
    assert "1090" not in result.anonymized
    assert set(result.mapping) == {"<OSOBA_1>", "<IBAN_1>"}


async def test_decomposed_name_is_masked_whole_when_it_ends_the_text():
    """The accent on the last letter of a message is part of the name, not text after it."""
    text = unicodedata.normalize("NFD", "Umowę podpisał Łukasz Kuś")
    result = await mask(text)
    assert result.anonymized == unicodedata.normalize("NFD", "Umowę podpisał <OSOBA_1>")
    assert result.mapping == {"<OSOBA_1>": unicodedata.normalize("NFD", "Łukasz Kuś")}


async def test_guard_span_one_past_the_end_is_refused():
    text = "Klient EMP-11111"
    with pytest.raises(ValueError, match="'fixed'"):
        await mask(text, [FixedGuard(Span(7, len(text) + 1, "PERSON"))])


async def test_guard_span_starting_at_minus_one_is_refused():
    """Python reads a start of -1 from the end, and the text went out with half of it repeated."""
    with pytest.raises(ValueError, match="'fixed'"):
        await mask("Klient EMP-11111 dzwonił.", [FixedGuard(Span(-1, 5, "PERSON"))])


async def test_custom_type_with_a_digit_is_refused():
    """TOKEN_PATTERN has no digits in a label: <EMP2_1> could never be restored."""
    with pytest.raises(ValueError, match="EMPLOYEE_ID"):
        await mask("Klient EMP-11111.", [FixedGuard(Span(7, 16, "EMP2"))], enabled={"EMP2"})


async def test_four_letter_known_value_inside_a_word_masks_the_whole_word():
    """ "Adam" known from an earlier turn: "Adamem" is the same man and must not go out."""
    text = "Spotkanie z Adamem."
    result = await mask(text, [], mapping={"<OSOBA_1>": "Adam"})
    assert result.anonymized == "Spotkanie z <OSOBA_2>."
    assert result.mapping["<OSOBA_2>"] == "Adamem"


def test_mask_known_values_masks_a_value_right_after_a_token():
    mapping = {"<OSOBA_1>": "Jan", "<OSOBA_2>": "Nowak"}
    assert mask_known_values("Pan <OSOBA_1>Nowak dzwonił.", mapping) == (
        "Pan <OSOBA_1><OSOBA_2> dzwonił."
    )


async def test_judge_finding_right_after_a_token_is_masked():
    """A guard that cut "Jan" out of "JanNowak" left "Nowak" for the judge; it must be masked."""
    judge = SpyJudge([("Nowak", "PERSON")])
    result = await mask("JanNowak dzwonił.", [FixedGuard(Span(0, 3, "PERSON"))], judge=judge)
    assert result.anonymized == "<OSOBA_1><OSOBA_2> dzwonił."
    assert is_lossless(result)


def test_mask_known_values_masks_a_value_right_before_a_token():
    mapping = {"<OSOBA_1>": "Jan", "<OSOBA_2>": "Nowak"}
    assert mask_known_values("Pan Nowak<OSOBA_1> dzwonił.", mapping) == (
        "Pan <OSOBA_2><OSOBA_1> dzwonił."
    )


async def test_judge_finding_right_before_a_token_is_masked():
    judge = SpyJudge([("Nowak", "PERSON")])
    result = await mask("NowakJan dzwonił.", [FixedGuard(Span(5, 8, "PERSON"))], judge=judge)
    assert result.anonymized == "<OSOBA_2><OSOBA_1> dzwonił."
    assert is_lossless(result)


async def test_judge_finding_of_bare_punctuation_masks_nothing():
    """A small model answering "..." would otherwise turn every ellipsis into a token."""
    text = "Czekam... i dalej czekam..."
    result = await mask(text, [], judge=SpyJudge([("...", "OTHER")]))
    assert result.anonymized == text
    assert result.judge_hits == 0


async def test_known_value_whose_word_is_masked_already_is_masked_in_place():
    """The rest of "TelevisionM12037" is a secret already, so the company is masked in place.

    Widening it over the whole word would cross the secret, and leaving it alone sent it in clear.
    """
    text = "Television TelevisionM12037"
    guards = [DenylistGuard(["Television"]), FixedGuard(Span(21, 27, "SECRET", 1.0))]
    result = await mask(text, guards)
    assert result.anonymized == "<FIRMA_1> <FIRMA_1><SEKRET_1>"


def test_mask_known_values_masks_a_short_value_opening_the_text():
    """The last letter of the text is not the letter before its first word."""
    masked = mask_known_values("Ewa dzwoniła wczoraj", {"<OSOBA_1>": "Ewa"})
    assert masked == "<OSOBA_1> dzwoniła wczoraj"


async def test_known_value_starting_with_punctuation_stops_there():
    """Mirror of "DD." in "DD.MM.YYYY": "(22) ..." after "tel" is not part of the word "tel"."""
    text = "Zadzwoń pod tel(22) 123 45 67."
    result = await mask(text, [], mapping={"<TELEFON_1>": "(22) 123 45 67"})
    assert result.anonymized == "Zadzwoń pod tel<TELEFON_1>."
    assert result.mapping == {"<TELEFON_1>": "(22) 123 45 67"}


async def test_known_value_at_the_end_of_the_first_word_takes_all_of_it():
    text = "SuperNowak dzwonił."
    result = await mask(text, [], mapping={"<OSOBA_1>": "Nowak"})
    assert result.anonymized == "<OSOBA_2> dzwonił."
    assert result.mapping["<OSOBA_2>"] == "SuperNowak"


async def test_known_value_inside_the_last_word_takes_all_of_it():
    """Masking "Kowalskieg" leaves "o" glued to the token and garbles the inflected name."""
    text = "Nie było Kowalskiego"
    result = await mask(text, [], mapping={"<OSOBA_1>": "Kowalski"})
    assert result.anonymized == "Nie było <OSOBA_2>"
    assert result.mapping["<OSOBA_2>"] == "Kowalskiego"


async def test_known_value_that_only_partly_overlaps_a_detection_is_masked_whole():
    """The name rule reads "Anna Nowak", and skipping the known value sent "z Kowalskich"."""
    text = "Pani Anna Nowak z Kowalskich"
    result = await mask(text, mapping={"<OSOBA_1>": "Nowak z Kowalskich"})
    assert result.anonymized == "Pani <OSOBA_2>"
    assert result.mapping["<OSOBA_2>"] == "Anna Nowak z Kowalskich"
    assert is_lossless(result)


async def test_two_known_values_that_overlap_are_masked_together():
    text = "Pani Anna Nowak z Kowalskich"
    mapping = {"<OSOBA_1>": "Anna Nowak", "<OSOBA_2>": "Nowak z Kowalskich"}
    result = await mask(text, [], mapping=mapping)
    assert result.anonymized == "Pani <OSOBA_3>"
    assert is_lossless(result)


async def test_known_value_crossing_a_surer_detection_keeps_the_rest_under_its_own_type():
    text = "Firma Nowak z Kowalskich"
    guard = FixedGuard(Span(0, 11, "ORG", 1.0))
    result = await mask(text, [guard], mapping={"<OSOBA_1>": "Nowak z Kowalskich"})
    assert result.anonymized == "<FIRMA_1> <OSOBA_2>"
    assert result.mapping["<OSOBA_2>"] == "z Kowalskich"
    assert is_lossless(result)


async def test_leftover_sharing_the_start_of_its_detection_is_not_masked_elsewhere():
    """The address rule starts at "NIP" and loses the number to the NIP rule: the bare word left
    over is masked in place, not in every sentence that mentions a NIP."""
    text = "Kontakt: jan@example.com\nNIP 526-000-00-05\nProszę podać NIP nabywcy."
    guard = FixedGuard(
        Span(9, 24, "EMAIL"), Span(29, 42, "NIP"), Span(25, 42, "ADDRESS", 0.8, "fixed")
    )
    result = await mask(text, [guard])
    assert result.anonymized == "Kontakt: <EMAIL_1>\n<ADRES_1> <NIP_1>\nProszę podać NIP nabywcy."


async def test_one_value_read_as_two_types_with_equal_confidence_takes_the_first_type():
    text = "Samochód WA 12345 stoi. Nr rej. WA 12345 zgłoszony."
    guard = FixedGuard(Span(9, 17, "POSTAL_CODE", 1.0), Span(32, 40, "PLATE", 1.0))
    result = await mask(text, [guard])
    assert result.anonymized == (
        "Samochód <KOD_POCZTOWY_1> stoi. Nr rej. <KOD_POCZTOWY_1> zgłoszony."
    )


async def test_judge_span_right_after_a_token_points_at_the_original_value():
    text = "JanNowak dzwonił."
    judge = SpyJudge([("Nowak", "PERSON")])
    result = await mask(text, [FixedGuard(Span(0, 3, "PERSON"))], judge=judge)
    assert [(text[s.start : s.end], s.source) for s in result.spans] == [
        ("Jan", "guard"),
        ("Nowak", "judge"),
    ]


async def test_existing_mapping_out_of_order_never_hands_out_a_used_token():
    """A conversation's mapping read back from storage need not be in numbering order."""
    existing = {"<EMAIL_2>": "c@d.example", "<EMAIL_1>": "a@b.example"}
    result = await mask("Nowy adres: e@f.example", mapping=existing)
    assert result.anonymized == "Nowy adres: <EMAIL_3>"
    assert result.mapping == {
        "<EMAIL_1>": "a@b.example",
        "<EMAIL_2>": "c@d.example",
        "<EMAIL_3>": "e@f.example",
    }


async def test_judge_finding_of_a_type_the_policy_disables_is_not_masked():
    text = "Kowalski zapłacił fakturę."
    judge = SpyJudge([("Kowalski", "PERSON")])
    result = await mask(text, [], judge=judge, enabled={"PESEL"})
    assert result.anonymized == text
    assert result.judge_hits == 0


async def test_judge_findings_are_counted_per_entity_type():
    judge = SpyJudge([("Nowak", "PERSON")])
    result = await mask("Nowak i znowu Nowak.", [], judge=judge)
    assert result.anonymized == "<OSOBA_1> i znowu <OSOBA_1>."
    assert result.entity_counts == {"PERSON": 2}


async def test_judge_error_under_block_reaches_the_caller_unchanged():
    error = JudgeError("remote judge failed: HTTP 401")

    class FailingJudge(SpyJudge):
        async def review(self, masked_text):
            raise error

    with pytest.raises(JudgeError) as raised:
        await anonymize(
            "PESEL 50810112354",
            enabled=ALL,
            guards=[RegexGuard()],
            judge=FailingJudge(),
            existing_mapping={},
            judge_fallback=JudgeFallback.BLOCK,
        )
    assert raised.value is error


def test_mask_known_values_ignores_entries_that_are_not_tokens():
    """Only a token can stand in for a value: anything else would be written into the text."""
    mapping = {"<OSOBA_1>": "Anna", "klient": "Nowak"}
    assert mask_known_values("Anna i Nowak.", mapping) == "<OSOBA_1> i Nowak."


def _fullwidth(text):
    return "".join(chr(ord(char) + 0xFEE0) for char in text)


@pytest.mark.parametrize(
    "spelling",
    [
        "ZIAJA",
        "Zia" + ZERO_WIDTH_SPACE + "ja",
        "Zia" + SOFT_HYPHEN + "ja",
        _fullwidth("Ziaja"),
    ],
)
async def test_a_known_name_spelled_differently_in_the_same_text_is_masked(spelling):
    """Once "pani Ziaja" has a token, the name in capitals or with a hidden character is hers."""
    result = await mask(f"Dzwoniła pani Ziaja. {spelling} prosi o oddzwonienie.")
    assert result.anonymized == "Dzwoniła pani <OSOBA_1>. <OSOBA_1> prosi o oddzwonienie."
    assert result.mapping == {"<OSOBA_1>": "Ziaja"}


@pytest.mark.parametrize(
    "spelling",
    [
        "WIŚNIEWSKA",
        unicodedata.normalize("NFD", "Wiśniewska"),
        "Wiś" + ZERO_WIDTH_SPACE + "niewska",
    ],
)
async def test_a_name_from_an_earlier_turn_is_masked_however_it_is_spelled(spelling):
    result = await mask(f"{spelling} oddzwoni jutro.", [], mapping={"<OSOBA_1>": "Wiśniewska"})
    assert result.anonymized == "<OSOBA_1> oddzwoni jutro."
    assert result.mapping == {"<OSOBA_1>": "Wiśniewska"}


@pytest.mark.parametrize("gap", [NBSP, "  ", "\n", "\t", " \n "])
def test_a_known_phrase_broken_across_other_whitespace_is_masked(gap):
    """A judge's finding comes back on the next turn wrapped onto two lines or typed twice."""
    mapping = {"<INNE_1>": "Kierownik Zmiany Nocnej"}
    replayed = mask_known_values(f"Dzwonił Kierownik{gap}Zmiany Nocnej.", mapping)
    assert replayed == "Dzwonił <INNE_1>."


def test_a_known_phrase_in_lowercase_is_masked():
    mapping = {"<INNE_1>": "Kierownik Zmiany Nocnej"}
    assert mask_known_values("dzwonił kierownik zmiany nocnej", mapping) == "dzwonił <INNE_1>"


async def test_a_one_word_name_written_in_lowercase_is_left_as_the_word_it_also_is():
    """Zima, Kot and Wilk are surnames and common nouns: "zima" in lowercase is winter."""
    result = await mask("Pan Zima mówi, że zima będzie długa.", [], mapping={"<OSOBA_1>": "Zima"})
    assert result.anonymized == "Pan <OSOBA_1> mówi, że zima będzie długa."


async def test_one_value_keeps_one_token_however_it_is_spelled():
    """The model would otherwise be told about two people, or two telephones, where there is one."""
    first = await mask("Jan Kowalski dzwonił z numeru 601 234 567.")
    second = await mask(
        f"JAN KOWALSKI dzwonił ponownie z numeru 601{NBSP}234{NBSP}567.", mapping=first.mapping
    )
    assert second.anonymized == "<OSOBA_1> dzwonił ponownie z numeru <TELEFON_1>."
    assert (
        second.mapping
        == first.mapping
        == {
            "<OSOBA_1>": "Jan Kowalski",
            "<TELEFON_1>": "601 234 567",
        }
    )


async def test_counts_follow_the_token_a_value_keeps():
    """A registration read as a postcode in the next turn keeps its token, and is counted as one."""
    guard = FixedGuard(Span(4, 12, "POSTAL_CODE"))  # "WA 12345"
    mapping = {"<REJESTRACJA_1>": "WA 12345"}
    result = await mask("Kod WA 12345 przy ulicy.", [guard], mapping=mapping)
    assert result.anonymized == "Kod <REJESTRACJA_1> przy ulicy."
    assert result.entity_counts == {"PLATE": 1}


async def test_two_passwords_that_differ_only_in_letter_case_are_two_secrets():
    """Restoring the old password where the model meant the new one would lock someone out."""
    text = 'Stare hasło: "Abc123!", nowe hasło: "abc123!".'
    result = await mask(text, [SecretsGuard()])
    assert result.anonymized == 'Stare hasło: "<SEKRET_1>", nowe hasło: "<SEKRET_2>".'
    assert mask_known_values("abc123!", {"<SEKRET_1>": "Abc123!"}) == "abc123!"


async def test_a_short_number_known_as_a_secret_is_not_masked_inside_dates_or_amounts():
    """Masking every 2025 as the PIN told the model the PIN was a year."""
    text = "Nowy PIN do karty: 2025. Karta ważna do 12.2025, wydana 15.03.2025, opłata 20250 zł."
    result = await mask(text, [SecretsGuard()])
    assert result.anonymized == (
        "Nowy PIN do karty: <SEKRET_1>. Karta ważna do 12.2025, wydana 15.03.2025, opłata 20250 zł."
    )


async def test_a_short_number_known_from_an_earlier_turn_stays_out_of_other_numbers():
    text = "Faktura FV/2025/03 z dnia 15.03.2025, termin 2025-04-15, zamówienie 120256."
    result = await mask(text, [], mapping={"<SEKRET_1>": "2025"})
    assert result.anonymized == text


async def test_a_short_number_known_as_a_secret_is_masked_where_it_stands_alone():
    result = await mask("PIN 4821. Powtarzam: 4821, nie 4822.", [], mapping={"<SEKRET_1>": "4821"})
    assert result.anonymized == "PIN <SEKRET_1>. Powtarzam: <SEKRET_1>, nie 4822."


async def test_a_secret_is_masked_whole_with_an_address_inside_it():
    """The model learnt that the password was the e-mail address followed by "#!"."""
    result = await mask('Hasło: "jan.kowalski@example.com#!"', [RegexGuard(), SecretsGuard()])
    assert result.anonymized == 'Hasło: "<SEKRET_1>"'
    assert result.mapping == {"<SEKRET_1>": "jan.kowalski@example.com#!"}


@pytest.mark.parametrize("values", [1, 40, 400])
def test_the_search_for_many_values_finds_what_the_search_value_by_value_finds(values):
    """Past a few dozen values the search switches to one pass over the text; nothing changes."""
    rng = random.Random(values)  # noqa: S311 - reproducible test data, not a secret
    alphabet = "abAB12 .-ąĄ"
    for _ in range(200):
        haystack = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        needles = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 7))) for _ in range(values)
        ]
        expected = [[m.start() for m in re.finditer(re.escape(n), haystack)] for n in needles]
        assert occurrences(haystack, needles) == expected


async def test_a_spelling_found_in_lowercase_keeps_a_token_of_its_own():
    """The capitalised name leaves lowercase words alone, so the lowercase spelling needs its own.

    Folded into the capitalised token, the codename went out in "feniks_db" and "feniksapi", and
    the next turn no longer knew it in lowercase at all.
    """
    guards = [DenylistGuard(["Feniks"])]
    text = "Projekt Feniks startuje. Repo feniks, baza feniks_db, serwis feniksapi."
    first = await mask(text, guards)
    assert first.anonymized == (
        "Projekt <FIRMA_1> startuje. Repo <FIRMA_2>, baza <FIRMA_2>_db, serwis <FIRMA_3>."
    )
    second = await mask("Baza feniks_db i serwis feniksapi.", [], mapping=first.mapping)
    assert second.anonymized == "Baza <FIRMA_2>_db i serwis <FIRMA_3>."


async def test_a_known_value_is_masked_wherever_the_old_exact_search_found_it():
    """A decomposed ogonek after "Kowalska" makes the guards read "Kowalską", a different word.

    The search through the guards' reading only adds matches; the value as written still counts.
    """
    text = "Dzwoniła Pani Kowalska. " + unicodedata.normalize("NFD", "Spotkałem się z Kowalską.")
    result = await mask(text, [], mapping={"<OSOBA_1>": "Kowalska"})
    assert result.anonymized == "Dzwoniła Pani <OSOBA_1>. " + unicodedata.normalize(
        "NFD", "Spotkałem się z <OSOBA_2>."
    )
    assert result.mapping["<OSOBA_2>"] == unicodedata.normalize("NFD", "Kowalską")


async def test_a_short_number_inside_a_word_is_masked_with_the_word():
    """Only a longer number protects a PIN's digits; "bank4821" is the PIN with a prefix."""
    text = "PIN 4821, zapisany jako bank4821 i 4821zł."
    result = await mask(text, [], mapping={"<SEKRET_1>": "4821"})
    assert result.anonymized == "PIN <SEKRET_1>, zapisany jako <SEKRET_2> i <SEKRET_3>."


async def test_a_lowercase_password_does_not_take_the_token_of_a_name_it_spells():
    """A passphrase "tomasz zieliński" and the person Tomasz Zieliński are two values."""
    first = await mask("Moje hasło: 'tomasz zieliński'", [SecretsGuard()])
    second = await mask("Tomasz Zieliński dzwonił.", mapping=first.mapping)
    assert second.anonymized == "<OSOBA_1> dzwonił."
    assert second.mapping["<OSOBA_1>"] == "Tomasz Zieliński"


async def test_values_of_a_custom_type_that_differ_in_letter_case_are_two_values():
    """Nothing says a custom identifier ignores case, so "aB3xY9" and "Ab3Xy9" stay apart."""
    guard = FixedGuard(Span(10, 16, "ORDER_REF"), Span(33, 39, "ORDER_REF"))
    text = "Zwrot dla aB3xY9, reklamacja dla Ab3Xy9."
    result = await mask(text, [guard], enabled={"ORDER_REF"})
    assert result.anonymized == "Zwrot dla <ORDER_REF_1>, reklamacja dla <ORDER_REF_2>."
    assert is_lossless(result)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("İlknur Yılmaz", "İLKNUR YILMAZ i ilknur yılmaz"),  # noqa: RUF001
        ("ΝΙΚΟΣ", "Pan ΝΙΚΟΣ i ΝΙΚΟΣ, IZMİR"),
    ],
)
def test_letter_case_folds_alike_in_the_value_and_in_the_text(value, text):
    """The Turkish dotted I and the Greek final sigma fold the same way wherever they stand."""
    masked = mask_known_values(text, {"<OSOBA_1>": value})
    assert value not in masked
    assert masked.count("<OSOBA_1>") == 2


async def test_a_known_name_split_over_two_lines_is_masked_whole():
    """The name rule reads "Jan" on its own line; the known "Jan Kowalski" takes it in."""
    text = "Umowa zawarta z Jan Kowalski.\n\nPodpis:\nJan\nKowalski"
    result = await mask(text)
    assert result.anonymized == "Umowa zawarta z <OSOBA_1>.\n\nPodpis:\n<OSOBA_1>"
    assert result.mapping == {"<OSOBA_1>": "Jan Kowalski"}


async def test_a_known_password_holding_an_address_is_masked_whole_in_a_later_turn():
    first = await mask("Hasło: jan.kowalski@example.com#!", [RegexGuard(), SecretsGuard()])
    second = await mask(
        "Czy jan.kowalski@example.com#! to dobre hasło?",
        [RegexGuard(), SecretsGuard()],
        mapping=first.mapping,
    )
    assert second.anonymized == "Czy <SEKRET_1> to dobre hasło?"


async def test_a_known_iban_broken_over_lines_is_masked_whole_around_a_card_read_inside_it():
    text = f"IBAN {IBAN}, kopia:\nPL61\n1090 1014 0000 0712 1981 2874 koniec"
    result = await mask(text, [RegexGuard()])
    assert result.anonymized == "IBAN <IBAN_1>, kopia:\n<IBAN_1> koniec"


async def test_a_known_identifier_written_in_lowercase_later_is_masked():
    """The lowercase exception is for names; a VIN or an e-mail address is the same in any case."""
    mapping = {"<VIN_1>": "WVWZZZ1JZXW000001", "<EMAIL_1>": "Jan.Kowalski@Example.com"}
    replayed = mask_known_values("auto wvwzzz1jzxw000001, mail jan.kowalski@example.com", mapping)
    assert replayed == "auto <VIN_1>, mail <EMAIL_1>"


def test_a_known_value_written_inside_a_typed_placeholder_is_masked_on_replay():
    """A placeholder the user typed stands for itself; it does not shelter a password."""
    mapping = {"<ADMIN_2024>": "<ADMIN_2024>", "<SEKRET_1>": "ADMIN_2024"}
    assert mask_known_values("Wpis <ADMIN_2024>.", mapping) == "Wpis <<SEKRET_1>>."


def test_the_search_for_values_sharing_their_first_letters_finds_every_occurrence():
    """Values that share four letters and differ in length go to the regex one by one."""
    haystack = "a" * 2_000 + "b" + "a" * 37
    needles = ["a" * (4 + k) for k in range(200)] + ["ab", "ba", "aaab"]
    expected = [[m.start() for m in re.finditer(re.escape(n), haystack)] for n in needles]
    assert occurrences(haystack, needles) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Z koperty: PIN/PUK 4821/36912874.",
        "Kody do dwóch kart: 4821-7319.",
        "Kod 4821:1 z instrukcji.",
    ],
)
async def test_a_short_number_joined_to_a_number_that_is_no_date_is_masked(text):
    """Only a date, a time or an amount keeps a PIN's digits readable; a PIN/PUK pair does not."""
    result = await mask(text, [], mapping={"<SEKRET_1>": "4821"})
    assert "4821" not in result.anonymized


@pytest.mark.parametrize(
    "text",
    [
        "Budżet na lata 2024" + EN_DASH + "2025.",
        "Termin 2025" + EN_DASH + "04" + EN_DASH + "15 bez zmian.",
        "Termin 15" + chr(0x2212) + "04" + chr(0x2212) + "2025.",
        "Spotkanie o 20:25.",
        "Faktura FV/2025/03 na 2025,50 zł.",
    ],
)
async def test_a_short_number_inside_a_date_time_or_amount_is_left_alone(text):
    """Typeset dashes join a date as a hyphen does: 2024 to 2025 is a span of years, not a PIN."""
    result = await mask(text, [], mapping={"<SEKRET_1>": "2025", "<SEKRET_2>": "2025"})
    assert result.anonymized == text


def test_a_value_written_like_a_token_is_masked_on_replay():
    """A password "<ADMIN_2024>" is no token, even once it is no longer held as a placeholder."""
    replayed = mask_known_values("Zostaw <ADMIN_2024> bez zmian.", {"<SEKRET_1>": "<ADMIN_2024>"})
    assert replayed == "Zostaw <SEKRET_1> bez zmian."


@pytest.mark.parametrize("first", ["<SEKRET_1>", "<FIRMA_1>"])
def test_a_spelling_held_as_a_password_and_as_a_company_is_searched_for_both_ways(first):
    """The company is found in any case, the password only as written, whichever came first."""
    tokens = ["<SEKRET_1>", "<FIRMA_1>"]
    tokens.remove(first)
    mapping = {first: "kot", tokens[0]: "kot"}
    assert mask_known_values("KOT, Kot, kot", mapping) == "<FIRMA_1>, <FIRMA_1>, <SEKRET_1>"


async def test_a_password_and_a_name_that_read_alike_get_two_tokens():
    """Sharing the password's token, the name lost its search in any letter case next turn."""
    guards = [RegexGuard(), PersonNameGuard(), SecretsGuard()]
    first = await mask("Hasło: Kowalski\nProszę o reset, Pan Kowalski czeka.", guards)
    assert first.anonymized == "Hasło: <SEKRET_1>\nProszę o reset, Pan <OSOBA_1> czeka."
    second = await mask("Pismo od KOWALSKI do kadr.", guards, mapping=first.mapping)
    assert second.anonymized == "Pismo od <OSOBA_1> do kadr."


def test_a_word_with_an_invisible_character_inside_is_masked_whole():
    text = f"Dzień dobry, Nowako{ZERO_WIDTH_SPACE}wi przekazałem."
    assert mask_known_values(text, {"<OSOBA_1>": "Nowak"}) == "Dzień dobry, <OSOBA_1> przekazałem."


def test_a_longer_value_across_a_line_break_is_placed_before_a_shorter_one_written_exactly():
    """ "Feniks" written exactly must not cut "Feniks Software" broken over two lines in half."""
    mapping = {"<FIRMA_1>": "Feniks Software", "<FIRMA_2>": "Feniks"}
    replayed = mask_known_values("Umowa z firmą Feniks\nSoftware i z Feniks.", mapping)
    assert replayed == "Umowa z firmą <FIRMA_1> i z <FIRMA_2>."


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("tel. 601234567", "tel. 601 234 567"),
        ("NIP 5260000005", "NIP 526-000-00-05"),
        (f"konto {IBAN.replace(' ', '')}", f"konto {IBAN}"),
    ],
)
async def test_an_identifier_keeps_its_token_with_its_groups_apart_or_together(first, second):
    one = await mask(first, [RegexGuard()])
    two = await mask(second, [RegexGuard()], mapping=one.mapping)
    assert one.anonymized.split()[-1] == two.anonymized.split()[-1]
    assert two.mapping == one.mapping


async def test_a_known_iban_written_in_groups_over_two_lines_is_masked():
    compact = IBAN.replace(" ", "")
    text = "konto PL61 1090 1014\n0000 0712 1981 2874 koniec"
    result = await mask(text, [], mapping={"<IBAN_1>": compact})
    assert result.anonymized == "konto <IBAN_1> koniec"


def test_an_identifier_read_without_its_groups_is_not_found_inside_a_longer_number():
    replayed = mask_known_values("numer 601 234 5678 wewnętrzny", {"<TELEFON_1>": "601234567"})
    assert replayed == "numer 601 234 5678 wewnętrzny"


async def test_every_occurrence_inside_one_long_word_is_masked_around_a_secret_in_it():
    """The word cannot be masked whole, so each value inside it is masked on its own."""
    text = "x" + "Kowalski" * 3 + "M12037" + "Kowalski"
    secret = FixedGuard(Span(25, 31, "SECRET"))
    result = await mask(text, [secret], mapping={"<OSOBA_1>": "Kowalski"})
    assert "Kowalski" not in result.anonymized
    assert "<SEKRET_1>" in result.anonymized
