"""Stage-1 + stage-2 pipeline behaviour. Uses stdlib-only guards so no spaCy model is needed."""

import time

import pytest

from polanon import Anonymizer, DenylistGuard, MockJudge, RegexGuard, SecretsGuard, default_guards
from polanon.guards.base import Guard
from polanon.judge.base import JudgeError
from polanon.pipeline import anonymize, mask_known_values
from polanon.policy import ENTITY_TYPES
from polanon.types import JudgeFallback, Span

ALL = set(ENTITY_TYPES)


def guards(denylist=()):
    return [RegexGuard(), SecretsGuard(), DenylistGuard(denylist)]


async def run(
    text, *, judge=None, mapping=None, denylist=(), enabled=ALL, fallback=JudgeFallback.SEND_STAGE1
):
    return await anonymize(
        text,
        enabled=enabled,
        guards=guards(denylist),
        judge=judge,
        existing_mapping=mapping or {},
        judge_fallback=fallback,
    )


async def test_masks_polish_identifiers_and_keeps_mapping():
    result = await run("PESEL 50810112354 i NIP 526-000-00-05")
    assert result.anonymized == "PESEL <PESEL_1> i NIP <NIP_1>"
    assert result.mapping == {"<PESEL_1>": "50810112354", "<NIP_1>": "526-000-00-05"}
    assert result.entity_counts == {"PESEL": 1, "NIP": 1}


async def test_same_value_twice_gets_the_same_token():
    result = await run("mail a@b.example oraz ponownie a@b.example")
    assert result.anonymized == "mail <EMAIL_1> oraz ponownie <EMAIL_1>"
    assert len(result.mapping) == 1


async def test_distinct_values_get_consecutive_numbers():
    result = await run("a@b.example, c@d.example")
    assert result.anonymized == "<EMAIL_1>, <EMAIL_2>"


async def test_existing_mapping_is_reused_and_numbering_continues():
    existing = {"<EMAIL_1>": "a@b.example", "<EMAIL_2>": "c@d.example"}
    result = await run("a@b.example pisze do e@f.example", mapping=existing)
    assert result.anonymized == "<EMAIL_1> pisze do <EMAIL_3>"
    assert result.mapping["<EMAIL_3>"] == "e@f.example"
    assert result.mapping["<EMAIL_2>"] == "c@d.example"


async def test_value_at_the_very_start_and_end_is_masked():
    result = await run("a@b.example w środku c@d.example")
    assert result.anonymized.startswith("<EMAIL_1>")
    assert result.anonymized.endswith("<EMAIL_2>")


async def test_restore_is_the_inverse_of_anonymize():
    text = "Faktura dla a@b.example, PESEL 50810112354, tel. 600 100 200."
    result = await run(text)
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


async def test_existing_tokens_in_the_input_are_left_alone():
    result = await run("Klient <OSOBA_1> prosi o kontakt")
    assert result.anonymized == "Klient <OSOBA_1> prosi o kontakt"
    assert result.mapping == {"<OSOBA_1>": "<OSOBA_1>"}  # stands for itself, see the invariants


def test_a_token_number_containing_an_identifier_does_not_seed_new_numbers():
    text = "Pole <OSOBA_50810112354> wypełnia Jan Kowalski."
    result = Anonymizer(default_guards(with_ner=False)).anonymize_sync(text)

    assert result.anonymized == "Pole <OSOBA_<PESEL_1>> wypełnia <OSOBA_1>."
    assert result.mapping["<OSOBA_1>"] == "Jan Kowalski"
    assert result.mapping["<PESEL_1>"] == "50810112354"
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


def test_a_token_number_beyond_the_integer_conversion_limit_does_not_crash():
    token = "<X_" + "1" * 5000 + ">"
    text = f"Jan Kowalski, ref {token}"
    result = Anonymizer(default_guards(with_ner=False)).anonymize_sync(text)

    assert result.anonymized == f"<OSOBA_1>, ref {token}"
    assert result.mapping["<OSOBA_1>"] == "Jan Kowalski"
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


@pytest.mark.parametrize(
    ("number", "next_number"),
    [("123456", 123457), ("1000000", 1), ("0000012", 1)],
)
async def test_token_reservations_are_limited_by_digit_count(number, next_number):
    token = f"<EMAIL_{number}>"
    text = f"{token} a@b.example"
    result = await run(text, enabled={"EMAIL"})

    assert result.anonymized == f"{token} <EMAIL_{next_number}>"
    assert result.mapping[f"<EMAIL_{next_number}>"] == "a@b.example"
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


async def test_allocation_skips_literal_tokens_beyond_six_digits():
    text = "<EMAIL_999999> i <EMAIL_1000000> oraz a@b.example"
    result = await run(text, enabled={"EMAIL"})

    assert result.anonymized == "<EMAIL_999999> i <EMAIL_1000000> oraz <EMAIL_1000001>"
    assert result.mapping["<EMAIL_1000000>"] == "<EMAIL_1000000>"
    assert result.mapping["<EMAIL_1000001>"] == "a@b.example"
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


def test_allocation_keeps_seven_digit_tokens_stable_across_turns():
    anonymizer = Anonymizer(default_guards(with_ner=False))
    first_text = "Pole <OSOBA_999999> wypełnia Jan Kowalski."
    second_text = "Umowę podpisała Anna Nowak."
    first = anonymizer.anonymize_sync(first_text)
    second = anonymizer.anonymize_sync(second_text, mapping=first.mapping)

    assert first.anonymized == "Pole <OSOBA_999999> wypełnia <OSOBA_1000000>."
    assert second.anonymized == "Umowę podpisała <OSOBA_1000001>."
    assert second.mapping["<OSOBA_1000000>"] == "Jan Kowalski"
    assert second.mapping["<OSOBA_1000001>"] == "Anna Nowak"
    assert Anonymizer.restore(first.anonymized, second.mapping) == first_text
    assert Anonymizer.restore(second.anonymized, second.mapping) == second_text
    # Known values stay masked even when no guard detects them in a later turn.
    assert mask_known_values("Jan Kowalski", second.mapping) == "<OSOBA_1000000>"


async def test_disabled_entity_types_are_not_masked():
    result = await run("PESEL 50810112354 i mail a@b.example", enabled={"EMAIL"})
    assert result.anonymized == "PESEL 50810112354 i mail <EMAIL_1>"


async def test_denylist_masks_company_names_case_insensitively():
    result = await run("Umowa z acme sp. z o.o.", denylist=["ACME"])
    assert result.anonymized == "Umowa z <FIRMA_1> sp. z o.o."


async def test_secrets_are_masked():
    result = await run("export API_KEY=abcdef123456 w configu")
    assert "abcdef123456" not in result.anonymized
    assert result.entity_counts["SECRET"] == 1


async def test_longer_span_wins_on_overlap():
    result = await run(
        "token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1"
    )
    assert len(result.spans) == 1
    assert result.spans[0].entity_type == "SECRET"


async def test_judge_findings_are_masked_and_counted():
    judge = MockJudge({"dyrektor jedynej huty w Stalowej Woli": "PERSON"})
    result = await run("To dyrektor jedynej huty w Stalowej Woli.", judge=judge)
    assert result.anonymized == "To <OSOBA_1>."
    assert result.judge_hits == 1
    assert result.judge_failed is False


async def test_judge_finding_replaces_every_occurrence():
    judge = MockJudge({"nasz kontrahent z Rzeszowa": "ORG"})
    result = await run("nasz kontrahent z Rzeszowa i znowu nasz kontrahent z Rzeszowa", judge=judge)
    assert result.anonymized == "<FIRMA_1> i znowu <FIRMA_1>"
    assert result.judge_hits == 2


async def test_judge_other_type_maps_to_inne_label():
    judge = MockJudge({"właściciel trzech aptek w Sanoku": "OTHER"})
    result = await run("Pyta właściciel trzech aptek w Sanoku.", judge=judge)
    assert "<INNE_1>" in result.anonymized


async def test_judge_finding_overlapping_a_token_is_skipped():
    judge = MockJudge({"<EMAIL_1>": "PERSON"})
    result = await run("kontakt a@b.example", judge=judge)
    assert result.anonymized == "kontakt <EMAIL_1>"
    assert result.judge_hits == 0


async def test_judge_failure_with_send_stage1_keeps_stage1_result():
    result = await run("PESEL 50810112354", judge=MockJudge(fail=True))
    assert result.anonymized == "PESEL <PESEL_1>"
    assert result.judge_failed is True


async def test_judge_failure_with_block_raises():
    with pytest.raises(JudgeError):
        await run("PESEL 50810112354", judge=MockJudge(fail=True), fallback=JudgeFallback.BLOCK)


async def test_judge_spans_are_reported_in_original_coordinates():
    judge = MockJudge({"szef zakładu": "PERSON"})
    text = "Dzwonił szef zakładu wczoraj."
    result = await run(text, judge=judge)
    judge_spans = [s for s in result.spans if s.source == "judge"]
    assert text[judge_spans[0].start : judge_spans[0].end] == "szef zakładu"


async def test_empty_text_produces_empty_result():
    result = await run("")
    assert result.anonymized == ""
    assert result.mapping == {}


async def test_anonymizer_facade_matches_pipeline():
    anon = Anonymizer(guards(), judge=None)
    result = await anon.anonymize("mail a@b.example")
    assert result.anonymized == "mail <EMAIL_1>"


async def test_polish_password_keyword_is_masked():
    result = await run("Login: jkowalski, hasło: Tajne123!")
    assert "Tajne123!" not in result.anonymized
    assert "jkowalski" not in result.anonymized


async def test_labelled_secret_outranks_a_statistical_name_guess():
    """A value behind "hasło:" is a credential, not a person, whichever guard sees it first."""
    from polanon.types import Span

    class NameGuard:
        name = "fake_ner"

        def detect(self, text: str, enabled: set[str]) -> list[Span]:
            start = text.index("Tajne123")
            return [Span(start, start + 8, "PERSON", 0.85, self.name)]

    result = await anonymize(
        "hasło: Tajne123",
        enabled=ALL,
        guards=[SecretsGuard(), NameGuard()],
        judge=None,
        existing_mapping={},
    )
    assert "<SEKRET_1>" in result.anonymized


async def test_masking_does_not_freeze_the_caller_while_the_guards_run():
    """The guards are CPU work inside an async call, so they belong in a thread.

    Without that, one long document stops every other request a server handles for as long as the
    guards take, which on 200 000 characters was over a second.
    """
    import asyncio
    import time

    from polanon.guards.base import Guard

    class SlowGuard(Guard):
        name = "slow"

        def detect(self, text: str, enabled: set[str]) -> list[Span]:
            time.sleep(0.3)
            return []

    progress = 0

    async def other_work() -> None:
        nonlocal progress
        while True:
            progress += 1
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(other_work())
    await asyncio.sleep(0.05)
    before = progress

    await anonymize(
        "Jan Kowalski",
        enabled={"PERSON"},
        guards=[SlowGuard()],
        judge=None,
        existing_mapping={},
    )

    ticker.cancel()
    assert progress - before > 5, "the event loop was starved while the guards ran"


async def test_a_document_with_many_entities_is_rebuilt_once():
    """The masked text used to be rebuilt per span, which is quadratic on a long upload."""
    text = "PESEL 50810112354 oraz PESEL 45831523452. " * 500

    result = await anonymize(
        text,
        enabled={"PESEL"},
        guards=[RegexGuard()],
        judge=None,
        existing_mapping={},
    )

    assert result.entity_counts["PESEL"] == 1000
    assert len(result.mapping) == 2  # the same two values, so the same two tokens throughout
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hasło Zima2026! do panelu", "Zima2026!"),
        ("password Zimowy-Pociag-2026 do bazy", "Zimowy-Pociag-2026"),
        ("Kod dostępu 9f2a11bc do panelu", "9f2a11bc"),
    ],
)
async def test_a_password_introduced_by_a_word_is_masked(text, expected):
    """Polish writes "hasło X" as often as "hasło: X"."""
    result = await run(text, enabled={"SECRET"})
    assert expected not in result.anonymized
    assert "<SEKRET_1>" in result.anonymized


@pytest.mark.parametrize(
    "text",
    [
        "Hasło zostało zmienione wczoraj przez administratora.",
        "Hasło musi zawierać wielkie litery.",
        "Hasłem jest ustawiane automatycznie",
    ],
)
async def test_ordinary_sentences_about_passwords_are_left_alone(text):
    """Without a colon there is nothing marking where the credential starts, so the word after
    "hasło" has to look like one: a password carries a digit or a symbol, a Polish word does not."""
    result = await run(text, enabled={"SECRET"})
    assert result.anonymized == text


class FirstOccurrenceGuard(Guard):
    """Recognises a value only the first time, the way a statistical detector sometimes does."""

    name = "first_occurrence"

    def __init__(self, value: str, entity_type: str) -> None:
        self.value = value
        self.entity_type = entity_type

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        start = text.find(self.value)
        if start == -1 or self.entity_type not in enabled:
            return []
        return [Span(start, start + len(self.value), self.entity_type, 0.85, self.name)]


async def test_a_value_detected_once_is_masked_everywhere_it_stands_alone():
    """Found on a real invoice: "AB4" was masked on the billing address and sent in the clear on
    the delivery address below it, and the check before sending refused the whole message."""
    text = "BILLING AB4 1TX\nDELIVERY AB4 1TX"

    result = await anonymize(
        text,
        enabled={"LOCATION"},
        guards=[FirstOccurrenceGuard("AB4", "LOCATION")],
        judge=None,
        existing_mapping={},
    )

    assert result.anonymized == "BILLING <MIEJSCE_1> 1TX\nDELIVERY <MIEJSCE_1> 1TX"
    assert result.entity_counts == {"LOCATION": 2}
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


async def test_a_value_known_from_earlier_in_the_conversation_is_masked_without_a_guard():
    result = await anonymize(
        "Wczoraj pisała Anna Wiśniewska.",
        enabled=ALL,
        guards=[],
        judge=None,
        existing_mapping={"<OSOBA_1>": "Anna Wiśniewska"},
    )
    assert result.anonymized == "Wczoraj pisała <OSOBA_1>."
    assert result.mapping == {"<OSOBA_1>": "Anna Wiśniewska"}


async def test_a_known_value_inside_a_longer_word_masks_the_whole_word():
    """Masking only the inside leaves "<OSOBA_1>ego"; leaving it made the final check refuse."""
    text = "Nie było Kowalskiego."
    result = await anonymize(
        text, enabled=ALL, guards=[], judge=None, existing_mapping={"<OSOBA_1>": "Kowalski"}
    )
    assert result.anonymized == "Nie było <OSOBA_2>."
    assert result.mapping["<OSOBA_2>"] == "Kowalskiego"
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


async def test_a_word_masked_by_mistake_no_longer_stops_the_whole_document():
    """Five sample invoices never went out: "Television" was taken for a company, and the product
    code "TelevisionM12037" in the same line still held it, so the final check refused them."""
    text = "12333 TelevisionM12037 Television M120 37 Silver"
    result = await anonymize(
        text,
        enabled={"ORG"},
        guards=[DenylistGuard(["Television"])],
        judge=None,
        existing_mapping={},
    )
    # Numbered in reading order: the product code comes first, although the denylist matched
    # only the bare word and the code joined through the known-value pass.
    assert result.anonymized == "12333 <FIRMA_1> <FIRMA_2> M120 37 Silver"
    assert not any(value in result.anonymized for value in result.mapping.values())
    assert Anonymizer.restore(result.anonymized, result.mapping) == text


async def test_a_known_value_at_the_end_of_a_longer_word_takes_the_start_of_it_too():
    text = "Dzwonił superNowak z działu handlowego."
    result = await anonymize(
        text, enabled=ALL, guards=[], judge=None, existing_mapping={"<OSOBA_1>": "Nowak"}
    )
    assert result.anonymized == "Dzwonił <OSOBA_2> z działu handlowego."
    assert result.mapping["<OSOBA_2>"] == "superNowak"


async def test_a_known_value_ending_in_punctuation_is_not_inside_a_word():
    """The Swiss sample invoice was refused over "DD." in "date: DD.MM.YYYY"."""
    result = await anonymize(
        "date: DD.MM.YYYY",
        enabled=ALL,
        guards=[],
        judge=None,
        existing_mapping={"<OSOBA_1>": "DD."},
    )
    assert result.anonymized == "date: <OSOBA_1>MM.YYYY"


async def test_a_short_known_value_inside_a_longer_word_is_left_for_the_final_check():
    """Widening "Jan" to every word that contains it would mask half a document."""
    text = "Janusz dzwonił."
    result = await anonymize(
        text, enabled=ALL, guards=[], judge=None, existing_mapping={"<OSOBA_1>": "Jan"}
    )
    assert result.anonymized == text


async def test_the_longest_known_value_is_placed_first():
    result = await anonymize(
        "Anna Wiśniewska i Anna",
        enabled=ALL,
        guards=[],
        judge=None,
        existing_mapping={"<OSOBA_1>": "Anna", "<OSOBA_2>": "Anna Wiśniewska"},
    )
    assert result.anonymized == "<OSOBA_2> i <OSOBA_1>"


async def test_numbering_still_follows_reading_order_when_values_repeat():
    text = "PESEL 50810112354, potem 45831523452, znowu 50810112354."
    result = await anonymize(
        text, enabled={"PESEL"}, guards=[RegexGuard()], judge=None, existing_mapping={}
    )
    assert result.anonymized == "PESEL <PESEL_1>, potem <PESEL_2>, znowu <PESEL_1>."


def test_mask_known_values_replaces_every_occurrence_with_its_token():
    mapping = {"<FIRMA_1>": "Projekt Atlas"}
    masked = mask_known_values("Projekt Atlas, potem znowu Projekt Atlas.", mapping)
    assert masked == "<FIRMA_1>, potem znowu <FIRMA_1>."


def test_mask_known_values_takes_the_whole_word_around_a_long_value():
    """Replayed text is never restored, so an inflected form can share the value's token."""
    masked = mask_known_values("Pismo od Kowalskiego.", {"<OSOBA_1>": "Kowalski"})
    assert masked == "Pismo od <OSOBA_1>."


def test_mask_known_values_leaves_a_short_value_inside_a_longer_word():
    assert mask_known_values("Janusz przyszedł.", {"<OSOBA_1>": "Jan"}) == "Janusz przyszedł."


def test_mask_known_values_never_rewrites_a_token_already_in_the_text():
    masked = mask_known_values("<OSOBA_1> pyta o OSOBA", {"<FIRMA_1>": "OSOBA"})
    assert masked == "<OSOBA_1> pyta o <FIRMA_1>"


def test_mask_known_values_places_the_longest_value_first():
    mapping = {"<OSOBA_1>": "Anna", "<OSOBA_2>": "Anna Wiśniewska"}
    assert mask_known_values("Anna Wiśniewska i Anna", mapping) == "<OSOBA_2> i <OSOBA_1>"


def test_a_known_short_number_in_a_long_line_of_numbers_is_masked_in_linear_time():
    """Each occurrence was checked against the whole run of joined numbers around it."""
    text = "Lata: " + ",".join(["2025"] * 10_000)
    assert len(text) > 50_000
    started = time.perf_counter()
    result = Anonymizer([]).anonymize_sync(text, {"<SEKRET_1>": "2025"})
    # A fifth of a second on a laptop and about a second on a CI runner measuring coverage; the
    # quadratic version took 36 seconds on the same laptop.
    assert time.perf_counter() - started < 5.0
    assert result.anonymized == "Lata: " + ",".join(["<SEKRET_1>"] * 10_000)


@pytest.mark.parametrize(
    "text",
    [
        "Umowa na okres 01.01.2025-31.12.2025.",
        "Rok szkolny 2024/2025.",
        "Rok szkolny 2025/26.",
        "Raport 2025-01-01/2025-12-31.",
        "Pismo z dnia 15.04.2025r.",
        "Umowę podpisano w 2025r.",
    ],
)
def test_a_known_pin_that_equals_a_year_leaves_dates_joined_to_dates_readable(text):
    """Two dates joined by "-" or "/" were read as one number, and the PIN in them as a PIN."""
    first = Anonymizer([SecretsGuard()]).anonymize_sync("Mój PIN do karty: 2025.")
    assert first.mapping == {"<SEKRET_1>": "2025"}
    assert Anonymizer([]).anonymize_sync(text, first.mapping).anonymized == text


@pytest.mark.parametrize(
    "text", ["Kody 2025-7319.", "Kody 2025/4821.", "PIN 2025rok.", "PIN 2025R."]
)
def test_a_known_pin_joined_to_what_is_no_date_is_still_masked(text):
    result = Anonymizer([]).anonymize_sync(text, {"<SEKRET_1>": "2025"})
    assert "2025" not in result.anonymized
