from polanon.types import Span, merge_spans


def test_merge_spans_keeps_longer_on_overlap():
    spans = [Span(0, 5, "EMAIL"), Span(0, 20, "SECRET"), Span(25, 30, "PESEL")]
    merged = merge_spans(spans)
    assert [(s.start, s.end, s.entity_type) for s in merged] == [
        (0, 20, "SECRET"),
        (25, 30, "PESEL"),
    ]


def test_merge_spans_labels_by_confidence_and_keeps_the_rest_masked():
    """A loose NER span around an e-mail: the address keeps its label and the name stays masked.

    The previous rule kept only the more confident span, which unmasked "Jan Kowalski" here.
    """
    text = "Kontakt: Jan Kowalski <jan.k@firma.example>"
    ner = Span(text.index("Jan"), len(text), "PERSON", 0.85, "spacy_ner")
    email = Span(text.index("jan.k"), text.index(">"), "EMAIL", 1.0, "regex_pl")
    merged = merge_spans([ner, email], text)
    assert [(text[s.start : s.end], s.entity_type) for s in merged] == [
        ("Jan Kowalski", "PERSON"),
        ("jan.k@firma.example", "EMAIL"),
    ]


def test_merge_spans_without_the_text_keeps_leftovers_as_they_are():
    ner = Span(0, 40, "PERSON", 0.85, "spacy_ner")
    email = Span(8, 30, "EMAIL", 1.0, "regex_pl")
    assert merge_spans([ner, email]) == [
        Span(0, 8, "PERSON", 0.85, "spacy_ner"),
        email,
        Span(30, 40, "PERSON", 0.85, "spacy_ner"),
    ]


def test_merge_spans_joins_overlapping_detections_of_the_same_type():
    """Two detectors that disagree on where a name ends found one person, not a shorter one."""
    text = "z panem kamilem bednarkiem"
    guard = Span(8, 26, "PERSON", 0.9, "names_pl")
    model = Span(8, 15, "PERSON", 0.85, "spacy_ner")
    assert merge_spans([model, guard], text) == [guard]

    left = Span(0, 10, "PERSON", 0.85, "spacy_ner")
    right = Span(5, 20, "PERSON", 0.9, "names_pl")
    assert merge_spans([left, right]) == [Span(0, 20, "PERSON", 0.9, "names_pl")]


def test_merge_spans_breaks_a_complete_tie_by_detection_order():
    """Nine digits with a valid REGON checksum also look like a phone number; the guard's order
    of patterns decides, not the alphabetical order of the type names."""
    regon = Span(0, 9, "REGON", 1.0, "regex_pl")
    phone = Span(0, 9, "PHONE", 1.0, "regex_pl")
    assert merge_spans([regon, phone]) == [regon]
    assert merge_spans([phone, regon]) == [phone]


def test_merge_spans_keeps_touching_detections_apart():
    first = Span(0, 5, "PESEL")
    second = Span(5, 10, "PESEL")
    assert merge_spans([second, first]) == [first, second]


def test_merge_spans_uses_length_only_to_break_confidence_ties():
    short = Span(0, 5, "EMAIL", 1.0, "regex_pl")
    long = Span(0, 20, "SECRET", 1.0, "secrets")
    assert merge_spans([short, long]) == [long]


def test_merge_spans_keeps_non_overlapping_detections():
    first = Span(0, 5, "EMAIL", 1.0)
    second = Span(10, 20, "PESEL", 1.0)
    assert merge_spans([second, first]) == [first, second]


def test_a_string_that_is_not_a_token_has_no_entity_type():
    from polanon.policy import entity_type_for_token

    assert entity_type_for_token("<OSOBA_1>") == "PERSON"
    assert entity_type_for_token("zwykły tekst") == "UNKNOWN"
    assert entity_type_for_token("<NIEZNANY_1>") == "UNKNOWN"


def test_the_facade_sweeps_a_restored_answer():
    from polanon import Anonymizer, RegexGuard
    from polanon.policy import ENTITY_TYPES

    anonymizer = Anonymizer([RegexGuard()], enabled=set(ENTITY_TYPES))
    mapping = {"<PESEL_1>": "90010112349"}

    # A value that came back from the mapping is expected and says nothing.
    assert anonymizer.sweep("Sprawa 90010112349 zamknięta.", mapping).is_clean

    # A token nobody allocated means the model invented a reference.
    invented = anonymizer.sweep("Sprawa <PESEL_9> zamknięta.", mapping)
    assert invented.unknown_tokens == ["<PESEL_9>"]

    # Personal data in the answer that was never in the question is worth a warning.
    surprise = anonymizer.sweep("Sprawa 90010112349 i 85031523454.", mapping)
    assert [s.entity_type for s in surprise.new_spans] == ["PESEL"]
    assert not surprise.is_clean


def test_merge_spans_breaks_a_tie_by_the_first_report_of_a_value_seen_twice():
    """Two guards reporting the same REGON must not hand the tie to the phone rule between them."""
    regon = Span(0, 9, "REGON", 1.0, "regex_pl")
    phone = Span(0, 9, "PHONE", 1.0, "regex_pl")
    regon_again = Span(0, 9, "REGON", 1.0, "custom")
    assert [s.entity_type for s in merge_spans([regon, phone, regon_again])] == ["REGON"]


def test_a_secret_is_masked_whole_with_what_was_detected_inside_it():
    """'hasło: "jan.kowalski@firma.pl#!"' went out as "<EMAIL_1>#!", most of the password."""
    text = 'hasło: "jan.kowalski@firma.pl#!"'
    secret = Span(8, 31, "SECRET", 0.9)
    email = Span(8, 29, "EMAIL", 1.0)
    assert merge_spans([email, secret], text) == [secret]


def test_a_detection_reaching_out_of_a_secret_keeps_its_own_part():
    secret = Span(0, 10, "SECRET", 0.9)
    phone = Span(5, 20, "PHONE", 1.0)
    merged = merge_spans([secret, phone])
    assert [(span.start, span.end, span.entity_type) for span in merged] == [
        (0, 5, "SECRET"),
        (5, 20, "PHONE"),
    ]


def test_a_stored_list_of_types_keeps_only_real_ones():
    from polanon.policy import enabled_entities

    assert enabled_entities("PESEL, EMAIL,,PERSONS ,IBAN") == {"PESEL", "EMAIL", "IBAN"}
