"""Score Presidio on the same annotated samples, set up for Polish with no own code.

Presidio is the usual answer to "how do I mask personal data before an LLM call", so it is the
baseline worth measuring against. It gets the same Polish spaCy model polanon uses, with the model's
labels mapped to Presidio's entities, and every built-in pattern recognizer that runs offline,
Presidio's own Polish PESEL recognizer among them, switched to Polish, with telephone numbers read
for the countries in the corpus. Dates are left out on both sides.

The scoring is benchmarks/run.py's, tilted towards Presidio where it differs: every finding counts
whatever its score, coverage is the union of all findings rather than what a conflict resolution
would keep, and findings of one type separated only by spaces are joined, as Presidio's anonymizer
joins them. With --lenient-phones the phone recognizer runs at leniency 0, which masks more, mostly
postcodes and account numbers read as possible telephone numbers.

One allowance is made for the corpus rather than for Presidio: its e-mail addresses use the reserved
top-level domain ".example", which Presidio checks against the public list and rejects. A real
address would pass, so addresses under the reserved domains are accepted here.

The comparison is not neutral ground for polanon: its guards were developed against these very
documents, and Presidio was not. A setup that adds an English or German model for the foreign
invoices, which needs further downloads, was not tried and would find some of their names.

Presidio is not a dependency of this project. To run the comparison:

    pip install presidio-analyzer
    python -m spacy download pl_core_news_md
    python benchmarks/compare_presidio.py [--markdown]
"""

from __future__ import annotations

import argparse
import inspect
import time
from typing import Any

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer import predefined_recognizers as predefined
from presidio_analyzer.nlp_engine import NlpEngineProvider
from run import report, run

from polanon import Guard, Span

#: Recognizers that call a cloud service, download a model, or need an extra library or a key.
#: Everything else Presidio ships is used.
_NEEDS_MORE_THAN_PRESIDIO = {
    "AzureAILanguageRecognizer",
    "AzureHealthDeidRecognizer",
    "AzureOpenAILangExtractRecognizer",
    "BasicLangExtractRecognizer",
    "GLiNERRecognizer",
    "HuggingFaceNerRecognizer",
    "LangExtractRecognizer",
    "MedicalNERRecognizer",
    "SpacyRecognizer",
    "StanzaRecognizer",
    "TransformersRecognizer",
}

#: The Polish model's labels as Presidio entities.
_SPACY_LABELS = {
    "persName": "PERSON",
    "placeName": "LOCATION",
    "geogName": "LOCATION",
    "orgName": "ORGANIZATION",
}

#: Presidio's entities as polanon's types; anything else counts as masked under another type.
_ENTITY_TYPES = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "ORGANIZATION": "ORG",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN_CODE": "IBAN",
    "IP_ADDRESS": "IP_ADDRESS",
    "MAC_ADDRESS": "MAC_ADDRESS",
    "PL_PESEL": "PESEL",
    "UK_POSTCODE": "POSTAL_CODE",
    "DE_PLZ": "POSTAL_CODE",
    "SE_ORGANISATIONSNUMMER": "COMPANY_ID",
    "US_BANK_NUMBER": "BANK_ACCOUNT",
    "DE_VAT_ID": "TAX_ID",
    "IT_VAT_CODE": "TAX_ID",
    "UK_VEHICLE_REGISTRATION": "PLATE",
    "DE_KFZ": "PLATE",
}


#: The countries the corpus's telephone numbers come from. Presidio's default list has none of them
#: but Britain, so a Polish user would set this.
_PHONE_REGIONS = ("PL", "GB", "SE", "CH", "DE", "US")

#: RFC 2606 domains the synthetic corpus writes its addresses under.
_RESERVED_DOMAINS = (".example", ".test", ".invalid", ".localhost")


class _EmailRecognizer(predefined.EmailRecognizer):
    """Presidio's e-mail recognizer, accepting the reserved domains as it accepts real ones."""

    def validate_result(self, pattern_text: str) -> bool | None:
        if pattern_text.lower().endswith(_RESERVED_DOMAINS):
            return True
        return super().validate_result(pattern_text)


class PresidioGuard(Guard):
    """Presidio's analyzer in the shape of a polanon guard, so the same scoring applies."""

    name = "presidio"

    def __init__(self, spacy_model: str, *, phone_leniency: int = 1) -> None:
        """Build the analyzer: the Polish model plus every offline pattern recognizer."""
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "pl", "model_name": spacy_model}],
                "ner_model_configuration": {"model_to_presidio_entity_mapping": _SPACY_LABELS},
            }
        )
        registry = RecognizerRegistry(supported_languages=["pl"])
        registry.add_recognizer(
            predefined.SpacyRecognizer(
                supported_language="pl", supported_entities=sorted(set(_SPACY_LABELS.values()))
            )
        )
        self.recognizers = _pattern_recognizers(phone_leniency)
        for recognizer in self.recognizers:
            registry.add_recognizer(recognizer)
        self.analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(), registry=registry, supported_languages=["pl"]
        )
        self.entities = sorted(set(self.analyzer.get_supported_entities("pl")) - {"DATE_TIME"})

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return Presidio's findings, joined across spaces the way its anonymizer joins them."""
        results = self.analyzer.analyze(text, language="pl", entities=self.entities)
        spans = sorted(
            Span(r.start, r.end, _ENTITY_TYPES.get(r.entity_type, "OTHER"), r.score, self.name)
            for r in results
            if r.end > r.start
        )
        return _joined_across_spaces(spans, text)


def _joined_across_spaces(spans: list[Span], text: str) -> list[Span]:
    """Join findings of one type that only spaces separate: "Nowak Transport Sp." and "z o.o."."""
    joined: list[Span] = []
    for span in spans:
        previous = next((s for s in reversed(joined) if s.entity_type == span.entity_type), None)
        gap = text[previous.end : span.start] if previous else ""
        if previous and previous.end <= span.start and gap and not gap.strip(" "):
            joined.remove(previous)
            span = Span(previous.start, span.end, span.entity_type, span.score, span.source)
        joined.append(span)
    return joined


def _pattern_recognizers(phone_leniency: int) -> list[Any]:
    """Return every built-in recognizer that needs nothing beyond Presidio, set to Polish."""
    recognizers = []
    for name in sorted(dir(predefined)):
        recognizer_class = getattr(predefined, name)
        if not name.endswith("Recognizer") or name in _NEEDS_MORE_THAN_PRESIDIO:
            continue
        if "supported_language" not in inspect.signature(recognizer_class).parameters:
            continue
        if recognizer_class is predefined.EmailRecognizer:
            recognizer_class = _EmailRecognizer
        if recognizer_class is predefined.PhoneRecognizer:
            recognizers.append(
                recognizer_class(
                    supported_language="pl",
                    supported_regions=_PHONE_REGIONS,
                    leniency=phone_leniency,
                )
            )
            continue
        recognizers.append(recognizer_class(supported_language="pl"))
    return recognizers


def main() -> int:
    """Score Presidio and print the same report as benchmarks/run.py."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacy-model", default="pl_core_news_md")
    parser.add_argument("--markdown", action="store_true", help="print a markdown table")
    parser.add_argument(
        "--lenient-phones", action="store_true", help="run the phone recognizer at leniency 0"
    )
    args = parser.parse_args()
    leniency = 0 if args.lenient_phones else 1
    guard = PresidioGuard(args.spacy_model, phone_leniency=leniency)
    started = time.perf_counter()
    totals, extra, documents = run([guard], union=True)
    elapsed = time.perf_counter() - started
    print(
        f"Presidio with {len(guard.recognizers)} pattern recognizers, {args.spacy_model} "
        f"and phone leniency {leniency}"
    )
    report(totals, extra, documents, args.markdown)
    print(f"{elapsed:.1f} s for the {documents} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
