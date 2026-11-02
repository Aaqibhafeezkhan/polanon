# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

The first release.

### Added

- **Reversible masking.** `Anonymizer` replaces personal data with tokens such as `<OSOBA_1>` and
  `<PESEL_1>`, `Anonymizer.restore` puts the originals back, and `StreamRestorer` does it while an
  answer streams, never splitting a token across chunks. Tokens are stable for as long as the
  mapping is passed along, and a token is never reused for another value. `anonymize_sync` serves
  scripts, web views and notebooks; `await anonymize` does the work in a worker thread.
- **Polish identifiers with checksums**: PESEL, NIP, REGON, ID card, IBAN and payment cards, plus
  KRS numbers and passports next to their label. `is_valid_pesel`, `is_valid_iban` and the
  per-country `IBAN_LENGTHS` can be imported from `polanon.guards.recognizers_pl`.
- **Contacts, devices and secrets**: e-mail addresses, phone numbers, street addresses and postcodes,
  IPv4 and IPv6, MAC addresses, VINs, vehicle registrations, birth dates (`BIRTH_DATE`), API keys,
  JWTs, private keys and password assignments. A secret is masked whole, with whatever was found
  inside it.
- **`PersonNameGuard`.** Polish first names in all seven cases, in lowercase, in capitals and without
  Polish letters, with the surname that agrees with them or after "pan", "pani", "dr". No language
  model needed.
- **`CompanyGuard`.** Companies by their legal form (`Sp. z o.o.`, `S.A.`, `sp. j.`, `s.c.`,
  `Fundacja`, `Kancelaria`), not by guessing a name word by word.
- **`InternationalGuard`.** Foreign invoices: EU, UK, Swiss and Norwegian VAT numbers, company
  register numbers, Bankgiro, Plusgiro, sort codes and account numbers, international phone numbers,
  foreign addresses and postcodes, companies by foreign legal form (Ltd, Inc, GmbH, AB), and names
  after a form of address, a label or a closing formula.
- **An optional language model.** `NerGuard` reads the NKJP labels Polish spaCy models emit
  (`persName`, `placeName`, `orgName`), which a pipeline configured for English labels never finds.
- **Every occurrence of a known value**, in every spelling a person reads as the same: capitals,
  line breaks and runs of spaces, no-break spaces, invisible characters, decomposed accents, and an
  identifier's groups written apart or together. `mask_known_values` applies a mapping to text
  masked earlier.
- **An optional second stage.** A judge reviews the *masked* text for what still identifies someone
  by context: `OllamaJudge` locally, `RemoteJudge` for testing, `MockJudge` in tests, and
  `judge_fallback` to decide what happens when it is down.
- **`Allowlist`** with `WELL_KNOWN_NAMES`: platforms and tools a model takes for organisations stay
  readable, in every Polish case form, without ever releasing an identifier.
- **`sweep`**, which reports tokens a model invented and personal data its answer introduced.
- **Choosing what to mask.** `DEFAULT_ENABLED` (every type but `DATE`), `ENTITY_TYPES`, and types of
  your own declared in `Guard.entity_types`. A name that is not a type raises instead of quietly
  switching masking off.
- **`polanon` on the command line**: masked text to stdout, the mapping to stderr.
- **A benchmark.** Ten synthetic documents, seven Polish and three foreign, with 164 annotated values;
  `benchmarks/run.py` scores full, partial and missed coverage, and Microsoft Presidio is measured on
  the same documents in `benchmarks/compare_presidio.py`. The tests turn the current numbers into
  floors.
- **Recipes for the Anthropic and OpenAI SDKs** in `examples/`, tested offline against the real SDKs.
- Python 3.11 to 3.13, no dependencies in the core, `ner` and `judge` extras, and type information
  (`py.typed`).
