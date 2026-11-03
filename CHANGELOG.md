# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **E-mail addresses with Polish letters** are masked whole: `zofia.żak@example.com` was masked
  as a name and half an address, and an address on an internationalised domain not at all.
  Contributed by ShahadatTest in [#14](https://github.com/boromat/polanon/pull/14).
- **A phone number after "tel." is labelled a phone**, not a REGON: one nine-digit number in ten
  passes the REGON checksum, and the model was told it was a company's register number.
  Contributed by ShahadatTest in [#22](https://github.com/boromat/polanon/pull/22); "kom."
  counts as a phone label as well, and "hotel" no longer does.

## [0.2.0] - 2026-11-03

Before this release a pre-release review went through the library looking for ways personal data
could get past it. Everything under *Fixed* is what it found; each fix comes with a test that
fails without it.

### Added

- `--file PATH` on the command line. polanon reads the file's bytes itself and tells UTF-8,
  UTF-16 and cp1250 apart, so Windows PowerShell 5.1 cannot garble Polish letters on the way in.
  Contributed by jay23606 in [#5](https://github.com/boromat/polanon/pull/5).
- `--mapping PATH` writes the mapping to a file as UTF-8 JSON, created readable by its owner
  only, instead of to stderr, which Windows PowerShell's `2>` wraps in error records.

### Fixed

- **A UTF-16 file went out unmasked.** Windows PowerShell 5.1 writes UTF-16 with `>`; decoded
  as cp1250 it was full of NUL characters that hid every value from the guards, and the command
  line printed it unmasked with exit status 0. Input is now decoded by its byte order mark,
  UTF-16 is recognised without one, a UTF-8 file with one cp1250 line is decoded line by line,
  and bytes that are none of these are refused with exit status 2.
- **The second of two IBANs went out whole** when they were grouped and written one per line or
  joined by "oraz".
- **A double surname was read by its last part only**, so "mają kowalską-nowak" went out where
  "mają kowalską" was masked.
- **A name on the cut of a long single-line text was missed by spaCy.** Text over 100,000
  characters with no line break was cut at exactly that length, through the middle of a name.
- **A card number after the year of a date**, "15.03.2024 4111 1111 1111 1111", went out whole.
- **Settings secrets named `SECRET_KEY`, `APP_KEY` or `DB_PASS`** were not recognised, and a
  labelled password containing "," or ";" was cut short with the rest sent.
- **A birth date with the year abbreviation glued on**, "ur. 15.03.1985r.", went out whole.
- **An address after a street word in an oblique case**, "przy ulicy Długiej 5", went out
  whole; KRS is now also found after "pod numerem" and "No.".
- **A denylist term copied from Word, a PDF or macOS** (an en dash, decomposed letters, a soft
  hyphen) matched nothing, not even the same text.
- **A judge error carried the HTTP error it came from**, so a logged traceback could show a
  proxy password or an API key. Judge replies with code fences inside their JSON, a brace in a
  sentence before it, or a quote that differs from the text only in its whitespace are now read
  instead of dropped, and a reply of a few thousand spaces no longer freezes the event loop.
- "Pana przesyłka została nadana" no longer masks the parcel as a person.
- Masked CRLF documents no longer come out as CR CR LF on Windows, and two rules that took
  quadratic time on long lines without spaces now run in linear time.

### Changed

- The README no longer recommends `Get-Content -Encoding UTF8 file | polanon` for Windows
  PowerShell 5.1: that pipe turns Polish letters into `?`. It recommends `--file` and
  `--mapping` instead.

## [0.1.1] - 2026-11-03

### Changed

- **Example data that belongs to nobody.** The PESEL numbers in the README, the examples, the tests
  and the benchmark had birth dates in the 1980s, 1990s and 2000s, so each could have been issued
  to a real person. They now carry birth dates before 1860, for which no PESEL was ever issued (the
  register started in 1979), or in the 2100s. Day, month, serial number and the checksum rule are
  unchanged, so detection is tested exactly as before, and the benchmark scores are the same.
  E-mail addresses and URLs on real domains (`firma.pl`, `b.pl`) moved to reserved ones
  (`example.com`, `.example`).

### Added

- A terminal recording at the top of the README, showing a document masked on the command line and
  an answer restored in Python.

## [0.1.0] - 2026-11-03

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
