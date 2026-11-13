# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[semantic versioning](https://semver.org/spec/v2.0.0.html); until 1.0, the second number goes up
only for a change that breaks something, as CONTRIBUTING.md describes.

## [Unreleased]

### Fixed

- **A surname on the next line or in the next cell** is masked without the language model:
  "z panem Janem\nKowalskim" in a hard-wrapped e-mail, "Jan;Kowalski" in a CSV export and
  "| Jan | Kowalski |" in a Markdown table kept the surname in clear (#15). On the next line it
  joins the first name; in the next cell it is masked on its own, so the row keeps its columns.
  Across a line or a cell the word must read as a surname, so "Imię: Jan\nNazwisko:", "Jan;Anna",
  "Jan;Kraków" and "Kamil\nNowak Transport" under a signature stay apart.
- **A place spelled like a first name** keeps the language model's label: "Wakacje na Kubie" went
  to the model as a person, since a first name outranked the model wherever the two overlapped
  (#16). A first name alone, with no surname, initial or form of address, now scores 0.8, below
  the model's 0.85, and is still masked as a person without the model. "Asia" alone is the
  continent, as "Maja" is the month; "Asia Kowalska" and "pani Asia" stay people.
- **A piece left over from a losing detection** is treated alike in every turn (#20). Masked in
  place in the first turn, it was a known value in the next, so the bare "NIP" that an address
  rule left behind went to the model as an address in every sentence. A piece with no digit of a
  type whose values always carry one, or with fewer than three letters and digits, now stays as
  written; any other piece, such as the name of a loose span around an e-mail address, is masked
  wherever it stands from the first turn on.
- **An academic title alone** stays readable: the Polish model tagged the "n." of "dr n. med." as a
  person. A span the model finds made only of titles such as dr, n., med., hab., prof., mgr, inż.
  and lek. is dropped, as form labels are; "dr n. med. Ewa Zawadzka" keeps the name masked.

## [0.3.3] - 2026-11-12

Fixes only, and nothing that worked before stops working: a patch release, as CONTRIBUTING.md
describes. More is masked than under 0.3.2, so a text may carry more tokens, and with
`WELL_KNOWN_NAMES` the names of a few public institutions stay readable.

### Fixed

- **A token-shaped number longer than six digits** in the input, such as `<OSOBA_50810112354>`,
  no longer sets the number of every later token, and one of thousands of digits no longer stops
  masking with a `ValueError`. A new token still skips every token the mapping holds, so
  `<OSOBA_999999>` followed by a new person never hands out one token twice. Contributed by Sonike
  in [#24](https://github.com/boromat/polanon/pull/24).
- **A spaCy model named as pip writes it**, such as `pl-core-news-md`, loads, and a model directory
  without `meta.json` and `config.cfg` is reported when the guards are built, without a download
  command that could not work. An installed model still wins over a file or folder of the same
  name in the working directory. Contributed by Sonike in
  [#27](https://github.com/boromat/polanon/pull/27).
- **A password named by a word or two before its colon**, "Hasło administratora: Zima2025!Lod",
  "hasło użytkownika admin:", "Hasło do panelu klienta:", is masked: only "do", "dla" or "od" and
  one word could stand there, so the value went out in clear. With words in between, the value
  must look like a password, so "Hasło administratora zostało zmienione." and a date after "Hasło
  ważne do:" stay as they are.
- **A PESEL with a typo right after its label** is masked: "PESEL: 15501416181", with a wrong
  check digit, and "pesel 78059938421", with day 99, went out in clear, since the checksum was
  checked before the label. Without the label the checksum still decides, so an order number of
  eleven digits stays as it is.
- **A feminine surname in another case** is masked after the full name: once "Natalia Bielska"
  was masked, "Bielskiej" and "Bielską" went out in clear, in the same text and in the next turn,
  since no case of a surname in -ska, -cka or -dzka but the nominative begins with it. In
  lowercase, "w bielskiej firmie" stays an adjective.
- **A street without "ul." and without a postcode** is masked after an address label, "Adres:
  Japczaka 17", "zam. Długa 5", "mieszka przy Długiej 5", and after "z", "na", "przy" or "do"
  with a town right after the number, "z Długiej 17 w Białymstoku". Both went out in clear.
  "Windows 11 w Polsce", "na Windows 11 w firmie" and "Excel 365 w biurze" stay as they are.
- **An address is found in linear time** in a long run of capitalised words: each word started a
  street that ran to the end of the run, so 5,000 words such as "Ulica Ulica ..." or "I I ..."
  took several seconds, and ten times as many a hundred times longer. A street's name now takes
  at most eight words and three titles, more than any Polish street needs.
- **ZUS, Płatnik and the institutions staff name every day** are on `WELL_KNOWN_NAMES`: with the
  allowlist, "Jak to rozliczyć w ZUS?" and "zna program Płatnik" went out as `<FIRMA_1>`, since
  the language model reads them as organisations. Added: ZUS, PUE ZUS, eZUS, Płatnik, KRUS, NFZ,
  GUS, KSeF, ePUAP, CEIDG and PFRON. The list releases names only, so no identifier is unmasked,
  and "pani Płatnik" stays a person.

## [0.3.2] - 2026-11-12

New features and fixes, and nothing that worked before stops working: a patch release, as
CONTRIBUTING.md describes. Without a key, tokens are numbered as before.

### Added

- **Stable pseudonyms** for search indexes and RAG: `Anonymizer(guards, pseudonyms=Pseudonyms(key))`
  computes every new token with HMAC-SHA256 under a secret key of at least 32 bytes, such as
  `<OSOBA_482193047556>`, instead of numbering it per conversation. The same value gets the same
  token in every document, process and question, with no shared state. A person's name is read in
  the nominative where its first name and surname agree, so "Jan Kowalski", "Jana Kowalskiego" and
  "z Janem Kowalskim" share a token, and an identifier is one value with or without its separators
  and country prefix. Tokens keep the `<LABEL_number>` shape, with 12 digits by default. The key is
  never shown, pickled or copied; `key_id` and `settings` say what an index was built with, and
  `scope=` keeps two indexes under one key apart. A collision gets four digits more and is recorded
  in `Pseudonyms(collisions=...)`.
- **Masking for an index**: `anonymizer.index_chunks(passages, store)` masks the passages of a
  document with one mapping and puts what each token hides into a `MappingStore`, a two-method
  protocol over the application's own encrypted table (`InMemoryMappingStore` for tests and
  examples), with a person's name in the nominative and a fingerprint to tell a collision.
  `anonymizer.mask_query(question, store)` gives a question the documents' tokens, also for a
  surname written alone, which no rule reads and the store keeps for every indexed person. `mask_chunks`, `stored_mapping`
  and `tokens_in` mask without a store and restore found passages and answers, also while they
  stream; each passage also comes with its mapping as written, to restore it exactly.
- **Haystack 3 components**, with `pip install "polanon[haystack]"`: `PolanonDocumentMasker` masks
  documents before the splitter and the embedder of an indexing pipeline, `PolanonQueryMasker`
  gives a question the documents' tokens, `PolanonRestorer` restores the replies and the documents
  found, and `PolanonStreamRestorer` restores a streamed answer as a streaming callback. Nothing
  that names a value reaches the document store: not the content, not `meta`, and no `id` that is
  a plain hash of an original text. What the tokens stand for goes to the application's mapping
  store only. The key is a Haystack `Secret`, read from `POLANON_KEY` and saved by name only.
  `examples/haystack_rag.py` runs offline; `examples/haystack_openai.py` uses OpenAI and is paid.

### Fixed

- **A token the mapping holds is left whole** when masked text is masked again, such as a passage
  from an index or history sent back. The digits of `<OSOBA_50810112354>` read as a PESEL, a
  language model read `OSOBA_8` as a place, a PIN label took the token after it for the PIN, and a
  known company called "Firma" was found inside `<FIRMA_1>`; each corrupted the token so restore no
  longer gave the value back. A placeholder that stands for itself still shelters nothing.
- **Titles stay out of a foreign name after a label.** "z. Hd. Frau Wiśniewska" made "Frau" the
  first name, so every later "Frau" in the letter was masked as her, and "Attn: Mr Oliver Bennett"
  got a token apart from the "Oliver Bennett" of its sign-off. With two titles, as in "Frau Prof.
  Anna Schmidt", the second one was masked as the person wherever it stood.

## [0.3.1] - 2026-11-11

### Fixed

- **A known value that only partly overlaps a new detection** is masked whole: "Nowak z
  Kowalskich" known from an earlier turn met the name rule's "Anna Nowak", and "z Kowalskich" went
  out in the clear. The same holds for two known values that overlap.
- **Two findings of the judge that overlap** are both masked. A small model that quoted "Ngozi
  Okonkwo-Brzęczyszczykiewicz" and "Okonkwo-Brzęczyszczykiewicz z kancelarii ..." had its first
  name sent, while `judge_hits` reported success.
- **A PIN that equals a year** stays out of two dates joined by "-" or "/", a span of years such as
  `2024/2025` and a date ending in `2025r.`: in `01.01.<SEKRET_1>-31.12.<SEKRET_1>` the dates around
  the token gave it away.
- **A known short number in one long line of numbers**, such as a minified JSON array, no longer
  makes masking quadratic: 50 KB took most of a minute, and now takes under a second.

## [0.3.0] - 2026-11-11

### Added

- **`MaskedTurn`**, from `anonymizer.turn()`: one object for everything a call to a model sends and
  gets back. It masks the history, the question and the results of tools with one mapping, so a
  value keeps one token across them, restores the arguments of tool calls and the answer, also
  while it streams, and reports invented tokens. Values the caller already knows are sensitive,
  such as its clients' names, go in `known=` and are masked wherever they appear. When a detector
  fails, every `mask` method raises `MaskingUnavailableError` and returns nothing half masked.
  PolCRM and PolSales each kept a copy of this as a `Masker` of their own.
- **`MaskedTurn.mask_messages()`** masks a whole history written for the Anthropic or the OpenAI
  API: text, `tool_use` input, `tool_result` and `"tool"` content, the arguments of `tool_calls`.
  Tool results and arguments written as JSON stay valid JSON, and ids and tool names are left
  alone. A value first recognised in a later message is masked in the earlier ones too. A block
  nothing can mask, such as an image or a model's signed reasoning, is refused, or left out with
  `unknown_blocks="drop"`.
- **`token_instructions("pl" | "en")`**: a note for the system prompt that tells the model what the
  tokens are and to copy them exactly, the one both apps wrote out for themselves.
- **`MaskingUnavailableError`**, the one error an app catches to refuse sending text it could not
  mask, and **`NerModelUnavailableError`**, the one of them raised when spaCy or its model is
  missing. **`ner_available()`** runs the same check as `default_guards()` without loading
  anything, for a settings page.

### Changed

- `default_guards(with_ner=True)` without spaCy or its model raises `NerModelUnavailableError`
  instead of a plain `ModuleNotFoundError`. It is a subclass of `ModuleNotFoundError` with the same
  message, so `except ModuleNotFoundError` and `except ImportError` still catch it. Through
  `MaskingUnavailableError` it is also a `RuntimeError`, so an `except RuntimeError` around
  `default_guards()` now catches it too, and a check of the exact type, such as
  `type(exc) is ModuleNotFoundError`, no longer matches. Nothing else in the 0.2.2 API changed:
  every name and signature is still there, and every benchmark document masks to the same text,
  tokens and spans, with the language model and without it.

## [0.2.2] - 2026-11-08

Found by masking synthetic CVs, profiles written as JSON and lists of repositories with the rules
alone.

### Added

- **Links to a person's profile** are a new built-in type, `LINK` (tokens `<LINK_1>`): LinkedIn
  (`linkedin.com/in/...`), GitHub, GitLab, Bitbucket, Codeberg, GitHub and GitLab Pages, Stack
  Overflow, Kaggle, Behance, Dribbble, Medium, Instagram, Facebook, X, YouTube and a few more.
  Only the link up to the handle is masked, so `<LINK_1>/weather-app` still names the project.
  A LinkedIn link carried the person's name, and a GitHub login reappeared in every repository.
  The paths of a site's own pages (`github.com/features`, `facebook.com/groups/...`,
  `youtube.com/watch`) stay readable. A guard that declares `LINK` as its own type keeps working.
- **The surname of a masked person and the handle of a masked profile link are masked on their
  own** wherever they appear in the text: "Kowalski" alone after "Jan Kowalski", and
  `"login": "jkowalski"`, `@jkowalski` and `jkowalski.github.io` after `github.com/jkowalski`, went
  out in the clear. A lone surname in lowercase stays an ordinary word, as any known one-word name
  does; a name only the language model found gives its surname only next to a first name the rules
  know, and the handles of well-known brands (`github.com/microsoft/...`) stay readable.
- **Popular first names missing from the list**: Oliwier, Cyprian, Maurycy, Celina, Ksenia,
  Lilianna, Matylda, Melania and Otylia, with their surnames: "Oliwier Nowak" went out whole.

### Fixed

- **An e-mail address with its "@" written out** is masked whole: `jan.kowalski [at] example.com`,
  `(at)`, `(małpa)`, also with `[dot]`. Only "jan" was masked, as a first name.
- **A street written without "ul."** is masked when a postcode follows it: in
  `Długa 5 m. 4, 00-238 Warszawa` only the postcode was. A word and a number with no postcode
  after them, such as `Windows 11`, stay readable.
- **A birth date written in English** is masked after its label: `Date of birth: 15 March 1985`,
  `DOB: March 15, 1985` and `born 3rd May 1985` went out whole. With `DATE` turned on, such dates
  are dates too.

## [0.2.1] - 2026-11-08

### Added

- **`polanon --version`** prints the installed version. Contributed by barkya28 in
  [#29](https://github.com/boromat/polanon/pull/29).
- **An eleventh benchmark document**, a monthly roster of surnames and periods, which keeps names
  beside ordinary numbers masked without a language model. The corpus now has 169 annotated
  values: the rules alone fully mask 86% of them and the rules with `pl_core_news_md` 98%.

### Fixed

- **E-mail addresses with Polish letters** are masked whole: `zofia.żak@example.com` was masked
  as a name and half an address, and an address on an internationalised domain not at all.
  Contributed by ShahadatTest in [#14](https://github.com/boromat/polanon/pull/14).
- **A phone number after "tel." is labelled a phone**, not a REGON: one nine-digit number in ten
  passes the REGON checksum, and the model was told it was a company's register number.
  Contributed by ShahadatTest in [#22](https://github.com/boromat/polanon/pull/22); "kom."
  counts as a phone label as well, and "hotel" no longer does.
- **A phone number written with a bare 48 in front** is masked whole: `48123456789`, as SMS
  gateways and CRM exports write it, went out unmasked, and in `48 123 456 789` the country code
  stayed in the clear. A valid PESEL that starts with 48 keeps its type, and the minutes of a time
  such as `10:48` are not pulled into the phone. Contributed by Aaqibhafeezkhan in
  [#26](https://github.com/boromat/polanon/pull/26).
- **A restored name with a case ending is not reported as new**: "dla <OSOBA_1>a" restored as
  "Jan Kowalskia" was flagged by the output sweep as personal data the model made up. A person,
  company or place known from the conversation now covers its Polish case endings, while a longer
  different name, letters glued on the left, and other types such as an address with a letter
  added are still reported. Contributed by bfs2021 in
  [#25](https://github.com/boromat/polanon/pull/25).
- **A month spelled like a first name is left alone in a date** written with dashes, slashes or
  dots: "Jan" in `15-Jan-2024`, `2024/Jan/15` or `15.Jan.2024` was masked as a person. The same
  separator has to stand on both sides, so a name beside an ordinary number, as in
  `Nowak Anna 03/2024`, is still masked. Contributed by ShahadatTest in
  [#23](https://github.com/boromat/polanon/pull/23).

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
