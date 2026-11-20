# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[semantic versioning](https://semver.org/spec/v2.0.0.html); until 1.0, the second number goes up
only for a change that breaks something, as CONTRIBUTING.md describes.

## [Unreleased]

### Added

- **`polanon-gateway --workers N` masks N calls at once, on as many cores.** The gateway masked
  one call at a time in one thread, since the spaCy model is not shared safely between threads,
  so one process used one core, and with the default guards a second tool waited about 3 s for
  every 100 KB the first sent. With `--workers N` each call is masked in the next free of N
  processes of the gateway's own, each with its own anonymizer and model, and the gateway's
  process keeps serving, streaming and restoring; four calls at once took as long as one on a
  machine with 16 threads. A process gets the request through a pipe and sends back the masked
  request with the call's mapping, and forgets the values once it is masked. Each one builds its
  anonymizer from the gateway's options and reads `POLANON_KEY` from the environment it inherits,
  so the key is never pickled and stable tokens are the same in every process. The workers load
  the model before the gateway accepts calls, and a gateway whose workers cannot start does not
  start either; a worker that stops fails the calls being masked at that moment with a 503, and
  the next call starts new ones; each ends with the gateway, also one killed outright. Every
  worker holds a model of its own, about 330 MB with `pl_core_news_md`, while the gateway's own
  process then loads none. The default stays one call at a time in the gateway's process, as
  before. `Gateway` takes `build`, which makes each worker's anonymizer, and `GatewayConfig`
  takes `workers`. uvicorn's own `--workers`, a whole gateway in each process, was measured too
  and left aside: its processes share one socket and take calls as they come, so a busy one
  still took a second call while another sat idle, and on Windows one of them failed to start
  on the shared socket.

## [0.4.1] - 2026-11-19

A patch release: the fixes from an independent review of 0.4.0, each reproduced before it was
fixed, and a few additions. Nothing that worked stops working the same way, and the stable tokens
are computed as before, so an index needs no rebuilding. With the default guards a long text is
masked in time that grows with its length again, 100 KB in 2.9 s instead of 13 s, and a document
of 3,200 passages is cut in 1.5 s instead of 19 s. Numbered tokens take the spellings the stable
ones take for one value, and a value written with letters that only look Latin keeps its token.
The command line restores masked text with `--restore`, the gateway hands a client the call's
mapping when started with `--pii-map`, and CI runs on Windows too.

### Added

- **`polanon --restore` and `polanon --json`.** The command line masked but could not put the
  originals back: `--restore --mapping mapping.json` now restores masked text, such as a model's
  answer, from arguments, `--file` or standard input, a person named in another case included,
  and leaves a token the mapping does not hold as it is. `--json` writes one JSON object to
  stdout for a script, the text and the mapping unless `--mapping` takes it, instead of the text
  there and the mapping on stderr. A mapping file that is not one is refused without quoting it.
- **The gateway hands a client the call's mapping when the operator allows it.** Started with
  `--pii-map`, it answers a request with `"return_pii_map": true` with the mapping of the call,
  token to original, under `return_pii_map`: in the answer of chat completions, the Responses
  API, Anthropic's Messages API and embeddings, and in the last event of a stream, which for
  chat completions is a chunk of its own before `[DONE]`. A token the client wrote itself is left
  out. The provider never sees the field, a value that is not true or false is refused before
  anything is sent, and without `--pii-map` a request for the mapping is refused, since with
  stable tokens it tells which token stands for which value. Proposed by macieksonik in
  [#30](https://github.com/boromat/polanon/pull/30), with the reasons, the design and the first
  code; the request there to send the system prompt unmasked is answered by `--allow` and
  `--allow-file`, which leave the assistant's and its company's names readable.
- **The Haystack maskers take `mode` and `ml_filter`.** `PolanonDocumentMasker` and
  `PolanonQueryMasker` built the spaCy guard with its defaults only, so a pipeline could not ask
  for the paranoid mode or leave the decision to the hand-written filters. Both now pass them on
  as `default_guards` takes them, refuse a wrong value when they are built, and keep them in a
  saved pipeline; a pipeline saved before loads with the defaults.
- **CI runs the tests on Windows.** The package says it runs on any OS, and its command line
  reads Windows's code pages and UTF-16 and writes UTF-8 to the Windows console, but every job
  ran on Linux. A job on `windows-latest` now runs the whole suite with the spaCy model.

### Fixed

- **A long text is masked in time that grows with its length, not with its square.** The ML
  filter of 0.4.0 read the whole text again for every span the spaCy model found: whether the
  text is English, how often the span stands in it, and the language of the span's line, which
  in a text on one line, as a tool's JSON is, is the whole text. 100 KB of Polish prose took
  13 s to mask instead of 2.5 s without the filter, and the gateway masks one call at a time.
  Each of these is now read once per text and per line, and 100 KB takes 2.9 s. What is masked
  does not change.
- **A PESEL right after its label is masked whole however its digits are grouped.** Only the
  whole number, the birth date apart from the serial number and one digit per box were read as
  a PESEL. Grouped 3-3-3-2, "PESEL 508 101 123 54" went out as `PESEL <TELEFON_1> 54`, its last
  two digits in clear, and split after the fifth digit, "PESEL: 50810 112354", it was not masked
  at all. After a PESEL label eleven digits split by single spaces or dashes anywhere are one
  PESEL; without the label such groups still are not, since an order number has their shape.
- **The command line prints the masked text before the mapping.** Standard output is buffered
  and standard error is not, so in a terminal, or with both sent to one file, the mapping came
  first. The masked text is now flushed before the mapping is written.
- **`--mapping` narrows a file that already exists to its owner.** The mapping, which holds the
  originals, was created readable by its owner only, but a file already there kept its mode, so
  one left at 0644 by an earlier run or another tool stayed readable by everyone on the machine.
  It is now set to 0600 before anything is written to it, where the OS has such modes.
- **A document in many passages is cut in time that grows with its length.** `mask_chunks` and
  `index_chunks` mask the passages as one text and cut it apart again, and each passage read
  every masked range of the whole document: 800 passages took 1.4 s, 3,200 took 19 s. Each
  passage now finds its own ranges by halving, and 3,200 take 1.5 s. The test that guarded this
  with a limit of 5 s for 800 passages failed under `pytest --cov` on slower machines, and
  passed while the time still grew with the square; it now checks that four times the passages
  take less than eight times as long.
- **A value with a token is masked when written with letters that only look Latin.** "Jan
  Kowalski" with a Cyrillic a and o in it reads as the same name to a person and a model, but
  it went out in clear, in the next text and in the same one, and the gateway's last check
  before a call leaves did not see it either. The search for values with a token now reads the
  Cyrillic and Greek letters that look Latin as the Latin ones, and such a spelling keeps the
  value's token. The stable tokens are computed as before, so an index needs no rebuilding.
- **Two spellings of one value share a numbered token where they share a stable one.** With
  numbered tokens "jan.kowalski@example.com" and "jan.kowalski [at] example.com" were
  `<EMAIL_1>` and `<EMAIL_2>`, and so were a phone with and without +48 and an account with
  and without "PL", while the stable tokens took each pair for one value. A value now keeps its
  token in every spelling its stable token covers; a person named in another case still gets
  the token with that case. The keys behind this and the lookalike letters are worked out once
  per call and forgotten with the call's other values, so a call of 400 KB of an AI agent's
  history is masked as fast as in 0.4.0.
- **A KRS number after its label is a KRS number.** One ten-digit number in eleven passes the
  NIP checksum, and "KRS 0000012345" went out as `KRS <NIP_1>`: masked, under the wrong name. A
  number right after a KRS label is now left to the KRS rule, unless a NIP label stands before
  it.
- **The ML filter reviews spans only with the spaCy model it was fitted on.** Its weights read
  the word vectors of `pl_core_news_md`, but it ran with any Polish model: on the vectors of
  `pl_core_news_lg`, another space, and on `pl_core_news_sm`, which has none, where the weights
  of the vectors silently counted for nothing. With another model the hand-written filters now
  decide alone, as with `ml_filter=False`; with `pl_core_news_md` nothing changes.
- **A zero-width space between a first name and a surname reads as a space.** The guards drop
  invisible characters, so "Jan" and "Kowalski" with only a zero-width space between them were
  read as one word, which no rule takes for a person, and only the spaCy model masked them.
  Where such a space stands between a lowercase letter and a capital, it is now read as the
  space it stands for; inside a word it is still dropped.
- **The tests make no network calls with Haystack installed.** Haystack's telemetry, on by
  default, looked up its server four times during `tests/test_haystack.py`, while CONTRIBUTING
  promises tests without network. The tests now switch it off, and the README and the Haystack
  page say how to switch it off where nothing but the masked calls should leave the machine.
- **A stream the provider breaks off keeps its last words.** When the provider broke off a
  streamed answer with a connection error, the gateway ended the stream without the text it was
  still holding back in case it began a token, so an answer that stopped on "Tak, <OSOBA"
  reached the client as "Tak, ". It now sends that text first, as it already did for a stream
  that ends without finishing. Nothing leaked, since what is held back is still masked.

## [0.4.0] - 2026-11-18

A minor release, since masked text changes shape: a person named again in another case goes out
as `<OSOBA_1:celownik>`, which `TOKEN_PATTERN` does not match. Code that reads masked text for
its tokens should use the new `CASED_TOKEN_PATTERN`; restoring needs no change, as every restore
already understood such tokens. The spaCy guard gets an ML filter, on by default: on OpenPII's
test part polanon still masks 3,733 of 5,255 values whole and raises 306 false alarms instead of
434, eight times fewer than Presidio's 2,641, and `ml_filter=False` switches it off. Logins
written from a name are masked whole, with their surname, and English words that are forms of a
first name stay words.

### Added

- **An ML filter for the spaCy guard, on by default.** In the normal mode a logistic regression
  drops more of what the Polish model reads as a name; `default_guards(ml_filter=False)` or
  `NerGuard(ml_filter=False)` leaves the decision to the hand-written filters alone. It weighs the span's shape, its words and the words around
  it, their vectors in the spaCy model, and whether a rule found something there, once every
  guard has read the text. It was fitted on OpenPII's Polish rows and KPWr's train part, which
  no benchmark reads, chosen by cross-validation by document, and its thresholds were set end to
  end on dev parts, with a margin. On a part held out until it was done, measured once, it masks
  the same 4,878 values whole and leaves 400 spans outside the annotations instead of 539; on
  OpenPII's test part, 306 instead of 434, the same 3,733 values; KPWr, the agent's calls and the
  sample documents are unchanged. It costs about half a millisecond a text, against twelve for
  the spaCy model, and it learns nothing from the texts it masks. `benchmarks/ml_filter_data.py`,
  `ml_filter_train.py` and `ml_filter_tune.py` rebuild it, and `run.py`, `run_agent.py` and
  `run_external.py` take `--no-ml-filter` or `--without-ml-filter` to compare. The masking reads
  its weights in plain Python; numpy only fits them. With it, polanon has 306 false alarms on
  OpenPII's test part to Presidio's 2,641, eight times fewer.

### Changed

- **One person is one number in every case, also without a key.** Each case of a name went out
  as a token of its own: "Jan Kowalski złożył wniosek. Proszę odpisać Janowi Kowalskiemu" became
  `<OSOBA_1> złożył wniosek. Proszę odpisać <OSOBA_2>`, and the model read two people. A form of
  a person who already has a token now takes that token with its case after a colon,
  `<OSOBA_1:celownik>`, so the model sees one person in the case the text used. Restoring already
  understood such tokens and declines the name back; a case is written only where declining the
  kept form gives exactly the text's spelling, so restoring the masked text still gives the text
  back, and any other form, one in capitals for instance, keeps a token of its own. The mapping
  holds no new entries for the other cases. Masked text may now hold such tokens:
  `CASED_TOKEN_PATTERN`, new in the package, matches a token with a case or without, where
  `TOKEN_PATTERN` matches one without. Text masked before and sent again keeps them whole, the
  token instructions tell the model what they mean, and a stable token from a key carries no
  case, since an index searches for it as it is.

### Fixed

- **English words that are forms of a first name stay words.** Three of the names 0.3.9 took
  from the PESEL register have a form that reads as an English word once a chat drops its
  Polish letters: Alona's "Alonę" is "alone", Nika's "Nice" is "nice" and Maxim's "Maxima" is
  "maxima". In lowercase text, an agent's English tools and logs among it, "alone", every word
  holding it such as "standalone", and "nice" were masked as people. Nothing leaked, but the
  model read a token where a word stood. Such forms are now treated as "Kalina" is, a name next
  to its surname and a word on its own, and every form of the register's names was checked
  against the English of Python's documentation. No benchmark loses a value, and on OpenPII one
  span outside the values goes.
- **A login written from a name is masked whole, and so is the surname in it.** "jan.nowak",
  "anna.kowalczyk", "t.malinowski" or "jan_kowalski", the way an account, a home directory or an
  address book writes a person, went out as `<OSOBA_1>.nowak`: the first name was masked and the
  dot ended it, so the surname reached the model next to the person's token, and a check that
  looks for the masked values in the outgoing text could not see it. A lowercase login of a first
  name or an initial, one dot, "_" or "-", and a word that reads as a surname is now one person.
  The surname in it, and in the local part of an e-mail address shaped the same way,
  "anna.kowalczyk@example.com", joins the known values, so "Kowalczyk" further down and the other
  logins made of it, "akowalczyk", are masked as well. A surname of five letters now gives a
  login when a dot, "_" or "-" stands next to it, "nowak_j"; glued straight on it still takes
  six, so "nowakowie" stays a word.
- **The label of a field no longer goes inside a person's token.** The Polish spaCy model reads
  a word of software English before a surname as a first name: "Login Kowalski", "Email Nowaka"
  and "Admin: Ola Kowalska" came back as one person each, so the model saw `<OSOBA_1> nie
  działa` where the text said that a login does not work. Nothing leaked, but the label was
  hidden with the name. Such words at the front of a person are now cut off, and the rest of the
  span is masked as before. The list of those words leaves out every word that is also a first
  name or a surname, so "Grant Nowak" stays whole, and the paranoid mode keeps the span as the
  model found it.

## [0.3.9] - 2026-11-17

What public Polish data showed polanon missing, fixed, and a benchmark that keeps measuring on
data polanon was never tuned on. On 1,000 Polish texts of ai4privacy's OpenPII that no fix was
made from, the values masked whole with `pl_core_news_md` rise from 63.2% in 0.3.8 to 71.0%,
with 435 spans outside them instead of 432; Presidio, set up the same way, masks 65.8% with
2,641. Driving licence numbers are a new type, e-mail addresses, streets and identifiers with a
typo after their label are masked whole, and 630 first names of the PESEL register are known
without a model. Nothing that worked before stops working, so this is a patch release, as
CONTRIBUTING.md describes.

### Added

- **Driving licence numbers, after their label**, as the new type `DRIVING_LICENCE` with the
  token `<PRAWO_JAZDY_1>`: "Numer prawa jazdy: 01234/15/1234", "driver's license number:
  D1234-5678-9012". No rule knew them, and an HR record or a rental agreement sent them in
  clear. The shape differs by country, so the label is what makes a run of letters and digits
  one, and it must hold a digit: "prawo jazdy kategorii B" names no number. The number may
  stand in brackets or after a dash, after "kat. B" or a remark such as "(jeśli dotyczy)",
  and a tachograph's driver card, "karta kierowcy", is read the same way.
- **First names from the PESEL register**: the 630 names borne by at least 1 000 people in
  Poland that the hand-written lists lacked, from the register's lists of 22 January 2025 that
  the Ministry of Digital Affairs publishes on dane.gov.pl under CC0. The names polanon knows
  without a language model now cover about 98% of women and 96% of men in the register, up
  from 87% and 86%: Kazimiera, Sylwester, Marlena, and the Ukrainian and Belarusian names
  Poland's residents bear, Oleksandr, Tetiana, Nataliia, written as the register writes them.
  A form that is also an ordinary word, "inna", "melisa", "sylwestra", "idę", is treated as
  "Kalina" is, and Max, Mark, Frank, Roger, Pascal, Christian and Aurora are left out as the
  English words an agent's tools and logs are full of. Without the model, people found whole in
  the training part of CLARIN-PL's KPWr rise from 39% to 45%, and in the agent benchmark
  "Erik Holm" opening an English sentence no longer leaks: 2 of 243 values reach the model with
  `pl_core_news_md`, down from 3.
- **A benchmark on public Polish data polanon was never tuned on**, `benchmarks/run_external.py`.
  It downloads the Polish rows of ai4privacy's OpenPII (CC-BY-4.0), the test part of CLARIN-PL's
  KPWr (CC-BY-3.0) and GUS's TERYT street catalogue (CC-BY-4.0) on first use, keeps none of them
  in the repository, and scores polanon, and Presidio if asked, as `benchmarks/run.py` does.
  OpenPII is shuffled with a fixed seed: the first 1,000 Polish texts were read to find what the
  fixes above fix, the next 1,000 never were. On those, with `pl_core_news_md`, polanon masks 71%
  of 5,255 values whole with 435 spans outside them, and Presidio 66% with 2,641; on KPWr both
  find 90% of people.

### Changed

- **The README shows polanon at work and how it compares.** The terminal demo, now an animated
  SVG drawn by `benchmarks/terminal_demo.py`, shows the gateway in Docker next to the library
  and ends on the scores against Presidio. `benchmarks/comparison_chart.py` draws those scores
  as a chart under "How it compares", and a button at the top opens the demo that runs polanon
  in the browser, with nothing sent anywhere.

### Fixed

- **An e-mail address with letters from outside Polish is masked whole.** Only Polish letters
  were accepted next to the Latin ones, so `françois.dupont@example.com` was masked from its "ç"
  on and "franç" went to the model; a German, Czech or Scandinavian name in an address lost the
  same way, and one made of such letters alone was not masked at all. Every letter of any
  alphabet now counts, in the local part, the domain and the bracketed `[at]` form. Found by
  measuring on the Polish rows of ai4privacy's OpenPII, where 24 of 537 addresses leaked so.
- **A NIP, ID card or payment card number right after its label is masked with a typo in it**,
  as a PESEL already was. Without the label the checksum decides, and a number one keystroke
  off its real value after "NIP:", "numer dowodu osobistego:" or "numer karty kredytowej" went
  out in clear. "Numer ubezpieczenia społecznego", which a translated form writes for the PESEL,
  is read as a PESEL label too. The same numbers without a label are left as they were.
- **A street with "nr" before its house number, and a street in form fields, are masked.**
  "przy ulicy Długiej nr 36" or "al. Mickiewicza numer 228": the "nr" ended the street, and the
  street with its number went out in clear. A form or a CRM export writes the street and the
  number in fields of their own, "Ulica: Kwiatowa", "Numer domu: 8A", "house number: 5", with
  no "ul." and no postcode next to them; such a field is masked once it has its colon, so
  "ulica handlowa nr 1" in a sentence stays as it is.
- **A name with initials is one person**: "George W. Bush" went out as `<OSOBA_1> <OSOBA_2>`
  and "C. Alexander i S. Ishikawa" as four people, since one detector stopped at an initial and
  another started after it. Nothing leaked, but the model read two people where there was one.
  Two person spans one space apart are now joined when the space follows an initial or comes
  right before one; two names side by side without an initial stay two.
- **Streets as the TERYT register names them are masked whole.** Every one of the 50 449
  street names in GUS's TERYT catalogue of 16 November 2026, written with its house number,
  was run through the rules, and 91.4% came out masked whole. Missed were numbers in a name
  ("ul. 1000-lecia", "10-go Pułku Piechoty", "83. Pułku Strzelców"), initials ("ul. A.
  Słonimskiego"), titles beyond the usual few ("inż.", "mgr", "adw.", "kmdr"), small words and
  years ("Nad Stawem", "Bohaterów Wojny 1920 r."), a name in quotes or opened by "im.", and
  the kinds "Wybrzeże", "Nabrzeże", "Trakt", "Pasaż", "Zaułek" and "bulw.". Now 95.3% are,
  98.0% of those with "ul.", with no new span on any other benchmark.

## [0.3.8] - 2026-11-17

The gateway as a Docker image, for a tool that is already written: one command, and the tool
gets a base URL and a key. The library is the same as in 0.3.7, so this is a patch release, as
CONTRIBUTING.md describes. `docker run ghcr.io/boromat/polanon-gateway:0.3` starts the gateway
with the spaCy model inside, and the README now opens with it and with the library side by side.

### Added

- **The gateway as a Docker image**, `ghcr.io/boromat/polanon-gateway`, with the spaCy model
  inside, for Intel and ARM machines, published with every release and tagged with the version,
  the minor version and `latest`. In the container the gateway listens on every interface, so it
  starts only with `POLANON_GATEWAY_KEY`; it runs as a user without privileges and reports its
  health to Docker. CI builds the image and tries it on every change.
- **Two ways to start, at the top of the README**: the gateway in Docker for a tool that is
  already written, Claude Code among them, and the library for one's own code.

## [0.3.7] - 2026-11-17

Anthropic's Messages API in the gateway, names the operator leaves readable, an account number
that no longer comes out half a card, and a name a tool result gives late that is masked where
it came earlier too; nothing that worked before stops working, so this is a patch release, as
CONTRIBUTING.md describes. `polanon-gateway` now masks `/v1/messages` too, streamed or not, for
an app on Anthropic's SDK or Claude Code, `--allow` and `--allow-file` leave the assistant's
own name and its company readable while the rest of the prompt is masked, and an agent's call
the last check refused for such a name now goes out masked.

### Added

- **The gateway speaks Anthropic's Messages API.** `polanon-gateway --upstream
  https://api.anthropic.com/v1` serves `/v1/messages`, streamed or not, and
  `/v1/messages/count_tokens`, for an app on Anthropic's SDK or Claude Code pointed at it. The
  system prompt, the messages, the tools with their `input_schema` and the metadata are masked;
  a tool's input comes back restored, in a stream too, and the note about tokens joins the system
  prompt. A thinking block passes as the model wrote it both ways, since the provider checks its
  signature, and the last check still searches it. The client's key passes on as it came, as
  `x-api-key` or as a bearer token; the operator's `POLANON_UPSTREAM_KEY` goes as `x-api-key`
  instead, and a gateway key is accepted either way and never sent on. A base URL ending in
  `/v1`, the one an OpenAI client is given, works too, although Anthropic's SDKs add `/v1`
  themselves and asked for `/v1/v1/messages`. `mask_messages_request`,
  `restore_messages_response` and `MessagesStream` in `polanon.gateway.payloads` do it without
  a server.
- **`polanon-gateway --allow NAME` and `--allow-file PATH`** leave readable the names the model
  has to see as they are: an assistant that speaks for a company has its own name and the
  company's in its prompt, and masked they made it introduce itself as a token. Unlike the
  well-known names, the operator's own release a person the name guard found too; a denylist
  entry stays masked, and everything else in the prompt is masked as before. The banner counts
  them. Asked for in [#30](https://github.com/boromat/polanon/pull/30), in place of sending the
  whole system prompt unmasked.

### Fixed

- **A value recognised late in a message is masked where it came earlier in it too.** An
  agent's tool returned JSON naming a person first in a list of tags, where nothing recognised
  the name on its own, and then in a sentence that did. The tags had been masked before the
  name was known, so the gateway's last check found it there and refused the call, and the
  agent got an error three times; a surname alone in such a place, which the check does not
  look for, would have gone out. `MaskedTurn.mask_messages()` now masks a message once more
  when the mapping grew while it was masked, not only after, the Responses API's input items
  too, and `mask_data()` does the same for a whole value. The guards remember what they found
  in a text, so the second look costs little more than the rules.
- **A path the gateway refuses is named in its 404 as the client sent it**, with its `/v1`, not
  as the gateway read it: a client that asked for `/v1/v1/messages` was told the gateway does
  not serve `/v1/messages`, the one path it had just been told it serves.
- **An account number with a typo is masked whole as an IBAN, and never as a card.** A request
  typed "konto PL61 1090 1014 0000 0712 1921 2974", a digit or two off a real account: the IBAN
  failed its checksum, and sixteen digits from its middle passed Luhn, so they were masked as a
  card while the country, the check digits and the last eight digits went out in the clear.
  After an account label, "konto", "nr konta", "rachunek bankowy", "IBAN" or "account", a
  number of its country's registered length, or the 26 digits of a Polish account, is now an
  IBAN whatever its checksum, as eleven digits after "PESEL" are a PESEL. Without a label the
  checksum still decides, but a card is no longer read out of a number shaped as an account.

## [0.3.6] - 2026-11-15

Two modes for the language model, a benchmark of an AI agent's calls, and far fewer words masked
for nothing; nothing that worked before stops working, so this is a patch release, as
CONTRIBUTING.md describes. The spaCy guard now leaves out what it reads as a name in English
prose, code and identifiers, a place typed in lowercase among them, so an agent's prompt, tools
and results stay readable: in the benchmark's calls 5 of 307 token uses hide nothing, against 110
of 410 in 0.3.5. `mode="paranoid"`, `polanon-gateway --paranoid` and `polanon --ner --paranoid`
mask all of it instead. A birthday under a JSON key and a login made of a known surname are now
masked, and the gateway can send images and files on unmasked when told to.

### Added

- **`polanon --ner --paranoid`** masks with the spaCy guard in `mode="paranoid"`. Without `--ner`
  the flag is refused, since it changes nothing without the model.

- **`polanon-gateway --paranoid`** masks with the spaCy guard in `mode="paranoid"`, and the banner
  says so. Without the model the flag changes nothing, and the banner says that too.

- **`mode="paranoid"` masks every name the spaCy model finds.** The filters that keep an AI
  agent's English, code and identifiers readable let a little through by design: a name opening
  an English sentence, a word in inline code, a place typed in lowercase. `default_guards(mode=
  "paranoid")` and `NerGuard(mode="paranoid")` keep every span with a letter in it instead. On
  the agent's calls of the benchmark only what no guard finds leaks, 1 of 243 marked
  occurrences against 3 in the default `mode="normal"`, and 386 of 696 token uses hide nothing,
  against 5 of 307. On the sample
  documents it masks the same 187 values, with 8 over-masked spans instead of 6. The rules are
  the same in both modes. `benchmarks/run.py` and `benchmarks/run_agent.py` take `--paranoid`.

- **A benchmark of an AI agent's calls, and of what comes back.** `benchmarks/agent_corpus.py`
  holds eight invented calls of a personal assistant as its client sends them to a model, seven
  to Chat Completions and one to the Responses API: an English system prompt with a Polish memory
  of its user, sixteen tool definitions, and the JSON its tools return from a smart home, a web
  search, a shell, a mailbox, a contact book and a file, with code, a traceback, a system log,
  keys, a lease with two PESEL numbers, and mail in German and Ukrainian among it. The 103 values
  that must not reach the model are marked where they are written. `benchmarks/run_agent.py`
  masks each call as the gateway does and counts what still reaches the model and the tokens
  that hide nothing, with their uses. It then restores the model's answer to each call, written
  with the call's tokens and the cases the Polish note asks for, as text, as a whole response and
  as a stream cut at every place, and scores each line and argument against what a reader should
  get back. With `pl_core_news_md`, 3 of 243 occurrences reach the model and 5 of 307 uses hide
  nothing, against 5 and 110 of 410 in 0.3.5; 39 of 40 lines come back exact and none wrong.
  `tests/test_agent_benchmark.py` holds all of it as bounds, and CI prints the tables.

- **The gateway can send what it cannot mask, when told to.** `polanon-gateway
  --pass-unsupported` sends an image, a file or another part it cannot mask to the provider as it
  is, unmasked, and masks the text around it, for an operator who wants the model to see a photo
  or a document sent to an agent. The banner says so. `MaskedTurn(unknown_blocks="keep")` and
  `mask_responses_request(pass_unsupported=True)` do the same in code. Without the flag such a
  call is still refused, and `--drop-unsupported` still leaves the content out; the two flags
  exclude each other.

### Changed

- **`WELL_KNOWN_NAMES` holds hardware an agent names**: Dell, Lenovo, Samsung, Roborock and HACS,
  the store of Home Assistant's add-ons. In the agent's calls of the benchmark "Dell" was masked
  as a company eight times in one call, "Roborock" as a person and "HACS" as a place.
- **`WELL_KNOWN_NAMES` holds the platforms an AI agent names**: OpenRouter, DeepSeek, Hugging
  Face, Ollama, LM Studio, Claude Code, Hermes Agent, Nous Research, Home Assistant and its "HA",
  GitHub, GitLab, Docker, Linux and Ubuntu. An agent driving a smart home named "HA" 117 times in
  one call, and the Polish model took it for a place each time: the model saw a token in every
  tool description and result in place of Home Assistant. With the list, as the gateway uses it,
  they go out as written.

- **spaCy reads a text once.** A chat sends its whole history with every call, and an agent its
  system prompt and the descriptions of its tools too: masking an agent's call of 105 messages
  took 28 seconds, 20 of them in spaCy reading some two thousand texts it had read in the call
  before. `NerGuard` now remembers what it found in up to 20 000 texts (`cache_texts`, 0 turns it
  off), under each text's SHA-256 digest, never the text itself; the same call took 13 seconds.
- **The search for known values is prepared once per set of values.** Every text of a call is
  masked and checked against the whole mapping, and the search for its few hundred values, with
  each value's keys, was built anew for each of some 2,700 texts. It is now kept for the latest
  values, and the gateway drops it after each call (`polanon._matching.forget`), so a call's
  values stay in memory no longer than the call. Masking the agent's call of 105 messages now
  takes 8 seconds and its last check 1 instead of 13 and 3.5.

### Fixed

- **The gateway's last check no longer refuses an agent's history over names from code.** With
  Home Assistant's tools, a word of a tool's JSON result the masking took for a name, "service"
  or "text", also stood in a JSON key, a tool call's id, a type such as "reasoning.text", a
  tool's name or the parameters a tool requires, which the masking leaves as they are, and every
  call was refused. The check now reads the payload as the masking does: a JSON text value by
  value, and those names from code, written without a space, not at all. Every other text, every
  value and a key with a space in it are searched as before.
- **A value found while earlier messages are masked once more** is masked in all of them. A
  message masked before the mapping last grew is masked again with the whole mapping, and doing
  so can make the mapping grow once more: in an agent's history of 115 messages a piece left over
  from a losing detection became a new value then, a message re-masked just before kept it, and
  the gateway's last check refused the call. `MaskedTurn.mask_messages` and the gateway's
  Responses API masking now repeat the pass until the mapping stops growing, at most four times
  (`polanon.turn.MASKING_PASSES`).
- **A number, a quote, a literal of code and a small English word** are no longer names. In an
  agent's tool results the Polish model tagged "2026" as a company, `File "` of a Python
  traceback and Python's "None" as people, and "has" and "before" on lines it read as Polish. A
  span of digits alone, one with a double quote, a literal such as "None", "True" or "null", and
  one English function word in lowercase are now dropped on a line of any language; the rules
  still find the numbers that matter. The benchmark is unchanged.
- **A model's English with a Polish quote in it** is read as English. An agent's model reasons in
  English and quotes the user in Polish, and one Polish letter in the quote made the whole line
  Polish, so "The user asks" and "the model" went out as tokens. A line is now Polish by its
  Polish words, a word with a Polish letter counting as one, and what stands in quotation marks
  is left out of the count. In a text that is plainly English as a whole, a model's reasoning or a
  manual, a line too short to show a language, such as "Possible causes" or "Max", is read as
  English too; in a Polish text it stays as it was. The benchmark is unchanged.
- **A piece of a longer word, a Python traceback and more small English words** are no longer
  names: "service" of `ha_call_service` was a company, "File" of `File "/srv/app.py", line 12` a
  place, and "while" in a Polish sentence a place too. A span glued to a letter, a digit or a "_"
  is dropped, but not after a line break written out as `\n`, as an escaped JSON string has it,
  so `Kontakt:\nAnna Lindqvist` still names her. The benchmark is unchanged.
- **A span that opens with an English small word** is no name on a line that shows no language
  either: a tool's description "That's the limit" gave the person "That's", masked 64 times in
  one agent's call once it was known; "This Week" was a company. The benchmark is unchanged.
- **A place or a company of one word in lowercase** is no longer masked, on any line. Polish
  writes both with a capital letter, and in lowercase the Polish model found English words of
  the manuals in an agent's tool results instead: "text", "context" and "service", masked 54, 25
  and 38 times in one call once known. The price is a place or a company typed in lowercase in a
  chat, "jadę do gdyni" or "pracuję w orlenie", which now goes out as written; a person in
  lowercase is still masked. The benchmark masks what it did, with one stray span fewer.
- **An English small word ends a name on every line.** A span that opens with one is dropped on
  an English or a Polish line too: "That's" stood after a dash in mid-sentence on an English line
  and was masked 76 times in one agent's call once known, and a Polish manual quoting English gave
  "Before" and "the model". A span that holds one in lowercase is cut there, and a piece stays
  when it has two capitalised words or a Polish first name: "Maria Lindqvist at Northwind
  Traders" gives both names, "tokens and passwords" none. "Will" and "May" are first names too,
  so a span they open with a capital letter is read as before.
- **A word in inline code is no name.** An agent's Polish answers named a button `Uncheck`, and
  the Polish model took it for a person on line after line: masked 85 times in one call once
  known. A span between two backticks of its line, a line break written out as `\n` ending one
  too, is now dropped, unless it holds a Polish first name, as the login `anna.nowak` does.
- **A span without a letter is no name.** Under a Python traceback in an agent's tool results the
  Polish model took the carets for companies and places, a Markdown heading's "####" for a
  company and the offset of a timestamp, "808+00:00", for a place. What used to be dropped for
  digits alone is now dropped for having no letter at all.
- **The English name of a language is no place.** A Polish manual in an agent's tool results
  spoke of "Polish" and "English" texts, and the Polish model took both for places, "Polish
  names" and "Polish-first" for companies, masked 33 times in one call once known. A place or a
  company that opens with the English name of a language or a nation is now dropped; a real one,
  "British Airways" or "Czech Republic", is no secret either. A person is left alone: "German" is
  a Polish first name too.
- **A word after a model number is no person.** An agent's tool results listed a laptop's "M4
  Max" and a vacuum's "S8 Pro", and the Polish model took "Max" for a person, masked 44 times in
  one call once known. A person right after a word of letters and digits is now dropped, unless
  it holds a Polish first name: "12A Maja Nowak" is a house number and a woman.
- **A first name that is also a word, before a common surname in a chat,** is masked with it:
  `maja nowak` and `marek nowak`, typed in lowercase, went out with only the surname masked. Nowak,
  Wójcik, Woźniak, Kowalczyk, Kaczmarek, Krawczyk and Mazur in lowercase now settle the first
  name before them, in agreement with it, and `10 marek w ofercie` is left alone. In text that
  writes names with capitals they settle nothing, so `ani Kowalczyk` stays "nor Kowalczyk", and
  `maja` after a day, "od", "do", "końca" or "połowie" is May, not Maja: `od 1 maja nowak` keeps
  its month. Nor does such a surname claim a word in the next cell of a table. A thirteenth
  benchmark document, `common_surnames.txt`, holds the cases. Contributed by louisss1016 in
  [#28](https://github.com/boromat/polanon/pull/28), closing
  [#13](https://github.com/boromat/polanon/issues/13).
- **A product's tier, an identifier of code and a word with a digit are no names.** In the
  agent's calls of the benchmark the Polish model took a vacuum's fan speed "Max" for a person,
  masked 57 times once known, Python's `JSONDecodeError` and `HTTPError` too, and an invoice
  number for a place. A span of one word is dropped when it is a tier such as Max, Pro or Ultra,
  ends as the name of an error does, holds a digit without a Polish first name, or is written as
  code writes a name, with two capitals inside, `MediaPlayerEntity`, or an abbreviation run into
  a word, `HTTPServer`. One capital inside is not enough: a handle or a small company's name,
  "AnnaLis" or "ZielonyDom", is written so too, and so are MacLeod and DeVito. A span
  is cut at a semicolon, a pipe and a tab as at a line break, so a CSV row read as one company
  gives its cells apart. Token uses that hide nothing fall from 83 of 382 to 22 of 321, with the
  same values leaking, and the sample documents score as before.
- **The English of software is no name.** An agent's answers name buttons and settings in
  English inside Polish sentences, and the Polish model took "Uncheck" for a person, masked 48
  times in one call once known, "Authorization", "Scheduling" and "Privacy Policy" too. A span
  whose every word is a word of menus, settings and manuals, on a list of some 670 written for
  this library, is dropped unless it holds a Polish first name. Words that are also first names
  or surnames, such as Price, Grant or Page, are left off the list, so Hugh Grant is a person.
- **Chinese, Japanese and Korean text is no name.** A page in Chinese in the agent's calls of
  the benchmark gave the Polish model a "person" of six Chinese characters, and "Home
  Assistant。" with the full stop, which the list of well-known names no longer matched. A span
  whose every letter is Chinese, Japanese or Korean is now dropped, and the punctuation of those
  languages is trimmed off the ends of every span, in either mode, as a space is.
- **The parties of a contract are no names.** A Polish contract writes its parties with a
  capital letter, as defined terms, and in the lease of the benchmark the Polish model took
  "Najemcy" for a company. Najemca, Wynajmujący, Zleceniodawca, Zleceniobiorca, Zamawiający,
  Wykonawca, Sprzedający, Kupujący, Pracodawca, Pracownik, Strony and the like, in their cases,
  join the form words a span made only of is dropped; the person who is a party is still masked.
- **A birth date after "birthday", "birth_date" or "Geburtsdatum" is masked.** A contact book in
  the agent's calls of the benchmark gave the label "birthday", and the dates after it went out
  in the clear. The rule for birth dates now also reads "birthday", "birth date", "birth_date",
  "date_of_birth", "dateOfBirth", "Geburtsdatum" and "urodziny", with a quote or "=" between
  the label and the date, as code and JSON write them.
- **A date under a key that names a birthday is a birth date.** A tool's JSON is masked value
  by value, so the guards saw "1961-11-21" without the key "birthday" before it, and the date
  went out in the clear; in the agent's calls of the benchmark two birthdays of a contact book
  did. `MaskedTurn.mask_data`, `mask_messages` and the gateway now read a value under such a
  key, "birthday", "birth_date", "dateOfBirth", "dob" or "data_urodzenia", after the label of a
  birth date. A date under any other key stays readable.
- **A login made of a known surname is the person.** A backup's settings in the agent's calls of
  the benchmark held "user.name=Tomasz Malinowski" and the directory "/home/tmalinowski": the
  name was masked, the login went out. A word in lowercase that is a known surname of six
  letters or more with an initial, a dot, "_", "-" or digits around it, "tmalinowski",
  "t.malinowski" or "malinowski85", is now masked with the name. Letters after the surname count
  only after a dot, "_" or "-", so "malinowskie" stays the adjective it may be, and the surname
  alone in lowercase is still left as it is, since Zima, Kot or Wilk are ordinary words too.

## [0.3.5] - 2026-11-14

A new way back for names and fixes for agents, and nothing that worked before stops working: a
patch release, as CONTRIBUTING.md describes. The Polish note of `token_instructions` now asks the
model to write a person's case after a token; a token without one restores as before. Where the
Polish model read English as names it now finds less, so an agent's system prompt comes out with
fewer tokens, and the gateway masks the prose of tools it used to send as written.

### Added

- **A person comes back in the case the model asks for.** A token stood for the form the text
  wrote, so "wniosek pracownika Jana Kowalskiego" gave an answer "Szanowny Panie Jana
  Kowalskiego". The model may now write the case after a colon, `<OSOBA_1:wołacz>`, and the name
  comes back declined: "Szanowny Panie Janie Kowalski". The Polish note of `token_instructions`
  asks for it, with `<OSOBA_1:mianownik>` in a tool's arguments, and every restore understands
  it: `MaskedTurn.restore` and `restore_data`, `Anonymizer.restore`, `StreamRestorer` when a
  token and its case arrive cut apart, the Haystack components and both APIs of the gateway, a
  tool's arguments staying valid JSON. Declined are names of one or two words the rules read for
  sure, a listed first name and a surname such as Kowalski, Nowak, Bednarek, Mickiewicz or Duda,
  written in capitals, in lowercase or without Polish letters as the value was. A surname alone
  that may be a man's or a woman's, a foreign name, a company and a place come back as written,
  and the case is dropped. A token without a case restores as before, and an invented one with
  a case is reported by `unknown_tokens` and `sweep` as the token it declines.

### Fixed

- **Ordinary English words** are no longer masked as names by the Polish model. Reading an agent's
  English system prompt through the gateway, it tagged "result", "NOT", "Safety" and "the user's
  own" as companies, people and places: 89 tokens in one call, a prompt the model could hardly
  read, and, since the same words stood in the tool descriptions, a call the gateway's last check
  refused. On a plainly English line, one with English words, few Polish ones and no Polish
  letter, a span the Polish model finds is kept only as a capitalised name in mid-sentence, such as
  "Maria Lindqvist" in "ask Maria Lindqvist for it". A name that opens an
  English sentence is left to the rules, which find a Polish first name anywhere. Polish lines,
  in lowercase or without diacritics, and an English spaCy model are untouched; the benchmark is
  unchanged.
- **Names from code and lines with no language** are no longer taken for people. An agent's
  tools have parameters such as `replace_all`, which the Polish model tagged as a person: the tool
  schema holds the name as a key, which cannot be masked, so the gateway refused every call. A
  span with a backtick, or with a "_" and no Polish first name, is now code on any line, so
  "jan_kowalski" stays masked. On a line too short to show a language, such as "Max results" or
  "Skills for", a span with a word in lowercase is kept only when it holds a Polish first name.
  A line counts as Polish when its Polish words outnumber its English ones, and "ten", "pan",
  "pod" and "ale", English words too, no longer count; after a colon on an English line, a name
  is kept: "Contact: John Smith". The benchmark masks what it did, with one stray span fewer.
- **The gateway masks the prose of tools and answer formats.** Tool descriptions went out as
  written, so a value they held reached the provider, and a value the messages held was found
  there by the last check, which refused the call: an agent with 18 tools could not get a single
  call through. Now the descriptions and titles of the tools, the legacy `functions`, a JSON-schema
  `response_format` and the Responses API's `text.format` are masked, and so are the values of
  `enum`, `const`, `default` and `examples`; a model answering with such a token gets it back in
  the arguments. Tool names, types and the names of parameters stay, so the provider still checks
  answers against the schema. A value found in a tool is masked in the messages, and one found in
  the messages in every tool.

## [0.3.4] - 2026-11-14

A new extra and fixes, and nothing that worked before stops working: a patch release, as
CONTRIBUTING.md describes. The gateway comes with `pip install "polanon[gateway]"`; `pip install
polanon` alone is unchanged and still needs nothing beyond the standard library. Where detection
missed or mislabelled a value it now differs, so a text may come out with other tokens: a first
name alone gives way to the language model's place or company, and a stray word left over where
two detections overlapped stays readable.

### Added

- **A masking gateway for tools that speak the OpenAI API**, run by whoever needs it on their own
  machine: `pip install "polanon[gateway]"`, then `polanon-gateway --upstream
  https://api.openai.com/v1`, and Hermes Agent, n8n, Open WebUI or LibreChat point their base URL at
  `http://127.0.0.1:8787/v1`. It serves `/v1/chat/completions` and the Responses API,
  `/v1/responses`, streamed or not and with tool calls, `/v1/embeddings` and `/v1/models`, and
  refuses every other endpoint, images and audio. With the Responses API the whole conversation
  comes with every call, as Hermes Agent sends it with `store: false`, and encrypted reasoning
  passes as it is; a call that leaves the conversation at the provider is refused. Every call
  is masked, checked once more for any original value before it leaves, and restored on the way
  back, a tool call's arguments as valid JSON. It keeps nothing on disk, listens on 127.0.0.1 and
  needs a key every client sends to listen anywhere else, takes keys from the environment only, and
  logs counts, never values. The library still imports no web framework: the gateway is a plain
  ASGI application with httpx, served by uvicorn.

- **A twelfth benchmark document**, a hospital discharge summary, health data being the most
  sensitive kind (#4): a patient with her PESEL, birth date, address and phone, her daughter, two
  doctors behind their titles and the hospital's address. Every value is made up, and the patient
  was born in 1847, so her PESEL belongs to nobody. The rules alone mask all of it but its two
  places; with `pl_core_news_md` all 18 values are masked, and the ward's name is read as a company.
  The corpus now has 187 annotated values: 161 masked with the rules alone (86%) and 184 with the
  model (98%), with 8 over-masked spans, the ward one of them.

### Fixed

- **A model's reasoning sent back with its answer** is masked by `MaskedTurn.mask_messages`:
  OpenRouter returns the reasoning next to the answer, in `reasoning` and `reasoning_details`, and
  a client that kept the answer sent it back with the history, restored, in the clear. Now
  `reasoning`, `reasoning_content` and the text and summary of every item of `reasoning_details`
  are masked, and encrypted reasoning passes as it is. Found by the gateway's last check, which
  refused the call, in a test against OpenRouter.
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
