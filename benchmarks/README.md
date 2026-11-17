# How well polanon detects

[`samples/`](samples/) holds thirteen synthetic documents. Ten are Polish: a CV, a VAT invoice, a
customer complaint, an IT support ticket, an employment contract, an internal IT note, chat
messages typed the way people type them, in lowercase and without Polish letters, a monthly
roster of surnames and periods, a hospital discharge summary, and chat lines with common
surnames after first names that are also words. Three are invoices from abroad:
British, Swedish and Swiss. [`gold/`](gold/) annotates the 190 values that
must be masked, and [`run.py`](run.py) reports whether each one was **fully** covered, because half
a masked PESEL is a leak rather than half a success. It scores the guards and the merging of
their findings, not the pass that masks every later occurrence of a value already found: through
`Anonymizer`, which also masks every later occurrence of a value it has found, the same documents
can only come out the same or better.

| Configuration | Fully masked | Partially masked | Missed | Over-masked spans |
|---|---:|---:|---:|---:|
| polanon, guards only, no model | 164 / 190 (86%) | 0 | 26 | 0 |
| polanon, guards + `pl_core_news_md` | **187 / 190 (98%)** | 0 | 3 | 6 |
| Presidio 2.2.366, set up for Polish with the same model | 127 / 190 (67%) | 28 | 35 | 53 |
| Presidio, phone recognizer at leniency 0 | 146 / 190 (77%) | 32 | 12 | 89 |

Presidio is the usual answer to "mask personal data before an LLM call", so it is the baseline.
[`compare_presidio.py`](compare_presidio.py) gives it the same Polish spaCy model, every offline recognizer it
ships switched to Polish, its own PESEL recognizer among them, and the corpus's phone regions, and
scores it in its favour: every finding counts whatever its score, coverage is the union of its
findings, and same-type findings separated by a space are joined as its anonymizer joins them. What
it misses is what it has no rule for: Polish names in their grammatical cases, companies by their
legal form, street addresses, Polish, Swedish and Swiss postcodes, Bankgiro and sort codes, and
secrets. At leniency 0 its phone recognizer gains 19 values, mostly postcodes and account numbers
read as telephone numbers. Most of its over-masked spans are findings at very low scores, which
count here because every finding does. A setup with an English or German model for the foreign
invoices was not tried and would find some of their names.

With polanon, identifiers, contact details, bank accounts, secrets and people are fully covered
without any model at all. Polish identifiers come from regex plus a checksum; foreign ones from the label or official
shape an invoice prints them in; people from `PersonNameGuard`, which knows Polish first names in
every grammatical case, and, on foreign letters, from a form of address, a label or a closing
formula. What the model adds is places and company names without a legal form.

<details>
<summary>Per entity type, with the model</summary>

| Entity | Annotated | Masked | Partial | Missed | Right type |
|---|---:|---:|---:|---:|---:|
| `PERSON` | 39 | 39 | 0 | 0 | 39 |
| `LOCATION` | 25 | 22 | 0 | 3 | 22 |
| `ADDRESS` | 17 | 17 | 0 | 0 | 17 |
| `ORG` | 17 | 17 | 0 | 0 | 17 |
| `POSTAL_CODE` | 16 | 16 | 0 | 0 | 16 |
| `EMAIL` | 15 | 15 | 0 | 0 | 15 |
| `PHONE` | 15 | 15 | 0 | 0 | 15 |
| `NIP` | 6 | 6 | 0 | 0 | 6 |
| `BANK_ACCOUNT` | 5 | 5 | 0 | 0 | 5 |
| `IBAN` | 5 | 5 | 0 | 0 | 5 |
| `PESEL` | 5 | 5 | 0 | 0 | 5 |
| `IP_ADDRESS` | 3 | 3 | 0 | 0 | 3 |
| `REGON` | 3 | 3 | 0 | 0 | 3 |
| `SECRET` | 3 | 3 | 0 | 0 | 3 |
| `TAX_ID` | 3 | 3 | 0 | 0 | 3 |
| `COMPANY_ID` | 2 | 2 | 0 | 0 | 2 |
| `MAC_ADDRESS` | 2 | 2 | 0 | 0 | 2 |
| `PLATE` | 2 | 2 | 0 | 0 | 2 |
| `PL_ID` | 2 | 2 | 0 | 0 | 2 |
| `BIRTH_DATE` | 1 | 1 | 0 | 0 | 1 |
| `VIN` | 1 | 1 | 0 | 0 | 1 |

</details>

**Read the number honestly.** Thirteen short documents are a regression check, not a research
benchmark, and the corpus was written by the same person who wrote the guards, who tuned them
against it; Presidio was never tuned on it. The misses are
instructive: the word *Kraków* is masked in one document and missed in another, and *Bristol* and
one *Zürich* are missed on the foreign invoices, because places come from a statistical model and not
a rule. So do the over-masked spans: *ksiegowosci* typed without Polish letters, which the model
takes for a place, the hospital ward *Oddział Chorób Wewnętrznych*, read as a company, and on the
foreign invoices country names, a Swedish line item, the month *August*, which a Polish model reads
as names, and the label phrase *Company Registered in England and Wales*, read as a company. Without
the model the discharge summary loses only its two places, *Białystok* and *Choroszcz*: the
patient, her daughter and both doctors, the PESEL, the birth date, the addresses, the phones and
the hospital's company name are all masked by rules.
[`tests/test_benchmark.py`](../tests/test_benchmark.py) turns these numbers into floors so they
cannot quietly get worse.

## An AI agent's calls

The documents above are what a person writes. An AI agent sends something else, and
[`agent_corpus.py`](agent_corpus.py) holds eight calls of an invented one, a personal assistant
on a home server, as its client sends them to a model: an English system prompt with a Polish
memory of its user, sixteen tool definitions, Polish questions and Markdown answers, and the JSON
its tools return from a smart home, a web search, a shell, a mailbox, a contact book and a file.
English stands next to Polish on one line, there is code, a traceback, a system log, a
configuration file with keys in it, a lease with two PESEL numbers, mail in German and Ukrainian,
a page in Chinese, identifiers such as `notify.mobile_app_pixel_7_tomka`, and every call sends
the history again. Seven calls use Chat Completions, one the Responses API. The 103 values that
must not reach the model are marked where they are written, and stand 243 times in the eight
calls; 27 more, such as a first name alone or the byline of a published review, may be masked or
not.

[`run_agent.py`](run_agent.py) masks each call as the gateway does and counts what still reaches
the model and the tokens that hide nothing, with the number of times the masked calls use them.
The second number is what over-masking costs the model: a button's "Uncheck" or a chip's "Max"
made a token is a word it cannot read, and a value masked once is masked wherever it stands.

| Configuration | Leaked | Tokens that hide nothing | Their uses |
|---|---:|---:|---:|
| polanon, guards only, no model | 35 of 243 | 0 | 0 of 266 |
| polanon 0.3.5, guards + `pl_core_news_md` | 5 of 243 | 37 | 110 of 410 |
| polanon, guards + `pl_core_news_md` | **2 of 243** | 5 | **5 of 309** |
| the same, `mode="paranoid"` | 1 of 243 | 240 | 386 of 697 |

Without a model the places, the companies without a legal form and the foreign names leak, and
nothing is masked for nothing. With the model one value leaks by design: "sopotu", a place typed
in lowercase in a chat, dropped since a place in lowercase stopped being one, so that "text" and
"service" of a manual are not. "Erik Holm" opening an English sentence, where a capitalised word
is read as English, leaked too until Erik joined the first names of the PESEL register. No guard
finds the second in any mode, a name written in Cyrillic.
Two birthdays under a contact's `birthday` key and a login made of a surname in a path,
`/home/tmalinowski`, leaked too until a key that names a birthday was read as its label and a
login made of a known surname was masked with the person. `mode="paranoid"` takes every name the
model finds and wastes more than half of the token uses, on "return", "message", "notify" and
"That's". On the thirteen documents it masks the same 187 values as the default, with 8
over-masked spans instead of 6. What the default still masks for nothing, five uses in all, is
three German nouns, the name of a speaker in a smart home and "thanks" in Ukrainian; before the
filters for products, code, the English of software, Chinese and the parties of a contract it
was 83 uses, 57 of them "Max".

Each call also has the model's answer, in
[`agent_corpus.ANSWERS`](agent_corpus.py): Polish lines and a tool call written with the call's
tokens, many with the case the Polish note of `token_instructions` asks for,
`<OSOBA_3:wołacz>`, and with the text a reader should get back written by hand, "Szanowna Pani
Renato Kubiak". The answer is put back three ways: as text, as a whole response, and as a stream
cut into pieces of 1 to 21 characters, so a token arrives cut at every place. Each of its 40
lines and arguments is exact, comes back as the call wrote it instead of in the case asked for,
or is wrong:

| Configuration | Text | Response | Stream (35, Chat Completions) |
|---|---:|---:|---:|
| polanon 0.3.5, guards + `pl_core_news_md` | 38 / 2 / 0 | 38 / 2 / 0 | 34 / 1 / 0 |
| polanon, guards + `pl_core_news_md` | **39 / 1 / 0** | **39 / 1 / 0** | **35 / 0 / 0** |
| the same, `mode="paranoid"` | 38 / 2 / 0 | 38 / 2 / 0 | 34 / 1 / 0 |

Nothing comes back wrong, and no token is left in an answer. What comes back as written is a
name polanon cannot decline for sure, "Olena Kovalenko" asked for in the genitive. A value that
leaked reached the model as it is, so the model writes its case itself and the line counts as
exact: the rules alone score 39 of 40, which says little.
[`tests/test_agent_benchmark.py`](../tests/test_agent_benchmark.py) holds these numbers as
bounds both ways.

## On data polanon was never tuned on

Documents written by the author of the guards say what the guards were tuned to do.
[`run_external.py`](run_external.py) measures polanon, and Presidio set up as above, on public
Polish data instead, downloaded on first use and never kept in this repository:

- the Polish rows of ai4privacy's
  [OpenPII](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m) (CC-BY-4.0),
  texts written by a language model with their values annotated. Shuffled with a fixed seed, the
  first 1,000 were read to find what polanon missed and fix it; the next 1,000, measured here,
  never were;
- the test part of CLARIN-PL's [KPWr](https://huggingface.co/datasets/clarin-pl/kpwr-ner)
  (CC-BY-3.0), 4,323 sentences of real Polish text, where people, cities, streets and companies
  are read;
- every street of GUS's [TERYT](https://eteryt.stat.gov.pl/) catalogue (CC-BY-4.0).

On the 1,000 OpenPII texts never read, with 5,255 values:

| Configuration | Masked whole | Spans outside the values |
|---|---:|---:|
| polanon, guards only, no model | 2,668 (51%) | 105 |
| polanon, guards + `pl_core_news_md` | **3,733 (71%)** | **306** |
| the same with its ML filter off | 3,733 (71%) | 434 |
| the same, `mode="paranoid"` | 3,780 (72%) | 868 |
| Presidio 2.2.366, set up for Polish with the same model | 3,456 (66%) | 2,641 |

| Value | Annotated | polanon, with the model | Presidio |
|---|---:|---:|---:|
| People | 1,558 | 79% | 70% |
| Streets with their house numbers | 701 | 64% | 33% |
| E-mail addresses | 554 | 99% | 99% |
| Cities | 503 | 78% | 81% |
| Phone numbers | 437 | 60% | 34% |
| Postcodes | 300 | 98% | 0% |
| Driving licences | 189 | 51% | 89% |
| Passports | 145 | 98% | 100% |
| PESEL, NIP, ID card and payment card numbers that pass their checksum | 74 | 100% | 85% |
| The same that fail it | 794 | 30% | 82% |

Most of OpenPII's numbers are random digits. A real PESEL, NIP, ID card or card number always
passes its checksum, so the two are counted apart: polanon masks every valid one, and one that
fails only after its label, where a typo is likely. Presidio reads random codes without a label
too, which wins it the driving licences, with patterns of low score that also take product codes
and order numbers. A span outside the values is not always an error: dates and ages are left out
of the values, so a masked date of birth counts there for both.

On KPWr only recall counts, since it names much more than these four kinds. Rules alone find 46%
of the people; with the same spaCy model both find 90% of 949, and of the 1,557 values polanon
masks 1,393 (89.5%) and Presidio 1,384 (88.9%), ahead on streets, 66 of 95 against 56, and
behind on companies, 63 of 76 against 65. Of the 50,449 streets in TERYT, written with a house
number in a form, before a postcode and in a sentence, the rules mask 95.3% whole in all three,
98.0% of those with "ul.".

## The ML filter

The normal mode drops what the Polish model reads as a name in English, code and identifiers by
rules written one at a time. The ML filter, on by default, adds a logistic regression that weighs,
for every span those rules keep, its shape, its words and the words around it, their vectors in the spaCy
model, and whether a rule of polanon found something there. It is fitted on data the benchmarks
above never use, and measured on data nothing else read:

- [`ml_filter_data.py`](ml_filter_data.py) runs the model over OpenPII's Polish rows and KPWr's
  train part and writes every span it finds with its features and a label taken from the
  annotations. OpenPII's rows after the two parts above are split into 6,000 that train and
  1,290 held out; KPWr's train part is split by document, a tenth held out and a tenth for
  tuning. A first name OpenPII left unannotated teaches nothing.
- [`ml_filter_train.py`](ml_filter_train.py) chooses the inputs and the penalty by five-fold
  cross-validation, the folds cut by document, and writes the weights to
  `polanon/guards/_ml_filter_weights.py`.
- [`ml_filter_tune.py`](ml_filter_tune.py) sets the threshold of each type end to end on the dev
  parts, the highest with which no value is masked whole less often than with the filters alone,
  times 0.75, and measures the held-out part once.

| | Masked whole | Spans outside the values |
|---|---:|---:|
| OpenPII held out, 1,290 texts: the hand-written filters alone | 4,878 | 539 |
| the same with the ML filter | 4,878 | **400** |
| OpenPII test part above: the hand-written filters alone | 3,733 | 434 |
| the same with the ML filter | 3,733 | **306** |

On KPWr's test part and its held-out documents it masks the same values whole, and on the agent's
calls and the sample documents it changes nothing: what wastes tokens there, the English of
software, the filters already drop. In cross-validation the word vectors and the context counted
most; a network with one hidden layer did 3% better than the regression and is not worth what it
would cost to read and to run in plain Python. The filter costs about half a millisecond a text,
against twelve for the spaCy model, and it remembers what the model read in the words of the
names it reviews, never the words themselves.

## Running it

```bash
python benchmarks/run.py                 # the rules alone
python benchmarks/run.py --ner           # with pl_core_news_md
python benchmarks/run.py --ner --markdown
python benchmarks/demo.py --ner          # print the masked documents
python benchmarks/run_agent.py --ner --show 20   # an agent's calls, and what hides nothing
python benchmarks/run_agent.py --ner --paranoid  # the same with mode="paranoid"
```

The Presidio comparison needs Presidio, which is not a dependency of this project:

```bash
pip install presidio-analyzer
python benchmarks/compare_presidio.py [--lenient-phones] [--markdown]
```

The data polanon was never tuned on is downloaded into `~/.cache/polanon/benchmarks`:

```bash
python benchmarks/run_external.py              # OpenPII and KPWr
python benchmarks/run_external.py --presidio   # and Presidio
python benchmarks/run_external.py --teryt      # and every TERYT street
```

A document worth keeping is welcome: put the text in `samples/`, its annotations in `gold/` with
the same name, and synthetic values only, with valid checksums where the type has one.
