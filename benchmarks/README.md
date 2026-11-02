# How well polanon detects

[`samples/`](samples/) holds ten synthetic documents. Seven are Polish — a CV, a VAT invoice, a
customer complaint, an IT support ticket, an employment contract, an internal IT note, and chat
messages typed the way people type them, in lowercase and without Polish letters — and three are
invoices from abroad: British, Swedish and Swiss. [`gold/`](gold/) annotates the 164 values that
must be masked, and [`run.py`](run.py) reports whether each one was **fully** covered, because half
a masked PESEL is a leak rather than half a success. It scores the guards and the merging of
their findings, not the pass that masks every later occurrence of a value already found: through
`Anonymizer` the same documents come out a little better, 162 of 164 with the model.

| Configuration | Fully masked | Partially masked | Missed | Over-masked spans |
|---|---:|---:|---:|---:|
| polanon, guards only, no model | 140 / 164 (85%) | 0 | 24 | 0 |
| polanon, guards + `pl_core_news_md` | **161 / 164 (98%)** | 0 | 3 | 7 |
| Microsoft Presidio 2.2, set up for Polish with the same model | 106 / 164 (65%) | 24 | 34 | 46 |
| Presidio, phone recognizer at leniency 0 | 122 / 164 (74%) | 30 | 12 | 71 |

Presidio is the usual answer to "mask personal data before an LLM call", so it is the baseline.
[`compare_presidio.py`](compare_presidio.py) gives it the same Polish spaCy model, every offline recognizer it
ships switched to Polish, its own PESEL recognizer among them, and the corpus's phone regions, and
scores it in its favour: every finding counts whatever its score, coverage is the union of its
findings, and same-type findings separated by a space are joined as its anonymizer joins them. What
it misses is what it has no rule for: Polish names in their grammatical cases, companies by their
legal form, street addresses, Polish, Swedish and Swiss postcodes, Bankgiro and sort codes, and
secrets. At leniency 0 its phone recognizer gains 16 values, mostly postcodes and account numbers
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
| `PERSON` | 29 | 29 | 0 | 0 | 29 |
| `LOCATION` | 23 | 20 | 0 | 3 | 20 |
| `ORG` | 16 | 16 | 0 | 0 | 16 |
| `ADDRESS` | 15 | 15 | 0 | 0 | 15 |
| `EMAIL` | 14 | 14 | 0 | 0 | 14 |
| `POSTAL_CODE` | 14 | 14 | 0 | 0 | 14 |
| `PHONE` | 12 | 12 | 0 | 0 | 12 |
| `NIP` | 6 | 6 | 0 | 0 | 6 |
| `BANK_ACCOUNT` | 5 | 5 | 0 | 0 | 5 |
| `IBAN` | 5 | 5 | 0 | 0 | 5 |
| `PESEL` | 4 | 4 | 0 | 0 | 4 |
| `IP_ADDRESS` | 3 | 3 | 0 | 0 | 3 |
| `REGON` | 3 | 3 | 0 | 0 | 3 |
| `SECRET` | 3 | 3 | 0 | 0 | 3 |
| `TAX_ID` | 3 | 3 | 0 | 0 | 3 |
| `COMPANY_ID` | 2 | 2 | 0 | 0 | 2 |
| `MAC_ADDRESS` | 2 | 2 | 0 | 0 | 2 |
| `PLATE` | 2 | 2 | 0 | 0 | 2 |
| `PL_ID` | 2 | 2 | 0 | 0 | 2 |
| `VIN` | 1 | 1 | 0 | 0 | 1 |

</details>

**Read the number honestly.** Ten short documents are a regression check, not a research
benchmark, and the corpus was written by the same person who wrote the guards, who tuned them
against it; Presidio was never tuned on it. The misses are
instructive: the word *Kraków* is masked in one document and missed in another, and *Bristol* and
one *Zürich* are missed on the foreign invoices, because places come from a statistical model and not
a rule. So do the over-masked spans: *ksiegowosci* typed without Polish letters, which the model
takes for a place, and on the foreign invoices country names, a Swedish line item, the month
*August*, which a Polish model reads as names, and the label phrase *Company Registered in England
and Wales*, read as a company. [`tests/test_benchmark.py`](../tests/test_benchmark.py) turns these numbers into
floors so they cannot quietly get worse.

## Running it

```bash
python benchmarks/run.py                 # the rules alone
python benchmarks/run.py --ner           # with pl_core_news_md
python benchmarks/run.py --ner --markdown
python benchmarks/demo.py --ner          # print the masked documents
```

The Presidio comparison needs Presidio, which is not a dependency of this project:

```bash
pip install presidio-analyzer
python benchmarks/compare_presidio.py [--lenient-phones] [--markdown]
```

A document worth keeping is welcome: put the text in `samples/`, its annotations in `gold/` with
the same name, and synthetic values only, with valid checksums where the type has one.
