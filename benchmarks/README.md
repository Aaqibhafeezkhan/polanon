# How well polanon detects

[`samples/`](samples/) holds twelve synthetic documents. Nine are Polish: a CV, a VAT invoice, a
customer complaint, an IT support ticket, an employment contract, an internal IT note, chat
messages typed the way people type them, in lowercase and without Polish letters, a monthly
roster of surnames and periods, and a hospital discharge summary. Three are invoices from abroad:
British, Swedish and Swiss. [`gold/`](gold/) annotates the 187 values that
must be masked, and [`run.py`](run.py) reports whether each one was **fully** covered, because half
a masked PESEL is a leak rather than half a success. It scores the guards and the merging of
their findings, not the pass that masks every later occurrence of a value already found: through
`Anonymizer`, which also masks every later occurrence of a value it has found, the same documents
can only come out the same or better.

| Configuration | Fully masked | Partially masked | Missed | Over-masked spans |
|---|---:|---:|---:|---:|
| polanon, guards only, no model | 161 / 187 (86%) | 0 | 26 | 0 |
| polanon, guards + `pl_core_news_md` | **184 / 187 (98%)** | 0 | 3 | 8 |
| Presidio 2.2.366, set up for Polish with the same model | 112 / 169 (66%) | 25 | 32 | 47 |
| Presidio, phone recognizer at leniency 0 | 128 / 169 (76%) | 29 | 12 | 76 |

Presidio was measured on the first eleven documents, before the discharge summary joined them, and
has not been run on the twelve; on those eleven polanon masks 145 of 169 with the rules alone and
166 of 169 with the model, with 7 over-masked spans.

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

**Read the number honestly.** Twelve short documents are a regression check, not a research
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
