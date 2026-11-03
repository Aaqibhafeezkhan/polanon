# polanon

[![CI](https://github.com/boromat/polanon/actions/workflows/ci.yml/badge.svg)](https://github.com/boromat/polanon/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/boromat/polanon/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/boromat/polanon/blob/main/LICENSE)

**Reversible PII masking for LLM calls, Polish-first.** Mask personal data before a prompt leaves
your network, send only tokens to the model, and put the originals back into its answer — even
while it streams. Pure Python: the standard library is the whole core.

![polanon masking a Polish invoice request in a terminal, then restoring the model's answer in Python](https://raw.githubusercontent.com/boromat/polanon/main/docs/demo.gif)

```bash
pip install polanon
```

```python
from polanon import Anonymizer, default_guards

anonymizer = Anonymizer(default_guards(with_ner=False))

result = anonymizer.anonymize_sync(
    "Proszę o korektę faktury dla Jana Kowalskiego, PESEL 50810112354, "
    "konto PL61109010140000071219812874, mail jan.kowalski@example.com."
)
print(result.anonymized)
# Proszę o korektę faktury dla <OSOBA_1>, PESEL <PESEL_1>, konto <IBAN_1>, mail <EMAIL_1>.

answer = "Korekta dla <OSOBA_1> (PESEL <PESEL_1>) jest gotowa."  # what the model replied
print(Anonymizer.restore(answer, result.mapping))
# Korekta dla Jana Kowalskiego (PESEL 50810112354) jest gotowa.
```

The model only ever sees `<OSOBA_1>`. The mapping that turns it back into a name never leaves your
process. `anonymize_sync` works in scripts, Django and Flask views, Celery tasks and notebooks; in
async code, `await anonymizer.anonymize(...)` does the same work in a worker thread.

## Why another PII library

General-purpose tools are built around English. Polish data needs three things they do not do out
of the box:

- **Checksums.** A PESEL, NIP, REGON, ID card number, IBAN or payment card is validated, not just
  pattern-matched: the checksum rejects most random digit strings — for a PESEL, together with the
  birth date, about 98% of them — while a valid number always passes. KRS numbers, which have no
  public checksum, and passports are recognised next to their label as well.
- **Seven grammatical cases.** "Jan Kowalski", "Jana Kowalskiego" and "z Janem Kowalskim" are the
  same person. Polish first names are recognised in every case, in lowercase and typed without
  Polish letters, together with the surname that agrees with them, without any language model.
- **Polish model labels.** Polish spaCy models tag people as `persName`, not `PERSON`. A pipeline
  configured for English labels finds no names at all, and says nothing about it.

It also reads foreign invoices: EU, UK and Swiss VAT numbers, company register numbers, Bankgiro,
Plusgiro and sort codes, international phone numbers, street addresses and postcodes.

## Install

| You want | Install |
|---|---|
| Identifiers, contacts, secrets, Polish names, companies by legal form | `pip install polanon` |
| Also places, and names and organisations no rule can see | `pip install "polanon[ner]"` then `python -m spacy download pl_core_news_md` |
| The stage-2 judges (a local Ollama model, or a remote one) | `pip install "polanon[judge]"` |

Then use `default_guards()` with the model, or `default_guards(with_ner=False)` without it.
Another spaCy model is one argument away: `default_guards(spacy_model="pl_core_news_lg")`.

## From the command line

Text comes from the arguments or, without any, from standard input. Masked text goes to stdout
and the mapping to stderr, so a pipeline can keep the two apart:

```bash
polanon "Jan Kowalski, PESEL 50810112354"
cat umowa.txt | polanon --ner > masked.txt 2> mapping.json
polanon --file umowa.txt --ner > masked.txt 2> mapping.json
```

A non-zero exit status means stderr holds an error rather than the mapping. In Windows
PowerShell 5.1, read the file with `Get-Content -Encoding UTF8 umowa.txt | polanon`: a plain
`cat` there re-encodes Polish letters before they reach the program, and a garbled name is not
recognised. `--file` avoids that: polanon reads the file's bytes itself, as UTF-8 or else cp1250,
and it cannot be combined with text arguments.

## Streaming

A provider streams arbitrary fragments, so `<OSOBA_1>` can arrive as `<OSO` and `BA_1>`. The
restorer holds back anything that could still grow into a token:

```python
restorer = Anonymizer.stream_restorer(result.mapping)
for chunk in stream:  # "Dzień dobry <OSO", "BA_1>, PESEL <PESEL", "_1> zweryfikowany."
    print(restorer.feed(chunk), end="")
print(restorer.flush())
# Dzień dobry Jana Kowalskiego, PESEL 50810112354 zweryfikowany.
```

## With Claude, GPT or a local model

Mask before the call, restore while the answer streams. With the Anthropic SDK:

```python
import anthropic
from polanon import Anonymizer, default_guards

anonymizer = Anonymizer(default_guards(with_ner=False))
masked = anonymizer.anonymize_sync(question)
restorer = Anonymizer.stream_restorer(masked.mapping)

with anthropic.Anthropic().messages.stream(
    model="claude-opus-5-5",
    max_tokens=64000,
    system="Personal data is replaced by tokens such as <OSOBA_1>. Copy a token exactly as written.",
    messages=[{"role": "user", "content": masked.anonymized}],
) as stream:
    for text in stream.text_stream:
        print(restorer.feed(text), end="")
print(restorer.flush())
```

Telling the model what the tokens are keeps it from rewriting them. The
[examples](https://github.com/boromat/polanon/tree/main/examples) go further:
`openai_chat.py` does the same for any OpenAI-compatible endpoint (OpenAI, OpenRouter, Azure, vLLM,
a local Ollama), and `anthropic_chat.py` holds a conversation whose history keeps masked text only,
so an answer restored for the reader is never sent back in the clear. Both are tested in CI against
the real SDKs, offline.

## What it detects

Checksum-validated Polish identifiers (PESEL, NIP, REGON, ID card, IBAN, payment cards), KRS numbers
next to their label, e-mail addresses and phone numbers, street addresses and postcodes, IPv4 and
IPv6, MAC addresses, vehicle registrations after their label ("nr rej."), VINs, passport numbers,
birth dates, API keys, tokens and passwords, Polish first names with their surnames, and company
names anchored on a legal form ("Sp. z o.o.", "S.A.", "Ltd", "GmbH"). With the `ner` extra, a spaCy
model adds places and the names nothing else marks. An optional second stage lets a small local
model review the *masked* text for what still identifies someone by context. Other dates are
recognised too, but left readable unless you ask for them.

Text copied from PDFs, Word or macOS is read as a person would read it: letters written as a base
letter plus an accent, no-break spaces and en dashes inside numbers, and invisible zero-width
characters do not hide a value. The mapping keeps each value as it was first written.

On the annotated sample documents in the repository, the rules alone fully mask 85% of the
values and the rules with `pl_core_news_md` mask 98%, with nothing left half-masked. Ten short
documents are a regression check, not a research benchmark: see the
[benchmark notes](https://github.com/boromat/polanon/blob/main/benchmarks/README.md) for what is
missed and why, and for Microsoft Presidio on the same documents with the same model: 65% fully
masked, 74% at its most lenient.

## Worth knowing

**Tokens are stable per conversation, not global.** Pass the mapping from the latest result to
the next call: the same value keeps the same token, and a token is never reused for another
value. The mapping you pass in is read, not changed, so keep the one each result returns:

```python
mapping = result.mapping
second = anonymizer.anonymize_sync("Ten sam PESEL 50810112354", mapping)
second.anonymized  # "Ten sam PESEL <PESEL_1>"
mapping = second.mapping  # everything so far, for the next turn
```

**A token restores the form it replaced.** Masking happens on the text as written, so a name masked
in the genitive comes back in the genitive. Ask the model to keep tokens where the grammar allows it
("dla <OSOBA_1>"), or restore into a template you control. Spellings that differ only in letter
case, in the kind of whitespace (a no-break space, a line break, a run of spaces) or in invisible
characters are the same value, and so are an identifier's groups written apart or together: they
share its token, and restore writes the spelling seen first. Passwords and custom types are the
exception, since two passwords that differ in case are two passwords, and a password never shares
a token with a name that reads alike. A one-word name found in all lowercase keeps a token of its
own too, for the reason below.

**Every occurrence of a masked value is masked**, also where no detector would recognise it on its
own: as written, in capitals, across a line break, with a zero-width character in it, with its
groups spaced differently, and around whatever a detector found inside it. Two exceptions keep ordinary words readable. A capitalised
one-word name is not matched in all lowercase, because Zima, Kot and Wilk are surnames and "zima"
is winter; identifiers such as a VIN or an e-mail address are matched in any case. A number shorter
than six digits, such as a PIN, is not matched inside a longer number, a date, a time or an
amount: "15.04.2025", "2024–2025", "2025,50" and "20250" stay readable when the PIN is 2025, while
"w 2025 roku" and "PIN/PUK 2025/36912874" do not.

**A token you write stays yours.** A placeholder such as `<PESEL_1>` in your own text is never handed
to real data. It joins the mapping as standing for itself, so the reservation holds in later turns
too.

**The mapping is the sensitive half.** `result.anonymized` holds a token for everything the
detectors found, and only what they missed in the clear; `result.mapping` is the key that undoes
it. Store the mapping the way you would store the original document.

**Replaying a conversation?** `mask_known_values(text, mapping)` masks every value a mapping already
holds in text that was masked earlier, such as history sent again after a new value got its token.
The same call checks a payload before it leaves: `mask_known_values(payload, mapping) == payload`
means no value the mapping holds is left where the library would mask it.

**Words of your own.** In-house names no rule can know, such as a project codename or a client,
go on a denylist and are masked wherever they appear, in any case and across line breaks:

```python
guards = default_guards(denylist=["Projekt Wichura", "Zakład Nowak"], with_ner=False)
```

**Checking the answer.** After restoring, `anonymizer.sweep(restored, mapping)` reports tokens the
model invented and personal data its answer added: `report.is_clean`, or `report.unknown_tokens`
and `report.new_spans` to look at.

## Choosing what to mask

`Anonymizer(guards)` masks `DEFAULT_ENABLED`: every type in `ENTITY_TYPES` except `DATE`, because a
date rarely identifies anyone and masking every deadline leaves the model unable to answer. A birth
date does identify someone, so a date after "ur.", "data urodzenia", "DOB" or "geb." is its own
type, `BIRTH_DATE`, and stays masked. Pass `enabled` to choose for yourself:

```python
from polanon import DEFAULT_ENABLED, ENTITY_TYPES, Anonymizer, default_guards

guards = default_guards(with_ner=False)
Anonymizer(guards, enabled=DEFAULT_ENABLED | {"DATE"})  # dates too
Anonymizer(guards, enabled={"PESEL", "NIP", "IBAN"})  # identifiers only
list(ENTITY_TYPES)  # every name you can use
```

A name that is neither built in nor declared in a guard's `entity_types` raises `ValueError`, so a
typo such as `"PERSONS"` cannot quietly switch masking off.

## A second opinion: the judge

Rules and a language model miss what identifies someone only by context: "the finance director of
the only steelworks in town". A judge is a small model that reads the *masked* text after stage 1
and flags such fragments. Run it locally with Ollama, so the review never leaves your network:

```python
import asyncio

from polanon import Anonymizer, JudgeError, default_guards
from polanon.judge import OllamaJudge

anonymizer = Anonymizer(
    default_guards(),
    judge=OllamaJudge(model="qwen2.5:1.5b"),
    judge_fallback="block",  # refuse to go on if the judge is down
)


async def main(text: str) -> None:
    try:
        result = await anonymizer.anonymize(text)
    except JudgeError:
        ...  # the judge was down, and "block" means nothing goes out


asyncio.run(main("..."))
```

It needs `pip install "polanon[ner,judge]"`, the spaCy model, a running Ollama and
`ollama pull qwen2.5:1.5b`. With `judge_fallback="send_stage1"` the call goes on when the judge
is down, and `result.judge_failed` says that stage 1 went out alone.

Judges are asynchronous, so `anonymize_sync` refuses an anonymizer that has one. `RemoteJudge` asks an
OpenAI-compatible endpoint instead; the masked text then leaves your network, which makes it a
testing aid rather than a privacy control.

## Names that should stay readable

A language model flags "Facebook" or "Google Ads" as an organisation, which masks the very thing a
question is about. Pass an allowlist and those detections are dropped before tokens are assigned:

```python
from polanon import WELL_KNOWN_NAMES, Allowlist, Anonymizer, default_guards

anonymizer = Anonymizer(default_guards(), allowlist=Allowlist(WELL_KNOWN_NAMES))
```

A one-word name also matches its Polish case forms ("Facebooku", "Canvie"). Only detections of
names are dropped, and only when every word in them is on the list; identifiers, the denylist and
values a conversation has already masked stay masked.

## The pieces

| Module | What it is for |
|---|---|
| `guards/` | deterministic detectors. Each returns `Span`s in original-text coordinates |
| `pipeline.py` | runs the guards in a thread, resolves overlaps, assigns stable tokens |
| `policy.py` | the entity taxonomy and the token naming (`<OSOBA_1>`, `<PESEL_1>`) |
| `judge/` | the optional second stage: a model reviews the masked text for what still identifies |
| `stream_restore.py` | puts originals back into a streamed reply without splitting a token |
| `output_sweep.py` | flags invented tokens and personal data the answer introduced |

## Adding a detector

Subclass `Guard`, return spans in original-text coordinates and respect the `enabled` set. A type of
your own goes in `entity_types`, which turns it on by default; it becomes the label of its tokens,
so it is written in uppercase letters and underscores:

```python
import re

from polanon import Anonymizer, Guard, Span, default_guards


class EmployeeIdGuard(Guard):
    name = "employee_id"
    entity_types = frozenset({"EMPLOYEE_ID"})

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        if "EMPLOYEE_ID" not in enabled:
            return []
        return [
            Span(m.start(), m.end(), "EMPLOYEE_ID", 1.0, self.name)
            for m in re.finditer(r"\bEMP-\d{5}\b", text)
        ]


anonymizer = Anonymizer([*default_guards(with_ner=False), EmployeeIdGuard()])
anonymizer.anonymize_sync("Wniosek EMP-11111").anonymized  # "Wniosek <EMPLOYEE_ID_1>"
```

The score decides who wins where detections overlap: 1.0 for a value a checksum, a label or an
official format vouches for, 0.85 for a statistical guess; the judge's findings score 0.5. A secret
is masked whole, with whatever was detected inside it. A span outside the text, a type spelled any
other way, or one spelled like a built-in token label (`OSOBA`, `DATA`) raises `ValueError` naming
your guard.

## Before you rely on it

Masking is pseudonymisation, not anonymisation in the legal sense, and no detector catches
everything. The [threat model](https://github.com/boromat/polanon/blob/main/docs/THREAT_MODEL.md) says what is covered, what is
not, and what stays your application's job, starting with keeping the mapping safe.

polanon was built as the masking engine of piigate, a self-hosted hub that lets a team use Claude
or GPT on real documents without sending personal data to the provider; the hub is not published
yet. The library knows nothing about it — no HTTP, no database, no settings — and a test fails if
it ever starts to.

## Contributing

Bug reports, missed values and new detectors are welcome: see
[CONTRIBUTING.md](https://github.com/boromat/polanon/blob/main/CONTRIBUTING.md). A way to get an original value past the
masking is a security report, made privately as [SECURITY.md](https://github.com/boromat/polanon/blob/main/SECURITY.md)
describes.

MIT licensed.
