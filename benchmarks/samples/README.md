# Synthetic samples

Fictional Polish documents used for demos, manual end-to-end checks and the annotated benchmark:
a CV, a VAT invoice, a customer complaint, an IT support ticket, an employment contract, an
internal IT note, and a handful of chat messages typed the way people type them — in lowercase,
without Polish letters — with a few sentences whose words only look like names. Three invoices
come from abroad, addressed to the same Polish companies: a British one in English, a Swedish one
with Bankgiro and Plusgiro accounts, and a Swiss one in German.

Every name, address and identifier is made up. The identifiers carry valid checksums so the guards
treat them as real, and the same fictional people and companies recur across documents the way they
would in one company's paperwork. No data about an actual person belongs in this directory.

`../gold/*.json` annotates each document with the values that must be masked and their entity type.
`../run.py` scores the guards against those annotations; `../demo.py` just prints the masked text.

`monthly_roster.txt` keeps surname-first monthly periods and names beside ordinary numbers
masked without a language model. The separate name-guard tests cover punctuated date formats.
