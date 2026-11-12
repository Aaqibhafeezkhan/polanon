# Threat model

What polanon protects against, and what it does not. Read it before relying on the library for
anything that matters. It is engineering documentation, not legal advice.

## What the library is for

An application wants a cloud language model to work on text that holds personal data. polanon
replaces that data with tokens such as `<OSOBA_1>` inside the application's own process, so only
tokens reach the provider, and puts the originals back into the answer, also while it streams.

## Trust boundaries

| Zone | Holds | Trusted? |
|---|---|---|
| Your process | the original text, the mapping, restored answers | yes |
| The mapping, wherever you keep it | token to original value | yes, and as sensitive as the documents |
| The LLM provider | masked text only | **no**, this is the boundary the library exists for |
| A stage-2 judge on Ollama | masked text, on your own machine | yes |
| `RemoteJudge` | masked text, sent to an API | **no**, a testing aid only |
| The key of `Pseudonyms` | what turns a guessed value into its stable token | yes, and as sensitive as the mapping |
| A search index with stable tokens | masked passages and their embeddings | **no** more than the provider: it is masked text |

## What is protected

- **Deterministic identifiers.** PESEL, NIP, REGON, ID card numbers, IBAN and payment cards are
  matched by pattern and confirmed by checksum, so detection is near-certain and false positives are
  rare. KRS numbers and passports are recognised next to their label.
- **Contact details and devices.** E-mail addresses, also with "[at]" written out, phone numbers,
  street addresses and postcodes, links to a person's profile (LinkedIn, GitHub and GitHub Pages,
  GitLab, social media) up to the handle, IPv4 and IPv6, MAC addresses, VINs, vehicle registrations
  after their label, birth dates after theirs, in Polish or English.
- **Credentials.** API keys, JWTs, private key blocks and `password=` style assignments, masked whole
  with whatever was found inside them, so pasted logs and configuration files do not leak secrets.
- **People and companies by rule.** Polish first names in all seven cases, in any letter case and
  without Polish letters, with the surname that agrees with them; companies by their legal form
  (`Sp. z o.o.`, `S.A.`, `Ltd`, `GmbH`); names on foreign documents after a form of address, a label
  or a closing formula; and whatever you put on a denylist.
- **Every later occurrence of a masked value.** Pass the mapping from the latest result to the next
  call and a value keeps its token wherever it appears again: in capitals, across a line break or a run of
  spaces, with a no-break space, an invisible character or a decomposed accent in it, and with an
  identifier's groups written apart or together. The surname of a masked person and the handle of a
  masked profile link count as masked values of their own, so "Kowalski" after "Jan Kowalski" and
  "jkowalski" after "github.com/jkowalski" are masked too. `mask_known_values` does the same for
  text masked earlier, such as conversation history sent again.
- **Your own placeholders.** A token such as `<PESEL_1>` already in your text is never handed to real
  data.
- **The answer.** `StreamRestorer` never splits a token across chunks, and `sweep` reports tokens the
  model invented and personal data it introduced.
- **The library's own output.** Errors and log lines name types, counts, lengths and guards, not
  values; a judge that fails says how, without quoting what it was sent or the key it used.

## What is not protected

- **An identifier that fails its checksum.** A PESEL, NIP, IBAN or ID card number with a wrong
  check digit is not masked, even right after its label: a mistyped number, or one an OCR pass
  misread, goes out as written. The checksum is what keeps random digits from being masked, and
  the price is this.
- **Names and places are found partly by rule, partly statistically.** A surname on its own in
  lowercase ("napisz do kowalskiego") or in a text where the full name never appears, a rare or
  foreign first name and every place name depend on the spaCy model, which leans on capital letters and misses what people type in lowercase. A miss
  is a detection gap, not a bypass, but the value still reaches the provider. The stage-2 judge
  exists to narrow this gap, not to close it.
- **A one-word name later written in lowercase.** A capitalised one-word name is not matched in all
  lowercase, because many Polish surnames are also ordinary words (Zima, Kot, Wilk): "kowalska
  dzwoniła" after "Pani Kowalska" goes out unless a detector recognises it on its own. Once one
  does, that spelling has a token of its own.
- **A handle or a personal domain on its own.** A login is recognised inside a link to a profile
  on a known site, and from there wherever it stands. A login with no such link anywhere in the
  text ("mój GitHub to jkowalski") and a personal domain ("jankowalski.dev") are ordinary words to
  the rules: put them on the denylist, or report them from a guard of your own, when you know them.
- **A short number inside a longer one.** A PIN or another number under six digits is not masked
  inside a date, an amount or a longer number ("15.04.2025" when the PIN is 2025), so the model does
  not learn which numbers look like it. Where it stands alone it is masked.
- **Foreign documents are read by their labels and formats.** Tax and company numbers, accounts,
  phones, addresses and postcodes are found where a document prints them next to a label or in an
  official shape. A name in running English or German prose, a city and any format the rules have
  never seen depend on the Polish language model, which reads those languages poorly.
- **Identification by context.** "The finance director of the only steelworks in Stalowa Wola" names
  a person without naming them. The judge is asked to catch exactly this; a small local model will
  not catch all of it.
- **Polish inflection of tokens.** A model may write `<OSOBA_1>a`. Restoration matches tokens
  exactly, so the suffix survives and the sentence reads oddly. It does not leak anything.
- **The mapping.** It is the key that undoes the masking. The library keeps it in memory and hands it
  to you; storing it encrypted, expiring it and keeping it out of logs is your application's job.
- **What you send yourself.** The library masks the text you pass it. A system prompt, a file name,
  tool arguments or metadata built from unmasked data are outside it. Before a call,
  `mask_known_values(payload, mapping) == payload` confirms that no value the mapping holds is left
  where the library would mask it.
- **An allowlist that is too generous.** A name on the allowlist reaches the provider whenever a
  model or the judge takes it for a name. The list cannot release an identifier, an e-mail address,
  a denylisted word or a value that already has a token; a client's name added to it is sent.
- **The provider's side.** Masked text and its context are still processed and probably logged by
  the provider, and message sizes and timing are visible to it.
- **The law.** Masking is pseudonymisation, not anonymisation in the GDPR sense: whoever holds the
  mapping holds personal data. So does whoever holds stable tokens and the key that made them.

## Stable pseudonyms for search indexes

With `Anonymizer(pseudonyms=Pseudonyms(key))` a token is no longer numbered per conversation: it is
computed from the value with HMAC-SHA256 under a key, so the same person has the same token in
every document of an index, in every question and in every conversation that uses the key. That
is what lets a masked question find the masked passages, and it changes the risks.

- **Linking.** Whoever reads masked text can tell which passages, questions and conversations are
  about the same person, without knowing who it is. The provider sees it across every call, and so
  does anyone with the index. Tokens numbered per conversation hid that between conversations.
  Use stable tokens only where the linking is the purpose, as for an index, and leave ordinary
  chats numbered per conversation. Give each company its own key, and an index that must not be
  linked to another its own `scope=`: the same value then gets unrelated tokens.
- **Frequency.** The most frequent token of a sales index is probably the biggest client, and a
  token that turns up with a known date, amount or place can be matched to a person by someone who
  knows those. Masking hides names, not the shape of the data around them.
- **The key.** With the key, anyone holding the index can compute the token of a guess and look
  for it. Identifiers fall first: all PESEL numbers of people born from 1900 to 2025 are about
  4.6×10⁸ guesses, which one CPU core computes in a quarter of an hour and a graphics card in well
  under a second. Common first names with common surnames follow. A leaked key is close to a leaked
  mapping for everything that can be enumerated: keep it in a secret manager or a KMS, apart from
  the index and the mapping store, out of the code and the logs. The library never writes it to a
  message, a representation or a pickle.
- **Why not a plain hash.** A hash with no key is the same computation without the secret: anyone
  holding the index can hash every PESEL, phone number or name from a list and read the tokens
  back. HMAC needs the key for each guess. A slow password hash on top of the key would delay
  enumeration after a leak, but every question and every passage computes hundreds of tokens, and
  the key is the protection that has to hold anyway.
- **A masking service is an oracle.** Whatever masks chosen text with the key confirms guesses:
  type a candidate PESEL into a question, read its token in the masked preview, look for it in the
  index. Restrict who can call it, rate-limit it and log it, as one would for the key itself.
- **Erasure.** Deleting a person's entry from the mapping store does not unlink them. The token is
  still in the index, and the key turns a guessed value back into it. To erase a person, delete or
  mask again every passage that holds `pseudonyms.token(value, type)`, and the surname's token
  where nobody else with that surname is indexed; retention that only deletes the mapping leaves
  the index linkable.
- **A new key.** Changing the key, the scope or the number of digits changes every token, so an
  index built before no longer matches the questions: build it again from the sources, or map the
  tokens one to one through the store. `pseudonyms.settings`, stored with the index, tells the
  two apart, and `key_id` names a key without revealing it.
- **Collisions.** Two values of one type can share a twelve-digit token, with a chance of about 1
  in 20,000 at 10,000 values and 1 in 200 at 100,000. Where they meet, in the store or in one text,
  the second gets four digits more and the collision is recorded; a question that collides with an
  indexed value finds nothing instead of the other person's passages. Two processes that index the
  two values at once are kept apart by the store's unique key on the token.
- **What the index holds.** The masked passages and their embeddings: an embedding can be inverted
  into an approximation of its text, so treat it as the masked text itself. A value no detector
  found is in both in the clear, as it would be in a prompt. The mapping store holds the originals
  and must be encrypted and kept apart from the index; a mapping copied into a passage's metadata
  is stored and searched with it.
- **Names the store already holds.** The store keeps the surname of every indexed person whose
  first name the rules know, and a question or a passage is looked up in it for every run of up to
  four words that could be a name. That masks a surname written alone, which no rule would, and it
  never releases one: the allowlist does not apply to a value the store holds. One surname is one
  token, so two people who share it are linked by it, as they are by the surname itself. A word
  that names no stored value, and a lone word in lowercase, are not masked this way.

## Residual risk to accept consciously

Detection is imperfect, so the right mental model is "much less personal data leaves, not none".
Measure on documents like yours before you rely on it: `benchmarks/run.py` scores the library on
annotated documents and shows what it misses.

## Reporting

See `SECURITY.md`. A way to get a value that has a token past the masking, or into a log or an
error, is a security bug and we want to hear about it privately.
