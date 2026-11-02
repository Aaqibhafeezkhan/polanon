# Security policy

polanon exists to keep personal data out of what an application sends to a language model. If you
find a way past that — a spelling, a format or an input that leaves a value with a token in the
clear, a restore that puts an original where it does not belong, or an error or log line that
carries one — please report it privately from the repository's **Security** tab (*Report a
vulnerability*) instead of opening a public issue, or write to kontaktboromat@gmail.com. A public
issue would publish the way past before there is a fix. You will get an acknowledgement within 7
days.

What the library guarantees, and what it does not, is in `docs/THREAT_MODEL.md`. In short:

- Masking runs in your process. The mapping from token to original never leaves it unless your code
  sends it somewhere.
- Every later occurrence of a value that has a token is masked, in each spelling the README lists.
- The library's own errors and log lines name types, lengths, statuses and guards, never a value.
- Names and places are found partly by statistics. A value no detector recognises is a detection
  gap rather than a bypass; we still want to know, and a case for `benchmarks/` shows it best.
