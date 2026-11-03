# Contributing

Bug reports, missed values, new detectors and new languages are all welcome.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ner,judge]"
python -m spacy download pl_core_news_md
pre-commit install
```

Add `pip install anthropic openai` if you touch `examples/`: their tests run against the real SDKs,
offline, and are skipped without them.

## Quality bar

- `ruff check .`, `ruff format --check .`, `mypy polanon` and `pytest --cov` pass locally. CI runs
  them, then runs the tests again against the built wheel on Python 3.11, 3.12 and 3.13, and masks a
  document with a wheel installed on its own.
- The core imports nothing but the standard library at module level; spaCy and httpx are loaded
  inside the functions that need them. `tests/test_layering.py` enforces it.
- Type hints on every signature, Google-style docstrings on the public API.
- Tests for every behaviour change, and no network in them: the judges are tested with
  `httpx.MockTransport`. Coverage stays at 90% or above.
- A change to detection is measured, not asserted: run `python benchmarks/run.py --ner` before and
  after, and add a case worth keeping to `benchmarks/samples/` and `benchmarks/gold/`.
  `tests/test_benchmark.py` holds the current numbers as floors.
- Conventional Commits in the imperative mood, one logical change each:
  `feat(guards): recognise Czech VAT numbers`, `fix(pipeline): ...`, `docs: ...`, `test: ...`.
- Synthetic data only. No real person's name, identifier, document or log belongs in an issue, a
  test or a commit.

## Changes written with AI tools

Welcome, on the same terms as any other change. Say in the pull request that an AI tool wrote or
helped write it, and which one, so the review knows where to look. You answer for the change as if
you had typed it yourself: you have read every line, run the quality bar above, and can answer
questions about it. A pull request that nobody answers for after a review is closed.

## A value that was not masked

Open a *Something was not masked* issue with invented text of the same shape: a made-up name, a
PESEL with a valid checksum (50810112354), a fictional company. The shortest text that still shows
the miss is the most useful. A way to get a value that already has a token past the masking is a
security report instead: see `SECURITY.md`.

## A new detector

Subclass `Guard`, return `Span`s in original-text coordinates and respect the `enabled` set; the
README's *Adding a detector* section has a complete example. Score 1.0 for what a checksum, a label
or an official format vouches for and 0.85 for a statistical guess, and add the type to
`polanon/policy.py` if it should be built in.
