"""The arbiter: pl_core_news_md and FastPDN side by side, and a logistic regression that decides.

The spaCy guard finds names with ``pl_core_news_md`` alone. The arbiter reads every line with it
too, and every Polish line also with FastPDN, CLARIN-PL's transformer for Polish named entities,
which finds many of the names md misses: a street written without "ul.", a foreign surname after
a first name, a district after a town. What either model finds goes through the hand-written
filters of the normal mode, and what is left is a candidate. A logistic regression, fitted by
``benchmarks/arbiter_train.py``, weighs each candidate with the ML filter's signs (its shape, its
words, their vectors in md, the words around it, whether a rule found something there) and with
what the two models said of it: which found it, with the same bounds or not, how sure FastPDN was,
and the fine KPWr category FastPDN gave it, such as a street, a town, a company, a country or a
piece of software.

Three rules keep it from doing worse than the spaCy guard:

- **people and places are never fewer**: one md finds and the ML filter keeps stays, so the
  arbiter only adds them;
- **organisations come from md only**: FastPDN proposes none, so a brand in a Polish sentence
  about the stock market is not masked; the arbiter may drop one of md's, as it drops a news
  site's name taken for a company;
- **a line FastPDN does not read is decided as the spaCy guard decides it**: English, code, JSON
  and headings, where FastPDN reads software words as companies and md's filters already decide.

It needs ``pip install "polanon[arbiter]"``, the spaCy model, and FastPDN exported once with
``python -m polanon.fastpdn``. On the benchmarks it masks more and raises fewer false alarms than
the spaCy guard; on an agent's recorded calls it took 1.15 times as long. See the README.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_left
from collections import OrderedDict
from itertools import accumulate
from pathlib import Path
from typing import Any

from polanon.guards import _arbiter_weights as fitted
from polanon.guards import ner
from polanon.guards._fastpdn import DEFAULT_DIRECTORY
from polanon.guards._span_features import (
    Word,
    line_bounds,
    ml_filter_keeps,
    span_context,
    span_features,
    span_vector,
)
from polanon.guards.base import Guard
from polanon.types import Span

#: The spaCy model the arbiter and its ML filter were fitted with: the only one it reads with.
SPACY_MODEL = "pl_core_news_md"

#: FastPDN's KPWr categories that polanon masks, as its entity types. Every other category, a
#: country, a nation, a piece of software, still counts among the arbiter's signs.
_PLACES = frozenset(
    {"nam_fac_road", "nam_fac_square", "nam_fac_park", "nam_fac_bridge", "nam_num_house"}
)


def fastpdn_type(group: str) -> str | None:
    """The entity type polanon gives a FastPDN category, or None for one it does not mask."""
    if group == "nam_liv_person":
        return "PERSON"
    if group.startswith("nam_loc") or group in _PLACES:
        return "LOCATION"
    if group.startswith("nam_org") and group != "nam_org_nation":
        return "ORG"
    return None


#: What FastPDN's pieces of a word carry at the edge of an entity: "Gdyni." and "**Do".
_EDGE = frozenset(".,;:!?()[]{}\"'„”“«»*")

#: Words FastPDN takes for companies in an agent's calls that name nobody.
_SOFTWARE = frozenset({"markdown", "nas"})

#: The entity types the never-fewer rule keeps.
_NAMES = frozenset({"PERSON", "LOCATION"})


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and (text[start].isspace() or text[start] in _EDGE):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in _EDGE):
        end -= 1
    return start, end


def _software(fragment: str) -> bool:
    words = [ner._bare(word).lower() for word in fragment.split()]
    return bool(words) and all(word in _SOFTWARE for word in words)


def polish_runs(piece: str) -> list[tuple[int, int]]:
    """The ranges of `piece` FastPDN reads: runs of consecutive lines that read as Polish."""
    runs: list[tuple[int, int]] = []
    start = 0
    for line in piece.splitlines(keepends=True):
        end = start + len(line)
        if line.strip() and ner._language(line) == ner._POLISH:
            if runs and runs[-1][1] == start:
                runs[-1] = (runs[-1][0], end)
            else:
                runs.append((start, end))
        start = end
    return runs


class _Index:
    """Ranges sorted by start, to find the ones over a span without reading them all."""

    def __init__(self, items: list[tuple[int, int, Any]]) -> None:
        self.items = sorted(items, key=lambda item: (item[0], item[1]))
        self.starts = [item[0] for item in self.items]
        self.reach = list(accumulate((item[1] for item in self.items), max))

    def over(self, start: int, end: int) -> list[tuple[int, int, Any]]:
        found = []
        at = bisect_left(self.starts, end) - 1
        while at >= 0 and self.reach[at] > start:
            if self.items[at][1] > start:
                found.append(self.items[at])
            at -= 1
        found.reverse()
        return found

    def any(self, start: int, end: int) -> bool:
        at = bisect_left(self.starts, end) - 1
        while at >= 0 and self.reach[at] > start:
            if self.items[at][1] > start:
                return True
            at -= 1
        return False


def _spans_index(spans: list[Span]) -> _Index:
    return _Index([(span.start, span.end, span) for span in spans])


class _Reading:
    """What both models found in one text, before the arbiter decides; no word of the text.

    It is remembered under the text's digest, as the spaCy guard remembers what it found, so it
    keeps where things stand and what the models read in them, never the letters.
    """

    def __init__(self) -> None:
        self.md_raw: list[Span] = []
        self.fp_raw: list[Span] = []
        self.md_kept: list[Span] = []
        self.fp_kept: list[Span] = []
        self.details: list[tuple[int, int, str, float]] = []
        self.words: list[tuple[int, int, Word]] = []
        self.runs: list[tuple[int, int]] = []
        self.candidates: list[tuple[Span, dict[str, str]]] = []
        self.text_english = False

    def index(self) -> None:
        """Index what was found by position, and keep only the words of the candidates."""
        places = _spans_index([span for span, _ in self.candidates])
        self.words = [word for word in self.words if places.any(word[0], word[1])]
        self.i_words = _Index(self.words)
        self.i_details = _Index([(s, e, (group, score)) for s, e, group, score in self.details])
        self.i_md_raw, self.i_fp_raw = _spans_index(self.md_raw), _spans_index(self.fp_raw)
        self.i_md_kept, self.i_fp_kept = _spans_index(self.md_kept), _spans_index(self.fp_kept)
        self.i_runs = _Index([(a, b, None) for a, b in self.runs])
        self.infos = {(span.start, span.end): info for span, info in self.candidates}


class _Text:
    """A text under review with what it reads from the whole of itself, computed once."""

    def __init__(self, text: str, reading: _Reading, rules: list[Span], vocab: Any) -> None:
        self.text = text
        self.reading = reading
        self.rules = _spans_index(rules)
        self.vocab = vocab
        self._repeated: dict[str, bool] = {}
        self._languages: dict[tuple[int, int], str] = {}

    def features(self, span: Span) -> dict[str, int]:
        """The ML filter's features of a span, as the spaCy guard computes them."""
        fragment = self.text[span.start : span.end]
        if fragment not in self._repeated:
            self._repeated[fragment] = self.text.count(fragment) > 1
        line = line_bounds(self.text, span.start)
        if line not in self._languages:
            self._languages[line] = ner._language(self.text[slice(*line)])
        return span_features(
            self.text,
            span,
            rules_found_it=self.rules.any(span.start, span.end),
            words=[word for _, _, word in self.reading.i_words.over(span.start, span.end)],
            vocab=self.vocab,
            text_english=self.reading.text_english,
            repeated=self._repeated[fragment],
            line_language=self._languages[line],
        )


def describe(span: Span, info: dict[str, str], text: _Text) -> tuple[list[str], list[str]]:
    """The names of a candidate's signs that are on, and FastPDN's categories over it."""
    reading = text.reading
    on = [name for name, value in text.features(span).items() if value]
    md, fp = "md" in info, "fp" in info
    over = reading.i_details.over(span.start, span.end)
    scores = [score for _, _, (_, score) in over]
    best = max(scores, default=0.0)
    bounds = (span.start, span.end)
    signs = {
        "src_md_only": md and not fp,
        "src_fp_only": fp and not md,
        "src_both": md and fp,
        "md_saw_raw": reading.i_md_raw.any(span.start, span.end),
        "fp_saw_raw": reading.i_fp_raw.any(span.start, span.end),
        "md_other_bounds": any((a, b) != bounds for a, b, _ in reading.i_md_kept.over(*bounds)),
        "fp_other_bounds": any((a, b) != bounds for a, b, _ in reading.i_fp_kept.over(*bounds)),
        "fp_score_99": best >= 0.99,
        "fp_score_90": 0.9 <= best < 0.99,
        "fp_score_70": 0.7 <= best < 0.9,
        "fp_score_low": 0 < best < 0.7,
        "fp_silent": not scores,
        "types_disagree": md and fp and info["md"] != info["fp"],
    }
    on += [name for name, value in signs.items() if value]
    return on, sorted({f"fpl:{group}" for _, _, (group, _) in over})


def probability(
    on: list[str], categories: list[str], context: list[str], kind: str, vector: list[float]
) -> float:
    """The arbiter's probability that a candidate is a name, from its signs that are on."""
    weights = fitted.WEIGHTS
    z = fitted.BIAS
    for name in on + [f"{kind}:{sign}" for sign in on] + context + categories:
        z += weights.get(name, 0.0)
    if vector:
        z += sum(weight * value for weight, value in zip(fitted.VECTOR, vector, strict=False))
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


class ArbiterGuard(Guard):
    """People, places and organisations from md and FastPDN, the arbiter deciding which stay."""

    name = "arbiter_ner"

    def __init__(
        self,
        fastpdn_dir: str | Path | None = None,
        *,
        cache_texts: int = ner.CACHE_TEXTS,
        threads: int | None = None,
        fastpdn: Any = None,
    ) -> None:
        """Remember where FastPDN is; both models load on first use.

        Args:
            fastpdn_dir: where `python -m polanon.fastpdn` exported FastPDN;
                ~/.cache/polanon/fastpdn by default.
            cache_texts: how many texts to remember what the models found in and what the arbiter
                kept of it, by the text's SHA-256 digest, never as written; 0 reads and decides
                every text every time.
            threads: how many threads onnxruntime may use for one text; its default otherwise.
            fastpdn: an object with FastPDN's ``entities(text)``, in place of the exported model.
        """
        self.fastpdn_dir = Path(fastpdn_dir) if fastpdn_dir is not None else DEFAULT_DIRECTORY
        self.cache_texts = cache_texts
        self.threads = threads
        self._fastpdn = fastpdn
        # The spaCy guard loads md and splits a span at line breaks and table cells: the arbiter
        # reads the model through it, in the paranoid mode, which filters nothing itself.
        self._spacy = ner.NerGuard(SPACY_MODEL, cache_texts=0, mode="paranoid", ml_filter=False)
        self._read: OrderedDict[str, _Reading] = OrderedDict()
        self._decided: OrderedDict[tuple[str, bytes], tuple[int, ...]] = OrderedDict()
        #: How benchmarks/arbiter_data.py reads the texts the arbiter learns from: FastPDN on
        #: every line, and its organisations proposed too, so the regression sees every
        #: candidate either model would make. Masking never reads that way.
        self._for_training = False

    def _models(self) -> tuple[Any, Any]:
        """The pipeline of md and FastPDN, each loaded once even when threads ask at once.

        Raises:
            RuntimeError: when spaCy, the model, onnxruntime, tokenizers or the export is missing.
        """
        nlp = self._spacy._get_pipeline()
        if self._fastpdn is None:
            with ner._LOAD_LOCK:
                if self._fastpdn is None:
                    from polanon.guards._fastpdn import FastPDN

                    self._fastpdn = FastPDN(self.fastpdn_dir, threads=self.threads)
        return nlp, self._fastpdn

    def detect(self, text: str, enabled: set[str]) -> list[Span]:
        """Return the candidates of the enabled types; the arbiter decides in :meth:`_review`."""
        wanted = ner.SUPPORTED & enabled
        if not wanted or not text.strip():
            return []
        reading = self._reading(text, reuse=self.cache_texts > 0)
        return [span for span, _ in reading.candidates if span.entity_type in wanted]

    def _reading(self, text: str, *, reuse: bool) -> _Reading:
        """What both models found in `text`, read again unless `reuse` and it was read lately.

        A reading is kept for the review that follows detection even with no texts remembered,
        as the spaCy guard keeps the words of the last texts for its ML filter.
        """
        digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
        if reuse:
            with ner._CACHE_LOCK:
                found = self._read.get(digest)
                if found is not None:
                    self._read.move_to_end(digest)
                    return found
        found = self._read_text(text)
        with ner._CACHE_LOCK:
            self._read[digest] = found
            while len(self._read) > max(self.cache_texts, ner._WORDS_KEPT):
                self._read.popitem(last=False)
        return found

    def _read_text(self, text: str) -> _Reading:
        """Run both models over `text` and keep what each found, filtered and not."""
        from spacy.util import filter_spans

        nlp, fastpdn = self._models()
        reading = _Reading()
        line_spans = self._spacy._line_spans
        for offset, piece in ner._chunks(text, ner.CHUNK_CHARS):
            if not piece.strip():
                continue
            document = nlp(piece)
            for entity in document.ents:
                kind = ner.LABEL_MAP.get(entity.label_)
                if kind:
                    reading.md_raw.extend(
                        line_spans(text, offset + entity.start_char, offset + entity.end_char, kind)
                    )
            for token in document:
                if not token.is_space:
                    reading.words.append(
                        (offset + token.idx, offset + token.idx + len(token.text), Word.of(token))
                    )
            proposed = []
            runs = [(0, len(piece))] if self._for_training else polish_runs(piece)
            reading.runs.extend((offset + a, offset + b) for a, b in runs)
            for run_start, run_end in runs:
                for entity in fastpdn.entities(piece[run_start:run_end]):
                    start, end = _trim(piece, run_start + entity.start, run_start + entity.end)
                    if start >= end:
                        continue
                    reading.details.append(
                        (offset + start, offset + end, entity.group, entity.score)
                    )
                    kind = fastpdn_type(entity.group)
                    if kind:
                        span = document.char_span(start, end, label=kind, alignment_mode="expand")
                        if span is not None:
                            proposed.append(span)
            for span in filter_spans(proposed):
                # Expanding to md's tokens brings back what was trimmed: "**Do" of "**Do x**".
                start, end = _trim(piece, span.start_char, span.end_char)
                if start >= end:
                    continue
                for part in line_spans(text, offset + start, offset + end, span.label_):
                    # A span FastPDN ran across a line break leaves "Gdyni." on its first line.
                    part_start, part_end = _trim(text, part.start, part.end)
                    if part_start < part_end and not ner._is_only_labels(text[part_start:part_end]):
                        reading.fp_raw.append(
                            Span(part_start, part_end, part.entity_type, part.score, part.source)
                        )
        reading.md_raw = [
            s
            for s in ner._join_hyphenated(text, reading.md_raw)
            if ner._has_a_letter(text[s.start : s.end])
        ]
        reading.fp_raw = [
            s
            for s in ner._join_hyphenated(text, reading.fp_raw)
            if ner._has_a_letter(text[s.start : s.end])
        ]
        reading.md_kept = ner._without_english_misreadings(text, reading.md_raw)
        reading.fp_kept = [
            span
            for span in ner._without_english_misreadings(text, reading.fp_raw)
            if not _software(text[span.start : span.end])
        ]
        by_bounds: dict[tuple[int, int], dict[str, str]] = {}
        for source, spans in (("fp", reading.fp_kept), ("md", reading.md_kept)):
            for span in spans:
                info = by_bounds.setdefault((span.start, span.end), {"type": span.entity_type})
                info[source] = span.entity_type
        for (start, end), info in sorted(by_bounds.items()):
            if info["type"] == "ORG" and "md" not in info and not self._for_training:
                continue  # an organisation only md proposes
            reading.candidates.append(
                (Span(start, end, info["type"], ner.NER_SCORE, self.name), info)
            )
        reading.text_english = ner._is_english_text(text)
        reading.index()
        return reading

    def _review(self, text: str, spans: list[Span], others: list[Span]) -> list[Span]:
        """Keep the candidates the arbiter and its rules keep, knowing what the other guards found.

        Called by the pipeline once every guard has read `text`. What it kept of a text it decided
        with the same spans is remembered, under the text's digest and where the spans stand.
        """
        if not spans:
            return spans
        digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
        key = (digest, ner._places(spans, others))
        if self.cache_texts > 0:
            with ner._CACHE_LOCK:
                kept_at = self._decided.get(key)
                if kept_at is not None:
                    self._decided.move_to_end(key)
                    return [spans[at] for at in kept_at]
        kept_at = self._decide(text, spans, others)
        if self.cache_texts > 0:
            with ner._CACHE_LOCK:
                self._decided[key] = kept_at
                while len(self._decided) > self.cache_texts:
                    self._decided.popitem(last=False)
        return [spans[at] for at in kept_at]

    def _decide(self, text: str, spans: list[Span], others: list[Span]) -> tuple[int, ...]:
        """The positions in `spans` of the spans to mask."""
        reading = self._reading(text, reuse=True)
        vocab = self._models()[0].vocab
        under_review = _Text(text, reading, others, vocab)
        kept = []
        for at, span in enumerate(spans):
            info = reading.infos.get((span.start, span.end))
            if info is None:
                kept.append(at)  # no candidate of this guard: more masked, nothing let through
                continue
            fragment = text[span.start : span.end]
            vector = span_vector(fragment, vocab)
            context = span_context(text, span)
            if not reading.i_runs.any(span.start, span.end):
                # A line FastPDN did not read: as the spaCy guard decides.
                if ml_filter_keeps(under_review.features(span), span.entity_type, vector, context):
                    kept.append(at)
                continue
            md_type = info.get("md")
            if md_type in _NAMES:
                as_md = Span(span.start, span.end, md_type, span.score, span.source)
                if ml_filter_keeps(under_review.features(as_md), md_type, vector, context):
                    kept.append(at)  # one the spaCy guard keeps is never dropped
                    continue
            on, categories = describe(span, info, under_review)
            chance = probability(on, categories, context, span.entity_type, vector)
            if chance >= fitted.THRESHOLDS.get(span.entity_type, 0.5):
                kept.append(at)
        return tuple(kept)
