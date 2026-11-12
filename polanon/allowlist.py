"""Names an organisation wants left alone: the platforms, tools and brands it talks about daily.

The Polish language model tags "Facebooka", "Canvie" and "Google Ads" as organisations and places,
so a marketing team asking how to plan a campaign sent "zrób post na <FIRMA_1>" and got an answer
about nothing. None of those names identifies anyone. An allowlist removes such detections before
tokens are assigned, and four limits keep it from becoming a way to unmask personal data:

* it only removes detections of names — people, organisations, places and the judge's context —
  never an identifier, an address, an e-mail or a secret, whatever text they cover;
* a detection is removed only when every word in it belongs to an allowed name, so a model span
  reading "Jan Kowalski z Facebooka" stays masked in full;
* a detection from the denylist, which includes any words a caller adds to mask in a single
  message, always stays;
* a person the Polish name guard found always stays. That guard needs evidence, a known first name
  or a form of address in front of the word, so "pani Ziaja" is a woman even where Ziaja is also an
  allowed brand.

Polish declines a brand like any other noun, so a one-word name also matches its case forms:
"Facebook" covers "Facebooka" and "Facebookiem", "Canva" covers "Canvie", "YouTube" covers
"YouTubie" and "YouTube'a". Matching ignores letter case and Polish diacritics.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from polanon._view import as_read
from polanon.guards.names_pl import decline_feminine, decline_masculine, fold
from polanon.types import Span

#: Entity types an allowlist may remove. Identifiers are absent on purpose.
NAME_TYPES = frozenset({"PERSON", "ORG", "LOCATION", "OTHER"})

#: Guards whose detections are never removed: what somebody asked to mask stays masked, and so does
#: a value the mapping store of an index already holds a token for.
PROTECTED_SOURCES = frozenset({"denylist", "mapping_store"})

#: Guards whose people are never removed either, because they report a person only on evidence. A
#: brand a statistical model mistakes for a person is still released.
_PERSON_EVIDENCE_SOURCES = frozenset({"names_pl"})

#: Platforms and tools an office or marketing team in Poland names every day. Nothing here is also
#: a first name ("Claude", "Miro") or an ordinary word ("Word", "Teams", "Make").
WELL_KNOWN_NAMES: tuple[str, ...] = (
    "Facebook",
    "Instagram",
    "Messenger",
    "WhatsApp",
    "Threads",
    "Meta",
    "Meta Ads",
    "Meta Business Suite",
    "LinkedIn",
    "TikTok",
    "YouTube",
    "Twitter",
    "Pinterest",
    "Snapchat",
    "Reddit",
    "Telegram",
    "Discord",
    "Google",
    "Google Ads",
    "Google Analytics",
    "Google Tag Manager",
    "Google Search Console",
    "Search Console",
    "Google Business Profile",
    "Google Merchant Center",
    "Google Maps",
    "Google Meet",
    "Google Drive",
    "Google Docs",
    "Google Sheets",
    "Looker Studio",
    "Gmail",
    "Gemini",
    "Chrome",
    "Android",
    "Microsoft",
    "Microsoft Teams",
    "Microsoft Ads",
    "Outlook",
    "Excel",
    "PowerPoint",
    "OneDrive",
    "SharePoint",
    "Bing",
    "Copilot",
    "Windows",
    "Apple",
    "iPhone",
    "Amazon",
    "Allegro",
    "Ceneo",
    "OLX",
    "eBay",
    "AliExpress",
    "Temu",
    "Shopify",
    "WooCommerce",
    "WordPress",
    "PrestaShop",
    "Wix",
    "Squarespace",
    "Canva",
    "Figma",
    "Adobe",
    "Photoshop",
    "Lightroom",
    "Premiere Pro",
    "CapCut",
    "Slack",
    "Notion",
    "Asana",
    "Trello",
    "ClickUp",
    "Jira",
    "Confluence",
    "Monday.com",
    "Zoom",
    "Calendly",
    "Dropbox",
    "Zapier",
    "HubSpot",
    "Mailchimp",
    "GetResponse",
    "Brevo",
    "Salesforce",
    "Pipedrive",
    "Semrush",
    "Ahrefs",
    "Senuto",
    "Hotjar",
    "OpenAI",
    "ChatGPT",
    "Anthropic",
    "Perplexity",
    "Midjourney",
    "DALL-E",
    "Stable Diffusion",
    "DeepL",
)

#: Words that may join two allowed names inside one detection: "Facebooku i Instagramie".
_CONNECTORS = frozenset({"i", "oraz", "lub", "albo", "czy", "and", "or", "&", "+"})

#: Case endings written after an apostrophe or a hyphen: "Google'a", "ChatGPT-em".
_ATTACHED_ENDINGS = ("a", "u", "owi", "em", "iem", "ie", "e", "y", "i", "om", "ach", "ami")

#: A name shorter than this, counted in letters and digits, would allow far more than it names.
_MIN_NAME_LENGTH = 2

_SEPARATORS = re.compile(r"[\s/]+")
_EDGE_PUNCTUATION = ".,;:!?()[]{}\"'„”«»"


def _words(text: str) -> list[str]:
    """Split a name or a detection into folded words, trimming punctuation at their edges.

    Both are read the way the guards read text, so "Face" + soft hyphen + "booku", copied from a
    web page, is still Facebook.
    """
    words: list[str] = []
    for raw in _SEPARATORS.split(as_read(text).replace("\u2019", "'")):
        word = fold(raw.strip(_EDGE_PUNCTUATION))
        if word:
            words.append(word)
    return words


def _case_forms(word: str) -> set[str]:
    """Return a one-word name together with the forms Polish declension gives it."""
    forms = {word}
    forms.update(word + mark + ending for mark in "'-" for ending in _ATTACHED_ENDINGS)
    if not word[-1].isalpha() or len(word) < 3:
        return forms
    if word.endswith("a"):
        forms.update(fold(form) for form in decline_feminine(word))
    elif word.endswith("e"):
        forms.update(fold(form) for form in decline_masculine(word[:-1]))  # YouTube → YouTubie
    elif word[-1] not in "iouy":
        forms.update(fold(form) for form in decline_masculine(word))
    return forms


class Allowlist:
    """Public names that are never masked, matched word by word and in every case form."""

    def __init__(self, names: Iterable[str]) -> None:
        """Build the allowlist; blank lines and names under two characters are ignored."""
        self._phrases: set[tuple[str, ...]] = set()
        for name in names:
            words = _words(name)
            if sum(character.isalnum() for word in words for character in word) < _MIN_NAME_LENGTH:
                continue
            if len(words) == 1:
                self._phrases.update((form,) for form in _case_forms(words[0]))
            else:
                self._phrases.add(tuple(words))
        self._longest = max((len(phrase) for phrase in self._phrases), default=0)

    def __bool__(self) -> bool:
        """Return whether the allowlist names anything at all."""
        return bool(self._phrases)

    def allows(self, fragment: str) -> bool:
        """Return whether every word of `fragment` belongs to an allowed name.

        Connectors such as "i" or "and" may join two names, but a fragment has to contain at least
        one name to be allowed.
        """
        words = _words(fragment)
        if not words or not self._phrases:
            return False
        # reached[i] is None when the first i words cannot be explained, False when only
        # connectors explain them so far, and True once at least one allowed name did.
        reached: list[bool | None] = [False] + [None] * len(words)
        for index, word in enumerate(words):
            state = reached[index]
            if state is None:
                continue
            if word in _CONNECTORS:
                reached[index + 1] = state or bool(reached[index + 1])
            for length in range(1, min(self._longest, len(words) - index) + 1):
                if tuple(words[index : index + length]) in self._phrases:
                    reached[index + length] = True
        return reached[-1] is True

    def exempts(self, entity_type: str, fragment: str, source: str = "") -> bool:
        """Return whether a detection of `fragment` as `entity_type` should be left unmasked."""
        if entity_type not in NAME_TYPES or source in PROTECTED_SOURCES:
            return False
        if entity_type == "PERSON" and source in _PERSON_EVIDENCE_SOURCES:
            return False
        return self.allows(fragment)

    def drop_allowed(self, spans: list[Span], text: str) -> list[Span]:
        """Return the spans that remain once the detections of allowed names are removed."""
        if not self._phrases:
            return spans
        return [
            span
            for span in spans
            if not self.exempts(span.entity_type, text[span.start : span.end], span.source)
        ]
