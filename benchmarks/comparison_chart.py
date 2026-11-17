"""Draw the animated comparison of polanon and Presidio for the README, in English and Polish.

    python benchmarks/comparison_chart.py

writes docs/comparison.svg and docs/comparison.pl.svg. The numbers are fixed here, as measured on
data never used to tune polanon: the Polish rows of ai4privacy's OpenPII held out for testing and
the test part of CLARIN-PL's KPWr, both tools with spaCy's pl_core_news_md, Presidio as
benchmarks/compare_presidio.py sets it up. The SVG animates with CSS alone, so GitHub shows it as
an image; with reduced motion asked for, the first scene stands still.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

DOCS = Path(__file__).resolve().parent.parent / "docs"

WIDTH, HEIGHT = 960, 540
SCENE = 6.0  # seconds each scene is shown
SCENES = 3
LOOP = SCENE * SCENES

#: (label key, polanon, Presidio): fully masked shares in percent, as measured.
BY_TYPE = [
    ("postcodes", 98, 0),
    ("addresses", 64, 33),
    ("phones", 60, 34),
    ("people", 79, 70),
    ("places", 78, 81),
    ("licences", 51, 89),
]
KPWR = [("people", 90, 90), ("streets", 69, 59), ("cities", 93, 93), ("companies", 83, 86)]
WHOLE = (100 * 3733 / 5255, 100 * 3456 / 5255)  # OpenPII test part: values masked whole
FALSE_ALARMS = (435, 2641)  # spans over nothing the annotations mark

TEXT = {
    "en": {
        "title": "polanon vs Presidio on Polish personal data",
        "s1": "Held-out Polish texts: 1,000 from ai4privacy OpenPII, 5,255 values",
        "whole": "Values masked whole",
        "alarms": "False alarms: spans over nothing personal",
        "more": "points more masked whole",
        "fewer": "6× fewer",  # noqa: RUF001 - the multiplication sign, meant
        "fewer_note": "false alarms than Presidio",
        "s2": "The same texts, by kind of value",
        "postcodes": "Postcodes",
        "addresses": "Street addresses",
        "phones": "Phone numbers",
        "people": "People",
        "places": "Places",
        "licences": "Driving licences",
        "streets": "Streets",
        "cities": "Cities",
        "companies": "Companies",
        "s3": "Real Polish text: CLARIN-PL KPWr, 4,323 sentences",
        "tie": "A tie where the same spaCy model does the work",
        "footer": "Data never used for tuning. Both tools with spaCy pl_core_news_md; "
        "presidio-analyzer 2.2.366 set up for Polish.",
        "thousands": ",",
        "decimal": ".",
    },
    "pl": {
        "title": "polanon i Presidio na polskich danych osobowych",
        "s1": "Polskie teksty odłożone do testu: 1000 z ai4privacy OpenPII, 5255 wartości",
        "whole": "Wartości zamaskowane w całości",
        "alarms": "Fałszywe alarmy: zamaskowane coś, co nie jest daną osobową",
        "more": "punktu procentowego więcej w całości",
        "fewer": "6× mniej",  # noqa: RUF001 - the multiplication sign, meant
        "fewer_note": "fałszywych alarmów niż Presidio",
        "s2": "Te same teksty według rodzaju danych",
        "postcodes": "Kody pocztowe",
        "addresses": "Adresy",
        "phones": "Telefony",
        "people": "Osoby",
        "places": "Miejsca",
        "licences": "Prawa jazdy",
        "streets": "Ulice",
        "cities": "Miasta",
        "companies": "Firmy",
        "s3": "Prawdziwe polskie teksty: CLARIN-PL KPWr, 4323 zdania",
        "tie": "Remis tam, gdzie pracuje ten sam model spaCy",
        "footer": "Dane, na których polanon nie był strojony. Oba narzędzia z modelem spaCy "
        "pl_core_news_md; presidio-analyzer 2.2.366 ustawiony dla polskiego.",
        "thousands": " ",
        "decimal": ",",
    },
}

POLANON, PRESIDIO = "polanon", "Presidio"

STYLE = """
.bg{fill:#ffffff}.edge{stroke:#d0d7de}.fg{fill:#1f2328}.muted{fill:#59636e}
.track{fill:#eff2f5}.pol{fill:url(#pol)}.pre{fill:#8c959f}.win{fill:#1a7f37}.lose{fill:#59636e}
.dot{fill:#d0d7de}.on{fill:#2da44e}
@media (prefers-color-scheme:dark){
.bg{fill:#0d1117}.edge{stroke:#30363d}.fg{fill:#e6edf3}.muted{fill:#9198a1}
.track{fill:#161b22}.pre{fill:#6e7681}.win{fill:#3fb950}.lose{fill:#9198a1}.dot{fill:#30363d}
.on{fill:#3fb950}}
text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,
sans-serif}
.bar{transform-box:fill-box;transform-origin:0 50%}
.s2,.s3{opacity:0}
"""


def _pct(at: float) -> str:
    return f"{100 * at / LOOP:.2f}%"


def _scene_keyframes(index: int) -> str:
    """Visible during its own slot, fading in and out."""
    start, end = index * SCENE, (index + 1) * SCENE
    frames = [(0.0, 0), (start, 0), (start + 0.4, 1), (end - 0.4, 1), (end, 0), (LOOP, 0)]
    if index == 0:
        frames = [(0.0, 1), (end - 0.4, 1), (end, 0), (LOOP - 0.4, 0), (LOOP, 1)]
    body = "".join(f"{_pct(t)}{{opacity:{o}}}" for t, o in frames)
    return (
        f"@keyframes scene{index}{{{body}}}.s{index + 1}{{animation:scene{index} {LOOP}s infinite}}"
    )


def _grow(name: str, scene: int, delay: float, *, fade: bool = False) -> str:
    """A bar that grows, or a label that fades in, once its scene is up, and resets after it."""
    start, end = scene * SCENE + 0.3 + delay, (scene + 1) * SCENE
    if fade:
        shown, hidden = "opacity:1", "opacity:0"
    else:
        shown, hidden = "transform:scaleX(1)", "transform:scaleX(0)"
    frames = [
        (0.0, hidden),
        (start, hidden),
        (start + (0.4 if fade else 0.9), shown),
        (end - 0.05, shown),
        (end, hidden),
        (LOOP, hidden),
    ]
    body = "".join(f"{_pct(t)}{{{style}}}" for t, style in frames)
    timing = "linear" if fade else "ease-out"
    return f"@keyframes {name}{{{body}}}.{name}{{animation:{name} {LOOP}s {timing} infinite}}"


class Chart:
    """One language's SVG, built part by part with the CSS that animates it."""

    def __init__(self, language: str) -> None:
        """Start an empty chart in `language`, "en" or "pl"."""
        self.t = TEXT[language]
        self.parts: list[str] = []
        self.styles: list[str] = [STYLE]
        self.count = 0

    def text(self, x: float, y: float, content: str, cls: str, size: int, **attrs: str) -> None:
        """Add a line of text; attributes are written with hyphens for underscores."""
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        self.parts.append(
            f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}"{extra}>'
            f"{escape(content)}</text>"
        )

    def bar(
        self, x: float, y: float, w: float, h: float, cls: str, scene: int, delay: float
    ) -> None:
        """Add a bar that grows from the left when its scene comes up."""
        self.count += 1
        name = f"g{self.count}"
        self.styles.append(_grow(name, scene, delay))
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{max(w, 0.01):.1f}" height="{h}" rx="{min(h / 2, 5)}" '
            f'class="bar {cls} {name}"/>'
        )

    def label(
        self,
        x: float,
        y: float,
        content: str,
        cls: str,
        size: int,
        scene: int,
        delay: float,
        **attrs: str,
    ) -> None:
        """Add text that fades in when its scene comes up."""
        self.count += 1
        name = f"f{self.count}"
        self.styles.append(_grow(name, scene, delay, fade=True))
        self.text(x, y, content, f"{cls} {name}", size, **attrs)

    def number(self, value: float) -> str:
        """A whole number with the language's thousands separator."""
        return f"{value:,.0f}".replace(",", self.t["thousands"])

    def pair(
        self,
        y: float,
        label: str,
        ours: float,
        theirs: float,
        top: float,
        scene: int,
        delay: float,
        *,
        unit: str = "%",
        lower_is_better: bool = False,
    ) -> None:
        """One comparison: a label and two bars, polanon above Presidio, the winner in green."""
        left, width, h = 300, 520, 18
        self.text(left - 16, y + 24, label, "fg", 17, text_anchor="end", font_weight="600")
        for row, (value, cls) in enumerate(((ours, "pol"), (theirs, "pre"))):
            other = theirs if row == 0 else ours
            top_y = y + row * (h + 6)
            self.parts.append(
                f'<rect x="{left}" y="{top_y}" width="{width}" height="{h}" rx="5" class="track"/>'
            )
            self.bar(left, top_y, width * value / top, h, cls, scene, delay + row * 0.15)
            better = value < other if lower_is_better else value > other
            shown = f"{value:.0f}{unit}" if unit == "%" else self.number(value)
            tone = "win" if better else ("fg" if value == other else "lose")
            self.label(
                left + width + 12,
                top_y + h - 3,
                shown,
                tone,
                15,
                scene,
                delay + 0.8,
                font_weight="700" if better else "500",
            )

    def draw(self) -> str:
        """The whole SVG document."""
        t = self.t
        self.parts.append(
            f'<rect x="1" y="1" width="{WIDTH - 2}" height="{HEIGHT - 2}" rx="14" '
            'class="bg edge" stroke-width="1.5"/>'
        )
        self.text(40, 52, t["title"], "fg", 26, font_weight="700")
        self.parts.append('<rect x="40" y="70" width="14" height="14" rx="3" class="pol"/>')
        self.text(60, 82, POLANON, "fg", 15, font_weight="600")
        self.parts.append('<rect x="150" y="70" width="14" height="14" rx="3" class="pre"/>')
        self.text(170, 82, PRESIDIO, "fg", 15, font_weight="600")

        # scene 1: the whole held-out set
        self.parts.append('<g class="s1">')
        self.text(40, 130, t["s1"], "muted", 16)
        self.pair(165, t["whole"], WHOLE[0], WHOLE[1], 100, 0, 0.0)
        name, note = (part.strip() for part in t["alarms"].split(":", 1))
        self.pair(
            255,
            name,
            FALSE_ALARMS[0],
            FALSE_ALARMS[1],
            FALSE_ALARMS[1],
            0,
            0.6,
            unit="",
            lower_is_better=True,
        )
        self.text(284, 300, note, "muted", 13, text_anchor="end")
        points = f"+{WHOLE[0] - WHOLE[1]:.1f}".replace(".", t["decimal"])
        self.label(300, 380, points, "win", 40, 0, 1.5, font_weight="800")
        self.label(300, 408, t["more"], "muted", 14, 0, 1.6)
        self.label(620, 380, t["fewer"], "win", 40, 0, 1.9, font_weight="800")
        self.label(620, 408, t["fewer_note"], "muted", 14, 0, 2.0)
        self.parts.append("</g>")

        # scene 2: by kind of value
        self.parts.append('<g class="s2">')
        self.text(40, 130, t["s2"], "muted", 16)
        for row, (key, ours, theirs) in enumerate(BY_TYPE):
            self.pair(145 + row * 51, t[key], ours, theirs, 100, 1, row * 0.25)
        self.parts.append("</g>")

        # scene 3: real Polish text
        self.parts.append('<g class="s3">')
        self.text(40, 130, t["s3"], "muted", 16)
        for row, (key, ours, theirs) in enumerate(KPWR):
            self.pair(165 + row * 62, t[key], ours, theirs, 100, 2, row * 0.3)
        self.label(40, 445, t["tie"], "muted", 15, 2, 1.8, font_style="italic")
        self.parts.append("</g>")

        # progress dots and footer
        for index in range(SCENES):
            x = WIDTH / 2 - 20 + index * 20
            self.parts.append(f'<circle cx="{x}" cy="478" r="5" class="dot"/>')
            self.parts.append(f'<circle cx="{x}" cy="478" r="5" class="on s{index + 1}"/>')
        self.text(40, 515, t["footer"], "muted", 12)

        self.styles += [_scene_keyframes(i) for i in range(SCENES)]
        self.styles.append(
            "@media (prefers-reduced-motion:reduce){*{animation:none!important}"
            ".s1{opacity:1}.s2,.s3{opacity:0}}"
        )
        gradient = (
            '<defs><linearGradient id="pol" x1="0" x2="1"><stop offset="0" stop-color="#2da44e"/>'
            '<stop offset="1" stop-color="#1f883d"/></linearGradient></defs>'
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(t["title"])}">'
            f"<title>{escape(t['title'])}</title>{gradient}"
            f"<style>{''.join(self.styles)}</style>{''.join(self.parts)}</svg>\n"
        )


def main() -> None:
    """Write both languages' SVGs into docs/."""
    for language, name in (("en", "comparison.svg"), ("pl", "comparison.pl.svg")):
        (DOCS / name).write_text(Chart(language).draw(), encoding="utf-8", newline="\n")
        print("written", DOCS / name)


if __name__ == "__main__":
    main()
