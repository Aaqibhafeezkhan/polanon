"""Draw the README's terminal demo, in English and Polish, as an animated SVG.

    python benchmarks/terminal_demo.py

writes docs/demo.svg and docs/demo.pl.svg, and the button to the demo in the browser,
docs/try-demo.svg and docs/try-demo.pl.svg. Three scenes play in a loop: the library in Python,
the gateway in Docker in front of a tool, and the comparison with Presidio. Every output shown is
what polanon prints: the masked texts and the restored answers come from Anonymizer, the banner
and the log line from polanon-gateway, the scores from benchmarks/run_external.py --presidio. The
SVG animates with CSS alone, so GitHub shows it as an image; with reduced motion asked for, the
first scene stands still, complete.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

DOCS = Path(__file__).resolve().parent.parent / "docs"

WIDTH, HEIGHT = 960, 500
LEFT, FIRST, LINE_H = 24, 74, 25
CHAR_W, FONT = 8.95, 15
TYPE_SPEED = 0.022  # seconds per typed character
AFTER_TYPED, AFTER_OUTPUT, HOLD, FADE = 0.3, 0.25, 3.6, 0.4
MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace"

#: A piece of a line: its text and its colour class.
Piece = tuple[str, str]
#: A line: whether it is typed, and its pieces; the prompt of a typed line is shown at once.
Line = tuple[bool, list[Piece]]

STYLE = f"""
.win{{fill:#0d1117;stroke:#30363d}}.bar{{fill:#161b22}}.title{{fill:#8b949e}}
.p{{fill:#3fb950}}.c{{fill:#c9d1d9}}.s{{fill:#a5d6ff}}.t{{fill:#ffa657;font-weight:700}}
.m{{fill:#8b949e}}.g{{fill:#3fb950;font-weight:700}}.w{{fill:#f0f6fc}}
.cover{{fill:#0d1117;transform-box:fill-box;transform-origin:100% 50%;transform:scaleX(0)}}
.cursor{{fill:#c9d1d9;opacity:0}}
text{{font-family:{MONO}}}
"""


def _tokens(text: str, base: str) -> list[Piece]:
    """A text with every <TOKEN> in it set apart."""
    pieces: list[Piece] = []
    rest = text
    while "<" in rest and ">" in rest[rest.index("<") :]:
        start = rest.index("<")
        end = rest.index(">", start) + 1
        if start:
            pieces.append((rest[:start], base))
        pieces.append((rest[start:end], "t"))
        rest = rest[end:]
    if rest:
        pieces.append((rest, base))
    return pieces


def _scenes(language: str) -> list[list[Line]]:
    en = language == "en"
    python: list[Line] = [
        (
            False,
            [
                (
                    "# in Python: mask, call the model, restore its answer"
                    if en
                    else "# w Pythonie: zamaskuj, wyślij do modelu, przywróć odpowiedź",
                    "m",
                )
            ],
        ),
        (True, [(">>> ", "p"), ("from polanon import Anonymizer, default_guards", "c")]),
        (True, [(">>> ", "p"), ("anonymizer = Anonymizer(default_guards(with_ner=False))", "c")]),
        (
            True,
            [
                (">>> ", "p"),
                ("masked = anonymizer.anonymize_sync(", "c"),
                ('"Zwrot dla Jana Kowalskiego, PESEL 50810112354, na konto PL61', "s"),
            ],
        ),
        (True, [("    ", "c"), ('109010140000071219812874."', "s"), (")", "c")]),
        (True, [(">>> ", "p"), ("masked.anonymized", "c")]),
        (False, _tokens("'Zwrot dla <OSOBA_1>, PESEL <PESEL_1>, na konto <IBAN_1>.'", "s")),
        (
            True,
            [
                (">>> ", "p"),
                ("answer = ", "c"),
                ('"Zwrot dla <OSOBA_1> zatwierdzony."', "s"),
                ("  # " + ("what the model replied" if en else "tak odpowiedział model"), "m"),
            ],
        ),
        (True, [(">>> ", "p"), ("Anonymizer.restore(answer, masked.mapping)", "c")]),
        (False, [("'Zwrot dla Jana Kowalskiego zatwierdzony.'", "s")]),
        (False, [("", "c")]),
        (
            False,
            [
                (
                    "# the model only ever saw <OSOBA_1>"
                    if en
                    else "# model widział tylko <OSOBA_1>",
                    "m",
                )
            ],
        ),
    ]
    gateway: list[Line] = [
        (
            False,
            [
                (
                    "# or without code: the gateway in Docker, for Claude Code, n8n, Open WebUI"
                    if en
                    else "# albo bez kodu: bramka w Dockerze, dla Claude Code, n8n, Open WebUI",
                    "m",
                )
            ],
        ),
        (
            True,
            [
                ("$ ", "p"),
                (
                    "docker run -p 127.0.0.1:8787:8787 "
                    "-e POLANON_UPSTREAM_KEY -e POLANON_GATEWAY_KEY \\",
                    "c",
                ),
            ],
        ),
        (
            True,
            [
                ("    ", "c"),
                ("ghcr.io/boromat/polanon-gateway:0.4 --upstream https://api.openai.com/v1", "c"),
            ],
        ),
        (
            False,
            [
                (
                    "polanon gateway on http://0.0.0.0:8787/v1 -> "
                    "https://api.openai.com/v1; masking",
                    "m",
                )
            ],
        ),
        (False, [("with rules and pl_core_news_md; clients send the gateway key", "m")]),
        (
            False,
            [
                (
                    "polanon.gateway POST /chat/completions -> 200 in 0.87s, masked {'PERSON': 1, "
                    "'PESEL': 1, 'PHONE': 1}",
                    "m",
                )
            ],
        ),
        (False, [("", "c")]),
        (False, [("# the tool sent" if en else "# narzędzie wysłało", "m")]),
        (
            False,
            [
                (
                    "Odpisz Janowi Kowalskiemu, PESEL 50810112354, tel. 512 345 678, "
                    "że korekta jest "
                    "gotowa.",
                    "w",
                )
            ],
        ),
        (False, [("# the provider got" if en else "# dostawca dostał", "m")]),
        (
            False,
            _tokens(
                "Odpisz <OSOBA_1>, PESEL <PESEL_1>, tel. <TELEFON_1>, że korekta jest gotowa.", "w"
            ),
        ),
        (
            False,
            [
                (
                    "# the tool got back, the name in the case the model asked for"
                    if en
                    else "# narzędzie dostało, z nazwiskiem w przypadku, o który prosił model",
                    "m",
                )
            ],
        ),
        (False, [("Szanowny Panie Janie Kowalski, korekta jest gotowa.", "s")]),
    ]
    header = (
        "OpenPII, 1,000 held-out Polish texts" if en else "OpenPII, 1000 polskich tekstów do testu"
    )
    rows = [
        ("values masked whole" if en else "zamaskowane w całości", "71%", "66%", True),
        ("false alarms" if en else "fałszywe alarmy", "306", "2,641" if en else "2 641", True),
        ("postcodes" if en else "kody pocztowe", "98%", "0%", True),
        ("street addresses" if en else "adresy", "64%", "33%", True),
        ("phone numbers" if en else "telefony", "60%", "34%", True),
    ]
    compare: list[Line] = [
        (
            False,
            [
                (
                    "# and against Presidio, on Polish data polanon was never tuned on"
                    if en
                    else "# i wobec Presidio, na polskich danych, "
                    "na których polanon nie był strojony",
                    "m",
                )
            ],
        ),
        (True, [("$ ", "p"), ("python benchmarks/run_external.py --presidio", "c")]),
        (False, [(f"{header:<44}", "w"), (f"{'polanon':>10}", "g"), (f"{'Presidio':>12}", "m")]),
    ]
    for label, ours, theirs, better in rows:
        compare.append(
            (
                False,
                [
                    (f"  {label:<42}", "c"),
                    (f"{ours:>10}", "g" if better else "c"),
                    (f"{theirs:>12}", "m"),
                ],
            )
        )
    compare += [
        (False, [("KPWr, " + ("real Polish text" if en else "prawdziwe polskie teksty"), "w")]),
        (
            False,
            [
                (f"  {'people' if en else 'osoby':<42}", "c"),
                (f"{'90%':>10}", "c"),
                (f"{'90%':>12}", "c"),
            ],
        ),
        (False, [("", "c")]),
        (
            False,
            [
                (
                    "# 8x fewer false alarms, and a tie where the same spaCy model does the work"
                    if en
                    else "# 8 razy mniej fałszywych alarmów, "
                    "remis tam, gdzie pracuje ten sam spaCy",
                    "m",
                )
            ],
        ),
    ]
    return [python, gateway, compare]


class Demo:
    """One language's demo, built line by line with the CSS that plays it."""

    def __init__(self, language: str) -> None:
        """Lay out the scenes of `language`, "en" or "pl", one after another."""
        self.language = language
        self.parts: list[str] = []
        self.styles: list[str] = [STYLE]
        self.n = 0
        self.scenes = _scenes(language)
        self.windows: list[tuple[float, float]] = []
        clock = 0.6
        self.starts: list[list[float]] = []
        for scene in self.scenes:
            begin = clock
            starts = []
            for typed, pieces in scene:
                starts.append(clock)
                if typed:
                    clock += len("".join(p for p, _ in pieces[1:])) * TYPE_SPEED + AFTER_TYPED
                else:
                    clock += AFTER_OUTPUT
            clock += HOLD
            self.windows.append((begin, clock))
            self.starts.append(starts)
            clock += FADE + 0.3
        self.loop = clock

    def _pct(self, seconds: float) -> str:
        return f"{100 * seconds / self.loop:.3f}%"

    def _frames(self, prefix: str, points: list[tuple[float, str]], timing: str = "linear") -> str:
        self.n += 1
        name = f"{prefix}{self.n}"
        body = "".join(f"{self._pct(t)}{{{d}}}" for t, d in points)
        self.styles.append(
            f"@keyframes {name}{{{body}}}"
            f".{name}{{animation:{name} {self.loop:.2f}s {timing} infinite}}"
        )
        return name

    def _text(self, x: float, y: float, text: str, cls: str) -> None:
        if not text:
            return
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{FONT}" '
            f'textLength="{len(text) * CHAR_W:.1f}" lengthAdjust="spacingAndGlyphs" '
            f'xml:space="preserve">{escape(text)}</text>'
        )

    def draw(self) -> str:
        """The whole SVG document."""
        title = "polanon: reversible PII masking for LLM calls"
        if self.language == "pl":
            title = "polanon: odwracalne maskowanie danych osobowych dla LLM"
        self.parts.append(
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{HEIGHT - 1}" rx="12" class="win"/>'
            f'<path d="M0.5 12.5a12 12 0 0 1 12-12h{WIDTH - 25}a12 12 0 0 1 12 12'
            f'v24h-{WIDTH - 1}z" '
            'class="bar"/>'
        )
        for index, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
            self.parts.append(f'<circle cx="{22 + index * 20}" cy="18.5" r="6" fill="{color}"/>')
        self.parts.append(
            f'<text x="{WIDTH / 2}" y="23" class="title" font-size="14" text-anchor="middle">'
            f"{escape(title)}</text>"
        )
        for index, scene in enumerate(self.scenes):
            begin, end = self.windows[index]
            shown = self._frames(
                "sc",
                [
                    (0, "opacity:0"),
                    (max(begin - 0.01, 0), "opacity:0"),
                    (begin, "opacity:1"),
                    (end, "opacity:1"),
                    (end + FADE, "opacity:0"),
                    (self.loop, "opacity:0"),
                ],
            )
            hidden = ' opacity="0"' if index else ""
            self.parts.append(f'<g class="{shown}"{hidden}>')
            for row, (typed, pieces) in enumerate(scene):
                self._line(row, typed, pieces, self.starts[index][row], end)
            self.parts.append("</g>")
        self.styles.append("@media (prefers-reduced-motion:reduce){*{animation:none!important}}")
        label = escape(title)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{label}">'
            f"<title>{label}</title><style>{''.join(self.styles)}</style>"
            f"{''.join(self.parts)}</svg>\n"
        )

    def _line(self, row: int, typed: bool, pieces: list[Piece], start: float, end: float) -> None:
        """One line: it appears at `start`; a typed one is revealed a character at a time."""
        y = FIRST + row * LINE_H
        appear = self._frames(
            "ln",
            [
                (0, "opacity:0"),
                (start, "opacity:0"),
                (start + 0.01, "opacity:1"),
                (self.loop, "opacity:1"),
            ],
        )
        self.parts.append(f'<g class="{appear}">')
        column = 0
        for text, cls in pieces:
            self._text(LEFT + column * CHAR_W, y, text, cls)
            column += len(text)
        if typed:
            prompt = len(pieces[0][0])
            chars = column - prompt
            duration = chars * TYPE_SPEED
            x = LEFT + prompt * CHAR_W
            cover = self._frames(
                "cv",
                [
                    (0, "transform:scaleX(1)"),
                    (start, "transform:scaleX(1)"),
                    (start + duration, "transform:scaleX(0)"),
                    (self.loop, "transform:scaleX(0)"),
                ],
                timing=f"steps({chars},end)",
            )
            self.parts.append(
                f'<rect x="{x:.1f}" y="{y - FONT}" width="{chars * CHAR_W + 2:.1f}" '
                f'height="{LINE_H - 2}" class="cover {cover}"/>'
            )
            cursor = self._frames(
                "cu",
                [
                    (0, "opacity:0;transform:translateX(0px)"),
                    (start, "opacity:1;transform:translateX(0px)"),
                    (start + duration, f"opacity:1;transform:translateX({chars * CHAR_W:.1f}px)"),
                    (
                        start + duration + AFTER_TYPED * 0.8,
                        f"opacity:0;transform:translateX({chars * CHAR_W:.1f}px)",
                    ),
                    (self.loop, f"opacity:0;transform:translateX({chars * CHAR_W:.1f}px)"),
                ],
                timing=f"steps({chars},end)",
            )
            self.parts.append(
                f'<rect x="{x:.1f}" y="{y - FONT + 1}" width="{CHAR_W:.1f}" height="{FONT + 3}" '
                f'class="cursor {cursor}"/>'
            )
        self.parts.append("</g>")


SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"

#: The button's title and note with the width each takes in Segoe UI, which every other font is
#: fitted to, so the layout holds wherever the button is shown.
BUTTON = {
    "en": ("Try polanon in your browser", 226.0, "No install. Nothing leaves the tab.", 186.0),
    "pl": (
        "Wypróbuj polanon w przeglądarce",
        276.0,
        "Bez instalacji. Nic nie wychodzi z karty.",
        212.0,
    ),
}
BUTTON_LOOP = 4.4  # seconds for the name to be masked and come back


def button(language: str) -> str:
    """The animated button that leads from the README to the demo in the browser.

    A name in its chip is masked into its token and back, so the button shows what the demo does.
    """
    title, title_w, note, note_w = BUTTON[language]
    text_x, chip_w, chip_h = 68, 122, 32
    chip_x = text_x + max(title_w, note_w) + 22
    width = round(chip_x + chip_w + 14)
    chip_y, middle = (64 - chip_h) / 2, chip_x + chip_w / 2
    loop = f"{BUTTON_LOOP}s"
    style = (
        f"text{{font-family:{SANS}}}.mono{{font-family:{MONO}}}"
        f".a{{animation:a {loop} ease-in-out infinite}}"
        f".b{{opacity:0;animation:b {loop} ease-in-out infinite}}"
        f".scan{{opacity:0;animation:scan {loop} linear infinite}}"
        "@keyframes a{0%,36%{opacity:1}44%,86%{opacity:0}94%,100%{opacity:1}}"
        "@keyframes b{0%,36%{opacity:0}44%,86%{opacity:1}94%,100%{opacity:0}}"
        "@keyframes scan{0%,28%{opacity:0;transform:translateX(0)}30%{opacity:1}"
        f"44%{{opacity:1;transform:translateX({chip_w + 3}px)}}"
        f"46%,100%{{opacity:0;transform:translateX({chip_w + 3}px)}}}}"
        "@media (prefers-reduced-motion:reduce){.a,.b,.scan{animation:none}.a{opacity:0}"
        ".b{opacity:1}}"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="64" '
        f'viewBox="0 0 {width} 64" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title><defs>"
        '<linearGradient id="bg" x1="0" x2="0" y1="0" y2="1">'
        '<stop offset="0" stop-color="#1f2630"/><stop offset="1" stop-color="#0d1117"/>'
        "</linearGradient>"
        '<linearGradient id="edge" x1="0" x2="1">'
        '<stop offset="0" stop-color="#3fb950"/><stop offset="1" stop-color="#58a6ff"/>'
        "</linearGradient>"
        '<linearGradient id="tile" x1="0" x2="1" y1="0" y2="1">'
        '<stop offset="0" stop-color="#4ac45f"/><stop offset="1" stop-color="#1f883d"/>'
        "</linearGradient>"
        f'<clipPath id="chip"><rect x="{chip_x}" y="{chip_y}" width="{chip_w}" '
        f'height="{chip_h}" rx="8"/></clipPath>'
        f"</defs><style>{style}</style>"
        f'<rect x=".75" y=".75" width="{width - 1.5}" height="62.5" rx="14" fill="url(#bg)" '
        'stroke="url(#edge)" stroke-width="1.5"/>'
        '<rect x="12" y="12" width="40" height="40" rx="10" fill="url(#tile)"/>'
        '<path d="M27.5 22.5v19l15-9.5z" fill="#ffffff"/>'
        f'<text x="{text_x}" y="29" font-size="17" font-weight="700" fill="#f0f6fc" '
        f'textLength="{title_w}" lengthAdjust="spacingAndGlyphs">{escape(title)}</text>'
        f'<text x="{text_x}" y="47" font-size="12.5" fill="#8b949e" '
        f'textLength="{note_w}" lengthAdjust="spacingAndGlyphs">{escape(note)}</text>'
        f'<rect x="{chip_x}" y="{chip_y}" width="{chip_w}" height="{chip_h}" rx="8" '
        'fill="#0d1117" stroke="#30363d"/>'
        '<g clip-path="url(#chip)">'
        f'<text class="mono a" x="{middle}" y="{chip_y + 21}" text-anchor="middle" '
        'font-size="13" fill="#c9d1d9">Jan Kowalski</text>'
        f'<text class="mono b" x="{middle}" y="{chip_y + 21}" text-anchor="middle" '
        'font-size="13" font-weight="700" fill="#ffa657">&lt;OSOBA_1&gt;</text>'
        f'<rect class="scan" x="{chip_x - 3}" y="{chip_y}" width="3" height="{chip_h}" '
        'fill="#3fb950"/>'
        "</g></svg>\n"
    )


def main() -> None:
    """Write both languages' demos and buttons into docs/."""
    for language, suffix in (("en", ""), ("pl", ".pl")):
        drawn = {
            f"demo{suffix}.svg": Demo(language).draw(),
            f"try-demo{suffix}.svg": button(language),
        }
        for name, svg in drawn.items():
            (DOCS / name).write_text(svg, encoding="utf-8", newline="\n")
            print("written", DOCS / name)


if __name__ == "__main__":
    main()
