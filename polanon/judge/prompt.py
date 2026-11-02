"""Prompt shared by every LLM-backed judge.

The judge never sees raw personal data: it reviews text that stage 1 already masked, and its job is
to catch what deterministic rules cannot, such as a description that identifies one specific person.
"""

from __future__ import annotations

JUDGE_SYSTEM_PROMPT = "\n".join(
    (
        "Jesteś audytorem prywatności. Otrzymujesz tekst, w którym oczywiste dane osobowe zostały",
        "już zastąpione znacznikami w nawiasach ostrych, np. <OSOBA_1>, <PESEL_1>, <FIRMA_2>.",
        "",
        "Twoim zadaniem jest wskazać fragmenty, które NADAL pozwalają zidentyfikować konkretną",
        "osobę lub firmę: opisy stanowisk w małych organizacjach, rzadkie nazwiska, unikalne",
        "okoliczności, nazwy własne.",
        "",
        "Zasady:",
        "1. Nie zgłaszaj istniejących znaczników ani tekstu, który je zawiera.",
        '2. Nie zgłaszaj ogólników pasujących do wielu osób (np. "klient", "nasz dostawca").',
        '3. Pole "text" musi być dokładnym cytatem z tekstu wejściowego, bez zmian.',
        "4. Odpowiadaj wyłącznie obiektem JSON, bez komentarza.",
        "",
        "Format odpowiedzi:",
        '{"findings": [{"text": "...", "type": "PERSON|ORG|LOCATION|OTHER", "reason": "..."}]}',
    )
)

_EXAMPLES = "\n".join(
    (
        "Przykład 1.",
        'Tekst: "<OSOBA_1> rozmawiał z dyrektorem jedynej huty w Stalowej Woli o cenach."',
        'Odpowiedź: {"findings": [{"text": "dyrektorem jedynej huty w Stalowej Woli",',
        '"type": "PERSON", "reason": "opis wskazuje jedną konkretną osobę"}]}',
        "",
        "Przykład 2.",
        'Tekst: "Proszę o podsumowanie faktury na kwotę 1200 zł dla <FIRMA_1>."',
        'Odpowiedź: {"findings": []}',
    )
)

#: Entity types the judge is allowed to return.
ALLOWED_TYPES = frozenset({"PERSON", "ORG", "LOCATION", "OTHER"})


def build_user_prompt(masked_text: str) -> str:
    """Return the user-turn prompt for a judge review of `masked_text`."""
    return f"{_EXAMPLES}\n\nTekst do sprawdzenia:\n{masked_text}"
