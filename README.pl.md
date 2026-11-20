# polanon

[![CI](https://github.com/boromat/polanon/actions/workflows/ci.yml/badge.svg)](https://github.com/boromat/polanon/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/boromat/polanon/blob/main/pyproject.toml)
[![Licencja: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/boromat/polanon/blob/main/LICENSE)

[English](https://github.com/boromat/polanon/blob/main/README.md) | **Polski**

**Odwracalne maskowanie danych osobowych w zapytaniach do LLM, przede wszystkim dla języka polskiego.**
Zamaskuj dane osobowe, zanim prompt opuści twoją sieć, wyślij do modelu same tokeny, a oryginały
wstaw z powrotem do jego odpowiedzi, nawet wtedy, gdy przychodzi strumieniowo. Czysty Python: rdzeń
nie potrzebuje niczego poza biblioteką standardową.

[Pisali o nas](#pisali-o-polanon)

<p align="center"><a href="https://boromat.pl/demo/"><img src="https://raw.githubusercontent.com/boromat/polanon/main/docs/try-demo.pl.svg" alt="Wypróbuj polanon w przeglądarce"></a></p>

**Bez pisania kodu**, z bramką w Dockerze. Claude Code, n8n, Open WebUI, LibreChat, Hermes Agent
i każde inne narzędzie, które mówi API OpenAI albo Anthropic, działają jak dotąd, a każde
wywołanie jest maskowane, zanim wyjdzie:

```bash
export POLANON_UPSTREAM_KEY=sk-...                 # klucz twojego dostawcy
export POLANON_GATEWAY_KEY=$(openssl rand -hex 16)  # klucz, który wyślą twoje narzędzia
docker run --rm -p 127.0.0.1:8787:8787 -e POLANON_UPSTREAM_KEY -e POLANON_GATEWAY_KEY \
    ghcr.io/boromat/polanon-gateway:0.4 --upstream https://api.openai.com/v1
```

Potem ustaw w narzędziu adres bazowy `http://127.0.0.1:8787/v1`, a jako klucz API wartość
`POLANON_GATEWAY_KEY`. Jedna bramka obsługuje jednego dostawcę: dla Claude Code uruchom drugą,
z `--upstream https://api.anthropic.com/v1`, kluczem Anthropic i `-p 127.0.0.1:8788:8787`, a potem
`ANTHROPIC_BASE_URL=http://127.0.0.1:8788 ANTHROPIC_API_KEY=$POLANON_GATEWAY_KEY claude`.
Wszystkie narzędzia, bramkę bez Dockera i to, co robi z wywołaniem, opisuje
[rozdział o bramce](#bramka-dla-każdego-narzędzia-zgodnego-z-api-openai).

**W Pythonie**, we własnym kodzie: zamaskuj tekst, wyślij go, przywróć odpowiedź.

<p align="center"><img src="https://raw.githubusercontent.com/boromat/polanon/main/docs/demo.pl.svg" alt="polanon w terminalu: maskowanie polskiego tekstu w Pythonie i przywracanie odpowiedzi, bramka w Dockerze przed narzędziem i wyniki wobec Presidio"></p>

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

answer = "Korekta dla <OSOBA_1> (PESEL <PESEL_1>) jest gotowa."  # tak odpowiedział model
print(Anonymizer.restore(answer, result.mapping))
# Korekta dla Jana Kowalskiego (PESEL 50810112354) jest gotowa.
```

Model widzi wyłącznie `<OSOBA_1>`. Mapowanie, które zamienia ten token z powrotem na imię
i nazwisko, nigdy nie opuszcza twojego procesu. `anonymize_sync` działa w skryptach, widokach Django
i Flaska, zadaniach Celery i notatnikach; w kodzie asynchronicznym `await anonymizer.anonymize(...)`
wykonuje tę samą pracę w wątku roboczym.

## Po co kolejna biblioteka do danych osobowych

Narzędzia ogólnego przeznaczenia powstają z myślą o języku angielskim. Polskie dane wymagają trzech
rzeczy, których te narzędzia nie zapewniają od razu po instalacji:

- **Sumy kontrolne.** PESEL, NIP, REGON, numer dowodu osobistego, IBAN czy numer karty płatniczej
  jest walidowany, a nie tylko dopasowywany do wzorca: suma kontrolna odrzuca większość przypadkowych
  ciągów cyfr (w przypadku PESEL-u, razem z datą urodzenia, około 98% z nich), a prawidłowy numer
  jest zawsze akceptowany. Numery KRS, które nie mają publicznej sumy kontrolnej, oraz paszporty też
  są rozpoznawane, gdy występują obok swojej etykiety. Jedenaście cyfr tuż po słowie PESEL jest
  maskowanych nawet z literówką.
- **Siedem przypadków gramatycznych.** „Jan Kowalski”, „Jana Kowalskiego” i „z Janem Kowalskim” to
  ta sama osoba. Polskie imiona są rozpoznawane w każdym z nich, także pisane małymi literami i bez
  polskich znaków, razem z nazwiskiem, które się z nimi zgadza, bez żadnego modelu językowego.
- **Polskie etykiety w modelach.** Polskie modele spaCy oznaczają osoby jako `persName`, a nie
  `PERSON`. Pipeline skonfigurowany dla angielskich etykiet nie znajduje żadnych imion ani nazwisk
  i w żaden sposób o tym nie informuje.

Radzi sobie też z zagranicznymi fakturami: numery VAT z UE, Wielkiej Brytanii i Szwajcarii, numery
rejestrowe spółek, Bankgiro, Plusgiro i sort code, międzynarodowe numery telefonów, adresy i kody
pocztowe.

## Na tle innych

<p align="center"><img src="https://raw.githubusercontent.com/boromat/polanon/main/docs/comparison.pl.svg" alt="polanon i Presidio na polskich danych, na których polanon nie był strojony: wartości zamaskowane w całości, fałszywe alarmy, rodzaje danych i osoby w prawdziwych tekstach"></p>

|  | polanon | [Presidio](https://github.com/data-privacy-stack/presidio) | [pii-veil](https://pypi.org/project/pii-veil/) | [piimask](https://pypi.org/project/piimask/) |
|---|---|---|---|---|
| Przywracanie danych w odpowiedzi modelu | Tak, ze stabilnych tokenów takich jak `<OSOBA_1>`, także strumieniowo | Nie jest wbudowane: jego deanonymizer odszyfrowuje fragmenty na znanych pozycjach; dokumentacja zawiera przykład, który przywraca dane w odpowiedzi LLM | Tak, ze stabilnych tokenów i zapisanego mapowania | Tak, ze stabilnych tokenów |
| Polskie identyfikatory z sumą kontrolną | PESEL, NIP, REGON, dowód osobisty, IBAN, karty płatnicze | PESEL; IBAN i karty płatnicze w wersji ogólnej | PESEL, NIP, REGON, IBAN, karty płatnicze; dowód osobisty i paszport tylko według wzorca | Brak; IBAN i karty płatnicze w wersji ogólnej |
| Polskie imiona i nazwiska w każdym przypadku gramatycznym | Dla imion z listy i zgodnych z nimi nazwisk, bez modelu językowego; resztę dokłada spaCy | Z polskim modelem spaCy i zmapowaną etykietą `persName` | Nie są wykrywane | Nie są wykrywane; jego README sugeruje dodanie modelu NER |
| Czego potrzebuje rdzeń | Niczego poza biblioteką standardową | spaCy, numpy, pydantic i kilku innych pakietów; modelu językowego tylko do imion i nazwisk oraz miejsc | pii-core i typer | Niczego poza biblioteką standardową |
| Na polskich danych, na których żadne nie było strojone ([OpenPII, KPWr](https://github.com/boromat/polanon/blob/main/benchmarks/README.md#on-data-polanon-was-never-tuned-on)) | 71% z 5255 wartości zamaskowanych w całości z `pl_core_news_md`, 306 fałszywych alarmów; 90% osób w prawdziwych tekstach | 66%, 2641 fałszywych alarmów; 90% osób (presidio-analyzer 2.2.366, ten sam model) | Nie mierzono | Nie mierzono |
| Na własnych [przykładowych dokumentach](https://github.com/boromat/polanon/blob/main/benchmarks/README.md) biblioteki polanon, na których była strojona | 98% zamaskowanych w całości z `pl_core_news_md`, 86% przy samych regułach | 67% z tym samym modelem | Nie mierzono | Nie mierzono |
| Licencja | MIT | MIT | Apache-2.0 | MIT |

Porównanie na podstawie własnej dokumentacji i kodu źródłowego każdego z projektów, stan na
3 listopada 2026 r.; pomiary z 17 listopada 2026 r. Uruchomiono tylko Presidio, skryptami
z `benchmarks/`. Trzynaście własnych dokumentów biblioteki polanon nie jest neutralnym terenem:
napisał je jej autor i dostrajał na nich reguły, a Presidio dostał tylko polski model, także do
trzech faktur z zagranicy. Neutralne są polskie wiersze OpenPII od ai4privacy i KPWr od CLARIN-PL:
na mierzonych częściach polanon nigdy nie był strojony, a różnica maleje tam do pięciu punktów
i osiem razy mniej fałszywych alarmów, z remisem na osobach, gdzie pracuje ten sam model spaCy.
Presidio wygrywa na losowych kodach bez etykiety, które czyta wzorcami o niskiej pewności,
oznaczającymi też kody produktów i numery zamówień. Każde z tych narzędzi ma swoje miejsce. Presidio
służy do wielu języków, danych ustrukturyzowanych i integracji z Azure, pii-veil do polskich
identyfikatorów na licencji Apache-2.0, gdy imiona i nazwiska nie mają znaczenia, a piimask do
identyfikatorów niezależnych od języka (e-mail, numery telefonów, karty płatnicze, IBAN, adresy IP,
adresy URL) i amerykańskich numerów ubezpieczenia społecznego, ze strategiami podstawiania
fałszywych wartości, haszowania i częściowego maskowania oraz zaszyfrowanym sejfem. polanon jest dla
polskiego tekstu wysyłanego do modelu, gdy imiona i nazwiska w nim zawarte oraz przywrócona
odpowiedź liczą się tak samo jak numery.

## Instalacja

| Czego potrzebujesz | Co zainstalować |
|---|---|
| Identyfikatory, dane kontaktowe, sekrety, polskie imiona i nazwiska, nazwy firm rozpoznawane po formie prawnej | `pip install polanon` |
| Dodatkowo miejsca oraz imiona, nazwiska i organizacje, których nie wychwyci żadna reguła | `pip install "polanon[ner]"`, a potem `python -m spacy download pl_core_news_md` |
| Sędziowie etapu 2 (lokalny model w Ollamie albo zdalny) | `pip install "polanon[judge]"` |
| Komponenty do potoków [Haystacka 3](#z-haystackiem) | `pip install "polanon[haystack]"` |
| [Bramka](#bramka-dla-każdego-narzędzia-zgodnego-z-api-openai) dla narzędzi korzystających z API OpenAI | `pip install "polanon[gateway]"` |
| Bramka jako obraz Dockera, z modelem spaCy w środku | `docker pull ghcr.io/boromat/polanon-gateway:0.4` |

Następnie użyj `default_guards()` z modelem albo `default_guards(with_ner=False)` bez niego.
Inny model spaCy to kwestia jednego argumentu: `default_guards(spacy_model="pl_core_news_lg")`.
Działają też nazwy pakietów skopiowane z pip (`pl-core-news-lg`). Model z lokalnego katalogu można
podać jako ścieżkę; katalog musi zawierać `meta.json` i `config.cfg`. Brak pakietu albo katalogu
z modelem jest zgłaszany już przy budowaniu detektorów, a sam model ładuje się przy pierwszym użyciu.
Zainstalowany pakiet modelu ma pierwszeństwo przed lokalną ścieżką o tej samej nazwie.

## Z wiersza poleceń

Tekst jest czytany z argumentów, z pliku podanego w `--file` albo ze standardowego wejścia.
Zamaskowany tekst trafia na stdout, a mapowanie na stderr albo do pliku podanego w `--mapping`:

```bash
polanon --version  # wypisuje zainstalowaną wersję
polanon "Jan Kowalski, PESEL 50810112354"
polanon --file umowa.txt --ner --mapping mapping.json > masked.txt
cat umowa.txt | polanon --ner > masked.txt 2> mapping.json
polanon --restore --mapping mapping.json --file answer.txt  # przywrócona odpowiedź modelu
polanon --json "Jan Kowalski, PESEL 50810112354"  # {"text": "...", "mapping": {...}}
```

`--restore` wstawia oryginały z powrotem do zamaskowanego tekstu, na przykład odpowiedzi modelu,
z mapowaniem zapisanym przez `--mapping`, także osobę w innym przypadku; token, którego mapowanie
nie zna, zostaje bez zmian. `--json` wypisuje zamiast tego jeden obiekt JSON na stdout, dla
skryptów: tekst, a także mapowanie, chyba że trafia do pliku z `--mapping`.

Niezerowy kod wyjścia oznacza, że coś poszło nie tak, a komunikat na stderr wyjaśnia, co. `--file`
sam czyta bajty pliku i odróżnia od siebie UTF-8, UTF-16 (tak zapisuje Windows PowerShell 5.1
przy `>`) i cp1250; tekst, którego nie da się jednoznacznie zdekodować, jest odrzucany, zamiast być
maskowany po omacku. `--ner --paranoid` maskuje każdą nazwę, którą znajdzie model, także
w angielskim tekście i w kodzie, tak jak [`mode="paranoid"`](#co-maskować).

W systemie Windows zamiast potoków i przekierowań lepiej używać `--file` i `--mapping`. Windows
PowerShell 5.1 przekodowuje tekst przekazywany potokiem do programu na ASCII, więc polskie litery
docierają jako `?`, a zniekształcone imię i nazwisko nie zostaje rozpoznane; do tego jego `2>`
opakowuje stderr w rekordy błędów. Jeśli mimo to używasz potoku, najpierw wykonaj
`$OutputEncoding = [System.Text.UTF8Encoding]::new($false)`; w `cmd.exe` wykonaj `chcp 65001`.

## Strumieniowanie

Dostawca modelu przesyła odpowiedź strumieniowo w dowolnych fragmentach, więc `<OSOBA_1>` może
przyjść jako `<OSO` i `BA_1>`. Obiekt przywracający przytrzymuje wszystko, co może się jeszcze
okazać częścią tokenu:

```python
restorer = Anonymizer.stream_restorer(result.mapping)
for chunk in stream:  # "Dzień dobry <OSO", "BA_1>, PESEL <PESEL", "_1> zweryfikowany."
    print(restorer.feed(chunk), end="")
print(restorer.flush())
# Dzień dobry Jana Kowalskiego, PESEL 50810112354 zweryfikowany.
```

## Z Claude'em, GPT albo modelem lokalnym

Maskuj przed wywołaniem, przywracaj w trakcie strumieniowania odpowiedzi. Z SDK Anthropic:

```python
import anthropic
from polanon import Anonymizer, default_guards

anonymizer = Anonymizer(default_guards(with_ner=False))
masked = anonymizer.anonymize_sync(question)
restorer = Anonymizer.stream_restorer(masked.mapping)

with anthropic.Anthropic().messages.stream(
    model="claude-opus-5-5",
    max_tokens=64000,
    system="Dane osobowe zastąpiono tokenami takimi jak <OSOBA_1>. Przepisuj token dokładnie tak, jak go zapisano.",
    messages=[{"role": "user", "content": masked.anonymized}],
) as stream:
    for text in stream.text_stream:
        print(restorer.feed(text), end="")
print(restorer.flush())
```

Gdy powiesz modelowi, czym są tokeny, nie będzie ich przerabiał.
[Przykłady](https://github.com/boromat/polanon/tree/main/examples) pokazują więcej: `openai_chat.py`
robi to samo dla dowolnego endpointu zgodnego z OpenAI (OpenAI, OpenRouter, Azure, vLLM, lokalna
Ollama), a `anthropic_chat.py` prowadzi rozmowę, której historia zawiera wyłącznie zamaskowany
tekst, więc odpowiedź przywrócona dla czytelnika nigdy nie jest odsyłana jawnym tekstem. Oba
przykłady są testowane w CI z prawdziwymi SDK, offline.

## Tura rozmowy z narzędziami

Tura rozmowy z narzędziami wysyła więcej niż samo pytanie: także historię i wynik każdego
narzędzia, które wywoła model. `anonymizer.turn()` maskuje to wszystko jednym mapowaniem, wstawia
oryginały z powrotem do argumentów narzędzi i do odpowiedzi, a gdy czegoś nie da się zamaskować,
odmawia, zamiast to wysłać:

```python
import json

import anthropic
from polanon import Anonymizer, MaskingUnavailableError, default_guards, token_instructions

anonymizer = Anonymizer(default_guards(with_ner=False))
client = anthropic.Anthropic()


def chat(history: list[dict], clients: list[str]) -> None:
    """Jedna tura. `history` kończy się pytaniem; SYSTEM, TOOLS i run_tool są twoje."""
    try:
        turn = anonymizer.turn(known=clients, unknown_blocks="drop")
        messages = turn.mask_messages(history)
        while True:
            restorer = turn.stream_restorer()
            with client.messages.stream(
                model="claude-opus-5-5",
                max_tokens=64000,
                system=SYSTEM + "\n\n" + token_instructions("pl"),
                tools=TOOLS,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    print(restorer.feed(text), end="")
                response = stream.get_final_message()
            print(restorer.flush())
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                return
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, turn.restore_data(block.input))
                    masked = json.dumps(turn.mask_data(result, default=str), ensure_ascii=False)
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": masked}
                    )
            messages.append({"role": "user", "content": results})
    except MaskingUnavailableError:
        print("Maskowanie jest włączone, ale nie działa, więc nic więcej nie trafiło do modelu.")
```

`turn.stream_restorer()` wstawia oryginały, gdy każda odpowiedź spływa strumieniem, także token
przecięty między dwoma zdarzeniami, tak jak w sekcji [Strumieniowanie](#strumieniowanie). Testy
uruchamiają ten przykład dokładnie w tej postaci z prawdziwym SDK `anthropic`, offline, więc nie
rozjedzie się z biblioteką.

**Tura czy niższe API.** Sięgnij po turę zawsze, gdy model dostaje listę wiadomości, z narzędziami
albo bez: trzyma jedno mapowanie dla nich wszystkich, maskuje wartość znalezioną w późniejszej
wiadomości także we wcześniejszych i odmawia przy każdym błędzie, a bez niej to wszystko piszesz
sam. `anonymize_sync()`, `Anonymizer.restore()` i `StreamRestorer` wywołuj bezpośrednio dla
pojedynczego tekstu, na przykład dokumentu albo promptu, który składasz sam, gdy potrzebujesz
pozycji wykrytych wartości albo gdy ma działać sędzia drugiego etapu, którego tura nie przyjmuje.

**Jedno mapowanie na turę.** Klient w historii, w pytaniu i w wyniku narzędzia to ten sam
`<FIRMA_1>`. Żeby tokeny przechodziły z tury na turę, zacznij następną od
`anonymizer.turn(mapping=turn.mapping)`.

**Co maskuje `mask_messages`.** Wiadomości w formacie Anthropic albo OpenAI: ich tekst, `input`
bloku `tool_use`, wyniki narzędzi i argumenty `tool_calls`; JSON zostaje poprawny, a identyfikatory
i nazwy narzędzi bez zmian. Wartość rozpoznana dopiero w pytaniu jest maskowana także tam, gdzie
wcześniej w historii nikt jej nie rozpoznał. Blok, którego nie da się zamaskować, na przykład obraz
albo podpisane rozumowanie modelu, powoduje odmowę albo jest pomijany przy
`unknown_blocks="drop"`. `mask_data` maskuje teksty wewnątrz wartości; klucze i liczby zostają bez
zmian, więc dane osobowe trzymaj w wartościach tekstowych, a obiekt taki jak data potrzebuje
`default=str`.

**Wartości, które już znasz.** `known=` przyjmuje to, o czym wiesz, że jest wrażliwe, na przykład
nazwy klientów z bazy. Są maskowane wszędzie, gdzie się pojawią, także gdy są na allowliście.

**Odmowa zamiast wycieku.** Każdy błąd maskowania kończy się `MaskingUnavailableError` i nic
zamaskowanego tylko w części nie wraca. `default_guards()` zgłasza `NerModelUnavailableError`, który
też nim jest, gdy brakuje spaCy albo modelu, a `ner_available()` pozwala stronie ustawień sprawdzić,
czy są zainstalowane. Gdy przyjdzie odpowiedź, `turn.unknown_tokens(answer)` wypisuje tokeny
wymyślone przez model, a `turn.sweep(restored)` szuka też danych osobowych, które dodała odpowiedź.

## Maskowanie dla RAG i indeksów wyszukiwania

Tokeny numerowane w każdym wywołaniu od nowa nic nie znaczą poza swoją rozmową: umowa zamaskowana
do indeksu wyszukiwania i pytanie zamaskowane tydzień później mogą nazwać `<OSOBA_1>` dwie różne
osoby, a wyszukiwanie nic nie znajdzie. Daj anonimizatorowi tajny klucz, a każdy token będzie
obliczany: z HMAC-SHA256 wartości pod tym kluczem, taki sam w każdym dokumencie, każdym procesie
i każdym pytaniu, bez żadnego wspólnego stanu. „Jan Kowalski”, „Jana Kowalskiego” i „z Janem
Kowalskim” dostają jeden token, tak samo NIP zapisany z myślnikami i bez nich.

```python
import os

from polanon import Anonymizer, InMemoryMappingStore, Pseudonyms, default_guards, stored_mapping

key = bytes.fromhex(os.environ["POLANON_KEY"])  # 32 losowe bajty na firmę, z magazynu sekretów
store = InMemoryMappingStore()  # w produkcji twoja zaszyfrowana tabela, poza indeksem
pseudonyms = Pseudonyms(key, collisions=store.collisions)
anonymizer = Anonymizer(default_guards(with_ner=False), pseudonyms=pseudonyms)

document = anonymizer.index_chunks(
    [
        "Umowę z Firmą Kot Sp. z o.o. podpisał Jan Kowalski.",
        "Pełnomocnikiem Jana Kowalskiego jest Anna Nowak, tel. 600 100 200.",
    ],
    store,
)
for text in document.texts:  # same tokeny: to idzie do embeddera i do indeksu
    index.add(text, embed(text))  # index i embed są twoje

question = anonymizer.mask_query("Kto jest pełnomocnikiem Jana Kowalskiego?", store)
passages = index.search(embed(question.anonymized))  # pytanie ma tokeny fragmentów

mapping = question.mapping | stored_mapping(store, question.anonymized, *passages)
restorer = Anonymizer.stream_restorer(mapping)  # przywraca odpowiedź, jak w Strumieniowaniu
```

Testy uruchamiają ten przykład dokładnie w tej postaci, z zastępczym indeksem i embedderem. Po
kolei:

**Nic w indeksie nie nazywa wartości.** `document.texts` zawiera tokeny w miejscu wartości, a to,
co każdy token oznacza, trafia do magazynu. Nigdy nie kopiuj mapowania do metadanych fragmentu,
które są zapisywane i przeszukiwane razem z nim. `document.mappings[i]` przywraca fragment `i`
dokładnie tak, jak został napisany, „Jana Kowalskiego” w dopełniaczu; trzymaj je przy fragmencie,
zaszyfrowane, poza indeksem.

**Własny magazyn.** `MappingStore` ma dwie metody: `get(tokens)` i `add(entries)`, która zapisuje
tylko tokeny, których jeszcze nie ma, atomowo dla każdego tokenu, jak robi to unikalny klucz na
tokenie. Dla pozostałych zwraca to, co już ma. Szyfruj `StoredValue.value`, bo to są dane osobowe;
`fingerprint` bez klucza niczego nie zdradza. Trzymaj obok `store.collisions` i przekazuj go do
każdego `Pseudonyms`: są tam tylko tokeny i odciski, nic więcej.

**W jakiej formie wraca nazwisko.** Jeden token oznacza „Jan Kowalski” i „Jana Kowalskiego”.
Magazyn trzyma mianownik, bo model wpisuje token we własne zdanie, a „Umowę podpisał Jana
Kowalskiego” byłoby błędem bez względu na to, który dokument trafił do indeksu pierwszy. Fragment
przywrócony własnym mapowaniem zachowuje przypadek, w którym go napisano.

**Nazwiska, które magazyn już zna.** Samo nazwisko, jak w „Co podpisał Kowalczyk?”, to nazwa,
której nie odczyta żadna reguła. `index_chunks` zapisuje w magazynie nazwisko każdej osoby, której
imię znają reguły, obok pełnego imienia i nazwiska, a `mask_query` i `index_chunks` sprawdzają w
magazynie każdy ciąg do czterech słów i maskują te, dla których ma token. Pytanie dostaje wtedy
token nazwiska, a nazwisko zamaskowane w jednym dokumencie pozostaje zamaskowane w następnym. Przy
magazynie, który jest na to za wolny, przekaż `lookup=False`.

**Klucz.** Co najmniej 32 losowe bajty, jeden klucz na firmę i osobny `scope=` dla indeksu, którego
nie wolno powiązać z innym. Trzymaj go w menedżerze sekretów, poza indeksem, kodem i logami;
`Pseudonyms` nigdy go nie pokazuje, nie serializuje ani nie kopiuje. Nowy klucz, zakres albo liczba
cyfr zmienia każdy token, więc indeks trzeba zbudować od nowa: zapisz `pseudonyms.settings` razem z
indeksem i porównuj je przy starcie. Żeby usunąć dane osoby, usuń fragmenty zawierające
`pseudonyms.token(name, "PERSON")`, a także te z tokenem jej nazwiska, jeśli w indeksie nie ma
nikogo innego o tym nazwisku. Samo usunięcie wpisów z magazynu nie zrywa powiązania, bo klucz
zamienia odgadniętą wartość z powrotem w token.

**Kolizje.** Przy dwunastu cyfrach dwie wartości jednego typu dzielą token z szansą około 1 do
20 000 przy 10 000 wartości i 1 do 200 przy 100 000. Gdziekolwiek takie dwie wartości się spotkają,
w magazynie albo w jednym tekście, ta, która przyszła druga, dostaje cztery cyfry więcej.

**Ograniczenia.** Imię albo nazwisko, którego reguły nie umieją odmienić, na przykład rzadkie imię
albo „Krawca” od „Krawiec”, i nazwa firmy w innym przypadku, „Firmą Kot” obok „Firma Kot”, mają token na
każdą formę, co kosztuje chybione wyszukiwanie, ale niczego nie ujawnia. Samotne słowo pisane małą
literą nigdy nie jest sprawdzane w magazynie. A jeden token wszędzie to cel i zarazem cena: kto czyta
wiele zamaskowanych fragmentów, widzi, które dotyczą tej samej osoby. Co to oznacza, opisuje
[model zagrożeń](https://github.com/boromat/polanon/blob/main/docs/THREAT_MODEL.md#stable-pseudonyms-for-search-indexes).

## Z Haystackiem

`pip install "polanon[haystack]"` dodaje komponenty, które wpinają maskowanie w potok
[Haystacka 3](https://haystack.deepset.ai/). Klucz jest czytany ze zmiennej `POLANON_KEY`: 32
losowe bajty zapisane szesnastkowo:

```python
from haystack import Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from polanon import InMemoryMappingStore
from polanon.integrations.haystack import (
    PolanonDocumentMasker,
    PolanonQueryMasker,
    PolanonRestorer,
    PolanonStreamRestorer,
)

mappings = InMemoryMappingStore()  # w produkcji twoja zaszyfrowana tabela

indexing = Pipeline()
indexing.add_component("mask", PolanonDocumentMasker(store=mappings))
indexing.add_component("split", DocumentSplitter(split_by="period"))
indexing.add_component("embed", document_embedder)  # dowolny embedder: dostaje zamaskowany tekst
indexing.add_component("write", DocumentWriter(document_store))
indexing.connect("mask", "split")
indexing.connect("split", "embed")
indexing.connect("embed", "write")

asking = Pipeline()  # z twoim embedderem, retrieverem, budowniczym promptu i generatorem
asking.add_component("mask", PolanonQueryMasker(store=mappings))  # pierwszy
asking.add_component("restore", PolanonRestorer(store=mappings))  # po generatorze
asking.connect("mask.query", "embed.text")
asking.connect("mask.query", "prompt.query")
asking.connect("retrieve.documents", "restore.documents")
asking.connect("llm.replies", "restore.replies")
asking.connect("mask.mapping", "restore.mapping")

stream = PolanonStreamRestorer(mappings, on_text=lambda text: print(text, end=""))
asking.run({"mask": {"query": question, "stream": stream}, "llm": {"streaming_callback": stream}})
```

[`examples/haystack_rag.py`](https://github.com/boromat/polanon/blob/main/examples/haystack_rag.py)
to całość, offline, z zastępczym embedderem i generatorem Haystacka; testy go uruchamiają.
[`examples/haystack_openai.py`](https://github.com/boromat/polanon/blob/main/examples/haystack_openai.py)
to samo z embedderem i modelem OpenAI: jest płatny, więc testy tylko go budują.

**Najpierw maskowanie.** Maskownik dokumentów stoi przed splitterem, więc czyta cały dokument i
maskuje firmę czy adres tam, gdzie stoją: pocięta najpierw na kropkach „Firmą Kot Sp. z o.o.”
rozpadła się i poszła niezamaskowana. Fragmenty, które przychodzą już pocięte, są maskowane razem,
gdy wskazują jedno źródło w `source_id`, `parent_id` albo `__parent_id`. Tnij po słowach, zdaniach,
kropkach albo akapitach: splitter liczący znaki przecina tokeny na pół, a połówka niczego nie trafia.

**Nic w magazynie dokumentów nie nazywa wartości.** Zamaskowane dokumenty są nowe: tokeny w treści
i w każdym tekście `meta`, także w kluczach zagnieżdżonych słowników i w elementach zbiorów. `id`
dokumentu to zwykły SHA-256 jego tekstu, więc każdy napis w `meta` w tym kształcie, czyli to, co
zapisują tam splittery, wskazuje na zamaskowany odpowiednik swojego dokumentu albo staje się
odciskiem z kluczem. `id` policzone przez Haystacka jest liczone od nowa z zamaskowanego dokumentu, a
nadane przez ciebie jest maskowane jak tekst i zostaje, więc upsert i usuwanie po `id` dalej działają.
To, co oznaczają tokeny, trafia wyłącznie do magazynu mapowań, nigdy do `meta`. Dokument z `blob` albo z
embeddingiem jest odrzucany, bo oba powstały z oryginału. Testy sprawdzają każdy dokument w
magazynie i zapisany przez niego plik pod kątem każdej oryginalnej wartości.

**Odpowiedź dla czytelnika.** `PolanonQueryMasker` wysyła zamaskowane pytanie do embeddera i do
budowniczego promptu, a swoje `mapping` do `PolanonRestorer`, który przywraca odpowiedzi i
znalezione dokumenty. Jego dokumenty służą tylko do wyświetlenia: nigdy nie zapisuj ich do magazynu
i nie wysyłaj do modelu. `PolanonStreamRestorer` przywraca odpowiedź w strumieniu jako
`streaming_callback` generatora; przy `Pipeline.stream()` podawaj mu fragmenty sam:
`stream.feed(chunk.content)`. Wartości, o których wiesz, że są wrażliwe, na przykład nazwy klientów,
podajesz w czasie uruchomienia, `{"mask": {"known": clients}}`, więc nigdy nie trafiają do zapisanego
potoku.

**Detektory.** Oba maskowniki używają samych reguł, chyba że dostaną `with_ner=True`; wtedy
`spacy_model`, `mode` i `ml_filter` działają tak jak w `default_guards`:
`PolanonDocumentMasker(store=mappings, with_ner=True, mode="paranoid")`.

**Zapisywanie potoku.** Klucz to `Secret` Haystacka, a zapisany potok zawiera tylko nazwę jego
zmiennej. Magazyn mapowań zapisuje się przez własne `to_dict`, które powinno mówić, dokąd się łączy,
a nigdy, co przechowuje; `InMemoryMappingStore` go nie ma, więc potok z nim budujesz w kodzie.
Haystack 3 wczytuje klasy tylko z własnych modułów, więc dopuść ten moduł i moduł swojego magazynu:
`Pipeline.loads(text, allowed_modules=["polanon.integrations.haystack", "myapp.vault"])`.

**Content tracing zostaw wyłączony.** Przy `HAYSTACK_CONTENT_TRACING_ENABLED` Haystack zapisuje
to, co dostaje każdy komponent, a maskownik dostaje oryginały. Nieudane uruchomienie zachowuje swoje
wejścia w snapshocie błędu, zapisywanym na dysk przy `HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED`. Oba
są domyślnie wyłączone.

**Telemetria samego Haystacka jest domyślnie włączona.** Haystack wysyła na własne serwery dane o
tym, jak używane są jego potoki, chyba że ustawisz `HAYSTACK_TELEMETRY_ENABLED=False`; polanon nie
wysyła niczego. Tam, gdzie z maszyny mają wychodzić tylko zamaskowane zapytania, ustaw tę zmienną.

## Bramka dla każdego narzędzia zgodnego z API OpenAI

Narzędzia takie jak Hermes Agent, n8n, Open WebUI czy LibreChat rozmawiają z modelem przez API
OpenAI. `polanon-gateway` staje między takim narzędziem a dostawcą, na komputerze tego, kto ją
uruchamia: każda wiadomość jest maskowana, zanim wyjdzie, a odpowiedź wraca z oryginalnymi
wartościami, strumieniowo albo nie, razem z wywołaniami narzędzi.

```bash
pip install "polanon[gateway,ner]"
python -m spacy download pl_core_news_md
polanon-gateway --upstream https://api.openai.com/v1
```

Potem ustaw w narzędziu adres bazowy `http://127.0.0.1:8787/v1` i zostaw jego klucz API: bramka
przekazuje go dostawcy. Model lokalny działa tak samo, dla Ollamy `--upstream
http://localhost:11434/v1`.

Albo w Dockerze, z modelem spaCy w środku, tak samo na komputerach z procesorem Intel i ARM:

```bash
docker run --rm -p 127.0.0.1:8787:8787 -e POLANON_UPSTREAM_KEY -e POLANON_GATEWAY_KEY \
    ghcr.io/boromat/polanon-gateway:0.4 --upstream https://api.openai.com/v1
```

W kontenerze bramka słucha na wszystkich interfejsach, więc startuje tylko
z `POLANON_GATEWAY_KEY`, kluczem, który każde narzędzie wysyła potem jako swój klucz API,
i potrzebuje `POLANON_UPSTREAM_KEY`, klucza dostawcy, który wysyła zamiast niego.
`-p 127.0.0.1:8787:8787` zostawia port na tym komputerze. Każda opcja `polanon-gateway` idzie
po nazwie obrazu. Model na tym samym komputerze, na przykład Ollama, to `--upstream
http://host.docker.internal:11434/v1`, a na Linuksie dodatkowo
`--add-host=host.docker.internal:host-gateway`. Obraz ma tagi z wersją, na przykład `0.4.1`,
z wersją pomniejszą, `0.4`, i `latest`.

| Narzędzie | Gdzie wpisać adres bazowy |
|---|---|
| Hermes Agent | `base_url` modelu albo nazwanego dostawcy w `config.yaml` |
| n8n | Base URL w poświadczeniach OpenAI |
| Open WebUI | adres połączenia z API OpenAI |
| LibreChat | `baseURL` własnego endpointu w `librechat.yaml` |
| aplikacja na SDK Anthropic, Claude Code | `base_url` albo `ANTHROPIC_BASE_URL` jako `http://127.0.0.1:8787` |

Narzędzie, które mówi Messages API Anthropic, potrzebuje osobnej bramki, uruchomionej
z `--upstream https://api.anthropic.com/v1`; jego klucz, wysłany jako `x-api-key` albo token
bearer, idzie dalej tak jak klucz OpenAI, a adres bazowy z `/v1` na końcu też działa.

Co robi z jednym wywołaniem:

- Obsługuje `/v1/chat/completions`, Responses API, `/v1/responses`, i Messages API Anthropic,
  `/v1/messages`, strumieniowo albo nie, z `/v1/messages/count_tokens`, oraz `/v1/embeddings`
  i `/v1/models`, a każdy inny endpoint odrzuca, więc nic, czego nie umie zamaskować, nie
  wychodzi w postaci, w jakiej przyszło.
- Przy Messages API maskowane są prompt systemowy, wiadomości, narzędzia i metadane, a notatka
  o tokenach trafia do promptu systemowego. Blok myślenia (`thinking`) przechodzi tak, jak
  napisał go model, z zamaskowanego tekstu: dostawca sprawdza jego podpis. Ostatnia kontrola
  i tak go przeszukuje.
- Przy Responses API cała rozmowa przychodzi z każdym wywołaniem, tak jak wysyła ją Hermes Agent
  ze `store: false`. Wywołanie, które zostawia ją u dostawcy, przez `previous_response_id` albo
  `conversation`, albo prosi o odpowiedź `background` do odebrania później, przyniosłoby tokeny,
  których nikt nie umie przywrócić, więc jest odrzucane. Zaszyfrowane rozumowanie przechodzi bez
  zmian: dostawca zapieczętował je z zamaskowanego tekstu.
- Maskuje każdą wiadomość, wynik narzędzia i wcześniejsze wywołanie narzędzia, a także tekst
  w opisach narzędzi i formatu odpowiedzi: opisy, tytuły i przykładowe wartości. Nazwy narzędzi,
  typy i nazwy parametrów zostają, więc narzędzia dalej działają. Pole `name` uczestnika pomija.
- Obrazu, pliku ani dźwięku nie da się zamaskować, więc takie wywołanie jest odrzucane;
  `--drop-unsupported` pomija taką treść i wysyła resztę, a `--pass-unsupported` wysyła ją do
  dostawcy taką, jaka jest, bez maskowania, razem z nazwą pliku, gdy chcesz, żeby model ją widział.
- Zanim cokolwiek wyjdzie, przeszukuje całe zapytanie jeszcze raz pod kątem każdej oryginalnej
  wartości i odmawia wysyłki, jeśli któraś wciąż tam jest. Nie przeszukuje nazw z kodu, których
  maskowanie nie rusza, takich jak nazwy pól, identyfikatory i nazwy narzędzi.
- Do wiadomości systemowej dopisuje notatkę, która mówi modelowi, czym są tokeny: `--instructions
  en` albo `none`, żeby jej nie dodawać.
- `--paranoid` maskuje każdą nazwę, którą znajdzie model spaCy, tak jak `mode="paranoid"`.
  Domyślnie pomija to, co znajdzie w angielskim tekście, w kodzie i identyfikatorach, bo prompt,
  narzędzia i logi agenta są pełne angielskich słów, które polski model bierze za nazwy.
- `--allow NAZWA`, podane tyle razy, ile trzeba, i `--allow-file ŚCIEŻKA`, plik z jedną nazwą
  w linii, zostawiają czytelne nazwy, które model musi widzieć takimi, jakie są, na przykład
  imię asystenta i firmę, w imieniu której mówi; tak podana osoba też zostaje czytelna.
  Reszta promptu jest maskowana jak dotąd.
- `--pii-map` pozwala klientowi, który prowadzi własny rejestr, audyt albo magazyn mapowań,
  poprosić o mapowanie wywołania: z `"return_pii_map": true` w żądaniu odpowiedź niesie
  `"return_pii_map": {"<OSOBA_1>": "Jan Kowalski"}`, a strumień niesie je w ostatnim zdarzeniu.
  Dostawca nigdy nie widzi tego pola. Bez tej opcji takie żądanie jest odrzucane.
- Niczego nie zapisuje na dysku. Narzędzie wysyła całą rozmowę przy każdym wywołaniu, a każde
  wywołanie jest maskowane i przywracane osobno. Z `POLANON_KEY` ustawionym na 64 znaki
  szesnastkowe dana osoba ma jeden token w każdym wywołaniu i w każdym przypadku gramatycznym.

Domyślnie bramka maskuje jedno wywołanie naraz, we własnym procesie: model spaCy nie jest
bezpiecznie współdzielony między wątkami, a Python i tak wykonuje naraz jeden wątek procesu.
Z domyślnymi strażnikami zamaskowanie 100 KB trwa około 3 s, więc drugie narzędzie czeka na
pierwsze. `--workers N` maskuje N wywołań naraz, każde w osobnym procesie z własną kopią modelu,
na tylu rdzeniach, a własny proces bramki dalej obsługuje połączenia, strumienie i przywracanie;
na komputerze z 16 wątkami cztery wywołania naraz trwały tyle co jedno. Więcej workerów niż
rdzeni nic nie daje.

Każdy worker trzyma własny model spaCy: z `pl_core_news_md` to około 330 MB na każdego workera,
a własny proces bramki nie wczytuje wtedy modelu i zostaje przy około 55 MB. Jeden proces zajmuje
około 340 MB, `--workers 2` około 710 MB, a `--workers 4` około 1,4 GB. W Dockerze opcja idzie po nazwie obrazu, a limit pamięci,
na przykład `docker run --memory`, musi zostawić miejsce dla każdego workera. Workery wczytują
model, zanim bramka zacznie przyjmować wywołania, i kończą pracę razem z nią. Wywołanie trafia
do workera i wraca zamaskowane, razem z mapowaniem, przez potok na tym samym komputerze, a worker
zapomina wartości, gdy tylko je zamaskuje. Każdy worker czyta `POLANON_KEY` ze środowiska, które
dziedziczy, więc stabilne tokeny są takie same, niezależnie od tego, który worker zamaskował
wywołanie. Worker, który przestanie działać, na przykład zabity z braku pamięci, psuje wywołania
maskowane w tej chwili: są odrzucane z kodem 503 i nic nie zostaje wysłane, a SDK OpenAI
i Anthropic ponawiają taki błąd; następne wywołanie uruchamia nowe workery.

Klucze zostają u tego, kto uruchamia bramkę, i są brane wyłącznie ze zmiennych środowiskowych.
Domyślnie bramka przekazuje dalej klucz narzędzia; `POLANON_UPSTREAM_KEY` wysyła zamiast niego
inny, jako token bearer albo, do Anthropic, jako `x-api-key`. Bramka słucha na `127.0.0.1`.
Żeby była dostępna z innego komputera, ustaw `--host` i `POLANON_GATEWAY_KEY`, który każdy
klient musi wtedy wysyłać, jako token bearer albo jako `x-api-key`: bez tego klucza bramka odmawia
startu. Log podaje ścieżkę, status, czas i liczbę zamaskowanych wartości każdego typu, nigdy samą
wartość.

Nie ma żadnej usługi w chmurze: bramka działa tam, gdzie ją zainstalowano, a kto ją uruchamia,
odpowiada za ten komputer, swoje klucze i dostawcę, do którego idą wywołania.
[Model zagrożeń](https://github.com/boromat/polanon/blob/main/docs/THREAT_MODEL.md#the-gateway)
opisuje, przed czym bramka chroni, a przed czym nie.

## Co wykrywa

Polskie identyfikatory walidowane sumą kontrolną (PESEL, NIP, REGON, dowód osobisty, IBAN, karty
płatnicze), numery KRS obok swojej etykiety, adresy e-mail i numery telefonów, adresy i kody
pocztowe, linki do profilu osoby (LinkedIn, GitHub, GitHub Pages, GitLab, media społecznościowe) aż
po nazwę użytkownika, IPv4 i IPv6, adresy MAC, numery rejestracyjne pojazdów po etykiecie
(„nr rej.”), numery VIN, numery paszportów, numery praw jazdy po etykiecie, daty urodzenia, klucze API, tokeny i hasła, polskie
imiona razem z nazwiskami oraz nazwy firm rozpoznawane po formie prawnej („Sp. z o.o.”, „S.A.”,
„Ltd”, „GmbH”). Gdy zamaskowana zostanie osoba albo link do profilu, wszędzie w tym samym tekście
maskowane jest też samo nazwisko albo sama nazwa użytkownika: „Kowalski” po „Jan Kowalski”,
„jkowalski” w polu JSON po „github.com/jkowalski”. Z dodatkiem `ner` model spaCy dokłada miejsca
i nazwy, których nie wykrywa nic innego. Opcjonalny drugi etap pozwala małemu lokalnemu modelowi
przejrzeć *zamaskowany* tekst pod kątem tego, co z kontekstu wciąż pozwala kogoś zidentyfikować.
Pozostałe daty też są rozpoznawane, ale zostają czytelne, chyba że poprosisz o ich maskowanie.

Tekst skopiowany z PDF-ów, Worda albo macOS jest czytany tak, jak przeczytałby go człowiek: litery
zapisane jako litera bazowa plus znak diakrytyczny, spacje niełamiące i półpauzy wewnątrz liczb oraz
niewidoczne znaki o zerowej szerokości nie ukrywają wartości. Mapowanie przechowuje każdą wartość
w postaci, w jakiej została zapisana za pierwszym razem.

Na oznaczonych przykładowych dokumentach z repozytorium same reguły maskują w całości 86% wartości,
a reguły z `pl_core_news_md` 98%, przy czym nic nie zostaje zamaskowane tylko w części. Trzynaście
krótkich dokumentów to test regresji, a nie benchmark badawczy:
[notatki z benchmarku](https://github.com/boromat/polanon/blob/main/benchmarks/README.md) opisują,
co zostaje przeoczone i dlaczego, i podają wynik Presidio na nich z tym samym modelem: 67%
zamaskowanych w całości, 77% przy najłagodniejszym sposobie liczenia. Osiem
wymyślonych wywołań agenta AI, z jego angielskim promptem, narzędziami i ich JSON-em, kodem,
logami, umową najmu i pocztą w trzech językach, mierzy, ile kosztuje ruch agenta:
z `pl_core_news_md` do modelu trafiają 2 z 243 oznaczonych wystąpień, a 5 z 309 użyć tokenów nic
nie ukrywa, tylko zasłania słowo, którego model nie może przeczytać. Odpowiedzi modelu wracają ze
wszystkimi tokenami przywróconymi, jako tekst, jako odpowiedź i jako strumień, a 39 z ich 40 linii
brzmi dokładnie tak, jak napisałby je człowiek.

## Warto wiedzieć

**Tokeny są stabilne w obrębie rozmowy, a nie globalnie**, chyba że podasz klucz, jak dla
[indeksu wyszukiwania](#maskowanie-dla-rag-i-indeksów-wyszukiwania). Przekaż mapowanie z ostatniego
wyniku do kolejnego wywołania: ta sama wartość zachowuje ten sam token, a token nigdy nie zostaje
ponownie użyty dla innej wartości. Przekazane mapowanie jest tylko odczytywane, a nie zmieniane, więc
zachowuj to, które zwraca każdy wynik:

```python
mapping = result.mapping
second = anonymizer.anonymize_sync("Ten sam PESEL 50810112354", mapping)
second.anonymized  # "Ten sam PESEL <PESEL_1>"
mapping = second.mapping  # wszystko do tej pory, na kolejną turę rozmowy
```

**Token przywraca formę, którą zastąpił, chyba że model poda przypadek.** Maskowanie działa na
tekście w takiej postaci, w jakiej jest zapisany, więc imię i nazwisko zamaskowane w dopełniaczu
wraca w dopełniaczu. Model może dopisać po dwukropku przypadek, którego wymaga jego zdanie, a imię
i nazwisko wróci odmienione: `Szanowny Panie <OSOBA_1:wołacz>` daje „Szanowny Panie Janie
Kowalski” tam, gdzie tekst mówił „Jana Kowalskiego”. Prosi o to `token_instructions("pl")`,
a rozumie to każde przywracanie, także strumień i bramka. Odmieniane są imiona i nazwiska z jednego
lub dwóch słów, które reguły czytają na pewno: imię z listy i nazwisko w rodzaju Kowalski, Nowak,
Bednarek, Mickiewicz albo Duda. Samo nazwisko, które może należeć do mężczyzny albo kobiety, obce
nazwisko, firma i miejsce wracają tak, jak były zapisane, a przypadek znika. Inne przypadki osoby,
która ma już token, dostają jej numer i swój przypadek: po „Jan Kowalski” forma „Janowi
Kowalskiemu” idzie jako `<OSOBA_1:celownik>`, więc model widzi jedną osobę, a przywracanie odmienia
imię i nazwisko z powrotem dokładnie do formy z tekstu. Forma, której odmiana nie odtwarza litera
w literę, na przykład zapisana wielkimi literami, dostaje własny token, a stały token z kluczem
nie ma przypadku. `CASED_TOKEN_PATTERN` rozpoznaje token z przypadkiem i bez. Zapisy różniące się
tylko wielkością liter, rodzajem białych znaków (spacja
niełamiąca, podział wiersza, kilka spacji z rzędu) albo niewidocznymi znakami to ta sama wartość;
tak samo identyfikator, którego grupy zapisano osobno albo razem: wszystkie te zapisy dzielą jeden
token, a przywracanie wstawia ten, który pojawił się jako pierwszy. Wyjątkiem są hasła i typy
własne, bo dwa hasła różniące się wielkością liter to dwa różne hasła, a hasło nigdy nie dzieli
tokenu z podobnie wyglądającą nazwą. Jednowyrazowa nazwa znaleziona w zapisie samymi małymi
literami też dostaje własny token, z powodu opisanego niżej.

**Każde wystąpienie zamaskowanej wartości jest maskowane**, także tam, gdzie żaden detektor nie
rozpoznałby jej samodzielnie: w oryginalnym zapisie, wielkimi literami, rozdzielona podziałem
wiersza, ze znakiem o zerowej szerokości w środku, z literami cyrylicy albo greki, które wyglądają
jak łacińskie, z inaczej rozdzielonymi grupami oraz w całości,
razem z tym, co detektor znalazł w jej wnętrzu. Dwa wyjątki sprawiają, że zwykłe słowa pozostają
czytelne. Jednowyrazowa nazwa pisana wielką literą nie jest dopasowywana, gdy występuje zapisana
samymi małymi literami, bo Zima, Kot i Wilk to nazwiska, a „zima” to pora roku; identyfikatory
takie jak VIN czy adres e-mail są dopasowywane bez względu na wielkość liter. Liczba krótsza niż
sześć cyfr, na przykład PIN, nie jest dopasowywana wewnątrz dłuższej liczby, daty, godziny ani
kwoty: „15.04.2025”, „2024-2025”, „2025,50” i „20250” zostają czytelne, gdy PIN to 2025,
a „w 2025 roku” i „PIN/PUK 2025/36912874” już nie.

**Token, który sam wpiszesz, pozostaje twój.** Znacznik taki jak `<PESEL_1>` w twoim własnym
tekście nigdy nie zostanie przydzielony prawdziwym danym. Trafia do mapowania jako odpowiadający
samemu sobie, więc ta rezerwacja obowiązuje także w kolejnych turach.

**Mapowanie to wrażliwa połowa.** `result.anonymized` zawiera token w miejscu wszystkiego, co
znalazły detektory, a jawnym tekstem tylko to, co przeoczyły; `result.mapping` to klucz, który to
odwraca. Przechowuj mapowanie tak samo jak oryginalny dokument.

**Wysyłasz rozmowę ponownie?** `mask_known_values(text, mapping)` maskuje w tekście zamaskowanym
wcześniej każdą wartość, którą mapowanie już zawiera, na przykład w historii wysyłanej ponownie po
tym, jak nowa wartość dostała swój token. To samo wywołanie sprawdza treść przed wysłaniem:
`mask_known_values(payload, mapping) == payload` oznacza, że żadna wartość z mapowania nie została
w miejscu, w którym biblioteka by ją zamaskowała.

**Własne słowa.** Wewnętrzne nazwy, których żadna reguła nie może znać, takie jak kryptonim projektu
albo nazwa klienta, trafiają na denylistę i są maskowane wszędzie, gdzie się pojawią, bez względu na
wielkość liter, także rozdzielone podziałem wiersza:

```python
guards = default_guards(denylist=["Projekt Wichura", "Zakład Nowak"], with_ner=False)
```

**Sprawdzanie odpowiedzi.** Po przywróceniu `anonymizer.sweep(restored, mapping)` zgłasza tokeny
wymyślone przez model i dane osobowe, które dodała jego odpowiedź: `report.is_clean`, a jeśli
chcesz się im przyjrzeć, `report.unknown_tokens` i `report.new_spans`.

## Co maskować

`Anonymizer(guards)` maskuje `DEFAULT_ENABLED`: każdy typ z `ENTITY_TYPES` oprócz `DATE`, bo data
rzadko kogoś identyfikuje, a po zamaskowaniu każdego terminu model nie jest w stanie odpowiedzieć.
Data urodzenia natomiast kogoś identyfikuje, więc data po „ur.”, „data urodzenia”, „DOB” albo
„geb.” ma własny typ, `BIRTH_DATE`, i pozostaje zamaskowana. Przekaż `enabled`, żeby wybrać
samodzielnie:

```python
from polanon import DEFAULT_ENABLED, ENTITY_TYPES, Anonymizer, default_guards

guards = default_guards(with_ner=False)
Anonymizer(guards, enabled=DEFAULT_ENABLED | {"DATE"})  # także daty
Anonymizer(guards, enabled={"PESEL", "NIP", "IBAN"})  # tylko identyfikatory
list(ENTITY_TYPES)  # wszystkie nazwy, których można użyć
```

Nazwa, która nie jest ani wbudowana, ani zadeklarowana w `entity_types` któregoś detektora,
powoduje wyjątek `ValueError`, więc literówka w rodzaju `"PERSONS"` nie może po cichu wyłączyć
maskowania.

Z modelem spaCy `mode` ustala, ile z tego, co model uznaje za nazwę, zostaje zamaskowane. Polski
model bierze angielskie słowa za nazwy, a prompt, narzędzia, kod i logi agenta AI są ich pełne:
„return”, „message” czy „Uncheck” z przycisku stają się tokenami, których model nie przeczyta.
Domyślny `mode="normal"` pomija to, co model znajdzie w angielskim tekście, w kodzie
i identyfikatorach, oraz miejsce zapisane małą literą; ceną jest nazwisko otwierające angielskie
zdanie albo „jadę do gdyni”, które wychodzą bez zmian. `mode="paranoid"` maskuje wszystko:

```python
guards = default_guards(mode="paranoid")
```

Na wywołaniach agenta AI z `benchmarks/` tryb „normal” przepuszcza 3 z 243 oznaczonych wystąpień,
a 2% użyć tokenów nic nie ukrywa; „paranoid” przepuszcza 1, imię zapisane cyrylicą, którego nie
znajduje żaden strażnik, ale nic nie ukrywa ponad połowa użyć. Reguły są w obu trybach te same, więc bez modelu
tryb niczego nie zmienia.

W trybie normalnym filtr ML odrzuca więcej z tego, co model uznaje za nazwę: to regresja
logistyczna, która waży fragment, jego słowa i słowa wokół niego, ich wektory oraz to, co
znalazły tam reguły, dopasowana na publicznych polskich danych. Na każdym benchmarku maskuje
w całości te same wartości, a fragmentów, które niczego nie ukrywają, zostawia mniej więcej
o jedną czwartą mniej: na części OpenPII odłożonej do samego końca 400 zamiast 539. Dokłada około
pół milisekundy do tekstu, który model czyta dwanaście, i niczego nie uczy się z maskowanych
tekstów: trafia do paczki jako gotowe wagi. Jego wagi czytają wektory `pl_core_news_md`, więc
przy innym modelu spaCy decydują same ręczne filtry. Jest domyślnie włączony; jak go dopasowano
i zmierzono, opisują
[notatki z benchmarku](https://github.com/boromat/polanon/blob/main/benchmarks/README.md#the-ml-filter),
a tak zostawia się decyzję samym ręcznym filtrom:

```python
guards = default_guards(ml_filter=False)
```

## Druga opinia: sędzia

Reguły i model językowy przeoczają to, co pozwala kogoś zidentyfikować wyłącznie z kontekstu:
„dyrektor finansowy jedynej huty w mieście”. Sędzia to mały model, który czyta *zamaskowany* tekst
po etapie 1 i oznacza takie fragmenty. Uruchom go lokalnie w Ollamie, a przegląd nigdy nie opuści
twojej sieci:

```python
import asyncio

from polanon import Anonymizer, JudgeError, default_guards
from polanon.judge import OllamaJudge

anonymizer = Anonymizer(
    default_guards(),
    judge=OllamaJudge(model="qwen2.5:1.5b"),
    judge_fallback="block",  # nie idź dalej, gdy sędzia nie działa
)


async def main(text: str) -> None:
    try:
        result = await anonymizer.anonymize(text)
    except JudgeError:
        ...  # sędzia nie działał, a "block" oznacza, że nic nie wychodzi


asyncio.run(main("..."))
```

Potrzebne są `pip install "polanon[ner,judge]"`, model spaCy, działająca Ollama i
`ollama pull qwen2.5:1.5b`. Z `judge_fallback="send_stage1"` wywołanie idzie dalej, gdy sędzia nie
działa, a `result.judge_failed` informuje, że tekst wyszedł po samym etapie 1.

Sędziowie są asynchroniczni, więc `anonymize_sync` odmawia pracy z anonimizatorem, który ma
sędziego. `RemoteJudge` wysyła zamiast tego zapytania do endpointu zgodnego z OpenAI; zamaskowany
tekst opuszcza wtedy twoją sieć, przez co jest to narzędzie do testów, a nie mechanizm ochrony
prywatności.

## Nazwy, które mają zostać czytelne

Model językowy oznacza „Facebook” czy „Google Ads” jako organizację, przez co maskuje dokładnie to,
czego dotyczy pytanie. Przekaż allowlistę, a te wykrycia zostaną odrzucone przed przydzieleniem
tokenów:

```python
from polanon import WELL_KNOWN_NAMES, Allowlist, Anonymizer, default_guards

anonymizer = Anonymizer(default_guards(), allowlist=Allowlist(WELL_KNOWN_NAMES))
```

`WELL_KNOWN_NAMES` zawiera platformy i narzędzia biurowe, urzędy i programy do rozliczeń, takie
jak ZUS, Płatnik, NFZ czy KSeF, oraz modele i narzędzia, które wymienia agent AI, takie jak
OpenRouter, DeepSeek i Home Assistant ze skrótem „HA”, a także sprzęt, taki jak Dell czy
Roborock. Jednowyrazowa nazwa pasuje też do swoich polskich form
fleksyjnych („Facebooku”, „Canvie”, „ZUS-ie”). Odrzucane są tylko wykrycia nazw i tylko wtedy, gdy każde ich słowo jest na liście; identyfikatory,
wpisy z denylisty i wartości zamaskowane już wcześniej w rozmowie pozostają zamaskowane.

## Z czego się składa

| Moduł | Do czego służy |
|---|---|
| `guards/` | deterministyczne detektory. Każdy zwraca obiekty `Span` we współrzędnych oryginalnego tekstu |
| `pipeline.py` | uruchamia detektory w wątku, rozstrzyga nakładające się wykrycia, przydziela stabilne tokeny |
| `policy.py` | taksonomia typów encji i nazewnictwo tokenów (`<OSOBA_1>`, `<PESEL_1>`) |
| `judge/` | opcjonalny drugi etap: model przegląda zamaskowany tekst pod kątem tego, co wciąż pozwala kogoś zidentyfikować |
| `stream_restore.py` | wstawia oryginały z powrotem do odpowiedzi przesyłanej strumieniowo, nie rozcinając tokenu |
| `turn.py` | maskuje jednym mapowaniem wszystko, co wysyła jedno wywołanie modelu, i przywraca to, co wraca |
| `pseudonyms.py` | oblicza stałe tokeny z tajnego klucza, takie same w każdym dokumencie i procesie |
| `index.py` | maskuje dokumenty i pytania dla indeksu wyszukiwania, z magazynem tego, co kryją tokeny |
| `integrations/haystack.py` | komponenty Haystacka, które maskują dokumenty i pytania i przywracają odpowiedzi |
| `gateway/` | bramka zgodna z API OpenAI, uruchamiana na własnym komputerze |
| `output_sweep.py` | oznacza wymyślone tokeny i dane osobowe, które wprowadziła odpowiedź |

## Dodawanie detektora

Utwórz podklasę `Guard`, zwracaj fragmenty we współrzędnych oryginalnego tekstu i uwzględniaj zbiór
`enabled`. Własny typ trafia do `entity_types`, co domyślnie go włącza; staje się etykietą jego
tokenów, więc zapisuje się go wielkimi literami i znakami podkreślenia:

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

Gdy wykrycia się nakładają, o tym, które wygrywa, decyduje wynik (score): 1.0 dla wartości, za którą
ręczy suma kontrolna, etykieta albo oficjalny format, 0.85 dla statystycznego przypuszczenia;
znaleziska sędziego mają wynik 0.5. Polskie imię z nazwiskiem ma wynik 0.9, a samo imię 0.8, więc
przypuszczenie, że to miejsce, wygrywa z nim. Sekret jest maskowany w całości, razem ze wszystkim,
co wykryto w jego wnętrzu. Fragment wychodzący poza tekst, typ zapisany w jakikolwiek inny sposób
albo zapisany tak jak etykieta wbudowanego tokenu (`OSOBA`, `DATA`) powoduje wyjątek `ValueError` z
nazwą twojego detektora.

## Zanim zaczniesz na tym polegać

Maskowanie to pseudonimizacja, a nie anonimizacja w rozumieniu prawa, i żaden detektor nie wyłapuje
wszystkiego. [Model zagrożeń](https://github.com/boromat/polanon/blob/main/docs/THREAT_MODEL.md)
opisuje, co jest objęte ochroną, a co nie, i co pozostaje zadaniem twojej aplikacji, począwszy od
bezpiecznego przechowywania mapowania.

Biblioteka polanon powstała jako silnik maskowania dla [PolGate](https://github.com/boromat/polgate),
samodzielnie hostowanego huba, dzięki któremu zespół może pracować z Claude'em albo GPT na
prawdziwych dokumentach bez wysyłania danych osobowych do dostawcy modelu. Biblioteka nic o nim nie
wie: nie ma w niej HTTP, bazy danych ani ustawień, a jeśli kiedykolwiek zacznie, jeden
z testów nie przejdzie.

## Kto korzysta z polanon

[PolGate](https://github.com/boromat/polgate) to samodzielnie hostowany czat firmowy dla Claude'a
i GPT, zbudowany na bibliotece polanon: każda wiadomość i każdy przesłany dokument są maskowane na
serwerze, zanim dotrą do modelu, a odpowiedź wraca strumieniowo z oryginałami na swoich miejscach.
Dodaje to, czego zespół potrzebuje poza samą biblioteką: zaszyfrowany sejf na mapowanie, retencję,
która usuwa dane, dziennik audytu z typami i liczbą wystąpień oraz panel administratora do
zarządzania polityką, w tym allowlistą i denylistą.

[PolCV](https://github.com/boromat/polcv) pisze CV do polskich ogłoszeń o pracę i potrafi za pomocą
polanon zamaskować imię i nazwisko kandydata, dane kontaktowe i nazwy firm, zanim jakikolwiek tekst
trafi do modelu, a potem wstawić je z powrotem do odpowiedzi. Dodaje własny detektor dla tego, co
już wiadomo z profilu, na przykład dla linku do LinkedIna z imieniem i nazwiskiem, tak jak opisuje
to sekcja [Dodawanie detektora](https://github.com/boromat/polanon/blob/main/README.pl.md#dodawanie-detektora). polanon jest
tam opcjonalny: aplikacja instaluje go na żądanie, komunikuje się z nim przez niewielki most
JSON-lines i nie wysyła zupełnie nic, gdy maskowanie jest włączone, a polanon nie działa.

## Pisali o polanon

- [unknowNews](https://news.mrugalski.pl/post/833420581541117952/13-listopada-2026), newsletter Jakuba Mrugalskiego, 13 listopada 2026: „polanon -
  zamaskuj polskie dane osobowe przed wysłaniem ich do LLM-a”.
- [blog.prokulski.science](https://blog.prokulski.science/2026/11/16/pamiec-agentow-ai-polskie-pesel-e-w-promptach-i-wykresy-gestosci-w-power-bi/),
  cotygodniowy przegląd Łukasza Prokulskiego, 16 listopada 2026: jak polanon odwracalnie maskuje
  PESEL, NIP i IBAN, także imiona i nazwiska w każdym przypadku, zanim tekst trafi do modelu
  językowego.

## Współtworzenie

Mile widziane są zgłoszenia błędów i przeoczonych wartości oraz nowe detektory: zobacz
[CONTRIBUTING.md](https://github.com/boromat/polanon/blob/main/CONTRIBUTING.md). Sposób na
przemycenie oryginalnej wartości przez maskowanie to problem bezpieczeństwa: zgłoś go prywatnie, tak
jak opisuje [SECURITY.md](https://github.com/boromat/polanon/blob/main/SECURITY.md).

Licencja MIT.
