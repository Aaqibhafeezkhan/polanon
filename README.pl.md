# polanon

[![CI](https://github.com/boromat/polanon/actions/workflows/ci.yml/badge.svg)](https://github.com/boromat/polanon/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/boromat/polanon/blob/main/pyproject.toml)
[![Licencja: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/boromat/polanon/blob/main/LICENSE)

[English](https://github.com/boromat/polanon/blob/main/README.md) | **Polski**

**Odwracalne maskowanie danych osobowych w zapytaniach do LLM, przede wszystkim dla języka polskiego.**
Zamaskuj dane osobowe, zanim prompt opuści twoją sieć, wyślij do modelu same tokeny, a oryginały
wstaw z powrotem do jego odpowiedzi — nawet wtedy, gdy przychodzi strumieniowo. Czysty Python: rdzeń
nie potrzebuje niczego poza biblioteką standardową.

![polanon w terminalu maskuje polską prośbę o korektę faktury, a potem w Pythonie przywraca dane w odpowiedzi modelu](https://raw.githubusercontent.com/boromat/polanon/main/docs/demo.gif)

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
  ciągów cyfr — w przypadku PESEL-u, razem z datą urodzenia, około 98% z nich — a prawidłowy numer
  jest zawsze akceptowany. Numery KRS, które nie mają publicznej sumy kontrolnej, oraz paszporty też
  są rozpoznawane, gdy występują obok swojej etykiety.
- **Siedem przypadków gramatycznych.** „Jan Kowalski”, „Jana Kowalskiego” i „z Janem Kowalskim” to
  ta sama osoba. Polskie imiona są rozpoznawane w każdym z nich, także pisane małymi literami i bez
  polskich znaków, razem z nazwiskiem, które się z nimi zgadza — bez żadnego modelu językowego.
- **Polskie etykiety w modelach.** Polskie modele spaCy oznaczają osoby jako `persName`, a nie
  `PERSON`. Pipeline skonfigurowany dla angielskich etykiet nie znajduje żadnych imion ani nazwisk
  i w żaden sposób o tym nie informuje.

Radzi sobie też z zagranicznymi fakturami: numery VAT z UE, Wielkiej Brytanii i Szwajcarii, numery
rejestrowe spółek, Bankgiro, Plusgiro i sort code, międzynarodowe numery telefonów, adresy i kody
pocztowe.

## Na tle innych

|  | polanon | [Presidio](https://github.com/data-privacy-stack/presidio) | [pii-veil](https://pypi.org/project/pii-veil/) | [piimask](https://pypi.org/project/piimask/) |
|---|---|---|---|---|
| Przywracanie danych w odpowiedzi modelu | Tak, ze stabilnych tokenów takich jak `<OSOBA_1>`, także strumieniowo | Nie jest wbudowane: jego deanonymizer odszyfrowuje fragmenty na znanych pozycjach; dokumentacja zawiera przykład, który przywraca dane w odpowiedzi LLM | Tak, ze stabilnych tokenów i zapisanego mapowania | Tak, ze stabilnych tokenów |
| Polskie identyfikatory z sumą kontrolną | PESEL, NIP, REGON, dowód osobisty, IBAN, karty płatnicze | PESEL; IBAN i karty płatnicze w wersji ogólnej | PESEL, NIP, REGON, IBAN, karty płatnicze; dowód osobisty i paszport tylko według wzorca | Brak; IBAN i karty płatnicze w wersji ogólnej |
| Polskie imiona i nazwiska w każdym przypadku gramatycznym | Dla imion z listy i zgodnych z nimi nazwisk, bez modelu językowego; resztę dokłada spaCy | Z polskim modelem spaCy i zmapowaną etykietą `persName` | Nie są wykrywane | Nie są wykrywane; jego README sugeruje dodanie modelu NER |
| Czego potrzebuje rdzeń | Niczego poza biblioteką standardową | spaCy, numpy, pydantic i kilku innych pakietów; modelu językowego tylko do imion i nazwisk oraz miejsc | pii-core i typer | Niczego poza biblioteką standardową |
| Na [przykładowych dokumentach](https://github.com/boromat/polanon/blob/main/benchmarks/README.md) biblioteki polanon | 98% zamaskowanych w całości z `pl_core_news_md`, 86% przy samych regułach | 66% z tym samym modelem (presidio-analyzer 2.2.366) | Nie mierzono | Nie mierzono |
| Licencja | MIT | MIT | Apache-2.0 | MIT |

Porównanie na podstawie własnej dokumentacji i kodu źródłowego każdego z projektów, stan na
3 listopada 2026 r. Na przykładowych dokumentach uruchomiono tylko Presidio, skryptem
z `benchmarks/`, i nie są to neutralne warunki: jedenaście syntetycznych dokumentów napisał autor
biblioteki polanon, który dostrajał na nich jej reguły, a Presidio dostał tylko polski model, także
do trzech faktur z zagranicy. Każde z tych narzędzi ma swoje miejsce: Presidio — do wielu języków,
danych ustrukturyzowanych i integracji z Azure; pii-veil — do polskich identyfikatorów na licencji
Apache-2.0, gdy imiona i nazwiska nie mają znaczenia; piimask — do identyfikatorów niezależnych od
języka (e-mail, numery telefonów, karty płatnicze, IBAN, adresy IP, adresy URL) i amerykańskich
numerów ubezpieczenia społecznego, ze strategiami podstawiania fałszywych wartości, haszowania
i częściowego maskowania oraz zaszyfrowanym sejfem. polanon jest dla polskiego tekstu wysyłanego do
modelu, gdy imiona i nazwiska w nim zawarte oraz przywrócona odpowiedź liczą się tak samo jak numery.

## Instalacja

| Czego potrzebujesz | Co zainstalować |
|---|---|
| Identyfikatory, dane kontaktowe, sekrety, polskie imiona i nazwiska, nazwy firm rozpoznawane po formie prawnej | `pip install polanon` |
| Dodatkowo miejsca oraz imiona, nazwiska i organizacje, których nie wychwyci żadna reguła | `pip install "polanon[ner]"`, a potem `python -m spacy download pl_core_news_md` |
| Sędziowie etapu 2 (lokalny model w Ollamie albo zdalny) | `pip install "polanon[judge]"` |

Następnie użyj `default_guards()` z modelem albo `default_guards(with_ner=False)` bez niego.
Inny model spaCy to kwestia jednego argumentu: `default_guards(spacy_model="pl_core_news_lg")`.

## Z wiersza poleceń

Tekst jest czytany z argumentów, z pliku podanego w `--file` albo ze standardowego wejścia.
Zamaskowany tekst trafia na stdout, a mapowanie na stderr albo do pliku podanego w `--mapping`:

```bash
polanon --version  # wypisuje zainstalowaną wersję
polanon "Jan Kowalski, PESEL 50810112354"
polanon --file umowa.txt --ner --mapping mapping.json > masked.txt
cat umowa.txt | polanon --ner > masked.txt 2> mapping.json
```

Niezerowy kod wyjścia oznacza, że coś poszło nie tak, a komunikat na stderr wyjaśnia, co. `--file`
sam czyta bajty pliku i odróżnia od siebie UTF-8, UTF-16 (tak zapisuje Windows PowerShell 5.1
przy `>`) i cp1250; tekst, którego nie da się jednoznacznie zdekodować, jest odrzucany, zamiast być
maskowany po omacku.

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
której nie odczyta żadna reguła. `mask_query` i `index_chunks` sprawdzają w magazynie każdy ciąg
do czterech słów i maskują te, dla których ma token, więc pytanie dostaje token fragmentów, a
nazwisko zamaskowane w jednym dokumencie pozostaje zamaskowane w następnym. Przy magazynie, który
jest na to za wolny, przekaż `lookup=False`.

**Klucz.** Co najmniej 32 losowe bajty, jeden klucz na firmę i osobny `scope=` dla indeksu, którego
nie wolno powiązać z innym. Trzymaj go w menedżerze sekretów, poza indeksem, kodem i logami;
`Pseudonyms` nigdy go nie pokazuje, nie serializuje ani nie kopiuje. Nowy klucz, zakres albo liczba
cyfr zmienia każdy token, więc indeks trzeba zbudować od nowa: zapisz `pseudonyms.settings` razem z
indeksem i porównuj je przy starcie. Żeby usunąć dane osoby, usuń fragmenty zawierające
`pseudonyms.token(name, "PERSON")`. Samo usunięcie wpisu z magazynu nie zrywa powiązania, bo klucz
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

## Co wykrywa

Polskie identyfikatory walidowane sumą kontrolną (PESEL, NIP, REGON, dowód osobisty, IBAN, karty
płatnicze), numery KRS obok swojej etykiety, adresy e-mail i numery telefonów, adresy i kody
pocztowe, linki do profilu osoby (LinkedIn, GitHub, GitHub Pages, GitLab, media społecznościowe) aż
po nazwę użytkownika, IPv4 i IPv6, adresy MAC, numery rejestracyjne pojazdów po etykiecie
(„nr rej.”), numery VIN, numery paszportów, daty urodzenia, klucze API, tokeny i hasła, polskie
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
a reguły z `pl_core_news_md` — 98%, przy czym nic nie zostaje zamaskowane tylko w części. Jedenaście
krótkich dokumentów to test regresji, a nie benchmark badawczy:
[notatki z benchmarku](https://github.com/boromat/polanon/blob/main/benchmarks/README.md) opisują,
co zostaje przeoczone i dlaczego, i podają wynik Presidio na tych samych dokumentach z tym samym
modelem: 66% zamaskowanych w całości, 76% przy najłagodniejszym sposobie liczenia.

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

**Token przywraca formę, którą zastąpił.** Maskowanie działa na tekście w takiej postaci, w jakiej
jest zapisany, więc imię i nazwisko zamaskowane w dopełniaczu wraca w dopełniaczu. Poproś model, żeby
zostawiał tokeny tam, gdzie pozwala na to gramatyka („dla <OSOBA_1>”), albo przywracaj do szablonu,
nad którym masz kontrolę. Zapisy różniące się tylko wielkością liter, rodzajem białych znaków (spacja
niełamiąca, podział wiersza, kilka spacji z rzędu) albo niewidocznymi znakami to ta sama wartość;
tak samo identyfikator, którego grupy zapisano osobno albo razem: wszystkie te zapisy dzielą jeden
token, a przywracanie wstawia ten, który pojawił się jako pierwszy. Wyjątkiem są hasła i typy
własne, bo dwa hasła różniące się wielkością liter to dwa różne hasła, a hasło nigdy nie dzieli
tokenu z podobnie wyglądającą nazwą. Jednowyrazowa nazwa znaleziona w zapisie samymi małymi
literami też dostaje własny token, z powodu opisanego niżej.

**Każde wystąpienie zamaskowanej wartości jest maskowane**, także tam, gdzie żaden detektor nie
rozpoznałby jej samodzielnie: w oryginalnym zapisie, wielkimi literami, rozdzielona podziałem
wiersza, ze znakiem o zerowej szerokości w środku, z inaczej rozdzielonymi grupami oraz w całości,
razem z tym, co detektor znalazł w jej wnętrzu. Dwa wyjątki sprawiają, że zwykłe słowa pozostają
czytelne. Jednowyrazowa nazwa pisana wielką literą nie jest dopasowywana, gdy występuje zapisana
samymi małymi literami, bo Zima, Kot i Wilk to nazwiska, a „zima” to pora roku; identyfikatory
takie jak VIN czy adres e-mail są dopasowywane bez względu na wielkość liter. Liczba krótsza niż
sześć cyfr, na przykład PIN, nie jest dopasowywana wewnątrz dłuższej liczby, daty, godziny ani
kwoty: „15.04.2025”, „2024–2025”, „2025,50” i „20250” zostają czytelne, gdy PIN to 2025,
a „w 2025 roku” i „PIN/PUK 2025/36912874” — nie.

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
wymyślone przez model i dane osobowe, które dodała jego odpowiedź: `report.is_clean` albo — jeśli
chcesz się im przyjrzeć — `report.unknown_tokens` i `report.new_spans`.

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

Jednowyrazowa nazwa pasuje też do swoich polskich form fleksyjnych („Facebooku”, „Canvie”).
Odrzucane są tylko wykrycia nazw i tylko wtedy, gdy każde ich słowo jest na liście; identyfikatory,
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

Gdy wykrycia się nakładają, o tym, które wygrywa, decyduje wynik (score): 1.0 dla wartości, za
którą ręczy suma kontrolna, etykieta albo oficjalny format, 0.85 dla statystycznego przypuszczenia;
znaleziska sędziego mają wynik 0.5. Sekret jest maskowany w całości, razem ze wszystkim, co wykryto
w jego wnętrzu. Fragment wychodzący poza tekst, typ zapisany w jakikolwiek inny sposób albo
zapisany tak jak etykieta wbudowanego tokenu (`OSOBA`, `DATA`) powoduje wyjątek `ValueError`
z nazwą twojego detektora.

## Zanim zaczniesz na tym polegać

Maskowanie to pseudonimizacja, a nie anonimizacja w rozumieniu prawa, i żaden detektor nie wyłapuje
wszystkiego. [Model zagrożeń](https://github.com/boromat/polanon/blob/main/docs/THREAT_MODEL.md)
opisuje, co jest objęte ochroną, a co nie, i co pozostaje zadaniem twojej aplikacji — począwszy od
bezpiecznego przechowywania mapowania.

Biblioteka polanon powstała jako silnik maskowania dla [PolGate](https://github.com/boromat/polgate),
samodzielnie hostowanego huba, dzięki któremu zespół może pracować z Claude'em albo GPT na
prawdziwych dokumentach bez wysyłania danych osobowych do dostawcy modelu. Biblioteka nic o nim nie
wie — żadnego HTTP, żadnej bazy danych, żadnych ustawień — a jeśli kiedykolwiek zacznie, jeden
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

## Współtworzenie

Mile widziane są zgłoszenia błędów i przeoczonych wartości oraz nowe detektory: zobacz
[CONTRIBUTING.md](https://github.com/boromat/polanon/blob/main/CONTRIBUTING.md). Sposób na
przemycenie oryginalnej wartości przez maskowanie to problem bezpieczeństwa: zgłoś go prywatnie, tak
jak opisuje [SECURITY.md](https://github.com/boromat/polanon/blob/main/SECURITY.md).

Licencja MIT.
