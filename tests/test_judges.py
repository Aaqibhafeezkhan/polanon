"""LLM-backed judges: strict parsing and failure handling, all with mocked transports."""

import pytest

httpx = pytest.importorskip("httpx")


from polanon.judge.base import JudgeError
from polanon.judge.ollama import OllamaJudge
from polanon.judge.parsing import parse_findings
from polanon.judge.remote import RemoteJudge
from polanon.pipeline import anonymize

MASKED = "<OSOBA_1> to dyrektor jedynej huty w Stalowej Woli."
GOOD_JSON = (
    '{"findings": [{"text": "dyrektor jedynej huty w Stalowej Woli",'
    ' "type": "PERSON", "reason": "x"}]}'
)


def _ollama(body, status=200):
    def handler(request):
        return httpx.Response(status, json={"message": {"content": body}})

    return OllamaJudge(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _remote(body, status=200):
    def handler(request):
        return httpx.Response(status, json={"choices": [{"message": {"content": body}}]})

    return RemoteJudge(
        base_url="https://provider.test/v1",
        api_key="k",
        model="m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_parsing_keeps_only_verbatim_quotes():
    raw = (
        '{"findings": [{"text": "nie ma tego", "type": "PERSON"},'
        ' {"text": "jedynej huty", "type": "ORG"}]}'
    )
    assert [f.text for f in parse_findings(raw, MASKED)] == ["jedynej huty"]


def test_parsing_normalises_unknown_types_to_other():
    raw = '{"findings": [{"text": "jedynej huty", "type": "WHATEVER"}]}'
    assert parse_findings(raw, MASKED)[0].entity_type == "OTHER"


def test_parsing_sorts_longest_finding_first():
    raw = '{"findings": [{"text": "huty", "type": "ORG"}, {"text": "jedynej huty", "type": "ORG"}]}'
    assert next(f.text for f in parse_findings(raw, MASKED)) == "jedynej huty"


@pytest.mark.parametrize("raw", ["not json", '{"nope": 1}', "[]"])
def test_parsing_rejects_malformed_responses(raw):
    with pytest.raises(JudgeError):
        parse_findings(raw, MASKED)


async def test_ollama_judge_returns_findings():
    findings = await _ollama(GOOD_JSON).review(MASKED)
    assert findings[0].entity_type == "PERSON"


async def test_ollama_judge_raises_on_server_error():
    with pytest.raises(JudgeError):
        await _ollama(GOOD_JSON, status=500).review(MASKED)


async def test_ollama_judge_raises_on_unparseable_output():
    with pytest.raises(JudgeError):
        await _ollama("definitely not json").review(MASKED)


async def test_ollama_judge_raises_on_timeout():
    def handler(request):
        raise httpx.ReadTimeout("too slow")

    judge = OllamaJudge(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(JudgeError):
        await judge.review(MASKED)


async def test_remote_judge_returns_findings():
    findings = await _remote(GOOD_JSON).review(MASKED)
    assert findings[0].text == "dyrektor jedynej huty w Stalowej Woli"


async def test_remote_judge_raises_on_error_status():
    with pytest.raises(JudgeError):
        await _remote(GOOD_JSON, status=401).review(MASKED)


def test_a_one_letter_finding_is_dropped_before_it_shreds_the_text():
    """A standalone "i" or "w" would be masked at every occurrence in the message."""
    from polanon.judge.parsing import parse_findings

    text = "Klient <OSOBA_1> prosi o fakturę i potwierdzenie."
    assert parse_findings('{"findings":[{"text":"i","type":"OTHER"}]}', text) == []
    assert parse_findings('{"findings":[{"text":"ie","type":"OTHER"}]}', text) == []


def test_a_finding_that_is_only_an_existing_token_is_dropped():
    """It quotes the text verbatim and is already masked; acting on it would allocate a token
    for a token."""
    from polanon.judge.parsing import parse_findings

    text = "Klient <OSOBA_1> prosi o fakturę."
    assert parse_findings('{"findings":[{"text":"<OSOBA_1>","type":"PERSON"}]}', text) == []


def test_a_finding_long_enough_to_matter_survives():
    from polanon.judge.parsing import parse_findings

    text = "Klient <OSOBA_1> prosi o fakturę."
    findings = parse_findings('{"findings":[{"text":"Klient","type":"PERSON"}]}', text)
    assert [f.text for f in findings] == ["Klient"]


async def test_a_judge_that_returns_junk_cannot_garble_a_message():
    """End to end: fragments too short to identify anyone leave the stage-1 text untouched.

    Asserting the round trip proved nothing: restore undoes any substitution, however much of the
    message it shredded on the way to the provider.
    """
    from polanon import Anonymizer, MockJudge, RegexGuard
    from polanon.policy import ENTITY_TYPES

    original = "Sprawa dotyczy firmy i jej klienta, PESEL 90010112349."
    judge = MockJudge({"i": "OTHER", "je": "OTHER"})
    anonymizer = Anonymizer([RegexGuard()], judge=judge, enabled=set(ENTITY_TYPES))

    result = await anonymizer.anonymize(original)

    assert result.anonymized == "Sprawa dotyczy firmy i jej klienta, PESEL <PESEL_1>."
    assert result.judge_hits == 0


@pytest.mark.parametrize(
    "reply",
    [
        '```json\n{"findings": [{"text": "jedynej huty", "type": "ORG"}]}\n```',
        'Oto wynik: {"findings": [{"text": "jedynej huty", "type": "ORG"}]}',
    ],
)
def test_a_reply_in_a_code_fence_or_after_a_sentence_is_read(reply):
    """Chat models wrap JSON in fences; a judge that failed on every such reply was off."""
    assert [f.text for f in parse_findings(reply, MASKED)] == ["jedynej huty"]


@pytest.mark.parametrize("value", ["null", "2024", "true", '["huty"]'])
def test_a_finding_whose_text_is_not_text_is_dropped(value):
    """A null used to become the finding "None" and mask every None in a pasted snippet."""
    reply = '{"findings": [{"text": ' + value + ', "type": "OTHER"}]}'
    assert parse_findings(reply, "None 2024 True huty") == []


def test_a_finding_that_contains_a_token_is_dropped():
    reply = '{"findings": [{"text": "<OSOBA_1> to dyrektor", "type": "PERSON"}]}'
    assert parse_findings(reply, MASKED) == []


def test_a_reply_with_no_content_is_a_judge_error():
    """An OpenAI-compatible endpoint answers with null content for a refusal."""
    with pytest.raises(JudgeError, match="no text content"):
        parse_findings(None, MASKED)


def test_an_unreadable_reply_never_quotes_what_it_flagged():
    """The reply is made of the fragments the judge found identifying, and the error is logged."""
    cut_off = '{"findings": [{"text": "Brzęczyszczykiewiczem", "type": "PERSON", "re'
    with pytest.raises(JudgeError) as error:
        parse_findings(cut_off, "z Brzęczyszczykiewiczem")
    assert "Brzęczyszczykiewicz" not in str(error.value)


@pytest.mark.parametrize(
    "body", [[], "tekst", {"message": None}, {"message": {"content": 7}}, {"other": 1}]
)
async def test_an_ollama_body_of_the_wrong_shape_is_a_judge_error(body):
    """A TypeError escaped the fallback policy and crashed the turn instead."""

    def handler(request):
        return httpx.Response(200, json=body)

    judge = OllamaJudge(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(JudgeError):
        await judge.review(MASKED)


async def test_a_remote_judge_error_never_quotes_the_api_key():
    seen = []

    def handler(request):
        seen.append(request.headers["authorization"])
        return httpx.Response(500, text="upstream rejected Bearer sk-SECRET")

    judge = RemoteJudge(
        base_url="https://provider.test/v1",
        api_key="sk-SECRET\n",
        model="m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(JudgeError) as error:
        await judge.review(MASKED)
    assert "SECRET" not in str(error.value)
    assert seen == ["Bearer sk-SECRET"]  # the newline a key file ends with is gone


def test_the_judges_can_be_imported_from_the_judge_package():
    from polanon.judge import OllamaJudge as Ollama
    from polanon.judge import RemoteJudge as Remote

    assert (Ollama, Remote) == (OllamaJudge, RemoteJudge)


def test_a_judge_without_httpx_names_the_extra_to_install(monkeypatch):
    import sys

    import polanon.judge

    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError) as error:
        polanon.judge.OllamaJudge  # noqa: B018 - the attribute access is what is tested
    assert "polanon[judge]" in str(error.value)


def _both_judges(handler, base_url="https://provider.test/v1"):
    def client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return [
        OllamaJudge(base_url=base_url, client=client()),
        RemoteJudge(base_url=base_url, api_key="k", model="m", client=client()),
    ]


async def test_a_transport_failure_is_reported_by_its_type_not_its_message():
    """A transport error can quote the request it failed on, header and key included."""

    def handler(request):
        raise httpx.ConnectError("cannot reach the host; header was Bearer SENTINEL")

    for judge in _both_judges(handler):
        with pytest.raises(JudgeError, match="ConnectError") as error:
            await judge.review(MASKED)
        assert "SENTINEL" not in str(error.value)


async def test_an_http_error_is_reported_by_its_status_not_its_message():
    """httpx names the URL in the message, and a URL can carry a token."""

    def handler(request):
        return httpx.Response(401, text="rejected")

    for judge in _both_judges(handler, base_url="https://SENTINEL.provider.test/v1"):
        with pytest.raises(JudgeError, match="HTTP 401") as error:
            await judge.review(MASKED)
        assert "SENTINEL" not in str(error.value)


def test_a_fenced_reply_after_a_sentence_with_braces_is_read():
    """Only the fence tells where the JSON is: the first brace opens the sentence's own text."""
    reply = (
        'Use {text, type}: ```json\n{"findings": [{"text": "jedynej huty", "type": "ORG"}]}\n```'
    )
    assert [f.text for f in parse_findings(reply, MASKED)] == ["jedynej huty"]


def test_parsing_reads_a_lowercase_type():
    raw = '{"findings": [{"text": "jedynej huty", "type": "org"}]}'
    assert parse_findings(raw, MASKED)[0].entity_type == "ORG"


async def test_lowercase_person_finding_is_masked_when_only_people_are_enabled():
    """Read as OTHER, the finding falls outside the policy and the surname goes out."""
    judge = _ollama('{"findings": [{"text": "Kowalski", "type": "person"}]}')
    result = await anonymize(
        "Kowalski zapłacił fakturę.",
        enabled={"PERSON"},
        guards=[],
        judge=judge,
        existing_mapping={},
    )
    assert result.anonymized == "<OSOBA_1> zapłacił fakturę."


async def test_remote_reply_with_no_choices_is_a_judge_error():
    """A content filter answers with an empty "choices" list: a JudgeError, not an IndexError."""

    def handler(request):
        return httpx.Response(200, json={"choices": []})

    judge = RemoteJudge(
        base_url="https://provider.test/v1",
        api_key="k",
        model="m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(JudgeError, match="unexpected shape"):
        await judge.review(MASKED)


def test_deeply_nested_reply_is_a_judge_error():
    with pytest.raises(JudgeError):
        parse_findings("[" * 100_000 + "]" * 100_000, MASKED)
