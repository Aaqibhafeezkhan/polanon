"""The polanon-gateway command: its safe defaults, and that it never prints a key."""

import pickle
import sys

import pytest

pytest.importorskip("httpx")

from polanon.gateway.__main__ import BAD_SETTING, main

UPSTREAM = ["--upstream", "https://provider.test/v1", "--rules-only"]
KEY = "ab" * 32


class Served:
    """Stands in for uvicorn and keeps what it was asked to serve."""

    def __init__(self):
        self.calls = []

    def __call__(self, app, host, port):
        self.calls.append((app, host, port))

    @property
    def app(self):
        return self.calls[0][0]


def run(argv, environ=None):
    served = Served()
    status = main(argv, environ=environ or {}, serve=served)
    return status, served


def test_the_gateway_listens_on_this_machine_by_default(capsys):
    status, served = run(UPSTREAM)
    assert status == 0
    assert served.calls[0][1:] == ("127.0.0.1", 8787)
    banner = capsys.readouterr().err
    assert "http://127.0.0.1:8787/v1 -> https://provider.test/v1" in banner
    assert "rules alone" in banner
    assert "the client's key is passed on" in banner


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "gateway.lan"])  # noqa: S104
def test_listening_beyond_this_machine_without_a_key_is_refused(host, capsys):
    status, served = run([*UPSTREAM, "--host", host])
    assert status == BAD_SETTING
    assert served.calls == []
    assert "POLANON_GATEWAY_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("host", ["localhost", "::1", "127.0.0.2"])
def test_every_loopback_address_counts_as_this_machine(host):
    status, served = run([*UPSTREAM, "--host", host])
    assert status == 0
    assert served.calls[0][1] == host


def test_with_a_gateway_key_it_may_listen_on_the_network_and_no_key_is_printed(capsys):
    environ = {"POLANON_GATEWAY_KEY": "local-secret", "POLANON_UPSTREAM_KEY": "sk-operator"}
    status, served = run([*UPSTREAM, "--host", "0.0.0.0", "--port", "9000"], environ)  # noqa: S104
    assert status == 0
    assert served.calls[0][1:] == ("0.0.0.0", 9000)  # noqa: S104
    output = capsys.readouterr()
    assert "local-secret" not in output.err + output.out
    assert "sk-operator" not in output.err + output.out
    assert "clients send the gateway key" in output.err
    config = served.app._config
    assert config.gateway_key == "local-secret"
    assert config.upstream_key == "sk-operator"
    assert "local-secret" not in repr(config)


def test_the_operator_s_key_for_the_provider_is_named_in_the_banner(capsys):
    run(UPSTREAM, {"POLANON_UPSTREAM_KEY": "sk-operator"})
    assert "the provider gets the operator's key" in capsys.readouterr().err


def test_a_key_for_stable_tokens_turns_them_on():
    status, served = run(UPSTREAM, {"POLANON_KEY": KEY})
    assert status == 0
    assert served.app._anonymizer.pseudonyms is not None


@pytest.mark.parametrize("key", ["not hex", "abcd"])
def test_a_key_for_stable_tokens_that_is_not_one_is_refused_without_quoting_it(key, capsys):
    status, served = run(UPSTREAM, {"POLANON_KEY": key})
    assert status == BAD_SETTING
    assert served.calls == []
    err = capsys.readouterr().err
    assert "POLANON_KEY" in err
    assert key not in err


def test_the_options_reach_the_gateway():
    argv = [*UPSTREAM, "--instructions", "none", "--drop-unsupported", "--max-body-mb", "2"]
    argv += ["--timeout", "30", "--no-well-known-names"]
    _, served = run(argv)
    config = served.app._config
    assert config.instructions is None
    assert config.unsupported == "drop"
    assert config.max_body == 2 * 1024 * 1024
    assert config.timeout == 30
    assert served.app._anonymizer.allowlist is None


def test_content_that_cannot_be_masked_may_be_passed_and_the_banner_says_so(capsys):
    _, served = run([*UPSTREAM, "--pass-unsupported"])
    assert served.app._config.unsupported == "pass"
    assert "images and files go to the provider unmasked" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run([*UPSTREAM, "--pass-unsupported", "--drop-unsupported"])


def test_without_the_model_the_banner_says_places_are_not_masked(monkeypatch, capsys):
    monkeypatch.setattr("polanon.gateway._build.ner_available", lambda model: False)
    run(["--upstream", "https://provider.test/v1"])
    assert "places are not masked" in capsys.readouterr().err


def test_without_the_extra_the_command_says_what_to_install(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "polanon.gateway.app", None)
    status, served = run(UPSTREAM)
    assert status == BAD_SETTING
    assert served.calls == []
    assert 'pip install "polanon[gateway]"' in capsys.readouterr().err


def test_paranoid_mode_reaches_the_spacy_guard_and_the_banner(monkeypatch, capsys):
    from polanon.gateway.__main__ import _anonymizer, _parser

    installed = {"spacy", "pl_core_news_md"}
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: object() if name in installed else None
    )
    argv = ["--upstream", "https://provider.test/v1", "--paranoid"]
    anonymizer, detection = _anonymizer(_parser().parse_args(argv), {})
    assert anonymizer.guards[-1].mode == "paranoid"
    assert detection == "rules and pl_core_news_md in paranoid mode"
    normal, detection = _anonymizer(_parser().parse_args(argv[:2]), {})
    assert normal.guards[-1].mode == "normal"
    assert "paranoid" not in detection


def test_paranoid_mode_without_the_model_says_it_changes_nothing(capsys):
    status, _ = run([*UPSTREAM, "--paranoid"])
    assert status == 0
    assert "--paranoid needs the spaCy model" in capsys.readouterr().err


def test_the_operator_s_names_stay_readable_even_a_person(tmp_path, capsys):
    """An assistant that speaks for a company has a name and a company of its own in its prompt,
    and masked they made it answer as a token. The operator names them with --allow or a file
    of names, one per line, and they go out as written, a person too; anyone else is masked."""
    from polanon.gateway.__main__ import _anonymizer, _parser

    names = tmp_path / "names.txt"
    names.write_text("# who the assistant speaks for\nKwadrat Sp. z o.o.\n\n", encoding="utf-8")
    argv = [*UPSTREAM, "--allow", "Ola Nowak", "--allow-file", str(names)]
    anonymizer, detection = _anonymizer(_parser().parse_args(argv), {})
    text = "Asystentka Ola Nowak pisze w imieniu firmy Kwadrat Sp. z o.o. do Jana Kowalskiego."
    masked = anonymizer.anonymize_sync(text, {}).anonymized
    assert "Ola Nowak" in masked
    assert "Kwadrat Sp. z o.o." in masked
    assert "Kowalskiego" not in masked
    assert "2 of the operator's names left readable" in detection

    status, _ = run([*UPSTREAM, "--allow-file", str(tmp_path / "missing.txt")])
    assert status == BAD_SETTING
    assert "missing.txt" in capsys.readouterr().err


def test_the_operator_s_names_stay_readable_without_the_well_known_ones():
    from polanon.gateway.__main__ import _anonymizer, _parser

    argv = [*UPSTREAM, "--no-well-known-names", "--allow", "Ola Nowak"]
    anonymizer, _ = _anonymizer(_parser().parse_args(argv), {})
    masked = anonymizer.anonymize_sync("Ola Nowak pisze z Facebooka do Jana Kowalskiego.", {})
    assert masked.anonymized.startswith("Ola Nowak pisze")
    assert "Kowalskiego" not in masked.anonymized


def test_the_mapping_goes_back_to_clients_only_when_the_operator_says_so(capsys):
    status, served = run(UPSTREAM)
    assert status == 0
    assert served.app._config.pii_map is False
    assert "mapping" not in capsys.readouterr().err
    status, served = run([*UPSTREAM, "--pii-map"])
    assert status == 0
    assert served.app._config.pii_map is True
    assert "clients may ask for each call's mapping" in capsys.readouterr().err


def test_one_call_is_masked_at_a_time_in_the_gateway_s_own_process_by_default(capsys):
    status, served = run(UPSTREAM)
    assert status == 0
    assert served.app._config.workers == 1
    assert served.app._pool is None
    assert "processes" not in capsys.readouterr().err


def test_workers_mask_calls_side_by_side_and_never_carry_the_key(capsys):
    """Each process builds its anonymizer from the options and reads POLANON_KEY itself, so what
    is pickled to reach it holds the options and no key."""
    status, served = run(
        [*UPSTREAM, "--workers", "3", "--allow", "Ola Nowak"], {"POLANON_KEY": KEY}
    )
    assert status == 0
    assert served.app._config.workers == 3
    assert served.app._pool.workers == 3
    assert "masking in 3 processes with rules alone, stable tokens" in capsys.readouterr().err
    build = pickle.dumps(served.app._pool._build)
    assert b"Ola Nowak" in build
    assert KEY.encode() not in build
    assert bytes.fromhex(KEY) not in build


@pytest.mark.parametrize("count", ["0", "-1"])
def test_fewer_than_one_worker_is_refused(count, capsys):
    status, served = run([*UPSTREAM, "--workers", count])
    assert status == BAD_SETTING
    assert served.calls == []
    assert "workers is 1 or more" in capsys.readouterr().err
