"""The polanon-gateway command: its safe defaults, and that it never prints a key."""

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


def test_without_the_model_the_banner_says_places_are_not_masked(monkeypatch, capsys):
    monkeypatch.setattr("polanon.gateway.__main__.ner_available", lambda model: False)
    run(["--upstream", "https://provider.test/v1"])
    assert "places are not masked" in capsys.readouterr().err


def test_without_the_extra_the_command_says_what_to_install(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "polanon.gateway.app", None)
    status, served = run(UPSTREAM)
    assert status == BAD_SETTING
    assert served.calls == []
    assert 'pip install "polanon[gateway]"' in capsys.readouterr().err
