"""Start the masking gateway on this machine: ``polanon-gateway --upstream URL``.

    polanon-gateway --upstream https://api.openai.com/v1
    polanon-gateway --upstream http://localhost:11434/v1 --port 8788       # a local Ollama

The tool that talks to the model then points its base URL at ``http://127.0.0.1:8787/v1``. The
gateway listens on this machine only unless told otherwise, and then only with a key every client
must send. Keys come from the environment, never from the command line, where other users of the
machine could read them:

    POLANON_UPSTREAM_KEY   the provider's key, sent in place of the client's own
    POLANON_GATEWAY_KEY    the key clients must send; required to listen beyond this machine
    POLANON_KEY            64 hexadecimal characters: stable tokens, the same in every call
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from polanon import (
    WELL_KNOWN_NAMES,
    Allowlist,
    Anonymizer,
    Pseudonyms,
    default_guards,
    ner_available,
)

#: Exit status for a setting the gateway refuses to start with.
BAD_SETTING = 2

_INSTALL_HINT = 'the gateway needs its extra: pip install "polanon[gateway]"'

Serve = Callable[[Any, str, int], None]


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    serve: Serve | None = None,
) -> int:
    """Parse the settings, build the gateway and serve it until it is stopped.

    Args:
        argv: the arguments; None reads them from the command line.
        environ: where the keys are read from; None reads the process's environment.
        serve: runs the application; None serves it with uvicorn.

    Returns:
        The exit status: 0 after a clean stop, 2 for a setting the gateway refuses.
    """
    args = _parser().parse_args(argv)
    env = os.environ if environ is None else environ
    try:
        from polanon.gateway.app import Gateway, GatewayConfig
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return BAD_SETTING
    gateway_key = env.get("POLANON_GATEWAY_KEY") or None
    if not _loopback(args.host) and gateway_key is None:
        print(
            f"refusing to listen on {args.host} without a key: anyone who reaches it could send "
            "through it. Set POLANON_GATEWAY_KEY, or keep the default --host 127.0.0.1.",
            file=sys.stderr,
        )
        return BAD_SETTING
    try:
        anonymizer, detection = _anonymizer(args, env)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return BAD_SETTING
    config = GatewayConfig(
        upstream=args.upstream,
        upstream_key=env.get("POLANON_UPSTREAM_KEY") or None,
        gateway_key=gateway_key,
        instructions=None if args.instructions == "none" else args.instructions,
        unsupported="drop" if args.drop_unsupported else "refuse",
        max_body=args.max_body_mb * 1024 * 1024,
        timeout=args.timeout,
    )
    app = Gateway(anonymizer, config)
    print(_banner(args, config, detection), file=sys.stderr)
    (serve or _uvicorn)(app, args.host, args.port)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polanon-gateway",
        description="Mask calls to an OpenAI-compatible API on this machine, restore the answers.",
    )
    parser.add_argument(
        "--upstream",
        required=True,
        help="the provider's API root, such as https://api.openai.com/v1",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8787, help="port to listen on (default: 8787)")
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="mask with the rules alone, without the spaCy model, even when it is installed",
    )
    parser.add_argument("--spacy-model", default="pl_core_news_md", help="the spaCy model to load")
    parser.add_argument(
        "--no-well-known-names",
        action="store_true",
        help="also mask names on the well-known list, such as Facebook, Excel or ZUS",
    )
    parser.add_argument(
        "--instructions",
        choices=("pl", "en", "none"),
        default="pl",
        help="language of the note telling the model what the tokens are, or none",
    )
    parser.add_argument(
        "--drop-unsupported",
        action="store_true",
        help="leave out content that cannot be masked, such as images, rather than refuse the call",
    )
    parser.add_argument(
        "--max-body-mb", type=int, default=10, help="largest request, in MB (default: 10)"
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="seconds to wait for the provider"
    )
    return parser


def _anonymizer(args: argparse.Namespace, env: Mapping[str, str]) -> tuple[Anonymizer, str]:
    """The anonymizer every call is masked with, and a few words on how it detects.

    Raises:
        ValueError: for a POLANON_KEY that is not 64 hexadecimal characters; the key itself is never
            part of the message.
    """
    with_ner = not args.rules_only and ner_available(args.spacy_model)
    detection = f"rules and {args.spacy_model}" if with_ner else "rules alone"
    if not args.rules_only and not with_ner:
        detection += " (spaCy or the model is not installed: places are not masked)"
    pseudonyms = None
    if env.get("POLANON_KEY"):
        try:
            pseudonyms = Pseudonyms(bytes.fromhex(env["POLANON_KEY"]))
        except ValueError:
            raise ValueError(
                "POLANON_KEY must be at least 64 hexadecimal characters, such as the output of "
                'python -c "import secrets; print(secrets.token_hex(32))"'
            ) from None
        detection += ", stable tokens"
    allowlist = None if args.no_well_known_names else Allowlist(WELL_KNOWN_NAMES)
    anonymizer = Anonymizer(
        default_guards(with_ner=with_ner, spacy_model=args.spacy_model),
        allowlist=allowlist,
        pseudonyms=pseudonyms,
    )
    return anonymizer, detection


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _banner(args: argparse.Namespace, config: Any, detection: str) -> str:
    """One line on what the gateway does, with no key in it."""
    if config.gateway_key:
        keys = "clients send the gateway key"
    elif config.upstream_key:
        keys = "the provider gets the operator's key"
    else:
        keys = "the client's key is passed on"
    return (
        f"polanon gateway on http://{args.host}:{args.port}/v1 -> {args.upstream}; "
        f"masking with {detection}; {keys}"
    )


def _uvicorn(app: Any, host: str, port: int) -> None:  # pragma: no cover - starts a real server
    try:
        import uvicorn
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        raise SystemExit(BAD_SETTING) from None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(app, host=host, port=port, server_header=False, proxy_headers=False)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
