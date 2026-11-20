"""Starts a stack from deploy/ with a stand-in provider and checks that its calls are masked.

    python deploy/tests/check_stack.py open-webui

It needs Docker with Compose and the image `polanon-gateway:ci`, built from this repository with
`docker build -t polanon-gateway:ci .`. The tool itself, Open WebUI, is not started: its image is
several gigabytes and changes outside this repository. What is checked is the stack's wiring:
where the stack points the tool, with which key, and that a call sent there reaches the provider
masked and comes back restored.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent

# Made up for the test and used nowhere else.
ENVIRONMENT = {
    "POLANON_UPSTREAM": "http://upstream:9000/v1",
    "POLANON_UPSTREAM_KEY": "stack-test-upstream-key",
    "POLANON_GATEWAY_KEY": "stack-test-gateway-key",
    "STACK_TESTS": str(TESTS),
}

# For each stack: the service that is the tool, the variables that carry the gateway's address
# and key to it, and what else the tool must be told so that nothing goes around the gateway.
TOOLS = {
    "open-webui": (
        "open-webui",
        "OPENAI_API_BASE_URL",
        "OPENAI_API_KEY",
        {"ENABLE_OLLAMA_API": "false"},
    ),
}


def project(stack: str) -> str:
    """The Compose project the test runs under: never the stack's own.

    Run under the stack's own name on a machine where the stack is up, `up` would put the test's
    image and keys in place of the real gateway, and `down --volumes` would delete Open WebUI's
    accounts and chats.
    """
    return f"polanon-stack-test-{stack}"


def compose(stack: str, *arguments: str, stdin: str | None = None) -> str:
    """Run `docker compose` on the stack with the test's services laid over it."""
    command = [
        "docker",
        "compose",
        "--project-name",
        project(stack),
        "--file",
        str(DEPLOY / stack / "docker-compose.yml"),
        "--file",
        str(TESTS / "compose.test.yml"),
        *arguments,
    ]
    # A fixed command; the stack's name must be one of TOOLS, which main() checks first.
    done = subprocess.run(  # noqa: S603
        command,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, **ENVIRONMENT},
        check=False,
    )
    if done.returncode:
        sys.exit(f"{' '.join(command)}\n{done.stdout}\n{done.stderr}")
    return done.stdout


def check_wiring(stack: str) -> tuple[str, str]:
    """Check where the stack points its tool, and return that address and key."""
    service, url_variable, key_variable, also = TOOLS[stack]
    config = json.loads(compose(stack, "config", "--format", "json"))
    assert config["name"] == project(stack), config["name"]
    services = config["services"]
    tool = services[service]["environment"]
    assert tool[url_variable] == "http://gateway:8787/v1", tool[url_variable]
    assert tool[key_variable] == ENVIRONMENT["POLANON_GATEWAY_KEY"], "the tool got another key"
    for variable, value in also.items():
        assert tool.get(variable) == value, f"{variable} is {tool.get(variable)!r}, not {value!r}"
    assert ENVIRONMENT["POLANON_UPSTREAM_KEY"] not in json.dumps(services[service]), (
        "the tool was given the provider's key"
    )
    assert not services["gateway"].get("ports"), "the gateway's port is published"
    for port in services[service].get("ports", []):
        assert port.get("host_ip") == "127.0.0.1", f"published beyond this machine: {port}"
    return tool[url_variable], tool[key_variable]


def main() -> None:
    """Check the wiring, start the gateway and the stand-in, and send the calls through."""
    stack = sys.argv[1] if len(sys.argv) > 1 else "open-webui"
    if stack not in TOOLS:
        sys.exit(f"no such stack: {stack}; the stacks are {', '.join(TOOLS)}")
    base_url, key = check_wiring(stack)
    try:
        compose(stack, "up", "--detach", "--wait", "--wait-timeout", "180", "gateway", "upstream")
        result = compose(
            stack,
            "exec",
            "--no-TTY",
            "--env",
            f"TOOL_BASE_URL={base_url}",
            "--env",
            f"TOOL_API_KEY={key}",
            "--env",
            "UPSTREAM_SEEN_URL=http://upstream:9000/seen",
            "gateway",
            "python",
            "-",
            stdin=(TESTS / "probe.py").read_text(encoding="utf-8"),
        )
        print(result, end="")
        logs = compose(stack, "logs", "--no-color", "gateway")
        keys = (ENVIRONMENT["POLANON_UPSTREAM_KEY"], ENVIRONMENT["POLANON_GATEWAY_KEY"])
        for secret in ("50810112354", "Kowalski", *keys):
            assert secret not in logs, "the gateway's log holds a value or a key"
    finally:
        compose(stack, "down", "--volumes", "--timeout", "5")
    print(f"ok: {stack}")


if __name__ == "__main__":
    main()
