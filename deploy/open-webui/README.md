# Open WebUI behind the polanon gateway

For a company choosing a chat now, [PolGate](https://github.com/boromat/polgate) is the one built
for this: it masks uploaded documents as well as messages, keeps the mapping in an encrypted vault,
deletes on a schedule, and finds, exports and erases what it holds about one person.

This directory is for a team that already works in Open WebUI and wants to keep it: every message
is masked before it reaches the model, and every answer comes back with the original names and
numbers. One command starts both.

Text Open WebUI takes from an uploaded file and puts in the prompt reaches the gateway and is
masked like a message. An image is not text: the gateway refuses a call that carries one, or
leaves it out when started with `--drop-unsupported`. Embeddings Open WebUI computes for its own
document search are made inside Open WebUI by default; set to an external provider, that call
bypasses the gateway unless its URL is the gateway's too.

```bash
cp .env.example .env          # put your provider's key and a gateway key in .env
docker compose up -d
```

Open WebUI is then at <http://127.0.0.1:3000>. The first account created there is its admin.

## What is where

- **The gateway** holds the provider's key, `POLANON_UPSTREAM_KEY`. Its port is not published:
  only Open WebUI, on the same Compose network, reaches it.
- **Open WebUI** gets the gateway's address, `http://gateway:8787/v1`, and the gateway's key,
  `POLANON_GATEWAY_KEY`, in place of the provider's. It never sees the provider's key.
- Open WebUI's own Ollama connection is turned off, since calls through it would not be masked.
- Port 3000 is published on this machine alone. To let a team reach it, put a reverse proxy with
  TLS in front of it; the gateway speaks plain HTTP, and TLS is the operator's job.

## Another provider or model

Any provider with an OpenAI-compatible API works; set `POLANON_UPSTREAM` in `.env`:

| Provider | `POLANON_UPSTREAM` |
|---|---|
| OpenAI | `https://api.openai.com/v1` (the default) |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Ollama on this machine | `http://host.docker.internal:11434/v1`, and on Linux add `extra_hosts: ["host.docker.internal:host-gateway"]` to the gateway |

Options of `polanon-gateway`, such as `--allow-file` for names that must stay readable, go in the
gateway's `command:` in `docker-compose.yml`, after `--upstream`.

## Connections added later

The connection set here is the one Open WebUI is started with. A connection added later in
Admin Settings → Connections bypasses the gateway unless its URL is `http://gateway:8787/v1` too.
Open WebUI keeps connections in its database, so once an admin has changed them, the variables
above stop counting; check that page after an upgrade.

## Versions

The gateway is pinned to `0.4`, which takes patch releases. Open WebUI is on `main`, the tag its
own instructions use; put a version in its `image:` to stay on one.

## Tested

`deploy/tests/check_stack.py open-webui` runs in CI against the gateway's image built from the
same commit. It checks where this file points Open WebUI, with which key, and that the ports are
on this machine alone; then it starts the gateway with a stand-in provider and sends a call as
Open WebUI would, plain and streamed. The stand-in must not receive the invented person and PESEL
number, the answer must come back with them, and a call without the key must be refused. Open
WebUI's own image is not started there, so the test cannot show that its container reaches the
gateway; the probe calls the gateway's address from inside the gateway's container. The test runs
under a Compose project of its own, `polanon-stack-test-open-webui`, so running it beside a stack
that is up leaves that stack and its data alone.
