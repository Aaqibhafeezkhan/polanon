"""A masking gateway compatible with the OpenAI API, for whoever runs it on their own machine.

Tools that talk to an OpenAI-compatible API, such as Hermes Agent, n8n, Open WebUI or LibreChat,
point their base URL at the gateway instead of the provider. The gateway masks every message
before it leaves the machine, forwards the call to the provider the operator chose, and puts the
original values back into the answer, streamed or not, tool calls included. It keeps nothing on
disk: a client sends the whole conversation with every call, so each call is masked and restored
on its own.

Nothing here is a hosted service. Install it with ``pip install "polanon[gateway]"`` and start it
with ``polanon-gateway --upstream https://api.openai.com/v1``; the machine it runs on, its keys
and its logs are the operator's. :mod:`polanon.gateway.payloads` holds the masking of the payloads
and needs nothing beyond the standard library; :mod:`polanon.gateway.app` serves them over HTTP.
"""
