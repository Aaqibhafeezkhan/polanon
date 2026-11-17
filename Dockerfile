# The masking gateway in a container, with the spaCy model, so places and organisations are masked
# too. Inside the container the gateway listens on every interface, and so it starts only with a
# key every client has to send; publish the port on this machine alone:
#
#     docker run --rm -p 127.0.0.1:8787:8787 \
#         -e POLANON_UPSTREAM_KEY -e POLANON_GATEWAY_KEY \
#         ghcr.io/boromat/polanon-gateway --upstream https://api.openai.com/v1
#
# Every option of polanon-gateway goes after the image name; see polanon-gateway --help.

FROM python:3.12-slim-bookworm AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY polanon ./polanon
RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /dist

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY --from=build /dist /dist
RUN pip install "$(ls /dist/polanon-*.whl)[gateway,ner]" \
    && python -m spacy download pl_core_news_md \
    && rm -rf /dist \
    && useradd --system --uid 10001 --no-create-home polanon
USER polanon
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=4)"]
ENTRYPOINT ["polanon-gateway", "--host", "0.0.0.0", "--port", "8787"]

LABEL org.opencontainers.image.title="polanon-gateway" \
      org.opencontainers.image.description="Masks personal data in calls to an OpenAI-compatible API or Anthropic's Messages API, and restores the answers" \
      org.opencontainers.image.source="https://github.com/boromat/polanon" \
      org.opencontainers.image.licenses="MIT"
