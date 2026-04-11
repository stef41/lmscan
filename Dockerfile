FROM python:3.12-slim

LABEL org.opencontainers.image.title="lmscan" \
      org.opencontainers.image.description="AI text forensics — detect AI-generated text and fingerprint which LLM wrote it" \
      org.opencontainers.image.source="https://github.com/stef41/lmscan" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN pip install --no-cache-dir lmscan

ENTRYPOINT ["lmscan"]
