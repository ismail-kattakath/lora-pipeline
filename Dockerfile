FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip \
 && pip install --no-cache-dir --root-user-action=ignore .

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/lora-pipeline /usr/local/bin/lora-pipeline
COPY src/ src/

ENV PYTHONUNBUFFERED=1 \
    IMAGE_ROOT=/data \
    OLLAMA_HOST=http://ollama:11434

CMD ["lora-pipeline"]
