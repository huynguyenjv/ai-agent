FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080 \
    HF_HOME=/home/agent/.cache/huggingface

RUN useradd -m -u 1000 agent \
    && mkdir -p /home/agent/.cache/huggingface \
    && chown -R agent:agent /app /home/agent
USER agent

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python", "main.py"]
