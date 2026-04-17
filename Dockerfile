FROM python:3.11-slim

WORKDIR /app

<<<<<<< HEAD
# Install system dependencies
=======
>>>>>>> feature/new-architecture
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

<<<<<<< HEAD
# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8080"]

=======
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
>>>>>>> feature/new-architecture
