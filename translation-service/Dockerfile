# NLLB-200-3.3B translation service (CTranslate2, CUDA).
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY server.py .

ENV MODEL_DIR=/models/nllb-200-3.3B-ct2 \
    DEVICE=cuda \
    COMPUTE_TYPE=int8_float16

EXPOSE 8100
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100"]
