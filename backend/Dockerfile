FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# PyTorch CPU por defecto (imagen más ligera). Para embeddings en GPU usa TORCH_INDEX=cu124.
ARG TORCH_INDEX=cpu
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY ingest ./ingest
COPY eval ./eval
COPY sql ./sql

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=10 \
  CMD curl -fsS http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
