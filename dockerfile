FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./
COPY data.py ./
COPY train.py ./
COPY testing.py ./
COPY FL ./FL
COPY models ./models
COPY flower-config.toml /root/.flwr/config.toml

RUN pip install --upgrade pip setuptools wheel && \
    pip install .

CMD ["python", "--version"]
