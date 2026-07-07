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
COPY LICENSE ./
COPY data.py ./
COPY train.py ./
COPY testing.py ./
COPY FL ./FL
COPY models ./models
COPY flower-config.toml /root/.flwr/config.toml

RUN pip install --upgrade pip setuptools wheel
# flwr 1.32.1 and flwr-datasets 0.6.0 disagree on the rich version; install in separate
# steps so pip resolves them without failing the combined resolve.
RUN pip install "flwr[simulation]==1.32.1"
RUN pip install "flwr-datasets[vision]==0.6.0" "torch==2.8.0" "torchvision==0.23.0"
RUN pip install --no-deps .

CMD ["python", "--version"]
