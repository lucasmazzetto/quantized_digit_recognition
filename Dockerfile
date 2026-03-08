FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/neural_network

COPY requirements.txt /app/neural_network/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install \
      --index-url https://pypi.org/simple \
      -r /app/neural_network/requirements.txt

COPY . /app/neural_network

CMD ["python", "train.py"]
