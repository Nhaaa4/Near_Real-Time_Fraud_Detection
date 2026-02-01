FROM python:3.14.0-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    libssl-dev \
    libsasl2-dev \
    pkg-config \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set Java home
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Install librdkafka
RUN curl -fsSL https://github.com/confluentinc/librdkafka/archive/refs/tags/v2.8.0.tar.gz -o librdkafka.tar.gz && \
    tar -xzf librdkafka.tar.gz && \
    cd librdkafka-2.8.0 && \
    ./configure --enable-ssl --enable-sasl && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf librdkafka-2.8.0 librdkafka.tar.gz

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Run the consumer
CMD ["python", "main.py"]
