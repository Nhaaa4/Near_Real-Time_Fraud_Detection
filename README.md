# Near Real-Time Fraud Detection System

A scalable fraud detection system that processes financial transactions in near real-time using Kafka, HDFS, Spark, and machine learning.

## Architecture

**Data Pipeline:**
1. **Producer** → Generates synthetic transaction data and streams to Kafka
2. **Consumer** → Reads from Kafka and writes to HDFS (Parquet format)
3. **Inference** → Batch processes transactions from HDFS, applies ML model, stores predictions in PostgreSQL

**Tech Stack:** Kafka, HDFS, Spark, XGBoost, PostgreSQL, Airflow

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Kafka, HDFS, and PostgreSQL settings
```

### 2. Run the Pipeline

```bash
# Terminal 1: Start transaction producer
python sparks/producer.py

# Terminal 2: Start HDFS consumer
python sparks/consumer.py

# Terminal 3: Run batch inference (processes last hour)
python sparks/inferences.py
```

## Configuration

Edit [`.env`](.env) for your environment:

```env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=transactions
HDFS_URL=hdfs://192.168.232.129:9000
HDFS_PATH=/fraud_detection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=fraud_detection
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

## Project Structure

```
├── sparks/
│   ├── producer.py          # Kafka transaction producer
│   ├── consumer.py           # HDFS consumer (Spark Streaming)
│   └── inferences.py         # Batch fraud prediction engine
├── dags/
│   └── training_fraud_detection_dag.py  # Airflow training pipeline
├── DATA/
│   ├── raw/                  # Original dataset
│   ├── processed/            # Cleaned data
│   └── feature_engineered/   # ML-ready features
├── notebooks/                # Jupyter analysis notebooks
└── models/                   # Trained ML models
```

## Features

### Transaction Producer
- Generates realistic financial transactions with fraud patterns (1-2% fraud rate)
- Streams to Kafka topic with configurable rate
- Includes: account IDs, amounts, merchant categories, locations, risk scores

### Consumer Pipeline
- Spark Structured Streaming from Kafka to HDFS
- Parquet format with timestamp partitioning
- Checkpointing for fault tolerance

### Inference Engine
- Loads transactions from HDFS (configurable time window)
- Applies 30+ engineered features (temporal, aggregation, network, risk)
- XGBoost model predictions with probability scores
- Stores results in PostgreSQL for monitoring/analysis

### Model Training (Airflow)
- Automated training pipeline
- Data preprocessing and feature engineering
- Model versioning and evaluation

## Fraud Detection Features

The system analyzes:
- **Temporal patterns**: Hour, day, weekend transactions
- **Spending behavior**: Deviation from average, velocity
- **Network analysis**: Sender/receiver transaction patterns
- **Geographic anomalies**: Location-based risk scores
- **Account history**: Fraud rates, transaction counts

## Output

Predictions stored in PostgreSQL `fraud_predictions` table:
- Transaction details (ID, accounts, amount)
- Fraud prediction (0/1) and probability
- Model version and prediction timestamp

## Requirements

- Python 3.8+
- Apache Kafka
- HDFS (Hadoop)
- PostgreSQL
- PySpark 3.5+
- See [`requirements.txt`](requirements.txt) for full dependencies

## License

See [LICENSE](LICENSE)