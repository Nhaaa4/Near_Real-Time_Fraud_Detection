import datetime
import json
import os
import signal
import time
import logging
import random
import pandas as pd
from typing import Any, Dict, Optional
from confluent_kafka import Producer
from dotenv import load_dotenv
from faker import Faker
from jsonschema import FormatChecker, ValidationError, validate

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="/app/.env")

fake = Faker()

TRANSACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction_id": {"type": "string"},
        "sender_account": {"type": "string"},
        "receiver_account": {"type": "string"},
        "amount": {"type": "number", "minimum": 0.01, "maximum": 10000},
        "transaction_type": {"type": "string"},
        "merchant_category": {"type": "string"},
        "location": {"type": "string"},
        "device_used": {"type": "string"},
        "is_fraud": {"type": "integer", "minimum": 0, "maximum": 1},
        "fraud_type": {"type": "string"},
        "time_since_last_transaction": {"type": "number"},
        "spending_deviation_score": {"type": "number"},
        "velocity_score": {"type": "integer"},
        "geo_anomaly_score": {"type": "number"},
        "payment_channel": {"type": "string"},
        "ip_address": {"type": "string"},
        "device_hash": {"type": "string"},
        "timestamp": {"type": "string"}
    },
    "required": ["transaction_id", "sender_account", "receiver_account", "amount", "transaction_type", "timestamp", "is_fraud"]
}

class TransactionProducer:
    def __init__(self):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.kafka_username = os.getenv("KAFKA_USERNAME")
        self.kafka_password = os.getenv("KAFKA_PASSWORD")
        self.topic = os.getenv("KAFKA_TOPIC", 'transactions')
        self.running = False
        
        self.producer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': 'transaction-producer',
            'compression.type': 'gzip',
            'linger.ms': '5',
            'batch.size': 16384,
        }
        
        if self.kafka_username and self.kafka_password:
            self.producer_config.update({
                'security.protocol': 'SASL_SSL',
                'sasl.mechanism': 'PLAIN',
                'sasl.username': self.kafka_username,
                'sasl.password': self.kafka_password,
            })
        else:
            self.producer_config['security.protocol'] = 'PLAINTEXT'
        
        try:
            self.producer = Producer(self.producer_config)
            logger.info("Confluent Kafka Producer Initialized Successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize confluent kafka producer: {str(e)}")
            raise e
        
        # Predefined lists matching the dataset format
        self.sender_accounts = [f"SA{str(i).zfill(6)}" for i in range(1, 100001)]
        self.receiver_accounts = [f"RA{str(i).zfill(6)}" for i in range(1, 100001)]
        self.transaction_types = ['Online Purchase', 'Transfer', 'Withdrawal', 'Deposit', 'Payment']
        self.merchant_categories = ['Electronics', 'Grocery', 'Travel', 'Entertainment', 'Healthcare', 
                                   'Retail', 'Dining', 'Utilities', 'Fuel', 'Education']
        self.payment_channels = ['Mobile App', 'Web', 'ATM', 'POS', 'Branch']
        self.devices = ['Mobile', 'Desktop', 'Tablet', 'ATM', 'POS Terminal']
        self.locations = ['US-CA', 'US-NY', 'US-TX', 'US-FL', 'GB-LDN', 'CN-BJ', 'JP-TYO', 'IN-DEL', 'BR-SP', 'AU-SYD']
        
        # For fraud patterns (1-2% fraud rate)
        self.compromised_accounts = set(random.sample(self.sender_accounts, 500))  # ~0.5% of accounts
        self.high_risk_merchants = ['Electronics', 'Travel']
        
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")
            
    def validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        try:
            validate(
                instance=transaction,
                schema=TRANSACTION_SCHEMA,
                format_checker=FormatChecker()
            )
            return True
        except ValidationError as e:
            logger.error(f"Invalid transaction: {str(e.message)}")
            return False
            
    def generate_transaction(self) -> Optional[Dict[str, Any]]:
        # Generate base transaction with new format
        sender_account = random.choice(self.sender_accounts)
        receiver_account = random.choice(self.receiver_accounts)
        transaction_type = random.choice(self.transaction_types)
        merchant_category = random.choice(self.merchant_categories)
        location = random.choice(self.locations)
        device_used = random.choice(self.devices)
        payment_channel = random.choice(self.payment_channels)
        
        # Generate realistic scores
        amount = round(random.lognormvariate(5, 2), 2)  # Log-normal distribution
        amount = min(amount, 10000.0)  # Cap at 10000
        amount = max(amount, 0.01)  # Min 0.01
        
        spending_deviation_score = round(random.uniform(0, 10), 2)
        velocity_score = random.randint(0, 20)
        geo_anomaly_score = round(random.uniform(0, 1), 2)
        
        transaction = {
            'transaction_id': fake.uuid4(),
            'sender_account': sender_account,
            'receiver_account': receiver_account,
            'amount': amount,
            'transaction_type': transaction_type,
            'merchant_category': merchant_category,
            'location': location,
            'device_used': device_used,
            'is_fraud': 0,
            'fraud_type': '',
            'time_since_last_transaction': round(random.uniform(-2, 10), 2),
            'spending_deviation_score': spending_deviation_score,
            'velocity_score': velocity_score,
            'geo_anomaly_score': geo_anomaly_score,
            'payment_channel': payment_channel,
            'ip_address': fake.ipv4(),
            'device_hash': fake.sha256()[:16],
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
        # Fraud logic to maintain 1-2% fraud rate
        is_fraud = 0
        
        # Pattern 1: Compromised accounts with high amounts (0.5%)
        if sender_account in self.compromised_accounts and amount > 500:
            if random.random() < 0.10:  # 10% of these become fraud
                is_fraud = 1
                transaction['fraud_type'] = 'account_takeover'
                transaction['spending_deviation_score'] = round(random.uniform(7, 10), 2)
                transaction['velocity_score'] = random.randint(15, 20)
        
        # Pattern 2: High-risk merchant categories with unusual scores (0.3%)
        if not is_fraud and merchant_category in self.high_risk_merchants:
            if amount > 1000 and random.random() < 0.015:
                is_fraud = 1
                transaction['fraud_type'] = 'card_fraud'
                transaction['spending_deviation_score'] = round(random.uniform(6, 9), 2)
                transaction['geo_anomaly_score'] = round(random.uniform(0.7, 1.0), 2)
        
        # Pattern 3: Suspicious velocity patterns (0.3%)
        if not is_fraud and velocity_score > 15:
            if random.random() < 0.02:
                is_fraud = 1
                transaction['fraud_type'] = 'transaction_laundering'
                transaction['spending_deviation_score'] = round(random.uniform(5, 8), 2)
        
        # Pattern 4: Geographic anomalies (0.2%)
        if not is_fraud and geo_anomaly_score > 0.8:
            if random.random() < 0.025:
                is_fraud = 1
                transaction['fraud_type'] = 'identity_theft'
                transaction['velocity_score'] = random.randint(10, 20)
        
        # Pattern 5: Random fraud for variety (0.2%)
        if not is_fraud and random.random() < 0.002:
            is_fraud = 1
            transaction['fraud_type'] = 'synthetic_identity'
            transaction['spending_deviation_score'] = round(random.uniform(6, 10), 2)
            transaction['velocity_score'] = random.randint(12, 20)
            transaction['geo_anomaly_score'] = round(random.uniform(0.5, 1.0), 2)
        
        transaction['is_fraud'] = is_fraud
        
        if self.validate_transaction(transaction):
            return transaction
     
    def send_transaction(self) -> bool:
        try:
            transaction = self.generate_transaction()
            if not transaction:
               return False

            self.producer.produce(
                self.topic,
                key=transaction['transaction_id'],
                value=json.dumps(transaction).encode('utf-8'),
                callback=self.delivery_report
            )
            
            self.producer.poll(0)
            return True
        except Exception as e:
            logger.error(f"Error producing message: {str(e)}")
            return False
        
    def run_continous_production(self, interval: float=0.0):
        self.running = True
        logger.info(f"Starting producer for topic: {self.topic}")
        
        try:
            while(self.running):
                if self.send_transaction():
                    time.sleep(interval)
        finally:
            self.shutdown()
    
    def shutdown(self, signum=None, frame=None):
        if self.running:
            logger.info("Initiating Shutdown")
            self.running = False
            
            if self.producer:
                self.producer.flush(timeout=30)
                self.producer.close()
            logger.info("Producer stopped")

if __name__ == "__main__":
    logger.info("Using synthetic TransactionProducer")
    producer = TransactionProducer()
    producer.run_continous_production()