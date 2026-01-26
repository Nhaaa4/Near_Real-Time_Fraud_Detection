import json
import os
import logging
from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv
import signal
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="/app/.env")

class TransactionConsumer:
    def __init__(self):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
        self.topic = os.getenv('KAFKA_TOPIC', 'transactions')  # Read ML predictions
        self.group_id = os.getenv('KAFKA_CONSUMER_GROUP', 'fraud-alert-consumer')
        self.running = False
        
        self.consumer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'session.timeout.ms': 6000,
            'max.poll.interval.ms': 300000
        }
        
        # Add authentication if provided
        kafka_username = os.getenv("KAFKA_USERNAME")
        kafka_password = os.getenv("KAFKA_PASSWORD")
        
        if kafka_username and kafka_password:
            self.consumer_config.update({
                'security.protocol': 'SASL_SSL',
                'sasl.mechanism': 'PLAIN',
                'sasl.username': kafka_username,
                'sasl.password': kafka_password
            })
        
        self.consumer = Consumer(self.consumer_config)
        logger.info(f"Consumer initialized with bootstrap servers: {self.bootstrap_servers}")
        logger.info(f"Subscribing to topic: {self.topic}")
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Shutdown signal received, closing consumer...")
        self.running = False
        
    def process_message(self, message_value):
        """Process transaction with ML prediction"""
        try:
            transaction = json.loads(message_value)
            
            # Get prediction from ML inference
            prediction = transaction.get('prediction', 0)
            
            if prediction == 1:
                # FRAUD - Log with warning banner
                logger.warning("="*80)
                logger.warning("FRAUD DETECTED BY ML MODEL")
                logger.warning("="*80)
                logger.info(f"Transaction ID: {transaction.get('transaction_id')}")
                logger.info(f"Sender: {transaction.get('sender_account')} → Receiver: {transaction.get('receiver_account')}")
                logger.info(f"Amount: ${transaction.get('amount'):.2f} {transaction.get('transaction_type', 'N/A')}")
                logger.info(f"Merchant: {transaction.get('merchant_category')} | Channel: {transaction.get('payment_channel')}")
                logger.info(f"Device: {transaction.get('device_used')} | Location: {transaction.get('location')}")
                logger.info(f"Risk Scores - Deviation: {transaction.get('spending_deviation_score'):.2f}, "
                           f"Velocity: {transaction.get('velocity_score')}, "
                           f"Geo-Anomaly: {transaction.get('geo_anomaly_score'):.2f}")
                logger.warning(f"ML Prediction: FRAUD (Threshold: 0.60)")
                logger.warning("="*80)
            else:
                # LEGITIMATE - Log with info banner
                logger.info("-"*80)
                logger.info("LEGITIMATE TRANSACTION")
                logger.info("-"*80)
                logger.info(f"Transaction ID: {transaction.get('transaction_id')}")
                logger.info(f"Sender: {transaction.get('sender_account')} → Receiver: {transaction.get('receiver_account')}")
                logger.info(f"Amount: ${transaction.get('amount'):.2f} {transaction.get('transaction_type', 'N/A')}")
                logger.info(f"Merchant: {transaction.get('merchant_category')} | Channel: {transaction.get('payment_channel')}")
                logger.info(f"ML Prediction: LEGITIMATE (Threshold: 0.60)")
                logger.info("-"*80)
            
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON message: {e}")
            return False
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return False
    
    def consume(self):
        """Start consuming messages from Kafka"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            self.consumer.subscribe([self.topic])
            self.running = True
            logger.info(f"Started consuming from topic: {self.topic}")
            
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                    
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        logger.debug(f"Reached end of partition {msg.partition()} at offset {msg.offset()}")
                    elif msg.error():
                        logger.error(f"Kafka error: {msg.error()}")
                        raise KafkaException(msg.error())
                else:
                    # Process message
                    logger.debug(f"Received message from partition {msg.partition()} at offset {msg.offset()}")
                    self.process_message(msg.value().decode('utf-8'))
                    
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}")
            raise
        finally:
            logger.info("Closing consumer...")
            self.consumer.close()
            logger.info("Consumer closed successfully")

def main():
    logger.info("Starting Fraud Detection Consumer...")
    consumer = TransactionConsumer()
    
    try:
        consumer.consume()
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user")
    except Exception as e:
        logger.error(f"Consumer failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
