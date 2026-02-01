import json
import os
import logging
from datetime import datetime
from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, TimestampType
)
from pyspark.sql.functions import from_json, col
import signal
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="/app/.env")

class HDFSConsumer:
    """Consumer that writes Kafka transactions to HDFS for batch processing"""
    
    def __init__(self):
        self.hdfs_url = os.getenv('HDFS_URL', 'hdfs://namenode:9000')
        self.hdfs_path = f"{self.hdfs_url}/fraud_detection/transactions"
        self.kafka_bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
        self.kafka_topic = os.getenv('KAFKA_TOPIC', 'transactions')
        
        logger.info(f"Initializing HDFS Consumer")
        logger.info(f"HDFS URL: {self.hdfs_url}")
        logger.info(f"HDFS Path: {self.hdfs_path}")
        logger.info(f"Kafka Bootstrap: {self.kafka_bootstrap}")
        logger.info(f"Kafka Topic: {self.kafka_topic}")
        
        # Initialize Spark Session with HDFS support
        self.spark = self._init_spark_session()
        
    def _init_spark_session(self):
        """Initialize Spark session with Kafka and HDFS support"""
        try:
            spark = SparkSession.builder \
                .appName("KafkaToHDFS_FraudDetection") \
                .config("spark.jars.packages", 
                       "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
                .config("spark.hadoop.fs.defaultFS", self.hdfs_url) \
                .config("spark.hadoop.hadoop.user.name", "hadoop") \
                .config("spark.sql.streaming.checkpointLocation", 
                       f"{self.hdfs_path}/_checkpoint") \
                .getOrCreate()
            
            logger.info("Spark Session initialized successfully")
            return spark
        except Exception as e:
            logger.error(f"Error initializing Spark Session: {str(e)}")
            raise
    
    def get_transaction_schema(self):
        """Define schema for transaction data"""
        return StructType([
            StructField("transaction_id", StringType(), True),
            StructField("sender_account", StringType(), True),
            StructField("receiver_account", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("transaction_type", StringType(), True),
            StructField("merchant_category", StringType(), True),
            StructField("location", StringType(), True),
            StructField("device_used", StringType(), True),
            StructField("is_fraud", IntegerType(), True),
            StructField("fraud_type", StringType(), True),
            StructField("time_since_last_transaction", DoubleType(), True),
            StructField("spending_deviation_score", DoubleType(), True),
            StructField("velocity_score", IntegerType(), True),
            StructField("geo_anomaly_score", DoubleType(), True),
            StructField("payment_channel", StringType(), True),
            StructField("ip_address", StringType(), True),
            StructField("device_hash", StringType(), True),
            StructField("timestamp", StringType(), True)
        ])
    
    def stream_to_hdfs(self):
        """Stream transactions from Kafka to HDFS in Parquet format"""
        try:
            logger.info(f"Setting up Kafka stream from topic: {self.kafka_topic}")
            
            # Read from Kafka
            kafka_df = self.spark.readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", self.kafka_bootstrap) \
                .option("subscribe", self.kafka_topic) \
                .option("startingOffsets", "earliest") \
                .option("failOnDataLoss", "false") \
                .load()
            
            logger.info("Kafka stream configured successfully")
            
            # Parse JSON messages
            schema = self.get_transaction_schema()
            parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as json_str") \
                .select(from_json(col("json_str"), schema).alias("data")) \
                .select("data.*")
            
            logger.info("JSON parsing configured")
            
            # Write to HDFS in Parquet format
            # Partitioned by date for efficient querying
            logger.info(f"Starting stream to HDFS: {self.hdfs_path}")
            
            query = parsed_df.writeStream \
                .format("parquet") \
                .option("path", self.hdfs_path) \
                .option("checkpointLocation", f"{self.hdfs_path}/_checkpoint") \
                .partitionBy("timestamp") \
                .trigger(processingTime='30 seconds') \
                .outputMode("append") \
                .start()
            
            logger.info("=" * 80)
            logger.info("HDFS CONSUMER STARTED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Streaming from Kafka topic: {self.kafka_topic}")
            logger.info(f"Writing to HDFS: {self.hdfs_path}")
            logger.info(f"Format: Parquet (partitioned by timestamp)")
            logger.info(f"Batch interval: 30 seconds")
            logger.info("=" * 80)
            
            # Wait for termination
            query.awaitTermination()
            
        except Exception as e:
            logger.error(f"Error in stream_to_hdfs: {str(e)}")
            raise
    
    def batch_write_to_hdfs(self):
        """Alternative: Batch write using Confluent Kafka Consumer"""
        consumer_config = {
            'bootstrap.servers': self.kafka_bootstrap,
            'group.id': 'hdfs-consumer-group',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True
        }
        
        # Add authentication if provided
        kafka_username = os.getenv("KAFKA_USERNAME")
        kafka_password = os.getenv("KAFKA_PASSWORD")
        
        if kafka_username and kafka_password:
            consumer_config.update({
                'security.protocol': 'SASL_SSL',
                'sasl.mechanism': 'PLAIN',
                'sasl.username': kafka_username,
                'sasl.password': kafka_password
            })
        
        consumer = Consumer(consumer_config)
        consumer.subscribe([self.kafka_topic])
        
        batch = []
        batch_size = 1000
        running = True
        
        def signal_handler(signum, frame):
            nonlocal running
            logger.info("Shutdown signal received")
            running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info(f"Batch consumer started, writing to HDFS every {batch_size} messages")
        
        try:
            while running:
                msg = consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                        break
                
                try:
                    # Parse message
                    transaction = json.loads(msg.value().decode('utf-8'))
                    batch.append(transaction)
                    
                    # Write batch when size reached
                    if len(batch) >= batch_size:
                        self._write_batch_to_hdfs(batch)
                        batch = []
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
            
            # Write remaining messages
            if batch:
                self._write_batch_to_hdfs(batch)
                
        finally:
            consumer.close()
            logger.info("Consumer closed")
    
    def _write_batch_to_hdfs(self, batch):
        """Write a batch of transactions to HDFS"""
        try:
            df = self.spark.createDataFrame(batch)
            
            # Get current date for partitioning
            date_str = datetime.now().strftime("%Y%m%d")
            partition_path = f"{self.hdfs_path}/date={date_str}"
            
            # Write to HDFS
            df.write.mode("append").parquet(partition_path)
            
            logger.info(f"Written {len(batch)} transactions to HDFS: {partition_path}")
            
        except Exception as e:
            logger.error(f"Error writing batch to HDFS: {e}")

def main():
    logger.info("Starting HDFS Consumer for Fraud Detection")
    
    try:
        consumer = HDFSConsumer()
        
        # Use Spark Structured Streaming (recommended)
        logger.info("Using Spark Structured Streaming mode")
        consumer.stream_to_hdfs()
        
        # Alternative: Use batch mode (uncomment to use)
        # logger.info("Using batch write mode")
        # consumer.batch_write_to_hdfs()
        
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user")
    except Exception as e:
        logger.error(f"Consumer failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
