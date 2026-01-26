import logging
import os

from dotenv import load_dotenv
import joblib
import yaml

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, hour, dayofmonth, dayofweek, month, when, lit,
    log1p, to_timestamp, avg, stddev, count, approx_count_distinct, sum as spark_sum,
    lag, unix_timestamp
)
from pyspark.sql.window import Window
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import (StructType, StructField, StringType, IntegerType, DoubleType, TimestampType)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

class FraudDetectionInference:
    bootstrap_servers = None
    topic = None
    security_protocal = None
    sasl_mechanism = None
    username = None
    password = None
    sasl_jass_config = None
    
    def __init__(self, config_path="/app/config.yaml"):
        load_dotenv(dotenv_path='/app/.env')
        
        self.config = self._load_config(config_path)
        self.spark = self._init_spark_session()
        
        self.model = self._load_model(self.config["model"]["path"])
        self.broadcast_model = self.spark.sparkContext.broadcast(self.model)
        
        logger.debug(f"Environment variables laoded: {dict(os.environ)}")
    
    def _load_config(self, config_path):
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise e
        
    def _init_spark_session(self):
        try:
            packages = self.config.get("spark", {}).get("packages", "")
            builder = SparkSession.builder.appName("FraudDetectionInferenceStreaming")
            
            if packages:
                builder = builder.config('spark.jars.packages', packages)
            
            spark = builder.getOrCreate()
            logger.info("Spark Session initialized.")
            return spark
        except Exception as e:
            logger.error(f"Error Initializing Spark Session: {str(e)}")
            raise e
    
    def _load_model(self, model_path):
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def read_from_kafka(self):
        logger.info(f"Setting up Kafka read stream from topic: {self.config['kafka']['topic']}")
        
        kafka_config = self.config['kafka']
        kafka_bootstrap_servers = kafka_config.get("bootstrap_servers", "kafka:29092")
        kafka_topic = kafka_config.get("topic", "transactions")
        kafka_security_protocol = kafka_config.get("security_protocol", "PLAINTEXT")
        kafka_sasl_mechanism = kafka_config.get("sasl_mechanism", "PLAIN")
        kafka_username = kafka_config.get("username", "")
        kafka_password = kafka_config.get("password", "")
        
        self.bootstrap_servers = kafka_bootstrap_servers
        self.topic = kafka_topic
        self.security_protocol = kafka_security_protocol
        self.sasl_mechanism = kafka_sasl_mechanism
        self.username = kafka_username
        self.password = kafka_password
        
        # Only set JAAS config if username and password are provided
        if kafka_username and kafka_password:
            self.sasl_jaas_config = (
                f"org.apache.kafka.common.security.plain.PlainLoginModule required "
                f"username='{kafka_username}' password='{kafka_password}';"
            )
        else:
            self.sasl_jaas_config = None
        
        json_schema = StructType([
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
        
        # Build Kafka read stream options
        kafka_options = {
            "kafka.bootstrap.servers": kafka_bootstrap_servers,
            "subscribe": kafka_topic,
            "startingOffsets": "latest",
            "kafka.security.protocol": kafka_security_protocol
        }
        
        # Only add SASL options if credentials are provided
        if self.sasl_jaas_config:
            kafka_options["kafka.sasl.mechanism"] = kafka_sasl_mechanism
            kafka_options["kafka.sasl.jaas.config"] = self.sasl_jaas_config
        
        stream_reader = self.spark.readStream.format("kafka")
        for key, value in kafka_options.items():
            stream_reader = stream_reader.option(key, value)
        
        df = stream_reader.load()
            
        parsed_df = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), json_schema).alias("data")) \
            .select("data.*")
        
        return parsed_df
    
    def add_features(self, df):
        df = df.withColumn("timestamp_parsed", to_timestamp(col("timestamp")))
        
        # Basic time features
        df = df.withColumn("hour", hour(col("timestamp_parsed")))
        df = df.withColumn("day", dayofmonth(col("timestamp_parsed")))
        df = df.withColumn("day_of_week", dayofweek(col("timestamp_parsed")))
        df = df.withColumn("month", month(col("timestamp_parsed")))
        
        logger.info("Computing features for streaming inference...")
        
        # === STREAMING-FRIENDLY FEATURES (no aggregations needed) ===
        
        # 1. Amount-based features
        df = df.withColumn("amount_log", log1p(col("amount")))
        df = df.withColumn("amount_squared", col("amount") * col("amount"))
        df = df.withColumn("amount_per_velocity", col("amount") / (col("velocity_score") + 1))
        
        # 2. Risk score interactions
        df = df.withColumn("deviation_squared", col("spending_deviation_score") * col("spending_deviation_score"))
        df = df.withColumn("risk_score_total", col("spending_deviation_score") + col("geo_anomaly_score") + col("velocity_score"))
        df = df.withColumn("risk_score_product", col("spending_deviation_score") * col("geo_anomaly_score") * col("velocity_score"))
        
        # 3. Time-based features
        df = df.withColumn("is_night_transaction", when(col("hour").between(22, 6), 1).otherwise(0))
        df = df.withColumn("is_business_hours", when(col("hour").between(9, 17), 1).otherwise(0))
        df = df.withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))
        df = df.withColumn("is_month_end", when(col("day") >= 25, 1).otherwise(0))
        
        # 4. Transaction characteristics
        df = df.withColumn("is_self_transfer", when(col("sender_account") == col("receiver_account"), 1).otherwise(0))
        df = df.withColumn("is_high_amount", when(col("amount") > 1000, 1).otherwise(0))
        df = df.withColumn("is_round_amount", when(col("amount") % 100 == 0, 1).otherwise(0))
        
        # 5. Use data from raw input (these come from the producer)
        df = df.withColumn("transaction_gap", col("time_since_last_transaction"))
        df = df.withColumn("is_rapid_transaction", when(col("time_since_last_transaction") < 60, 1).otherwise(0))
        
        # 6. Velocity and anomaly ratios
        df = df.withColumn("velocity_to_geo_ratio", when(col("geo_anomaly_score") > 0, 
                                                          col("velocity_score") / col("geo_anomaly_score")).otherwise(0.0))
        df = df.withColumn("amount_deviation_ratio", col("amount") * col("spending_deviation_score"))
        
        # 7. Default values for historical features
        df = df.withColumn("sender_total_transaction", lit(1))
        df = df.withColumn("sender_avg_amount", col("amount")) 
        df = df.withColumn("sender_std_amount", lit(0.0))
        df = df.withColumn("sender_degree", lit(1))
        df = df.withColumn("sender_fraud_transaction", lit(0))
        df = df.withColumn("transaction_per_day", lit(1))
        df = df.withColumn("receiver_total_transaction", lit(1))
        df = df.withColumn("receiver_degree", lit(1))
        df = df.withColumn("receiver_fraud_transaction", lit(0))
        df = df.withColumn("sender_fraud_percentage", lit(0.0))
        df = df.withColumn("receiver_fraud_percentage", lit(0.0))
        df = df.withColumn("amount_to_avg_ratio", lit(1.0))
        
        logger.info("Feature engineering completed.")
        
        return df
    
    def run_inference(self):
        import pandas as pd
        
        df = self.read_from_kafka()
        feature_df = self.add_features(df)
        feature_df = feature_df.withWatermark("timestamp_parsed", "24 hours")
        broadcast_model = self.broadcast_model

        @pandas_udf("int")
        def predict_udf(
            sender_account: pd.Series,
            receiver_account: pd.Series,
            amount: pd.Series,
            transaction_type: pd.Series,
            merchant_category: pd.Series,
            location: pd.Series,
            device_used: pd.Series,
            time_since_last_transaction: pd.Series,
            spending_deviation_score: pd.Series,
            velocity_score: pd.Series,
            geo_anomaly_score: pd.Series,
            payment_channel: pd.Series,
            ip_address: pd.Series,
            device_hash: pd.Series,
            hour: pd.Series,
            day: pd.Series,
            day_of_week: pd.Series,
            month: pd.Series,
            amount_per_velocity: pd.Series,
            amount_log: pd.Series,
            amount_to_avg_ratio: pd.Series,
            transaction_per_day: pd.Series,
            transaction_gap: pd.Series,
            is_night_transaction: pd.Series,
            is_weekend: pd.Series,
            is_self_transfer: pd.Series,
            sender_degree: pd.Series,
            receiver_degree: pd.Series,
            sender_total_transaction: pd.Series,
            receiver_total_transaction: pd.Series,
            sender_avg_amount: pd.Series,
            sender_std_amount: pd.Series,
            sender_fraud_transaction: pd.Series,
            receiver_fraud_transaction: pd.Series,
            sender_fraud_percentage: pd.Series,
            receiver_fraud_percentage: pd.Series,
            deviation_squared: pd.Series
        ) -> pd.Series:
            
            input_df = pd.DataFrame({
                "sender_account": sender_account,
                "receiver_account": receiver_account,
                "amount": amount,
                "transaction_type": transaction_type,
                "merchant_category": merchant_category,
                "location": location,
                "device_used": device_used,
                "time_since_last_transaction": time_since_last_transaction,
                "spending_deviation_score": spending_deviation_score,
                "velocity_score": velocity_score,
                "geo_anomaly_score": geo_anomaly_score,
                "payment_channel": payment_channel,
                "ip_address": ip_address,
                "device_hash": device_hash,
                "hour": hour,
                "day": day,
                "day_of_week": day_of_week,
                "month": month,
                "amount_per_velocity": amount_per_velocity,
                "amount_log": amount_log,
                "amount_to_avg_ratio": amount_to_avg_ratio,
                "transaction_per_day": transaction_per_day,
                "transaction_gap": transaction_gap,
                "is_night_transaction": is_night_transaction,
                "is_weekend": is_weekend,
                "is_self_transfer": is_self_transfer,
                "sender_degree": sender_degree,
                "receiver_degree": receiver_degree,
                "sender_total_transaction": sender_total_transaction,
                "receiver_total_transaction": receiver_total_transaction,
                "sender_avg_amount": sender_avg_amount,
                "sender_std_amount": sender_std_amount,
                "sender_fraud_transaction": sender_fraud_transaction,
                "receiver_fraud_transaction": receiver_fraud_transaction,
                "sender_fraud_percentage": sender_fraud_percentage,
                "receiver_fraud_percentage": receiver_fraud_percentage,
                "deviation_squared": deviation_squared
            })
            probabilities = broadcast_model.value.predict_proba(input_df)[:, 1]
            threshold = 0.60  # Tuned based on precision/recall requirements
            predictions = (probabilities >= threshold).astype(int)
            return pd.Series(predictions)

        # Apply predictions to streaming DataFrame
        prediction_df = feature_df.withColumn("prediction", predict_udf(
            *[col(f) for f in [
                "sender_account", "receiver_account", "amount", "transaction_type",
                "merchant_category", "location", "device_used", "time_since_last_transaction",
                "spending_deviation_score", "velocity_score", "geo_anomaly_score",
                "payment_channel", "ip_address", "device_hash",
                "hour", "day", "day_of_week", "month",
                "amount_per_velocity", "amount_log", "amount_to_avg_ratio",
                "transaction_per_day", "transaction_gap", "is_night_transaction", "is_weekend",
                "is_self_transfer", "sender_degree", "receiver_degree", "sender_total_transaction",
                "receiver_total_transaction", "sender_avg_amount", "sender_std_amount",
                "sender_fraud_transaction", "receiver_fraud_transaction", "sender_fraud_percentage",
                "receiver_fraud_percentage", "deviation_squared"
            ]]
        ))

        # Build Kafka write stream options
        write_options = {
            "kafka.bootstrap.servers": self.bootstrap_servers,
            "topic": 'transactions',
            "checkpointLocation": "checkpoints/checkpoint"
        }
        
        # Only add SASL options if credentials are provided
        if self.sasl_jaas_config:
            write_options["kafka.security.protocol"] = self.security_protocol
            write_options["kafka.sasl.mechanism"] = self.sasl_mechanism
            write_options["kafka.sasl.jaas.config"] = self.sasl_jaas_config
        else:
            write_options["kafka.security.protocol"] = "PLAINTEXT"
        
        # Write results back to Kafka topic
        stream_writer = prediction_df.selectExpr(
            "CAST(transaction_id AS STRING) AS key",
            "to_json(struct(*)) AS value"
        ).writeStream.format("kafka").outputMode("update")
        
        for key, value in write_options.items():
            stream_writer = stream_writer.option(key, value)
        
        stream_writer.start().awaitTermination()


if __name__ == "__main__":
    # Initialize pipeline with configuration
    inference = FraudDetectionInference("/app/config.yaml")

    # Start streaming processing and block until termination
    inference.run_inference()