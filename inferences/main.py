import logging
import os
from datetime import datetime, timedelta
from typing import Dict

from dotenv import load_dotenv
import joblib
import yaml

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, hour, dayofmonth, dayofweek, month, when, lit,
    log1p, to_timestamp, avg, stddev, count, approx_count_distinct, 
    sum as spark_sum, lag, unix_timestamp, coalesce, current_timestamp, expr
)
from pyspark.sql.window import Window
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

class BatchFraudInference:
    def __init__(self, config_path="/app/config.yaml"):
        load_dotenv(dotenv_path='/app/.env')
        
        self.hdfs_url = os.getenv('HDFS_URL', 'hdfs://namenode:9000')
        self.hdfs_input_path = os.getenv('HDFS_PATH', '/fraud_detection/transactions')
        self.hdfs_output_path = os.getenv('HDFS_OUTPUT_PATH', '/fraud_detection/predictions')
        
        logger.info(f"HDFS URL: {self.hdfs_url}")
        logger.info(f"Input path: {self.hdfs_input_path}")
        logger.info(f"Output path: {self.hdfs_output_path}")
        
        self.config = self._load_config(config_path)
        self.spark = self._init_spark_session()
        
        self.model = self._load_model(self.config["model"]["path"])
        self.broadcast_model = self.spark.sparkContext.broadcast(self.model)
        
        logger.info("Batch inference initialized successfully")
    
    def _load_config(self, config_path):
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise e
        
    def _init_spark_session(self):
        try:
            spark = SparkSession.builder \
                .appName("BatchFraudInference") \
                .config("spark.hadoop.fs.defaultFS", self.hdfs_url) \
                .config("spark.hadoop.hadoop.user.name", "hadoop") \
                .config("spark.sql.shuffle.partitions", "200") \
                .config("spark.driver.memory", "2g") \
                .config("spark.executor.memory", "2g") \
                .getOrCreate()
            
            logger.info("Spark Session initialized for batch inference")
            return spark
        except Exception as e:
            logger.error(f"Error initializing Spark Session: {str(e)}")
            raise e
    
    def _load_model(self, model_path):
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def load_data_from_hdfs(self, time_window_hours=1):
        try:
            full_path = f"{self.hdfs_url}{self.hdfs_input_path}"
            logger.info(f"Reading data from HDFS: {full_path}")
            
            df = self.spark.read.option("basePath", full_path).parquet(f"{full_path}/timestamp=*")
            
            # Convert timestamp partition column (string) to timestamp type
            if "timestamp" in df.columns:
                df = df.withColumn("timestamp_parsed", to_timestamp(col("timestamp")))
                
                # Filter to recent data (last N hours)
                if time_window_hours > 0:
                    cutoff_time = current_timestamp() - expr(f"INTERVAL {time_window_hours} HOURS")
                    df = df.filter(col("timestamp_parsed") >= cutoff_time)
                    logger.info(f"Filtered to last {time_window_hours} hours")
            else:
                logger.warning("No timestamp column found, skipping time filtering")
            
            record_count = df.count()
            logger.info(f"Loaded {record_count} transactions from HDFS")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from HDFS: {str(e)}")
            raise
    
    def apply_full_feature_engineering(self, df):
        logger.info("Starting FULL feature engineering pipeline...")
        
        # Parse timestamp and extract temporal features
        if "timestamp_parsed" not in df.columns:
            df = df.withColumn("timestamp_parsed", to_timestamp(col("timestamp")))
        
        df = df.withColumn("hour", hour(col("timestamp_parsed")))
        df = df.withColumn("day", dayofmonth(col("timestamp_parsed")))
        df = df.withColumn("day_of_week", dayofweek(col("timestamp_parsed")))
        df = df.withColumn("month", month(col("timestamp_parsed")))
        
        # Sort by timestamp (CRITICAL for window functions)
        df = df.orderBy("timestamp_parsed")
        
        # Label encode categorical columns
        categorical_mappings = self._get_categorical_mappings()
        
        for col_name, mapping in categorical_mappings.items():
            if col_name in df.columns:
                expr = None
                for key, value in mapping.items():
                    if expr is None:
                        expr = when(col(col_name) == key, value)
                    else:
                        expr = expr.when(col(col_name) == key, value)
                expr = expr.otherwise(0)
                df = df.withColumn(col_name, expr)
        
        logger.info("Preprocessing completed")
    
        logger.info("Computing window-based aggregation features...")
        
        # Define window specifications (WORKS IN BATCH MODE!)
        sender_window = Window.partitionBy("sender_account").orderBy("timestamp_parsed") \
            .rangeBetween(Window.unboundedPreceding, Window.currentRow)
        # Window specifications for different operations
        # For aggregations that support frames
        sender_window_with_frame = Window.partitionBy("sender_account").orderBy("timestamp_parsed") \
            .rangeBetween(Window.unboundedPreceding, Window.currentRow)
        
        sender_day_window = Window.partitionBy("sender_account", "day").orderBy("timestamp_parsed") \
            .rangeBetween(Window.unboundedPreceding, Window.currentRow)
        
        receiver_window_with_frame = Window.partitionBy("receiver_account").orderBy("timestamp_parsed") \
            .rangeBetween(Window.unboundedPreceding, Window.currentRow)
        
        # For lag/lead functions that don't support frames
        sender_window_no_frame = Window.partitionBy("sender_account").orderBy("timestamp_parsed")
        
        # === Amount Features ===
        df = df.withColumn("amount_per_velocity", 
                          col("amount") / (col("velocity_score") + 1))
        df = df.withColumn("amount_log", log1p(col("amount")))
        
        # Real sender average (not default!)
        df = df.withColumn("sender_avg_amount", avg("amount").over(sender_window_with_frame))
        df = df.withColumn("amount_to_avg_ratio", 
                          col("amount") / coalesce(col("sender_avg_amount"), lit(1.0)))
        
        # === Frequency Features ===
        df = df.withColumn("transaction_per_day", count("*").over(sender_day_window))
        
        # Calculate real transaction gap (use window without frame for lag)
        df = df.withColumn("prev_timestamp", lag("timestamp_parsed").over(sender_window_no_frame))
        df = df.withColumn("transaction_gap", 
                          when(col("prev_timestamp").isNotNull(),
                               unix_timestamp("timestamp_parsed") - unix_timestamp("prev_timestamp"))
                          .otherwise(col("time_since_last_transaction")))
        
        # === Risk Features ===
        df = df.withColumn("is_night_transaction", 
                          when(col("hour").between(18, 24), 1).otherwise(0))
        df = df.withColumn("is_weekend", 
                          when(col("day_of_week").isin([1, 7]), 1).otherwise(0))
        df = df.withColumn("is_self_transfer", 
                          when(col("sender_account") == col("receiver_account"), 1).otherwise(0))
        
        # === Network Features (REAL VALUES!) ===
        df = df.withColumn("sender_degree", 
                          approx_count_distinct("receiver_account").over(sender_window_with_frame))
        df = df.withColumn("receiver_degree", 
                          approx_count_distinct("sender_account").over(receiver_window_with_frame))
        df = df.withColumn("sender_total_transaction", count("*").over(sender_window_with_frame))
        df = df.withColumn("receiver_total_transaction", count("*").over(receiver_window_with_frame))
        
        # === Aggregation Features (REAL VALUES!) ===
        df = df.withColumn("sender_std_amount", 
                          coalesce(stddev("amount").over(sender_window_with_frame), lit(0.0)))
        
        # === Fraud Features (from historical data) ===
        df = df.withColumn("sender_fraud_transaction", 
                          spark_sum(when(col("is_fraud") == 1, 1).otherwise(0)).over(sender_window_with_frame))
        df = df.withColumn("receiver_fraud_transaction", 
                          spark_sum(when(col("is_fraud") == 1, 1).otherwise(0)).over(receiver_window_with_frame))
        df = df.withColumn("sender_fraud_percentage", 
                          (col("sender_fraud_transaction") * 100.0 / col("sender_total_transaction")))
        df = df.withColumn("receiver_fraud_percentage", 
                          (col("receiver_fraud_transaction") * 100.0 / col("receiver_total_transaction")))
        
        # === Other Features ===
        df = df.withColumn("deviation_squared", 
                          col("spending_deviation_score") * col("spending_deviation_score"))
        
        # Drop temporary columns
        df = df.drop("prev_timestamp")
        
        logger.info("Feature engineering completed with window aggregates")
        logger.info("All features computed from streaming data using rolling windows")
        
        return df
    
    def _get_categorical_mappings(self) -> Dict[str, Dict]:
        """Get categorical mappings for label encoding"""
        return {
            "transaction_type": {
                "Online Purchase": 0, "Transfer": 1, "Withdrawal": 2, 
                "Deposit": 3, "Payment": 4
            },
            "merchant_category": {
                "Electronics": 0, "Grocery": 1, "Travel": 2, "Entertainment": 3,
                "Healthcare": 4, "Retail": 5, "Dining": 6, "Utilities": 7,
                "Fuel": 8, "Education": 9
            },
            "location": {
                "US-CA": 0, "US-NY": 1, "US-TX": 2, "US-FL": 3,
                "GB-LDN": 4, "CN-BJ": 5, "JP-TYO": 6, "IN-DEL": 7,
                "BR-SP": 8, "AU-SYD": 9
            },
            "device_used": {
                "Mobile": 0, "Desktop": 1, "Tablet": 2, "ATM": 3, "POS Terminal": 4
            },
            "payment_channel": {
                "Mobile App": 0, "Web": 1, "ATM": 2, "POS": 3, "Branch": 4
            }
        }
    
    def predict(self, df):
        logger.info("Running predictions with XGBoost model...")
        
        broadcast_model = self.broadcast_model
        
        MODEL_FEATURES = [
            "amount", "transaction_type", "merchant_category", "location", "device_used", 
            "time_since_last_transaction", "spending_deviation_score", "velocity_score", 
            "geo_anomaly_score", "payment_channel",
            "hour", "day", "day_of_week", "month",
            "amount_per_velocity", "amount_log", "amount_to_avg_ratio",
            "transaction_per_day", "transaction_gap", "is_night_transaction",
            "is_weekend", "is_self_transfer", "sender_degree", "receiver_degree",
            "sender_total_transaction", "receiver_total_transaction",
            "sender_avg_amount", "sender_std_amount", "sender_fraud_transaction",
            "receiver_fraud_transaction", "sender_fraud_percentage",
            "receiver_fraud_percentage", "deviation_squared"
        ]
        
        # Define prediction function with proper type hints
        @pandas_udf(DoubleType())
        def predict_fraud_probability(*cols: pd.Series) -> pd.Series:
            """Predict fraud probability with type-hinted columns"""
            try:
                input_df = pd.DataFrame({
                    feature: cols[i] for i, feature in enumerate(MODEL_FEATURES)
                })
                probabilities = broadcast_model.value.predict_proba(input_df)[:, 1]
                return pd.Series(probabilities)
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return pd.Series([0.0] * len(cols[0]) if len(cols) > 0 else [])
        
        # Apply predictions
        prediction_df = df.withColumn(
            "fraud_probability",
            predict_fraud_probability(*[col(f) for f in MODEL_FEATURES])
        ).withColumn(
            "fraud_prediction",
            when(col("fraud_probability") >= 0.5, 1).otherwise(0)
        ).withColumn(
            "prediction_timestamp",
            current_timestamp()
        ).withColumn(
            "model_version",
            lit("v1.0")
        )
        
        return prediction_df
    
    def save_predictions(self, df):
        """Save predictions back to HDFS"""
        
        output_path = f"{self.hdfs_url}{self.hdfs_output_path}"
        logger.info(f"Saving predictions to HDFS: {output_path}")
        
        # Select output columns
        output_df = df.select(
            "transaction_id",
            "sender_account",
            "receiver_account",
            "amount",
            "timestamp",
            "is_fraud",
            "fraud_prediction",
            "fraud_probability",
            "prediction_timestamp",
            "model_version"
        )
        
        # Write to HDFS in Parquet format
        output_df.write \
            .mode("append") \
            .partitionBy("prediction_timestamp") \
            .parquet(output_path)
        
        logger.info(f"Predictions saved to {output_path}")
    
    def run_batch_inference(self, time_window_hours=1):
        """
        Main execution: Load from HDFS, apply features, predict, save.
        """
        
        try:
            logger.info("=" * 80)
            logger.info("STARTING BATCH FRAUD INFERENCE FROM HDFS")
            logger.info("=" * 80)
            
            # Load data
            df = self.load_data_from_hdfs(time_window_hours)
            
            if df.count() == 0:
                logger.warning("No data found in HDFS for the specified time window")
                return
            
            # Apply full feature engineering
            df = self.apply_full_feature_engineering(df)
            
            # Run predictions
            prediction_df = self.predict(df)
            
            # Calculate metrics
            total_count = prediction_df.count()
            fraud_count = prediction_df.filter(col("fraud_prediction") == 1).count()
            fraud_rate = (fraud_count / total_count * 100) if total_count > 0 else 0
            
            logger.info("=" * 80)
            logger.info("BATCH INFERENCE COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Total transactions processed: {total_count:,}")
            logger.info(f"Predicted frauds: {fraud_count:,}")
            logger.info(f"Fraud rate: {fraud_rate:.2f}%")
            logger.info("=" * 80)
            
            # Show sample predictions BEFORE saving
            logger.info("\n" + "=" * 80)
            logger.info("SAMPLE FRAUD DETECTIONS:")
            logger.info("=" * 80)
            
            fraud_predictions = prediction_df.filter(col("fraud_prediction") == 1)
            fraud_sample_count = fraud_predictions.count()
            
            if fraud_sample_count > 0:
                logger.info(f"Showing {min(fraud_sample_count, 20)} fraud detections:\n")
                fraud_predictions.select(
                    "transaction_id", "sender_account", "receiver_account", 
                    "amount", "fraud_probability", "timestamp"
                ).show(20, truncate=False)
            else:
                logger.info("No fraudulent transactions detected in this batch")
            
            logger.info("=" * 80)
            logger.info("ALL TRANSACTIONS SAMPLE (10 records):")
            logger.info("=" * 80 + "\n")
            prediction_df.select(
                "transaction_id", "sender_account", "amount", 
                "fraud_prediction", "fraud_probability"
            ).show(10, truncate=False)
            
            # Save predictions
            self.save_predictions(prediction_df)
            
        except Exception as e:
            logger.error(f"Error in batch inference: {str(e)}")
            raise

if __name__ == "__main__":
    # Get time window from environment or use default (1 hour)
    time_window = int(os.getenv('BATCH_TIME_WINDOW_HOURS', '1'))
    
    logger.info(f"Starting batch inference with {time_window}-hour window")
    
    # Initialize and run
    inference = BatchFraudInference("/app/config.yaml")
    inference.run_batch_inference(time_window_hours=time_window)
    
    logger.info("Batch inference completed successfully")
