import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yaml
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Optional MLflow support
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Training will proceed without experiment tracking.")

# Configure dual logging to file and stdout with structured format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./fraud_detection_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FraudDetectionTraining:
    def __init__(self, config_path='config.yaml'):
        load_dotenv()

        # Load configuration
        self.config = self._load_config(config_path)
        
        # PostgreSQL configuration
        self.postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.postgres_port = os.getenv('POSTGRES_PORT', '5432')
        self.postgres_db = os.getenv('POSTGRES_DB', 'fraud_detection')
        self.postgres_user = os.getenv('POSTGRES_USER', 'postgres')
        self.postgres_password = os.getenv('POSTGRES_PASSWORD', 'postgres')

        # Setup MLflow if available
        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
        
        logger.info("FraudDetectionTraining initialized with PostgreSQL connection")
        logger.info(f"Database: {self.postgres_user}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}")

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info('Configuration loaded successfully from %s', config_path)
            return config
        except Exception as e:
            logger.error('Failed to load configuration: %s', str(e))
            raise

    def _setup_mlflow(self):
        try:
            tracking_uri = self.config.get('mlflow', {}).get('tracking_uri', 'http://localhost:5500')
            experiment_name = self.config.get('mlflow', {}).get('experiment_name', 'Near Real-Time Fraud Detection')
            
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info('MLflow configured: %s / %s', tracking_uri, experiment_name)
        except Exception as e:
            logger.warning('MLflow setup failed: %s. Continuing without tracking.', str(e))
    
    def _get_postgres_connection(self):
        try:
            conn = psycopg2.connect(
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
                user=self.postgres_user,
                password=self.postgres_password
            )
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise
    
    def load_data_from_postgres(self, table_name='fraud_predictions') -> pd.DataFrame:
        try:
            logger.info(f'Loading data from PostgreSQL table: {table_name}')
            
            conn = self._get_postgres_connection()
            
            # Query to load all necessary columns
            query = f"""
            SELECT 
                transaction_id,
                sender_account,
                receiver_account,
                amount,
                transaction_type,
                merchant_category,
                location,
                device_used,
                time_since_last_transaction,
                spending_deviation_score,
                velocity_score,
                geo_anomaly_score,
                payment_channel,
                ip_address,
                device_hash,
                timestamp,
                is_fraud
            FROM {table_name}
            ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                raise ValueError('No data loaded from PostgreSQL')
            
            if 'is_fraud' not in df.columns:
                raise ValueError('Target column "is_fraud" not found in data')
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Basic data quality checks
            fraud_rate = df['is_fraud'].mean() * 100
            logger.info('Data loaded from PostgreSQL: %d samples, %.2f%% fraud rate', len(df), fraud_rate)
            
            return df
            
        except Exception as e:
            logger.error('Failed to load data from PostgreSQL: %s', str(e))
            raise
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('Applying feature engineering...')
        df = df.copy()
        
        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek + 1  # 1-7 for Monday-Sunday
        df['month'] = df['timestamp'].dt.month
        
        # Label encode categorical features
        categorical_mappings = {
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
        
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
        
        ## Amount features
        df["amount_per_velocity"] = df["amount"] / (df["velocity_score"] + 1)
        df["amount_log"] = np.log1p(df["amount"])
        df["amount_to_avg_ratio"] = df["amount"] / df.groupby("sender_account")["amount"].transform("mean")
        
        ## Frequency features
        df["transaction_per_day"] = df.groupby(["sender_account", "day"])["amount"].transform("count")
        df["transaction_gap"] = df.groupby("sender_account")["timestamp"].diff().dt.total_seconds().fillna(0)
        
        ## Risk features
        df["is_night_transaction"] = df["hour"].between(18, 24).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(int)
        df["is_self_transfer"] = (df["sender_account"] == df["receiver_account"]).astype(int)
        
        ## Network features
        df["sender_degree"] = df.groupby("sender_account")["receiver_account"].transform("nunique")
        df["receiver_degree"] = df.groupby("receiver_account")["sender_account"].transform("nunique")
        df["sender_total_transaction"] = df.groupby("sender_account")["amount"].transform("count")
        df["receiver_total_transaction"] = df.groupby("receiver_account")["amount"].transform("count")
        
        ## Aggregation features
        df["sender_avg_amount"] = df.groupby("sender_account")["amount"].transform("mean")
        df["sender_std_amount"] = df.groupby("sender_account")["amount"].transform("std").fillna(0)
        
        ## Fraud features
        df["sender_fraud_transaction"] = df.groupby("sender_account")["is_fraud"].transform("sum")
        df["receiver_fraud_transaction"] = df.groupby("receiver_account")["is_fraud"].transform("sum")
        df["sender_fraud_percentage"] = (df["sender_fraud_transaction"] * 100 / df["sender_total_transaction"]).round(2)
        df["receiver_fraud_percentage"] = (df["receiver_fraud_transaction"] * 100 / df["receiver_total_transaction"]).round(2)
        df[["sender_fraud_percentage", "receiver_fraud_percentage"]] = df[["sender_fraud_percentage", "receiver_fraud_percentage"]].fillna(0)
        
        ## Other features
        df["deviation_squared"] = df["spending_deviation_score"] ** 2
        
        # Drop unnecessary columns
        columns_to_drop = ['transaction_id', 'sender_account', 'receiver_account', 
                          'ip_address', 'device_hash', 'timestamp']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Drop any rows with missing values in critical columns
        df = df.dropna(subset=['hour', 'day', 'day_of_week', 'month'])
        
        logger.info('Feature engineering completed: %d features created', df.shape[1] - 1)
        
        return df

    def train_model(self):
        try:
            logger.info('=' * 80)
            logger.info('STARTING MODEL TRAINING PIPELINE')
            logger.info('=' * 80)
            
            # Load data from PostgreSQL
            df = self.load_data_from_postgres()
            
            # Apply feature engineering
            df = self.apply_feature_engineering(df)
            
            # Prepare features and target
            y = df['is_fraud']
            X = df.drop(columns=['is_fraud'])
            
            logger.info(f'Training data shape: {X.shape}')
            logger.info(f'Feature names: {list(X.columns)}')
            
            # Data quality checks
            if y.sum() == 0:
                raise ValueError('No fraud samples in training data')
            if y.sum() < 10:
                logger.warning('Low fraud samples: %d. Model may not generalize well', y.sum())
            
            # Train/val/test split (70/15/15)
            random_state = self.config.get('model', {}).get('seed', 36)
            
            # First split: 70% train, 30% temp
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=0.30,
                stratify=y,
                random_state=random_state
            )
            
            # Second split: split temp into 50/50 for val and test (15% each)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=0.50,
                stratify=y_temp,
                random_state=random_state
            )
            
            logger.info(f'Train data shape: {X_train.shape}')
            logger.info(f'Val data shape: {X_val.shape}')
            logger.info(f'Test data shape: {X_test.shape}')
            
            # Start MLflow run if available
            if MLFLOW_AVAILABLE:
                mlflow.start_run(run_name=f"XGBoost_Fraud_Detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            try:
                # Calculate scale_pos_weight for class imbalance
                scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
                logger.info(f'Scale_pos_weight (class imbalance): {scale_pos_weight:.2f}')
                
                # Step 5: Model configuration (following notebook params)
                params = {
                    'n_estimators': 300,
                    'objective': 'binary:logistic',
                    'tree_method': 'hist',
                    'max_depth': 12,
                    'learning_rate': 0.05,
                    'reg_lambda': 3.6,
                    'reg_alpha': 3.6,
                    'scale_pos_weight': scale_pos_weight,
                    'eval_metric': 'aucpr',
                    'verbosity': 2,
                    'subsample': 0.8,
                    'n_jobs': -1,
                    'random_state': random_state
                }
                
                # Log parameters to MLflow
                if MLFLOW_AVAILABLE:
                    mlflow.log_params(params)
                    mlflow.log_params({
                        'train_samples': len(X_train),
                        'val_samples': len(X_val),
                        'test_samples': len(X_test),
                        'n_features': X_train.shape[1],
                        'fraud_rate_train': float(y_train.mean()),
                        'fraud_rate_val': float(y_val.mean()),
                        'fraud_rate_test': float(y_test.mean())
                    })
                
                # Step 6: Train model
                logger.info('=' * 80)
                logger.info('Training XGBoost model with MLflow tracking...')
                logger.info('=' * 80)
                
                model = XGBClassifier(**params)
                eval_set = [(X_train, y_train), (X_val, y_val)]
                model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
                
                logger.info('XGBoost training completed!')
                
                # ==================== VALIDATION SET EVALUATION ====================
                logger.info('')
                logger.info('=' * 80)
                logger.info('VALIDATION SET EVALUATION')
                logger.info('=' * 80)
                
                y_val_pred_proba = model.predict_proba(X_val)[:, 1]
                y_val_pred = model.predict(X_val)
                
                val_metrics = {
                    'val_precision': float(precision_score(y_val, y_val_pred)),
                    'val_recall': float(recall_score(y_val, y_val_pred)),
                    'val_f1_score': float(f1_score(y_val, y_val_pred)),
                    'val_auc_pr': float(average_precision_score(y_val, y_val_pred_proba)),
                    'val_auc_roc': float(roc_auc_score(y_val, y_val_pred_proba))
                }
                
                logger.info(f"Precision: {val_metrics['val_precision']:.4f}")
                logger.info(f"Recall:    {val_metrics['val_recall']:.4f}")
                logger.info(f"F1 Score:  {val_metrics['val_f1_score']:.4f}")
                logger.info(f"AUC-PR:    {val_metrics['val_auc_pr']:.4f}")
                logger.info(f"AUC-ROC:   {val_metrics['val_auc_roc']:.4f}")
                
                # Validation confusion matrix
                cm_val = confusion_matrix(y_val, y_val_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix - Validation Set')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                
                cm_val_path = 'confusion_matrix_val.png'
                plt.savefig(cm_val_path)
                plt.close()
                
                if MLFLOW_AVAILABLE:
                    mlflow.log_artifact(cm_val_path)
                
                # ==================== TEST SET EVALUATION ====================
                logger.info('')
                logger.info('=' * 80)
                logger.info('TEST SET EVALUATION')
                logger.info('=' * 80)
                
                y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                y_test_pred = model.predict(X_test)
                
                test_metrics = {
                    'test_precision': float(precision_score(y_test, y_test_pred)),
                    'test_recall': float(recall_score(y_test, y_test_pred)),
                    'test_f1_score': float(f1_score(y_test, y_test_pred)),
                    'test_auc_pr': float(average_precision_score(y_test, y_test_pred_proba)),
                    'test_auc_roc': float(roc_auc_score(y_test, y_test_pred_proba))
                }
                
                logger.info(f"Precision: {test_metrics['test_precision']:.4f}")
                logger.info(f"Recall:    {test_metrics['test_recall']:.4f}")
                logger.info(f"F1 Score:  {test_metrics['test_f1_score']:.4f}")
                logger.info(f"AUC-PR:    {test_metrics['test_auc_pr']:.4f}")
                logger.info(f"AUC-ROC:   {test_metrics['test_auc_roc']:.4f}")
                
                # Test confusion matrix
                cm_test = confusion_matrix(y_test, y_test_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens')
                plt.title('Confusion Matrix - Test Set')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                
                cm_test_path = 'confusion_matrix_test.png'
                plt.savefig(cm_test_path)
                plt.close()
                
                if MLFLOW_AVAILABLE:
                    mlflow.log_artifact(cm_test_path)
                
                # Combine metrics
                all_metrics = {**val_metrics, **test_metrics}
                
                # Log metrics to MLflow
                if MLFLOW_AVAILABLE:
                    mlflow.log_metrics(all_metrics)
                
                # ==================== SUMMARY ====================
                logger.info('')
                logger.info('=' * 80)
                logger.info('METRICS SUMMARY')
                logger.info('=' * 80)
                logger.info(f"{'Metric':<15} {'Validation':<15} {'Test':<15}")
                logger.info('-' * 80)
                logger.info(f"{'Precision':<15} {val_metrics['val_precision']:<15.4f} {test_metrics['test_precision']:<15.4f}")
                logger.info(f"{'Recall':<15} {val_metrics['val_recall']:<15.4f} {test_metrics['test_recall']:<15.4f}")
                logger.info(f"{'F1 Score':<15} {val_metrics['val_f1_score']:<15.4f} {test_metrics['test_f1_score']:<15.4f}")
                logger.info(f"{'AUC-PR':<15} {val_metrics['val_auc_pr']:<15.4f} {test_metrics['test_auc_pr']:<15.4f}")
                logger.info(f"{'AUC-ROC':<15} {val_metrics['val_auc_roc']:<15.4f} {test_metrics['test_auc_roc']:<15.4f}")
                logger.info('=' * 80)
                
                # Step 7: Save model
                model_dir = self.config.get('model', {}).get('path', 'models/best_fraud_detection_model.pkl')
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                
                joblib.dump(model, model_dir)
                logger.info(f'Model saved to {model_dir}')
                
                # Save model metadata
                metadata = {
                    'feature_names': X_train.columns.tolist(),
                    'n_features': len(X_train.columns),
                    'training_date': datetime.now().isoformat(),
                    'model_params': params,
                    'metrics': all_metrics,
                    'data_split': {
                        'train_samples': len(X_train),
                        'val_samples': len(X_val),
                        'test_samples': len(X_test)
                    }
                }
                
                metadata_path = model_dir.replace('.pkl', '_metadata.json')
                with open(metadata_path, 'w') as f:
                    import json
                    json.dump(metadata, f, indent=2)
                
                logger.info(f'Model metadata saved to {metadata_path}')
                
                # Log model to MLflow
                if MLFLOW_AVAILABLE:
                    mlflow.xgboost.log_model(
                        model,
                        'model',
                        registered_model_name='fraud_detection_xgboost'
                    )
                    mlflow.log_artifact(metadata_path)
                    run_id = mlflow.active_run().info.run_id
                    logger.info(f'MLflow Run ID: {run_id}')
                
                logger.info('=' * 80)
                logger.info('TRAINING PIPELINE COMPLETED SUCCESSFULLY')
                logger.info('=' * 80)
                
                return model, all_metrics
                
            finally:
                if MLFLOW_AVAILABLE:
                    mlflow.end_run()
                    
        except Exception as e:
            logger.error('Training failed: %s', str(e), exc_info=True)
            raise


def train_fraud_detection_model(config_path='config.yaml'):
    trainer = FraudDetectionTraining(config_path=config_path)
    model, metrics = trainer.train_model()
    
    logger.info('Weekly retraining completed with metrics: %s', metrics)
    return metrics

if __name__ == '__main__':
    # For standalone execution
    train_fraud_detection_model()
