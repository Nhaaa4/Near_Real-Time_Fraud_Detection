import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import joblib
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
    """
    Real-time fraud detection training system for transaction data.

    Components:
    - Data Loading: CSV-based feature-engineered data
    - Model Training: XGBoost optimized for real-time inference
    - Evaluation: Comprehensive metrics for fraud detection
    - MLflow Integration: Optional experiment tracking
    - Model Persistence: Serialization for deployment

    Designed for compatibility with Spark streaming inference.
    """

    def __init__(self, config_path='config.yaml', data_path='DATA/feature_engineering.csv'):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Path to YAML configuration file
            data_path: Path to feature-engineered CSV data
        """
        # Load environment variables
        load_dotenv(dotenv_path='.env')

        # Load configuration
        self.config = self._load_config(config_path)
        self.data_path = data_path

        # Setup MLflow if available
        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
        
        logger.info("FraudDetectionTraining initialized")

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info('Configuration loaded successfully from %s', config_path)
            return config
        except Exception as e:
            logger.error('Failed to load configuration: %s', str(e))
            raise

    def _setup_mlflow(self):
        """Setup MLflow experiment tracking if available."""
        try:
            tracking_uri = self.config.get('mlflow', {}).get('tracking_uri', 'file:./mlruns')
            experiment_name = self.config.get('mlflow', {}).get('experiment_name', 'fraud_detection')
            
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info('MLflow configured: %s / %s', tracking_uri, experiment_name)
        except Exception as e:
            logger.warning('MLflow setup failed: %s. Continuing without tracking.', str(e))

    def load_data(self) -> pd.DataFrame:
        """
        Load feature-engineered data from CSV.
        
        Returns:
            DataFrame with all engineered features and target variable
        """
        try:
            logger.info('Loading data from %s', self.data_path)
            df = pd.read_csv(self.data_path)
            
            if df.empty:
                raise ValueError('Loaded data is empty')
            
            if 'is_fraud' not in df.columns:
                raise ValueError('Target column "is_fraud" not found in data')
            
            # Basic data quality checks
            fraud_rate = df['is_fraud'].mean() * 100
            logger.info('Data loaded: %d samples, %.2f%% fraud rate', len(df), fraud_rate)
            logger.info('Features: %d', len(df.columns) - 1)
            
            return df
        except Exception as e:
            logger.error('Failed to load data: %s', str(e))
            raise

    def train_model(self):
        """
        Complete training pipeline for real-time fraud detection:
        
        1. Load feature-engineered data
        2. Train/test split with stratification
        3. Train XGBoost model
        4. Optimize decision threshold
        5. Evaluate performance
        6. Save model artifacts
        7. Track with MLflow (if available)
        
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        try:
            logger.info('Starting model training process')
            
            # Load data
            df = self.load_data()
            
            # Prepare features and target
            y = df['is_fraud']
            X = df.drop(columns=['is_fraud'])
            
            # Data quality checks
            if y.sum() == 0:
                raise ValueError('No fraud samples in training data')
            if y.sum() < 10:
                logger.warning('Low fraud samples: %d. Model may not generalize well', y.sum())
            
            # Train/test split
            test_size = self.config.get('model', {}).get('test_size', 0.2)
            random_state = self.config.get('model', {}).get('seed', 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                stratify=y,
                random_state=random_state
            )
            
            logger.info('Data split: %d train, %d test', len(X_train), len(X_test))
            
            # Start MLflow run if available
            if MLFLOW_AVAILABLE:
                mlflow.start_run(run_name=f"weekly_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            try:
                # Log dataset info
                if MLFLOW_AVAILABLE:
                    mlflow.log_params({
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'n_features': X_train.shape[1],
                        'fraud_rate_train': float(y_train.mean()),
                        'fraud_rate_test': float(y_test.mean())
                    })
                
                # Model configuration
                model_params = self.config.get('model', {}).get('params', {})
                
                # Calculate scale_pos_weight for class imbalance
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                
                xgb_params = {
                    'n_estimators': model_params.get('n_estimators', 300),
                    'max_depth': model_params.get('max_depth', 12),
                    'learning_rate': model_params.get('learning_rate', 0.05),
                    'subsample': model_params.get('subsample', 0.8),
                    'colsample_bytree': model_params.get('colsample_bytree', 0.8),
                    'scale_pos_weight': scale_pos_weight,
                    'reg_lambda': 3.6,
                    'reg_alpha': 3.6,
                    'eval_metric': 'aucpr',
                    'tree_method': model_params.get('tree_method', 'hist'),
                    'random_state': random_state,
                    'n_jobs': -1
                }
                
                # Log hyperparameters
                if MLFLOW_AVAILABLE:
                    mlflow.log_params(xgb_params)
                
                # Train model
                logger.info('Training XGBoost model...')
                model = XGBClassifier(**xgb_params)
                model.fit(X_train, y_train, 
                         eval_set=[(X_test, y_test)],
                         verbose=False)
                
                logger.info('Model training completed')
                
                # Predictions
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Optimize threshold on test set
                precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, y_pred_proba)
                f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]
                
                logger.info('Optimal threshold: %.4f', optimal_threshold)
                
                # Predictions with optimal threshold
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                
                # Calculate metrics
                metrics = {
                    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                    'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
                    'auc_pr': float(average_precision_score(y_test, y_pred_proba)),
                    'auc_roc': float(roc_auc_score(y_test, y_pred_proba)),
                    'optimal_threshold': float(optimal_threshold)
                }
                
                # Log metrics
                logger.info('Model Metrics: %s', metrics)
                if MLFLOW_AVAILABLE:
                    mlflow.log_metrics(metrics)
                
                # Create and save confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix - Weekly Retrain')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Legitimate', 'Fraud'])
                plt.yticks(tick_marks, ['Legitimate', 'Fraud'])
                
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, format(cm[i, j], 'd'),
                                ha='center', va='center',
                                color='white' if cm[i, j] > cm.max() / 2 else 'black')
                
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                
                cm_path = 'confusion_matrix.png'
                plt.savefig(cm_path)
                plt.close()
                
                if MLFLOW_AVAILABLE:
                    mlflow.log_artifact(cm_path)
                
                # Save model
                model_dir = self.config.get('model', {}).get('path', 'models/best_fraud_detection_model.pkl')
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                
                joblib.dump(model, model_dir)
                logger.info('Model saved to %s', model_dir)
                
                # Save feature names and threshold
                metadata = {
                    'feature_names': X_train.columns.tolist(),
                    'n_features': len(X_train.columns),
                    'optimal_threshold': optimal_threshold,
                    'training_date': datetime.now().isoformat(),
                    'model_params': xgb_params,
                    'metrics': metrics
                }
                
                metadata_path = model_dir.replace('.pkl', '_metadata.json')
                with open(metadata_path, 'w') as f:
                    import json
                    json.dump(metadata, f, indent=2)
                
                logger.info('Model metadata saved to %s', metadata_path)
                
                # Log model to MLflow
                if MLFLOW_AVAILABLE:
                    mlflow.xgboost.log_model(
                        model,
                        'model',
                        registered_model_name=self.config.get('mlflow', {}).get('registered_model_name', 'fraud_detection_xgboost')
                    )
                    mlflow.log_artifact(metadata_path)
                
                logger.info('Training pipeline completed successfully')
                
                return model, metrics
                
            finally:
                if MLFLOW_AVAILABLE:
                    mlflow.end_run()
                    
        except Exception as e:
            logger.error('Training failed: %s', str(e), exc_info=True)
            raise


def train_fraud_detection_model(config_path='config.yaml', data_path='DATA/feature_engineering.csv'):
    """
    Entry point for Airflow DAG task.
    
    This function is called by the weekly retraining DAG to update the fraud detection model.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to feature-engineered data
        
    Returns:
        Dictionary with training metrics
    """
    trainer = FraudDetectionTraining(config_path=config_path, data_path=data_path)
    model, metrics = trainer.train_model()
    
    logger.info('Weekly retraining completed with metrics: %s', metrics)
    return metrics


if __name__ == '__main__':
    # For standalone execution
    train_fraud_detection_model()