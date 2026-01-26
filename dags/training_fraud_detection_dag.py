from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from training_fraud_detection import train_fraud_detection_model

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1
}

def train_model_task(**context):
    """
    Weekly model retraining task.
    
    This function:
    1. Loads the latest feature-engineered data
    2. Trains a new XGBoost model
    3. Evaluates performance
    4. Saves the model to the models directory
    5. Logs metrics to MLflow (if available)
    """
    try:
        logger.info("Starting weekly fraud detection model retraining")
        
        # Train model using the latest data
        metrics = train_fraud_detection_model(
            config_path='/app/config.yaml',
            data_path='/app/DATA/feature_engineering.csv'
        )
        
        logger.info(f"Model retraining completed successfully")
        logger.info(f"Performance metrics: {metrics}")
        
        # Push metrics to XCom for downstream tasks
        context['ti'].xcom_push(key='training_metrics', value=metrics)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        raise

# Define the DAG
with DAG(
    dag_id='fraud_detection_weekly_retrain',
    default_args=default_args,
    description="Weekly retraining of fraud detection model for real-time transactions",
    schedule='0 3 * * 0',  # Every Sunday at 3:00 AM
    catchup=False,
    tags=['fraud-detection', 'ml', 'weekly-retrain', 'production']
) as dag:
    
    # Task 1: Validate environment and data
    validate_environment = BashOperator(
        task_id='validate_environment',
        bash_command="""
        echo "Validating training environment..."
        test -f /app/config.yaml || (echo "Config file missing" && exit 1)
        test -f /app/DATA/feature_engineering.csv || (echo "Training data missing" && exit 1)
        echo "✓ Environment validation passed"
        """
    )
    
    # Task 2: Train the model (weekly)
    training_task = PythonOperator(
        task_id='train_fraud_detection_model',
        python_callable=train_model_task,
        provide_context=True,
        execution_timeout=timedelta(hours=2)
    )
    
    # Task 3: Validate model output
    validate_model = BashOperator(
        task_id='validate_model',
        bash_command="""
        echo "Validating trained model..."
        test -f /app/models/best_fraud_detection_model.pkl || (echo "Model file not created" && exit 1)
        test -f /app/models/best_fraud_detection_model_metadata.json || (echo "Metadata file not created" && exit 1)
        echo "✓ Model validation passed"
        """
    )
    
    # Task 4: Cleanup temporary files
    cleanup_task = BashOperator(
        task_id='cleanup_temporary_files',
        bash_command="""
        echo "Cleaning up temporary files..."
        rm -f /app/confusion_matrix.png
        rm -rf /app/cache
        echo "✓ Cleanup completed"
        """,
        trigger_rule='all_done'  # Run even if upstream tasks fail
    )
    
    # Task 5: Send notification (placeholder)
    notify_completion = BashOperator(
        task_id='notify_completion',
        bash_command="""
        echo "========================================="
        echo "Weekly Model Retraining Completed"
        echo "Timestamp: $(date)"
        echo "Model: /app/models/best_fraud_detection_model.pkl"
        echo "========================================="
        # TODO: Add email/Slack notification here
        """,
        trigger_rule='all_success'
    )
    
    # Define task dependencies
    validate_environment >> training_task >> validate_model >> notify_completion
    training_task >> cleanup_task
    
    # Documentation
    dag.doc_md = """
    ## Fraud Detection Weekly Retraining Pipeline
    
    **Schedule**: Every Sunday at 3:00 AM (weekly)
    
    **Purpose**: Automatically retrain the fraud detection model with the latest data
    to keep predictions accurate as transaction patterns evolve.
    
    ### Pipeline Steps:
    
    1. **Validate Environment** 
       - Checks config and data files exist
       - Ensures prerequisites are met
    
    2. **Train Model**
       - Loads feature-engineered data from CSV
       - Trains XGBoost classifier
       - Optimizes decision threshold
       - Evaluates performance metrics
       - Saves model and metadata
       - Logs to MLflow (if available)
    
    3. **Validate Model**
       - Confirms model file was created
       - Checks metadata file exists
    
    4. **Cleanup**
       - Removes temporary files
       - Cleans cache directory
    
    5. **Notify**
       - Logs completion status
       - Can be extended to send alerts
    
    ### Model Updates:
    
    - New model automatically replaces old model at: `/app/models/best_fraud_detection_model.pkl`
    - Inference service picks up new model on next restart
    - Model versioning tracked in MLflow
    
    ### Monitoring:
    
    - View training metrics in MLflow UI
    - Check Airflow logs for detailed execution info
    - Monitor model performance over time
    
    ### Configuration:
    
    Edit `config.yaml` to adjust:
    - Model hyperparameters
    - Training data path  
    - MLflow tracking URI
    - Model save location
    """