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
    try:
        logger.info("Starting monthly fraud detection model retraining")
        logger.info("Loading data from PostgreSQL database...")
        
        # Train model using data from PostgreSQL
        metrics = train_fraud_detection_model(
            config_path='config.yaml'
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
    dag_id='fraud_detection_monthly_retrain',
    default_args=default_args,
    description="Monthly retraining of fraud detection model for near real-time transactions",
    schedule='0 3 1 * *',  # First day of every month at 3:00 AM
    catchup=False,
    tags=['fraud-detection', 'ml', 'monthly-retrain', 'production']
) as dag:
    
    # Task 1: Validate environment and data
    validate_environment = BashOperator(
        task_id='validate_environment',
        bash_command="""
        echo "Validating training environment..."
        test -f config.yaml || (echo "Config file missing" && exit 1)
        echo "PostgreSQL connection will be validated during training"
        echo "✓ Environment validation passed"
        """
    )
    
    # Task 2: Train the model (monthly)
    training_task = PythonOperator(
        task_id='train_fraud_detection_model',
        python_callable=train_model_task,
        provide_context=True,
        execution_timeout=timedelta(hours=3)
    )
    
    # Task 3: Validate model output
    validate_model = BashOperator(
        task_id='validate_model',
        bash_command="""
        echo "Validating trained model..."
        test -f models/best_fraud_detection_model.pkl || (echo "Model file not created" && exit 1)
        test -f models/best_fraud_detection_model_metadata.json || (echo "Metadata file not created" && exit 1)
        echo "Model validation passed"
        """
    )
    
    # Task 4: Cleanup temporary files
    cleanup_task = BashOperator(
        task_id='cleanup_temporary_files',
        bash_command="""
        echo "Cleaning up temporary files..."
        rm -f confusion_matrix.png
        rm -rf cache
        echo "Cleanup completed"
        """,
        trigger_rule='all_done'  # Run even if upstream tasks fail
    )
    
    # Task 5: Send notification (placeholder)
    notify_completion = BashOperator(
        task_id='notify_completion',
        bash_command="""
        echo "========================================="
        echo "Monthly Model Retraining Completed"
        echo "Timestamp: $(date)"
        echo "Model: models/best_fraud_detection_model.pkl"
        echo "========================================="
        """,
        trigger_rule='all_success'
    )
    
    # Define task dependencies
    validate_environment >> training_task >> validate_model >> notify_completion
    training_task >> cleanup_task
    
    # Documentation
    dag.doc_md = """
    ## Fraud Detection Monthly Retraining Pipeline
    
    **Schedule**: First day of every month at 3:00 AM (monthly)
    
    **Purpose**: Automatically retrain the fraud detection model with the latest data
    to keep predictions accurate as transaction patterns evolve.
    
    ### Pipeline Steps:
    
    1. **Validate Environment** 
       - Checks config and data files exist
       - Ensures prerequisites are met
    
    2. **Train Model**
       - Loads feature-engineered data from PostgreSQL database
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
    
    - New model automatically replaces old model at: `models/best_fraud_detection_model.pkl`
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