import mlflow
from mlflow import MlflowClient
import subprocess
import sys
import os
import logging
from datetime import datetime

from mlops_rakuten.config_manager import ConfigurationManager

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_training_output(stderr: str) -> dict:
    """Extrait les métriques du STDERR (logs)"""
    metrics = {"accuracy_train": None, "f1_train": None, "accuracy_val": None, "f1_val": None}
    
    for line in stderr.split('\n'):
        if "Train accuracy:" in line:
            metrics["accuracy_train"] = float(line.split("Train accuracy:")[-1].strip())
        elif "Train F1 macro:" in line:
            metrics["f1_train"] = float(line.split("Train F1 macro:")[-1].strip())
        elif "Validation accuracy:" in line:
            metrics["accuracy_val"] = float(line.split("Validation accuracy:")[-1].strip())
        elif "Validation F1 macro:" in line:
            metrics["f1_val"] = float(line.split("Validation F1 macro:")[-1].strip())
    
    # Chemins des fichiers
    if "text_classifier.pkl" in stderr:
        metrics["model_path"] = "/home/ubuntu/nov25cmlops_rakuten/models/text_classifier.pkl"
    if "metrics_val.json" in stderr:
        metrics["metrics_path"] = "/home/ubuntu/nov25cmlops_rakuten/reports/metrics_val.json"
    
    return metrics

def param_output():
    
    config_manager = ConfigurationManager()
    cfg = config_manager.get_model_trainer_config()
    
    param = {"C": cfg.C, "max_iter": cfg.max_iter, "use_class_weight": cfg.use_class_weight}
    
    return param 

def test_training():
    """Teste l'exécution de mlops_rakuten/main.py train"""
    logger.info("🚀 Test du pipeline de training")
    logger.info(f"📁 Répertoire courant: {os.getcwd()}")
    
    # Define tracking_uri (localhost)
    client = MlflowClient(tracking_uri="http://127.0.0.1:5001")
    
    # Define experiment name, run name and artifact_path name
    rakuten_experiment = mlflow.set_experiment("Rakuten_Models")
    run_name = "first_run"
    artifact_path = "SVC_rakuten"
        
    # Exécute le script
    result = subprocess.run(
        [sys.executable, "mlops_rakuten/main.py", "train"],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        timeout=3600
    )
    
    # Parametres du model
    params = param_output()
    logger.info( f"Parametres : {params}")

    # Métriques du model
    metrics_dict = parse_training_output(result.stderr)
    
    metrics_keys = ['accuracy_train', 'f1_train', 'accuracy_val', 'f1_val']
    metrics = {k: metrics_dict[k] for k in metrics_keys if k in metrics_dict}
    
    logger.info(f"📊 MÉTRIQUES: {metrics}")
    
    # Store information in tracking server
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model_SVC, 
            input_example=X_val, 
            artifact_path=artifact_path
        )
    
    
if __name__ == "__main__":
    test_training()