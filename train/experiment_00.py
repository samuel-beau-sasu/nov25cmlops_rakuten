import mlflow
from mlflow import MlflowClient
import subprocess
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
import re

from mlops_rakuten.config.config_manager import ConfigurationManager
from sklearn.svm import LinearSVC


# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_newest_matching(pattern: str, root_dir: Path) -> Path | None:
    """Trouve le fichier le plus récent correspondant au pattern (récursif)."""
    try:
        fichiers = list(root_dir.rglob(pattern))
        return max(                             # max() trouve la valeur maximale
            fichiers,            # 1. Liste des fichiers correspondants
            key=lambda f: f.stat().st_mtime     # 2. Critère de comparaison
        )
    except ValueError:  # max() sur séquence vide
        return None


def parse_training_output(stderr: str) -> dict:
    """Extrait les métriques du STDERR (logs)"""
    metrics = {"accuracy_train": 0.0, "f1_train": 0.0, "accuracy_val": 0.0, "f1_val": 0.0}
    
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
    # Chemins des fichiers
    if "text_classifier.pkl" in stderr:
        #metrics["model_path"] = "/home/ubuntu/Projet_MLOps/nov25cmlops_rakuten/models/2026-01-16T15-07-46/text_classifier.pkl"
        metrics["model_path"] = str(find_newest_matching("**/text_classifier.pkl", base_path))
    if "metrics_val.json" in stderr:
        #metrics["metrics_path"] = "/home/ubuntu/Projet_MLOps/nov25cmlops_rakuten/reports/2026-01-16T15-08-10/metrics_val.json"
        metrics["metrics_path"] = str(find_newest_matching("**/metrics_val.json", base_path))
    
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
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
    
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
    
    
    model_SVC = LinearSVC(C=1.0, max_iter=1000)
    # Store information in tracking server
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model_SVC, 
            #input_example=X_val, 
            artifact_path=artifact_path
        )
    
    
if __name__ == "__main__":
    test_training()