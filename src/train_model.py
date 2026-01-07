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



def test_training():
    """Teste l'exécution de mlops_rakuten/main.py train"""
    logger.info("🚀 Test du pipeline de training")
    logger.info(f"📁 Répertoire courant: {os.getcwd()}")
    
    config_manager = ConfigurationManager()
    cfg = config_manager.get_model_trainer_config()
    
    logger.info( f"[DEBUG CONFIG] C={cfg.C}, max_iter={cfg.max_iter}, use_class_weight={cfg.use_class_weight}")
    
    # Exécute le script
    result = subprocess.run(
        [sys.executable, "mlops_rakuten/main.py", "train"],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        timeout=3600
    )
    
    # Affiche les résultats
    logger.info(f"📊 Code de retour: {result.returncode}")
    logger.info(f"📤 STDOUT ({len(result.stdout)} chars):\n{result.stdout}")
    logger.info(f"📥 STDERR ({len(result.stderr)} chars):\n{result.stderr}")
    
    metrics = parse_training_output(result.stderr)  # Pas result.stdout !
    logger.info(f"📊 MÉTRIQUES: {metrics}")
    
    
    if result.returncode == 0:
        logger.info("✅ Training réussi !")
        
    else:
        logger.error("❌ Training échoué")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(test_training())