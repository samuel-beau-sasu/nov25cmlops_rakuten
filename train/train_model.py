import subprocess
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
import re

from mlops_rakuten.config.config_manager import ConfigurationManager

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
        #metrics["model_path"] = "/home/ubuntu/Projet_MLOps/nov25cmlops_rakuten/models/2026-01-16T15-07-46/text_classifier.pkl"
        metrics["model_path"] = str(find_newest_matching("**/text_classifier.pkl", base_path))
    if "metrics_val.json" in stderr:
        #metrics["metrics_path"] = "/home/ubuntu/Projet_MLOps/nov25cmlops_rakuten/reports/2026-01-16T15-08-10/metrics_val.json"
        metrics["metrics_path"] = str(find_newest_matching("**/metrics_val.json", base_path))
    
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