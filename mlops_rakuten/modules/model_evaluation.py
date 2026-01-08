import json
from pathlib import Path
import pickle

from loguru import logger
import mlflow
import numpy as np
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from mlops_rakuten.config.entities import ModelEvaluationConfig
from mlops_rakuten.utils import create_directories


class ModelEvaluation:
    """
    Étape d'évaluation du modèle Rakuten sur le jeu de validation.

    - Charge X_val et y_val (TF-IDF + label encoding)
    - Charge le modèle entraîné
    - Calcule des métriques sur le jeu de validation
    - Sauvegarde :
        - un fichier JSON contenant les métriques de validation
        - un rapport texte de classification
        - une matrice de confusion (numpy)
    """

    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config

    def run(self) -> Path:
        """
        Évalue le modèle sur le jeu de validation et retourne
        le chemin du fichier de métriques JSON.
        """
        logger.info("Démarrage de l'étape ModelEvaluation")

        cfg = self.config

        # 1. Charger les données de validation
        logger.info(f"Chargement de X_val depuis : {cfg.X_val_path}")
        X_val = sparse.load_npz(cfg.X_val_path)

        logger.info(f"Chargement de y_val depuis : {cfg.y_val_path}")
        y_val = np.load(cfg.y_val_path)

        logger.debug(f"X_val shape: {X_val.shape}")
        logger.debug(f"y_val shape: {y_val.shape}")

        # 2. Charger le modèle
        logger.info(f"Chargement du modèle depuis : {cfg.model_path}")
        with open(cfg.model_path, "rb") as f:
            model = pickle.load(f)

        # 3. Prédictions sur le jeu de validation
        logger.info("Prédiction sur le jeu de validation")
        y_pred = model.predict(X_val)

        # 4. Calcul des métriques
        val_accuracy = accuracy_score(y_val, y_pred)
        val_f1_macro = f1_score(y_val, y_pred, average="macro")
        val_f1_weighted = f1_score(y_val, y_pred, average="weighted")

        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation F1 macro: {val_f1_macro:.4f}")
        logger.info(f"Validation F1 weighted: {val_f1_weighted:.4f}")

        cls_report = classification_report(y_val, y_pred)

        cm = confusion_matrix(y_val, y_pred)

        # ─────────────────────────────────────────────────────────────
        # LOG TO MLFLOW 
        # ─────────────────────────────────────────────────────────────
        logger.info("Logging validation metrics to MLflow...")
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_f1_macro", val_f1_macro)
        mlflow.log_metric("val_f1_weighted", val_f1_weighted)

        # 5. Create directories
        create_directories([cfg.metrics_dir])

        # 6. Save metrics
        metrics = {
            "val_accuracy": val_accuracy,
            "val_f1_macro": val_f1_macro,
            "val_f1_weighted": val_f1_weighted,
        }

        logger.info(f"Sauvegarde des métriques vers : {cfg.metrics_path}")
        with open(cfg.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(cfg.metrics_path), artifact_path="metrics")

        # 7. Save classification report
        logger.info(f"Sauvegarde du rapport vers : {cfg.classification_report_path}")
        with open(cfg.classification_report_path, "w") as f:
            f.write(cls_report)

        mlflow.log_artifact(str(cfg.classification_report_path), artifact_path="reports")

        # 8. Save confusion matrix
        logger.info(f"Sauvegarde de la matrice vers : {cfg.confusion_matrix_path}")
        with open(cfg.confusion_matrix_path, "w") as f:
            f.write(str(cm))

        mlflow.log_artifact(str(cfg.confusion_matrix_path), artifact_path="reports")

        logger.success("ModelEvaluation terminée avec succès")
        return cfg.metrics_path