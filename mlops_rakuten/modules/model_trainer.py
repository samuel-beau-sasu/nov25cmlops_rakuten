import json
from pathlib import Path
import pickle
import yaml

from loguru import logger
import numpy as np
from scipy import sparse
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

from mlops_rakuten.config.entities import ModelTrainerConfig
from mlops_rakuten.utils import create_directories


class ModelTrainer:
    """
    Étape d'entraînement du modèle Rakuten.

    - Charge X_train et y_train transformés (TF-IDF + label encoding)
    - Instancie le modèle sklearn (par défaut LinearSVC)
    - Entraîne le modèle
    - Calcule quelques métriques sur le jeu d'entraînement
    - **LOGS DVC LINEAGE** : trace data hashes → model hash
    - Sauvegarde :
        - le modèle entraîné (text_classifier.pkl)
        - un fichier JSON décrivant la configuration d'entraînement
        - un fichier JSON contenant les métriques d'entraînement
        - un rapport texte de classification sur le train
    """

    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config

    def _get_dvc_lineage(self) -> dict:
        """
        Récupère les hashes DVC depuis dvc.lock pour tracer data → model.
        """
        dvc_lock_path = Path("dvc.lock")
        
        if not dvc_lock_path.exists():
            logger.warning("dvc.lock not found! Data lineage unavailable.")
            return {
                "data_input_hash": None,
                "preprocessed_data_hash": None,
                "train_features_hash": None,
                "status": "dvc_lock_not_found"
            }
        
        try:
            with open(dvc_lock_path, "r") as f:
                lock_data = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to read dvc.lock: {e}")
            return {"status": "dvc_lock_read_error"}
        
        stages = lock_data.get("stages", {})
        
        lineage = {
            "data_input_hash": None,
            "preprocessed_data_hash": None,
            "train_features_hash": None,
        }
        
        # Data input (rakuten_train_current.csv from preprocess deps)
        preprocess_stage = stages.get("preprocess", {})
        if preprocess_stage:
            for dep in preprocess_stage.get("deps", []):
                if "rakuten_train_current.csv" in dep.get("path", ""):
                    lineage["data_input_hash"] = dep.get("md5")
        
        # Preprocessed data (preprocessed_dataset.csv from preprocess outs)
        if preprocess_stage:
            for out in preprocess_stage.get("outs", []):
                if "preprocessed_dataset.csv" in out.get("path", ""):
                    lineage["preprocessed_data_hash"] = out.get("md5")
        
        # Train features (X_train_tfidf.npz from transform outs)
        transform_stage = stages.get("transform", {})
        if transform_stage:
            for out in transform_stage.get("outs", []):
                if "X_train_tfidf.npz" in out.get("path", ""):
                    lineage["train_features_hash"] = out.get("md5")
        
        logger.info(f"DVC lineage extracted: {lineage}")
        return lineage

    def _build_model(self):
        """
        Construit le modèle sklearn en fonction de la configuration.
        """
        cfg = self.config

        if cfg.model_type == "linear_svc":
            class_weight = "balanced" if cfg.use_class_weight else None
            logger.info(
                f"Instanciation d'un LinearSVC (C={cfg.C}, "
                f"max_iter={cfg.max_iter}, class_weight={class_weight})"
            )
            return LinearSVC(
                C=cfg.C,
                max_iter=cfg.max_iter,
                class_weight=class_weight,
            )

        if cfg.model_type == "logistic_regression":
            class_weight = "balanced" if cfg.use_class_weight else None
            logger.info(
                "Instanciation d'une LogisticRegression "
                f"(C={cfg.C}, max_iter={cfg.max_iter}, "
                f"class_weight={class_weight}, multi_class='multinomial')"
            )
            return LogisticRegression(
                C=cfg.C,
                max_iter=cfg.max_iter,
                class_weight=class_weight,
                n_jobs=-1,
            )

        raise ValueError(
            f"Type de modèle non supporté : '{cfg.model_type}'. "
            "Actuellement 'linear_svc' et 'logistic_regression' sont gérés."
        )

    def run(self) -> Path:
        """
        Entraîne le modèle et retourne le chemin du fichier modèle sauvegardé.
        """
        logger.info("Démarrage de l'étape ModelTrainer")

        cfg = self.config

        # 1. Charger les données d'entraînement
        logger.info(f"Chargement de X_train depuis : {cfg.X_train_path}")
        X_train = sparse.load_npz(cfg.X_train_path)

        logger.info(f"Chargement de y_train depuis : {cfg.y_train_path}")
        y_train = np.load(cfg.y_train_path)

        logger.debug(f"X_train shape: {X_train.shape}")
        logger.debug(f"y_train shape: {y_train.shape}")

        # 2. Construire le modèle
        model = self._build_model()

        # 3. Entraîner le modèle
        logger.info("Entraînement du modèle")
        model.fit(X_train, y_train)
        logger.success("Modèle entraîné avec succès")

        # 4. Évaluer sur le jeu d'entraînement (baseline)
        logger.info("Évaluation du modèle sur le jeu d'entraînement")
        y_pred_train = model.predict(X_train)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_f1_macro = f1_score(y_train, y_pred_train, average="macro")

        logger.info(f"Train accuracy: {train_accuracy:.4f}")
        logger.info(f"Train F1 macro: {train_f1_macro:.4f}")

        # Rapport de classification détaillé
        cls_report = classification_report(y_train, y_pred_train)

        # ─────────────────────────────────────────────────────────────
        # LOG TRAINING METRICS TO MLFLOW
        # ─────────────────────────────────────────────────────────────
        logger.info("Logging training metrics to MLflow...")
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_f1_macro", train_f1_macro)

        # ─────────────────────────────────────────────────────────────
        # 🔗 LOG DVC DATA LINEAGE TO MLFLOW (traçabilité data → model)
        # ─────────────────────────────────────────────────────────────
        logger.info("Extracting DVC data lineage...")
        lineage = self._get_dvc_lineage()
        
        # Log les hashes comme params (pour filtrer/chercher dans MLflow)
        if lineage.get("data_input_hash"):
            mlflow.log_param("dvc_data_input_hash", lineage["data_input_hash"])
            logger.info(f"✓ DVC data_input_hash: {lineage['data_input_hash']}")
        
        if lineage.get("train_features_hash"):
            mlflow.log_param("dvc_train_features_hash", lineage["train_features_hash"])
            logger.info(f"✓ DVC train_features_hash: {lineage['train_features_hash']}")
        
        # Log la lignée complète en artifact JSON
        lineage_artifact_path = cfg.model_dir / "dvc_lineage.json"
        with open(lineage_artifact_path, "w") as f:
            json.dump(lineage, f, indent=2)
        mlflow.log_artifact(str(lineage_artifact_path), artifact_path="lineage")
        logger.info(f"✓ DVC lineage logged to MLflow artifact: {lineage_artifact_path}")

        # 5. Save model
        create_directories([cfg.model_dir])

        logger.info(f"Sauvegarde du modèle vers : {cfg.model_path}")
        with open(cfg.model_path, "wb") as f:
            pickle.dump(model, f)

        # LOG MODEL ARTIFACT
        logger.info("Logging model artifact to MLflow...")
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="rakuten_classifier"
        )

        # 6. Save config
        model_config_path = cfg.model_dir / "model_config.json"
        logger.info(f"Sauvegarde de la configuration vers : {model_config_path}")

        model_config = {
            "model_type": cfg.model_type,
            "params": {
                "C": cfg.C,
                "max_iter": cfg.max_iter,
                "use_class_weight": cfg.use_class_weight,
            },
            "training_data": {
                "X_train_path": str(cfg.X_train_path),
                "y_train_path": str(cfg.y_train_path),
                "X_train_rows": X_train.shape[0],
            },
            "dvc_lineage": lineage,  # ← Inclure la traçabilité dans la config
        }

        with open(model_config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        mlflow.log_artifact(str(model_config_path), artifact_path="config")

        # 7. Save train metrics
        metrics_path = cfg.model_dir / "metrics_train.json"
        logger.info(f"Sauvegarde des métriques vers : {metrics_path}")

        metrics = {
            "train_accuracy": train_accuracy,
            "train_f1_macro": train_f1_macro,
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        # 8. Save train report
        cls_report_path = cfg.model_dir / "classification_report_train.txt"
        logger.info(f"Sauvegarde du rapport vers : {cls_report_path}")

        with open(cls_report_path, "w") as f:
            f.write(cls_report)

        mlflow.log_artifact(str(cls_report_path), artifact_path="reports")

        logger.success("ModelTrainer terminé avec succès")
        logger.success(
            f"Data lineage: {lineage['data_input_hash'][:8] if lineage.get('data_input_hash') else 'N/A'} "
            f"→ Features: {lineage['train_features_hash'][:8] if lineage.get('train_features_hash') else 'N/A'}"
        )
        return cfg.model_path