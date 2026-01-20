import json
from pathlib import Path
import pickle

from loguru import logger
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from mlops_rakuten.config.entities import ModelTrainerConfig
from mlops_rakuten.utils import create_directories



class ModelTrainer:
    """
    Une seule classe avec option MLflow via enable_mlflow.
    """
    
    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config
        logger.info(f"Initialisation ModelTrainer avec enable_mlflow={config.enable_mlflow}")
        
        if config.enable_mlflow:
            logger.info(f"Configuration MLflow - tracking_uri: {config.mlflow_tracking_uri}")
            self.client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    
    def _build_model(self):
        """Construit le modèle sklearn."""
        cfg = self.config
        
        # DEBUG important
        logger.info(f"Construction du modèle - type: '{cfg.model_type}'")
        logger.info(f"Paramètres - C: {cfg.C}, max_iter: {cfg.max_iter}, use_class_weight: {cfg.use_class_weight}")
        
        if cfg.model_type == "linear_svc":
            class_weight = "balanced" if cfg.use_class_weight else None
            logger.info(f"Instanciation d'un LinearSVC")
            return LinearSVC(
                C=cfg.C,
                max_iter=cfg.max_iter,
                class_weight=class_weight,
                random_state=42,
            )
        
        if cfg.model_type == "logistic_regression":
            class_weight = "balanced" if cfg.use_class_weight else None
            logger.info(f"Instanciation d'une LogisticRegression")
            return LogisticRegression(
                C=cfg.C,
                max_iter=cfg.max_iter,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=42,
            )
        
        # ERREUR - on ne devrait jamais arriver ici
        error_msg = f"Type de modèle non supporté : '{cfg.model_type}'. Attendu: 'linear_svc' ou 'logistic_regression'"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    def run(self) -> Path:
        """
        Entraîne le modèle avec le logging configuré.
        """
        logger.info("Démarrage de l'entraînement")
        cfg = self.config
        
        # Démarrer MLflow si activé
        if cfg.enable_mlflow:
            logger.info(f"MLflow - experiment: {cfg.mlflow_experiment_name}, run: {cfg.mlflow_run_name}")
            mlflow.set_experiment(cfg.mlflow_experiment_name)
            mlflow.start_run(run_name=cfg.mlflow_run_name)
        
        try:
            # 1. Charger les données
            logger.info(f"Chargement X_train: {cfg.X_train_path}")
            X_train = sparse.load_npz(cfg.X_train_path)
            
            logger.info(f"Chargement y_train: {cfg.y_train_path}")
            y_train = np.load(cfg.y_train_path)
            
            logger.info(f"Forme X_train: {X_train.shape}, y_train: {y_train.shape}")
            
            # 2. Construire le modèle (VÉRIFIEZ ICI !)
            logger.info("Construction du modèle...")
            model = self._build_model()
            
            if model is None:
                raise ValueError("Le modèle n'a pas été construit (model=None)")
            
            logger.info(f"Modèle construit: {type(model).__name__}")
            
            # 3. Entraîner
            logger.info("Début de l'entraînement...")
            model.fit(X_train, y_train)
            logger.success("Modèle entraîné avec succès")
            
            # 4. Évaluer
            logger.info("Évaluation sur le jeu d'entraînement...")
            y_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_pred)
            train_f1_macro = f1_score(y_train, y_pred, average="macro")
            
            logger.info(f"Accuracy (train): {train_accuracy:.4f}")
            logger.info(f"F1 Macro (train): {train_f1_macro:.4f}")
            
            # 5. Logging MLflow conditionnel
            if cfg.enable_mlflow:
                logger.info("Logging dans MLflow...")
                mlflow.log_params({
                    "model_type": cfg.model_type,
                    "C": cfg.C,
                    "max_iter": cfg.max_iter,
                    "use_class_weight": cfg.use_class_weight,
                })
                mlflow.log_metrics({
                    "train_accuracy": train_accuracy,
                    "train_f1_macro": train_f1_macro,
                })
                
                # Log du modèle avec input_example
                if sparse.issparse(X_train):
                    input_example = X_train[:1].toarray()
                else:
                    input_example = X_train[:1]
                
                mlflow.sklearn.log_model(
                    model, 
                    cfg.mlflow_artifact_path,
                    input_example=input_example
                )
                logger.success(f"Modèle loggé dans MLflow: {cfg.mlflow_artifact_path}")
            
            # 6. Sauvegarder localement (toujours)
            create_directories([cfg.model_dir])
            logger.info(f"Sauvegarde locale: {cfg.model_path}")
            
            with open(cfg.model_path, "wb") as f:
                pickle.dump(model, f)
            
            # 7. Sauvegarder métriques et rapport
            self._save_additional_files(model, X_train, y_train, y_pred)
            
            logger.success(f"✓ Modèle sauvegardé: {cfg.model_path}")
            return cfg.model_path
            
        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement: {e}")
            raise
            
        finally:
            if cfg.enable_mlflow:
                mlflow.end_run()
                logger.info("Run MLflow terminé")
    
    def _save_additional_files(self, model, X_train, y_train, y_pred):
        """Sauvegarde les fichiers supplémentaires."""
        cfg = self.config
        
        # Sauvegarder la configuration
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
            },
            "model_info": {
                "type": type(model).__name__,
                "n_features": X_train.shape[1],
                "n_classes": len(np.unique(y_train)),
            }
        }
        
        with open(cfg.model_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)
        
        # Sauvegarder les métriques
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred),
            "train_f1_macro": f1_score(y_train, y_pred, average="macro"),
        }
        
        with open(cfg.model_dir / "metrics_train.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Sauvegarder le rapport de classification
        cls_report = classification_report(y_train, y_pred)
        with open(cfg.model_dir / "classification_report_train.txt", "w") as f:
            f.write(cls_report)
        
        logger.info("Fichiers supplémentaires sauvegardés")