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
    Étape d'entraînement du modèle Rakuten.

    - Charge X_train et y_train transformés (TF-IDF + label encoding)
    - Instancie le modèle sklearn (par défaut LinearSVC)
    - Entraîne le modèle
    - Calcule quelques métriques sur le jeu d'entraînement
    - Sauvegarde :
        - le modèle entraîné (text_classifier.pkl)
        - un fichier JSON décrivant la configuration d'entraînement
        - un fichier JSON contenant les métriques d'entraînement
        - un rapport texte de classification sur le train
    """

    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config

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

        # 5. Créer le répertoire des modèles si besoin
        create_directories([cfg.model_dir])

        # 6. Sauvegarder le modèle
        logger.info(f"Sauvegarde du modèle vers : {cfg.model_path}")
        with open(cfg.model_path, "wb") as f:
            pickle.dump(model, f)

        # 7. Sauvegarder un fichier de configuration lisible
        model_config_path = cfg.model_dir / "model_config.json"
        logger.info(f"Sauvegarde de la configuration du modèle vers : {model_config_path}")

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
        }

        with open(model_config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        # 8. Sauvegarder les métriques d'entraînement
        metrics_path = cfg.model_dir / "metrics_train.json"
        logger.info(f"Sauvegarde des métriques d'entraînement vers : {metrics_path}")

        metrics = {
            "train_accuracy": train_accuracy,
            "train_f1_macro": train_f1_macro,
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # 9. Sauvegarder le rapport de classification texte
        cls_report_path = cfg.model_dir / "classification_report_train.txt"
        logger.info(f"Sauvegarde du rapport de classification (train) vers : {cls_report_path}")

        with open(cls_report_path, "w") as f:
            f.write(cls_report)

        logger.success("ModelTrainer terminé avec succès")
        return cfg.model_path

class ModelTrainer_mlflow:
    """
    Étape d'entraînement du modèle Rakuten avec intégration MLflow.

    - Charge X_train et y_train transformés (TF-IDF + label encoding)
    - Instancie le modèle sklearn (par défaut LinearSVC)
    - Entraîne le modèle
    - Calcule quelques métriques sur le jeu d'entraînement
    - Log tout dans MLflow (params, metrics, model, artifacts)
    - Sauvegarde locale également
    """

    def __init__(
        self, 
        config: ModelTrainerConfig,
        #enable_mlflow: bool = True
    ) -> None:
        self.config = config
        
        # Initialiser le client MLflow si activé
        if self.config.enable_mlflow:
            self.client = MlflowClient(tracking_uri=self.config.mlflow_tracking_uri)
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI: {self.config.mlflow_tracking_uri}")
            

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

        raise ValueError(
            f"Type de modèle non supporté : '{cfg.model_type}'. "
            "Actuellement seul 'linear_svc' est géré."
        )

    def _prepare_params(self) -> dict:
        """
        Prépare les paramètres à logger dans MLflow.
        """
        cfg = self.config
        params = {
            "model_type": cfg.model_type,
            "C": cfg.C,
            "max_iter": cfg.max_iter,
            "use_class_weight": cfg.use_class_weight,
        }
        return params

    def _prepare_metrics(self, train_accuracy: float, train_f1_macro: float) -> dict:
        """
        Prépare les métriques à logger dans MLflow.
        """
        metrics = {
            "train_accuracy": train_accuracy,
            "train_f1_macro": train_f1_macro,
        }
        return metrics

    def run(self) -> Path:
        """
        Entraîne le modèle avec tracking MLflow et retourne le chemin du fichier modèle sauvegardé.
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

        # 3. Préparer les paramètres pour MLflow
        params = self._prepare_params()

        # 4. Démarrer le run MLflow
        logger.debug(f"affiche la valeur de  enable_mlflow: {cfg.enable_mlflow}")
        if cfg.enable_mlflow:
            mlflow.set_experiment(cfg.mlflow_experiment_name)
            logger.info(f"MLflow experiment: {cfg.mlflow_experiment_name}")
        
        # Contexte MLflow ou contexte vide
        mlflow_context = (
            mlflow.start_run(run_name=cfg.mlflow_run_name) 
            if cfg.enable_mlflow 
            else self._null_context()
        )
        
        logger.debug(f"affiche la valeur de  enable_mlflow: {cfg.enable_mlflow}")
        with mlflow_context as run:
            if cfg.enable_mlflow:
                logger.info(f"MLflow run ID: {run.info.run_id}")
                logger.info(f"MLflow run name: {cfg.mlflow_run_name}")
                
                # Logger les paramètres
                mlflow.log_params(params)
                logger.info("Paramètres loggés dans MLflow")

            # 5. Entraîner le modèle
            logger.info("Entraînement du modèle")
            model.fit(X_train, y_train)
            logger.success("Modèle entraîné avec succès")

            # 6. Évaluer sur le jeu d'entraînement
            logger.info("Évaluation du modèle sur le jeu d'entraînement")
            y_pred_train = model.predict(X_train)

            train_accuracy = accuracy_score(y_train, y_pred_train)
            train_f1_macro = f1_score(y_train, y_pred_train, average="macro")

            logger.info(f"Train accuracy: {train_accuracy:.4f}")
            logger.info(f"Train F1 macro: {train_f1_macro:.4f}")

            # Rapport de classification détaillé
            cls_report = classification_report(y_train, y_pred_train)

            # Préparer les métriques
            metrics = self._prepare_metrics(train_accuracy, train_f1_macro)

            if cfg.enable_mlflow:
                # Logger les métriques
                mlflow.log_metrics(metrics)
                logger.info("Métriques loggées dans MLflow")

            # 7. Créer le répertoire des modèles si besoin
            create_directories([cfg.model_dir])

            # 8. Sauvegarder le modèle localement
            logger.info(f"Sauvegarde du modèle vers : {cfg.model_path}")
            with open(cfg.model_path, "wb") as f:
                pickle.dump(model, f)

            # 9. Logger le modèle dans MLflow
            if cfg.enable_mlflow:
                logger.info("Logging du modèle dans MLflow")
                
                # Créer un input_example (première ligne de X_train)
                # Note: MLflow préfère des arrays denses pour input_example
                if sparse.issparse(X_train):
                    input_example = X_train[:1].toarray()
                else:
                    input_example = X_train[:1]
                
                mlflow.sklearn.log_model(
                    sk_model=model,
                    #artifact_path=self.mlflow_artifact_path,
                    artifact_path=cfg.mlflow_artifact_path,
                    input_example=input_example,
                )
                logger.success(f"Modèle loggé dans MLflow sous '{cfg.mlflow_artifact_path}'")

            # 10. Sauvegarder un fichier de configuration lisible
            model_config_path = cfg.model_dir / "model_config.json"
            logger.info(f"Sauvegarde de la configuration du modèle vers : {model_config_path}")

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
            }

            with open(model_config_path, "w") as f:
                json.dump(model_config, f, indent=2)

            # 11. Sauvegarder les métriques d'entraînement
            metrics_path = cfg.model_dir / "metrics_train.json"
            logger.info(f"Sauvegarde des métriques d'entraînement vers : {metrics_path}")

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # 12. Sauvegarder le rapport de classification texte
            cls_report_path = cfg.model_dir / "classification_report_train.txt"
            logger.info(f"Sauvegarde du rapport de classification (train) vers : {cls_report_path}")

            with open(cls_report_path, "w") as f:
                f.write(cls_report)

            # 13. Logger les artifacts dans MLflow
            if cfg.enable_mlflow:
                logger.info("Logging des artifacts dans MLflow")
                mlflow.log_artifact(str(model_config_path))
                mlflow.log_artifact(str(metrics_path))
                mlflow.log_artifact(str(cls_report_path))
                logger.success("Artifacts loggés dans MLflow")

        logger.success("ModelTrainer terminé avec succès")
        return cfg.model_path

    @staticmethod
    def _null_context():
        """Context manager vide pour quand MLflow est désactivé."""
        from contextlib import nullcontext
        return nullcontext()
