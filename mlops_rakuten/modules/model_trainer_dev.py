import json
from pathlib import Path
import pickle
from scipy import sparse
import numpy as np

from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from mlops_rakuten.config.entities import ModelTrainerConfig
from mlops_rakuten.utils import create_directories
from mlops_rakuten.utils.logging.factory import create_logger

from pathlib import Path
from typing import Optional
from .mlflow_logger import MLflowLogger
from .local_logger import LocalLogger
from .base_logger import BaseLogger

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from .base_logger import BaseLogger

import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Any, Dict
from .base_logger import BaseLogger

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

class BaseLogger(ABC):
    """Interface pour tous les systèmes de logging."""
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log des paramètres."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log des métriques."""
        pass
    
    @abstractmethod
    def log_model(self, model: Any, artifact_path: str, input_example=None) -> None:
        """Log d'un modèle."""
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: Path) -> None:
        """Log d'un fichier artifact."""
        pass
    
    @abstractmethod
    def start_run(self, run_name: str = None) -> Any:
        """Démarre un run."""
        pass
    
    @abstractmethod
    def end_run(self) -> None:
        """Termine le run."""
        pass

class MLflowLogger(BaseLogger):
    """Logger qui utilise MLflow."""
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = "default"):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.active_run = None
    
    def start_run(self, run_name: str = None):
        self.active_run = mlflow.start_run(run_name=run_name)
        return self.active_run
    
    def end_run(self):
        if self.active_run:
            mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(metrics)
    
    def log_model(self, model: Any, artifact_path: str, input_example=None) -> None:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            input_example=input_example
        )
    
    def log_artifact(self, local_path: Path) -> None:
        mlflow.log_artifact(str(local_path))

class LocalLogger(BaseLogger):
    """Logger qui sauvegarde localement (pour quand MLflow est désactivé)."""
    
    def __init__(self, base_dir: Path = Path("logs")):
        self.base_dir = base_dir
        self.run_dir = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def start_run(self, run_name: str = None) -> Dict:
        run_name = run_name or f"run_{self.run_id}"
        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_info = {
            "run_id": self.run_id,
            "run_name": run_name,
            "start_time": datetime.now().isoformat()
        }
        
        # Sauvegarder les infos du run
        with open(self.run_dir / "run_info.json", "w") as f:
            json.dump(self.run_info, f, indent=2)
        
        return self.run_info
    
    def end_run(self) -> None:
        if self.run_dir:
            self.run_info["end_time"] = datetime.now().isoformat()
            with open(self.run_dir / "run_info.json", "w") as f:
                json.dump(self.run_info, f, indent=2)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.run_dir:
            return
        params_path = self.run_dir / "params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if not self.run_dir:
            return
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    
    def log_model(self, model: Any, artifact_path: str, input_example=None) -> None:
        if not self.run_dir:
            return
        # Créer le sous-dossier pour l'artifact
        artifact_dir = self.run_dir / artifact_path
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle
        model_path = artifact_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Sauvegarder l'input example si fourni
        if input_example is not None:
            import numpy as np
            example_path = artifact_dir / "input_example.npy"
            np.save(example_path, input_example)
    
    def log_artifact(self, local_path: Path) -> None:
        if not self.run_dir or not local_path.exists():
            return
        
        # Copier le fichier dans le dossier du run
        dest_path = self.run_dir / local_path.name
        import shutil
        shutil.copy2(local_path, dest_path)

def create_logger(
    enable_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "default",
    local_logs_dir: Path = Path("logs")
    ) -> BaseLogger:
    """Factory pour créer le logger approprié."""
    if enable_mlflow:
        return MLflowLogger(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name
        )
    else:
        return LocalLogger(base_dir=local_logs_dir)


class ModelTrainer:
    """
    Classe unique pour l'entraînement avec logging configurable.
    Supporte MLflow ou logging local via composition.
    """
    
    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config
        
        # Créer le logger approprié (composition)
        self.logger = create_logger(
            enable_mlflow=config.enable_mlflow,
            tracking_uri=config.mlflow_tracking_uri,
            experiment_name=config.mlflow_experiment_name,
            local_logs_dir=config.local_logs_dir  # Ajouter à la config si nécessaire
        )
        
        logger.info(f"Logger initialisé: {type(self.logger).__name__}")
    
    def _build_model(self):
        """Construit le modèle sklearn (inchangé)."""
        cfg = self.config
        
        if cfg.model_type == "linear_svc":
            class_weight = "balanced" if cfg.use_class_weight else None
            logger.info(f"Instanciation d'un LinearSVC (C={cfg.C}, max_iter={cfg.max_iter})")
            return LinearSVC(
                C=cfg.C,
                max_iter=cfg.max_iter,
                class_weight=class_weight,
            )
        
        if cfg.model_type == "logistic_regression":
            class_weight = "balanced" if cfg.use_class_weight else None
            logger.info(f"Instanciation d'une LogisticRegression (C={cfg.C}, max_iter={cfg.max_iter})")
            return LogisticRegression(
                C=cfg.C,
                max_iter=cfg.max_iter,
                class_weight=class_weight,
                n_jobs=-1,
            )
        
        raise ValueError(f"Type de modèle non supporté : '{cfg.model_type}'")
    
    def _prepare_params(self) -> dict:
        """Prépare les paramètres pour le logging."""
        cfg = self.config
        return {
            "model_type": cfg.model_type,
            "C": cfg.C,
            "max_iter": cfg.max_iter,
            "use_class_weight": cfg.use_class_weight,
        }
    
    def run(self) -> Path:
        """
        Entraîne le modèle avec le logging configuré.
        """
        logger.info("Démarrage de l'étape ModelTrainer")
        cfg = self.config
        
        # 1. Démarrer le run (MLflow ou local)
        run_context = self.logger.start_run(run_name=cfg.mlflow_run_name)
        logger.info(f"Run démarré: {run_context.get('run_id', 'local')}")
        
        try:
            # 2. Charger les données
            logger.info(f"Chargement de X_train depuis: {cfg.X_train_path}")
            X_train = sparse.load_npz(cfg.X_train_path)
            
            logger.info(f"Chargement de y_train depuis: {cfg.y_train_path}")
            y_train = np.load(cfg.y_train_path)
            
            # 3. Construire et entraîner le modèle
            model = self._build_model()
            logger.info("Entraînement du modèle")
            model.fit(X_train, y_train)
            logger.success("Modèle entraîné avec succès")
            
            # 4. Évaluer
            y_pred_train = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_pred_train)
            train_f1_macro = f1_score(y_train, y_pred_train, average="macro")
            
            metrics = {
                "train_accuracy": train_accuracy,
                "train_f1_macro": train_f1_macro,
            }
            
            # 5. Logging via le logger configuré (polymorphisme)
            params = self._prepare_params()
            self.logger.log_params(params)
            self.logger.log_metrics(metrics)
            
            # 6. Sauvegarder localement (toujours nécessaire)
            create_directories([cfg.model_dir])
            with open(cfg.model_path, "wb") as f:
                pickle.dump(model, f)
            
            # 7. Logger le modèle
            if sparse.issparse(X_train):
                input_example = X_train[:1].toarray()
            else:
                input_example = X_train[:1]
            
            self.logger.log_model(
                model=model,
                artifact_path=cfg.mlflow_artifact_path,
                input_example=input_example
            )
            
            # 8. Sauvegarder les fichiers supplémentaires
            self._save_additional_files(model, X_train, y_train, y_pred_train)
            
            # 9. Logger les artifacts
            for file_path in [
                cfg.model_dir / "model_config.json",
                cfg.model_dir / "metrics_train.json",
                cfg.model_dir / "classification_report_train.txt"
            ]:
                if file_path.exists():
                    self.logger.log_artifact(file_path)
            
            logger.success("ModelTrainer terminé avec succès")
            return cfg.model_path
            
        finally:
            # Toujours terminer le run
            self.logger.end_run()
    
    def _save_additional_files(self, model, X_train, y_train, y_pred_train):
        """Sauvegarde les fichiers supplémentaires (inchangé)."""
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
        }
        
        with open(cfg.model_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)
        
        # Sauvegarder les métriques
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "train_f1_macro": f1_score(y_train, y_pred_train, average="macro"),
        }
        
        with open(cfg.model_dir / "metrics_train.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Sauvegarder le rapport de classification
        cls_report = classification_report(y_train, y_pred_train)
        with open(cfg.model_dir / "classification_report_train.txt", "w") as f:
            f.write(cls_report)