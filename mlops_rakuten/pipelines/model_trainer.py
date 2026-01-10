from pathlib import Path
from loguru import logger
import yaml
import json

# ═══════════════════════════════════════════════════════════════════════════
# 1. DAGSHUB INIT - DOIT ÊTRE EN PREMIER (avant mlflow)
# ═══════════════════════════════════════════════════════════════════════════
import dagshub
dagshub.init(
    repo_owner='shiff-oumi',
    repo_name='nov25cmlops_rakuten_dag',
    mlflow=True
)

# 2. MLFLOW SETUP - après DagsHub
import mlflow

from mlops_rakuten.config.config_manager import ConfigurationManager
from mlops_rakuten.modules.model_trainer import ModelTrainer


def get_data_lineage() -> dict:
    """
    Récupère les hashes clés pour tracer data → model
    Lien direct entre version data et version modèle
    """
    dvc_lock_path = Path("dvc.lock")
    
    if not dvc_lock_path.exists():
        logger.warning("dvc.lock not found!")
        return {"status": "unknown"}
    
    with open(dvc_lock_path, "r") as f:
        lock_data = yaml.safe_load(f)
    
    stages = lock_data.get("stages", {})
    
    lineage = {
        "data_input_hash": None,          # rakuten_train_current.csv
        "preprocessed_data_hash": None,   # preprocessed_dataset.csv
        "train_features_hash": None,      # X_train_tfidf.npz (input du modèle)
        "model_hash": None,               # text_classifier.pkl (output du modèle)
    }
    
    # 1. Data input - depuis preprocess deps
    preprocess_stage = stages.get("preprocess", {})
    if preprocess_stage:
        for dep in preprocess_stage.get("deps", []):
            path = dep.get("path", "")
            if "rakuten_train_current.csv" in path:
                lineage["data_input_hash"] = dep.get("md5")
                logger.debug(f"Found data_input_hash from preprocess deps: {lineage['data_input_hash']}")
    
    # 2. Preprocessed data - depuis preprocess outs
    if preprocess_stage:
        for out in preprocess_stage.get("outs", []):
            path = out.get("path", "")
            if "preprocessed_dataset.csv" in path:
                lineage["preprocessed_data_hash"] = out.get("md5")
                logger.debug(f"Found preprocessed_data_hash from preprocess outs: {lineage['preprocessed_data_hash']}")
    
    # 3. Train features - depuis transform outs
    transform_stage = stages.get("transform", {})
    if transform_stage:
        for out in transform_stage.get("outs", []):
            path = out.get("path", "")
            if "X_train_tfidf.npz" in path:
                lineage["train_features_hash"] = out.get("md5")
                logger.debug(f"Found train_features_hash from transform outs: {lineage['train_features_hash']}")
    
    # 4. Model - depuis train outs (IMPORTANT: c'est l'output du stage train)
    train_stage = stages.get("train", {})
    if train_stage:
        for out in train_stage.get("outs", []):
            path = out.get("path", "")
            if "text_classifier.pkl" in path:
                lineage["model_hash"] = out.get("md5")
                logger.debug(f"Found model_hash from train outs: {lineage['model_hash']}")
    
    logger.info(f"Data lineage extracted: {lineage}")
    return lineage


class ModelTrainerPipeline:
    def run(self) -> Path:
        logger.info("Démarrage du pipeline ModelTrainer")

        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()

        step = ModelTrainer(config=model_trainer_config)
        model_path = step.run()

        logger.success(f"Pipeline ModelTrainer terminé. Modèle créé : {model_path}")
        return model_path


if __name__ == "__main__":
    # DagsHub a déjà setup mlflow, donc pas besoin de set_tracking_uri
    mlflow.set_experiment("rakuten_classification_v0")
    
    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE (with MLflow + DVC + DagsHub tracking)")
    logger.info("=" * 80)
    
    try:
        with mlflow.start_run(run_name="train_stage"):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            
            # ───────────────────────────────────────────────────────────────
            # LOG HYPERPARAMS
            # ───────────────────────────────────────────────────────────────
            config_manager = ConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            
            mlflow.log_param("model_type", model_trainer_config.model_type)
            mlflow.log_param("C", model_trainer_config.C)
            mlflow.log_param("max_iter", model_trainer_config.max_iter)
            mlflow.log_param("use_class_weight", model_trainer_config.use_class_weight)
            
            # ───────────────────────────────────────────────────────────────
            # LOG DATA LINEAGE (DVC tracking - essential!)
            # ───────────────────────────────────────────────────────────────
            logger.info("Logging data lineage from dvc.lock...")
            
            lineage = get_data_lineage()
            
            # Log les hashes clés comme params (pour pouvoir filtrer dans MLflow)
            mlflow.log_param("data_input_hash", lineage["data_input_hash"])
            mlflow.log_param("train_features_hash", lineage["train_features_hash"])
            mlflow.log_param("model_hash", lineage["model_hash"])
            
            # Log la lignée complète en artifact (pour la traçabilité)
            with open("dvc_lineage.json", "w") as f:
                json.dump(lineage, f, indent=2)
            mlflow.log_artifact("dvc_lineage.json", artifact_path="data")
            
            logger.info(f"Data input hash: {lineage['data_input_hash']}")
            logger.info(f"Model hash: {lineage['model_hash']}")
            
            # ───────────────────────────────────────────────────────────────
            # EXÉCUTER LE PIPELINE
            # ───────────────────────────────────────────────────────────────
            pipeline = ModelTrainerPipeline()
            model_path = pipeline.run()
            
            # ───────────────────────────────────────────────────────────────
            # LOG TAGS (pour la recherche rapide)
            # ───────────────────────────────────────────────────────────────
            mlflow.set_tag("stage", "train")
            mlflow.set_tag("pipeline", "full_dvc_dagshub")
            mlflow.set_tag("data_input_hash", lineage["data_input_hash"])
            mlflow.set_tag("model_version", lineage["model_hash"])
            
            logger.success("\n" + "=" * 80)
            logger.success(f"Training completed with full DVC + DagsHub traceability")
            logger.success(f"Run ID: {run_id}")
            logger.success(f"Data → Model lineage tracked in DagsHub")
            logger.success("=" * 80)
    
    except Exception as e:
        logger.exception("Error in training pipeline")
        raise