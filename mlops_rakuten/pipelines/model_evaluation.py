from pathlib import Path
import mlflow
from loguru import logger

from mlops_rakuten.config.config_manager import ConfigurationManager
from mlops_rakuten.modules.model_evaluation import ModelEvaluation


class ModelEvaluationPipeline:
    """
    Pipeline d'évaluation du modèle :
    - charge la configuration
    - exécute l'étape ModelEvaluation
    - renvoie le chemin du fichier de métriques de validation
    """

    def run(self) -> Path:
        logger.info("Démarrage du pipeline ModelEvaluation")

        config_manager = ConfigurationManager()
        model_eval_config = config_manager.get_model_evaluation_config()

        step = ModelEvaluation(config=model_eval_config)
        metrics_path = step.run()

        logger.success(
            f"Pipeline ModelEvaluation terminé. "
            f"Métriques de validation disponibles dans : {metrics_path}"
        )
        return metrics_path

if __name__ == "__main__":
    # 1. SETUP MLFLOW
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("rakuten_classification_v0")
    
    logger.info("=" * 80)
    logger.info("EVALUATION PIPELINE (with MLflow)")
    logger.info("=" * 80)
    
    try:
        # 2. OUVRIR 1 RUN MLFLOW
        with mlflow.start_run(run_name="evaluate_stage"):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            
            # 3. EXÉCUTER LE PIPELINE (dans le run ouvert!)
            pipeline = ModelEvaluationPipeline()
            metrics_path = pipeline.run()
            
            # 4. LOG TAGS
            mlflow.set_tag("stage", "evaluate")
            mlflow.set_tag("pipeline", "standalone")
            
            logger.success("\n" + "=" * 80)
            logger.success(f"Evaluation completed with MLflow")
            logger.success(f"Run ID: {run_id}")
            logger.success("=" * 80)
    
    except Exception as e:
        logger.exception("Error in evaluation pipeline")
        raise