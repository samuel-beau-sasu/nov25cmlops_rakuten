from loguru import logger
import typer
from typing import Dict, Any
from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline
from mlops_rakuten.pipelines.data_preprocessing import DataPreprocessingPipeline
from mlops_rakuten.pipelines.data_transformation import DataTransformationPipeline
from mlops_rakuten.pipelines.model_evaluation import ModelEvaluationPipeline
from mlops_rakuten.pipelines.model_trainer import ModelTrainerPipeline
from mlops_rakuten.pipelines.prediction import PredictionPipeline

app = typer.Typer()


def run_training_pipeline() -> Dict[str, Any]:
    """
    Fonction réutilisable pour exécuter le pipeline complet de training.
    
    Returns:
        Dict contenant les chemins des artefacts et les métriques
    """
    logger.info("Lancement du pipeline complet de préparation du dataset")
    
    results = {}
    
    try:
        # 1. Ingestion
        ingestion_pipeline = DataIngestionPipeline()
        ingestion_output_path = ingestion_pipeline.run()
        logger.info(f"Dataset fusionné disponible à : {ingestion_output_path}")
        results["ingestion_path"] = ingestion_output_path
        
        # 2. Prétraitement
        preprocessing_pipeline = DataPreprocessingPipeline()
        preprocessing_output_path = preprocessing_pipeline.run()
        logger.success(f"Dataset prétraité disponible à : {preprocessing_output_path}")
        results["preprocessing_path"] = preprocessing_output_path
        
        # 3. Transformation
        transformation_pipeline = DataTransformationPipeline()
        transformation_output_path = transformation_pipeline.run()
        logger.success(f"Dataset transformé disponible à : {transformation_output_path}")
        results["transformation_path"] = transformation_output_path
        
        # 4. Entraînement du modèle
        model_trainer_pipeline = ModelTrainerPipeline()
        model_path = model_trainer_pipeline.run()
        logger.success(f"Modèle entraîné disponible à : {model_path}")
        results["model_path"] = model_path
        
        # 5. Évaluation du modèle
        model_evaluation_pipeline = ModelEvaluationPipeline()
        metrics_path = model_evaluation_pipeline.run()
        logger.success(f"Métriques de validation disponibles dans : {metrics_path}")
        results["metrics_path"] = metrics_path
        
        # Ajouter des informations supplémentaires
        results["status"] = "success"
        results["message"] = "Pipeline de training complété avec succès"
        
        logger.success("✅ Pipeline complet terminé avec succès !")
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline de training : {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }


def run_prediction(texts: list[str]) -> list:
    """
    Fonction réutilisable pour effectuer des prédictions.
    
    Args:
        texts: Liste de textes à prédire
        
    Returns:
        Liste des prédictions
    """
    logger.info(f"Démarrage de l'inférence pour {len(texts)} texte(s)")
    
    try:
        pipeline = PredictionPipeline()
        predictions = pipeline.run(texts)
        logger.success(f"Prédictions effectuées : {predictions}")
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction : {str(e)}")
        raise


# ============= CLI avec Typer (backward compatible) =============

@app.command()
def train():
    """
    Point d'entrée CLI pour construire le dataset prêt pour la modélisation.
    """
    results = run_training_pipeline()
    
    if results["status"] == "success":
        logger.info("🎉 Training terminé via CLI")
        logger.info(f"Modèle : {results.get('model_path')}")
        logger.info(f"Métriques : {results.get('metrics_path')}")
    else:
        logger.error(f"❌ Échec du training : {results.get('message')}")
        raise typer.Exit(code=1)


@app.command()
def predict(text: str):
    """
    Effectue une prédiction à partir d'un texte.
    
    Exemple:
        python -m mlops_rakuten.main predict "Super aspirateur sans fil"
    """
    logger.info("Démarrage de l'inférence via CLI")
    
    predictions = run_prediction([text])
    logger.success(f"Texte : {text}")
    logger.success(f"prdtypecode prédit : {predictions[0]}")


if __name__ == "__main__":
    app()