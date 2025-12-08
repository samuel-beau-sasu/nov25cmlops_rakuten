from loguru import logger
import typer

from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline
from mlops_rakuten.pipelines.data_preprocessing import DataPreprocessingPipeline
from mlops_rakuten.pipelines.data_transformation import DataTransformationPipeline
from mlops_rakuten.pipelines.model_evaluation import ModelEvaluationPipeline
from mlops_rakuten.pipelines.model_trainer import ModelTrainerPipeline

app = typer.Typer()


@app.command()
def main():
    """
    Point d'entrée CLI pour construire le dataset prêt pour la modélisation.
    Enchaîne :
    - le pipeline d’ingestion des données
    - le pipeline de prétraitement des données
    """
    logger.info("Lancement du pipeline complet de préparation du dataset")

    # 1. Ingestion
    ingestion_pipeline = DataIngestionPipeline()
    ingestion_output_path = ingestion_pipeline.run()
    logger.info(f"Dataset fusionné disponible à : {ingestion_output_path}")

    # 2. Prétraitement
    preprocessing_pipeline = DataPreprocessingPipeline()
    preprocessing_output_path = preprocessing_pipeline.run()
    logger.success(f"Dataset prétraité disponible à : {preprocessing_output_path}")

    # 3. Transformation
    transformation_pipeline = DataTransformationPipeline()
    transformation_output_path = transformation_pipeline.run()
    logger.success(f"Dataset transformé disponible à : {transformation_output_path}")

    # 4. Entraînement du modèle
    model_trainer_pipeline = ModelTrainerPipeline()
    model_path = model_trainer_pipeline.run()
    logger.success(f"Modèle entraîné disponible à : {model_path}")

    # 5. Évaluation du modèle sur le jeu de validation
    model_evaluation_pipeline = ModelEvaluationPipeline()
    metrics_path = model_evaluation_pipeline.run()
    logger.success(f"Métriques de validation disponibles dans : {metrics_path}")


if __name__ == "__main__":
    app()
