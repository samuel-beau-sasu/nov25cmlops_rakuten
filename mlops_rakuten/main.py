from loguru import logger
import typer

from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline
from mlops_rakuten.pipelines.data_preprocessing import DataPreprocessingPipeline
from mlops_rakuten.pipelines.data_transformation import DataTransformationPipeline
from mlops_rakuten.pipelines.model_evaluation import ModelEvaluationPipeline
from mlops_rakuten.pipelines.model_trainer import ModelTrainerPipeline
from mlops_rakuten.pipelines.prediction import PredictionPipeline
from mlops_rakuten.pipelines.data_versioning import DataVersioningPipeline

app = typer.Typer()

@app.command()
def create_versions():
    """
    Crée les versions de données (v1.0, v2.0, v3.0).
    
    ⚠️  À exécuter UNE SEULE FOIS au début du projet.
    
    Exemple:
        python -m mlops_rakuten.main create-versions
    """
    logger.info("Création des versions de données")
    
    pipeline = DataVersioningPipeline()
    created_versions = pipeline.run()
    
    logger.success(f"{len(created_versions)} versions créées avec succès !")
    for version_dir in created_versions:
        logger.info(f"{version_dir}")

@app.command()
def train():
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


@app.command()
def predict(text: str):
    """
    Effectue une prédiction à partir d'un texte.

    Exemple:
    python -m mlops_rakuten.dataset predict "Super aspirateur sans fil"
    """
    logger.info("Démarrage de l'inférence via CLI")

    pipeline = PredictionPipeline()
    pred = pipeline.run([text])

    logger.success(f"Texte : {text}")
    logger.success(f"prdtypecode prédit : {pred[0]}")


if __name__ == "__main__":
    app()
