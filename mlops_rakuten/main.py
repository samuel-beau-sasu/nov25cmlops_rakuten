from pathlib import Path

from loguru import logger
import typer

from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline
from mlops_rakuten.pipelines.data_preprocessing import DataPreprocessingPipeline
from mlops_rakuten.pipelines.data_seeding import DataSeedingPipeline
from mlops_rakuten.pipelines.data_transformation import DataTransformationPipeline
from mlops_rakuten.pipelines.model_evaluation import ModelEvaluationPipeline
from mlops_rakuten.pipelines.model_trainer import ModelTrainerPipeline
from mlops_rakuten.pipelines.prediction import PredictionPipeline

app = typer.Typer()


def _run_training_chain() -> tuple[Path, Path, Path, Path]:
    """
    Exécute la chaîne standard d'entraînement:
    preprocessing -> transformation -> training -> evaluation

    Retourne:
      (preprocessed_path, transformed_path, model_path, metrics_path)
    """
    # 1. Prétraitement
    preprocessing_pipeline = DataPreprocessingPipeline()
    preprocessing_output_path = preprocessing_pipeline.run()
    logger.success(f"Dataset prétraité disponible à : {preprocessing_output_path}")

    # 2. Transformation
    transformation_pipeline = DataTransformationPipeline()
    transformation_output_path = transformation_pipeline.run()
    logger.success(f"Dataset transformé disponible à : {transformation_output_path}")

    # 3. Entraînement
    model_trainer_pipeline = ModelTrainerPipeline()
    model_path = model_trainer_pipeline.run()
    logger.success(f"Modèle entraîné disponible à : {model_path}")

    # 4. Évaluation
    model_evaluation_pipeline = ModelEvaluationPipeline()
    metrics_path = model_evaluation_pipeline.run()
    logger.success(f"Métriques de validation disponibles dans : {metrics_path}")

    return (
        preprocessing_output_path,
        transformation_output_path,
        model_path,
        metrics_path,
    )


@app.command()
def seed():
    """
    Point d'entrée CLI pour construire le dataset prêt pour la modélisation.
    Enchaîne :
    - le pipeline de seeding des données
    """
    logger.info("Lancement du pipeline de seeding des données")

    # Seeding
    seeding_pipeline = DataSeedingPipeline()
    seeding_output_path = seeding_pipeline.run()
    logger.info(f"Dataset initial disponible à : {seeding_output_path}")

    _run_training_chain()


@app.command()
def ingest(uploaded_csv_path: str):
    """
    Point d'entrée CLI pour ingérer et fusionner le nouveau dataset
    au dataset précédemment utilisé.
    Enchaîne :
    - le pipeline d’ingestion des données
    - le pipeline complet d'entraînement du modèle
    """
    logger.info("Lancement du pipeline complet (ingestion + entraînement)")

    ingestion_pipeline = DataIngestionPipeline()
    ingestion_output_path = ingestion_pipeline.run(uploaded_csv_path)
    logger.info(f"Dataset fusionné disponible à : {ingestion_output_path}")



@app.command()
def train():
    """
    Point d'entrée CLI pour construire le dataset prêt pour la modélisation.
    Enchaîne :
    - le pipeline de prétraitement des données
    - le pipeline de transformation des données
    - le pipeline d'entraînement du modèle
    - le pipeline d'évaluation du modèle
    """
    logger.info("Lancement du pipeline d'entraînement du modèle")
    _run_training_chain()


@app.command()
def predict(text: str, top_k: int = 5):
    """
    Effectue une prédiction à partir d'un texte.

    Exemple:
    python -m mlops_rakuten.dataset predict "Super aspirateur sans fil"
    """
    logger.info("Démarrage de l'inférence via CLI")

    pipeline = PredictionPipeline()
    results_per_text = pipeline.run(texts=[text], top_k=top_k)

    results = results_per_text[0]  # un seul texte

    logger.info(f"Texte : {text}")
    for r in results:
        pct = r["proba"] * 100
        logger.success(f"{r['prdtypecode']} - {r['category_name']} : {pct:.1f}%")


if __name__ == "__main__":
    app()
