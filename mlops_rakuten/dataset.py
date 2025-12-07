from loguru import logger
import typer

from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline
from mlops_rakuten.pipelines.data_preprocessing import DataPreprocessingPipeline

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


if __name__ == "__main__":
    app()
