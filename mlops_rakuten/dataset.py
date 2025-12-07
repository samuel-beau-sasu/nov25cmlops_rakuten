from loguru import logger
import typer

from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline

app = typer.Typer()


@app.command()
def main():
    """
    Point d'entrée CLI pour lancer le pipeline d'ingestion de données.
    """
    logger.info("Lancement de l’ingestion des données (pipeline)")
    pipeline = DataIngestionPipeline()
    output_path = pipeline.run()
    logger.success(f"Dataset final disponible à : {output_path}")


if __name__ == "__main__":
    app()
