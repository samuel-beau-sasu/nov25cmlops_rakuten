from pathlib import Path

from loguru import logger

from mlops_rakuten.config_manager import ConfigurationManager
from mlops_rakuten.modeling.data_ingestion import DataIngestion


class DataIngestionPipeline:
    """
    Pipeline d'ingestion des données :
    - charge la configuration
    - exécute l'étape DataIngestion
    - renvoie le chemin du dataset fusionné
    """

    def run(self) -> Path:
        logger.info("Démarrage du pipeline DataIngestion")

        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()

        step = DataIngestion(config=data_ingestion_config)
        output_path = step.run()

        logger.success(f"Pipeline DataIngestion terminé. Fichier créé : {output_path}")
        return output_path
