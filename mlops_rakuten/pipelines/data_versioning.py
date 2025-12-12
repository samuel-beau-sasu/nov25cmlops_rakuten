from pathlib import Path
from typing import List

from loguru import logger

from mlops_rakuten.config_manager import ConfigurationManager
from mlops_rakuten.modules.data_versioning import DataVersioning


class DataVersioningPipeline:
    """
    Pipeline de création des versions de données.
    
    Ce pipeline est exécuté UNE SEULE FOIS pour créer toutes les versions.
    Ensuite, on utilise DataIngestionPipeline avec différentes versions.
    """

    def run(self) -> Path:
        """
        Crée toutes les versions définies dans config.yml
        Retourne la liste des répertoires créés
        """
        logger.info("Démarrage du pipeline DataIngestion")

        config_manager = ConfigurationManager()
        versioning_configs = config_manager.get_data_versioning_configs()

        logger.info(f"\n {len(versioning_configs)} versions à créer\n")

        created_versions = []

        for config in versioning_configs:
            versioning = DataVersioning(config=config)
            version_dir = versioning.run()

            created_versions.append(version_dir)
            logger.success(f" Pipeline DataVersioning terminé. Sauvegardée dans: {version_dir}\n")


        return created_versions