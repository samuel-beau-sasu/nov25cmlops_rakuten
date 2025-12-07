# mlops_rakuten/config_manager.py
from pathlib import Path

from loguru import logger
import yaml

from mlops_rakuten.config import CONFIG_FILE_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR
from mlops_rakuten.entities import DataIngestionConfig


class ConfigurationManager:
    """
    Centralise la lecture du fichier config.yml
    et expose des méthodes pour construire les objets de config.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = CONFIG_FILE_PATH

        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Construit un DataIngestionConfig à partir de la section
        'data_ingestion' du YAML, en combinant avec RAW_DATA_DIR / PROCESSED_DATA_DIR.
        """
        c = self._config["data_ingestion"]

        x_path = RAW_DATA_DIR / c["x_train_filename"]
        y_path = RAW_DATA_DIR / c["y_train_filename"]
        output_path = PROCESSED_DATA_DIR / c["output_dataset_filename"]

        logger.debug(f"x_train_path resolved to: {x_path}")
        logger.debug(f"y_train_path resolved to: {y_path}")
        logger.debug(f"output_path resolved to: {output_path}")

        return DataIngestionConfig(
            x_train_path=x_path,
            y_train_path=y_path,
            output_path=output_path,
        )
