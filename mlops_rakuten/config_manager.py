# mlops_rakuten/config_manager.py
from pathlib import Path

from loguru import logger
import yaml

from mlops_rakuten.config import CONFIG_FILE_PATH, INTERIM_DATA_DIR, RAW_DATA_DIR
from mlops_rakuten.entities import DataIngestionConfig, DataPreprocessingConfig


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
        'data_ingestion' du YAML, en combinant avec RAW_DATA_DIR / INTERIM_DATA_DIR.
        """
        c = self._config["data_ingestion"]

        x_path = RAW_DATA_DIR / c["x_train_filename"]
        y_path = RAW_DATA_DIR / c["y_train_filename"]
        output_path = INTERIM_DATA_DIR / c["output_dataset_filename"]

        logger.debug(f"x_train_path resolved to: {x_path}")
        logger.debug(f"y_train_path resolved to: {y_path}")
        logger.debug(f"output_path resolved to: {output_path}")

        return DataIngestionConfig(
            x_train_path=x_path,
            y_train_path=y_path,
            output_path=output_path,
        )

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
        Construit un DataPreprocessingConfig à partir de la section
        'data_preprocessing' du YAML, en combinant avec INTERIM_DATA_DIR.
        """
        c = self._config["data_preprocessing"]

        input_path = INTERIM_DATA_DIR / c["input_dataset_filename"]
        output_path = INTERIM_DATA_DIR / c["output_dataset_filename"]

        logger.debug(f"input_dataset_path resolved to: {input_path}")
        logger.debug(f"output_dataset_path resolved to: {output_path}")

        return DataPreprocessingConfig(
            input_dataset_path=input_path,
            output_dataset_path=output_path,
            text_column=c["text_column"],
            target_column=c["target_column"],
            drop_na_text=c.get("drop_na_text", True),
            drop_na_target=c.get("drop_na_target", True),
            drop_duplicates=c.get("drop_duplicates", True),
            min_char_length=c.get("min_char_length", 10),
            max_char_length=c.get("max_char_length", 1000),
            min_alpha_ratio=c.get("min_alpha_ratio", 0.2),
        )
