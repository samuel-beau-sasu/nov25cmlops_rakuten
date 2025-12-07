# mlops_rakuten/config_manager.py
from pathlib import Path

from loguru import logger
import yaml

from mlops_rakuten.config import (
    CONFIG_FILE_PATH,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from mlops_rakuten.entities import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    DataTransformationConfig,
)


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

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Construit un DataTransformationConfig à partir de la section
        'data_transformation' du YAML, en combinant avec INTERIM_DATA_DIR /
        PROCESSED_DATA_DIR.

        Cette configuration permet de :
        - charger le dataset prétraité (CSV)
        - appliquer le split train/validation
        - vectoriser le texte avec TF-IDF
        - encoder la cible avec un LabelEncoder
        - sauvegarder tous les artefacts dans data/processed/
        """
        c = self._config["data_transformation"]

        input_path = INTERIM_DATA_DIR / c["input_dataset_filename"]
        output_dir = PROCESSED_DATA_DIR

        return DataTransformationConfig(
            input_dataset_path=input_path,
            output_dir=output_dir,
            test_size=c["test_size"],
            random_state=c["random_state"],
            stratify=c["stratify"],
            max_features=c["max_features"],
            ngram_min=c["ngram_min"],
            ngram_max=c["ngram_max"],
            lowercase=c["lowercase"],
            stop_words=c["stop_words"],
            vectorizer_path=output_dir / c["vectorizer_filename"],
            label_encoder_path=output_dir / c["label_encoder_filename"],
            class_mapping_path=output_dir / c["class_mapping_filename"],
            X_train_path=output_dir / c["X_train_filename"],
            X_val_path=output_dir / c["X_val_filename"],
            y_train_path=output_dir / c["y_train_filename"],
            y_val_path=output_dir / c["y_val_filename"],
            text_column=c["text_column"],
            target_column=c["target_column"],
        )
