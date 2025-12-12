# mlops_rakuten/config_manager.py
from pathlib import Path

from loguru import logger
import yaml

from mlops_rakuten.config import (
    CONFIG_FILE_PATH,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    VERSIONS_DATA_DIR
)
from mlops_rakuten.entities import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    DataTransformationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    PredictionConfig,
    DataVersioningConfig
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

        self.global_version = self._config["global"]["data_version"]
        logger.debug(f"Version data initialisé à la version: {self.global_version}")

    def get_data_versioning_configs(self) -> list[DataVersioningConfig]:
        """
        Retourne une liste de DataVersioningConfig pour toutes les versions
        définies dans config.yml.
        """
        config = self._config["data_versioning"]

        source_x = RAW_DATA_DIR / config["source_x_train"]
        source_y = RAW_DATA_DIR / config["source_y_train"]

        versioning_configs = []

        for version_spec in config["versions"]:
            version_name = version_spec["name"]
            version_dir = VERSIONS_DATA_DIR / version_name

            cfg = DataVersioningConfig(
                source_x_train_path=source_x,
                source_y_train_path=source_y,
                version_name=version_name,
                split_ratio=version_spec["split_ratio"],
                description=version_spec["description"],
                apply_drift=version_spec["apply_drift"],
                output_x_path=version_dir / "X_train.csv",
                output_y_path=version_dir / "Y_train.csv",
                output_metadata_path=version_dir / "metadata.json",
            )
            versioning_configs.append(cfg)

        logger.debug(f"Loaded {len(versioning_configs)} versioning configs")
        return versioning_configs

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Construit un DataIngestionConfig à partir de la section
        'data_ingestion' du YAML.
        
        MODIFIÉ : Charge depuis data/versions/{data_version}/ au lieu de data/raw/
        """
        c = self._config["data_ingestion"]
        
        # Récupérer la version choisie
        version_dir = VERSIONS_DATA_DIR / self.global_version
        
        x_path = version_dir / c["x_train_filename"]
        y_path = version_dir / c["y_train_filename"]
        output_path = INTERIM_DATA_DIR / c["output_dataset_filename"]

        logger.debug(f"x_train_path resolved to: {x_path}")
        logger.debug(f"y_train_path resolved to: {y_path}")
        logger.debug(f"output_path resolved to: {output_path}")

        return DataIngestionConfig(
            data_version=self.global_version, 
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

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Construit un ModelTrainerConfig à partir de la section
        'model_trainer' du YAML, en combinant avec PROCESSED_DATA_DIR / MODELS_DIR.

        Cette configuration permet de :
        - charger X_train / y_train transformés
        - instancier le modèle sklearn (LinearSVC, etc.)
        - sauvegarder le modèle entraîné et ses hyperparamètres.
        """
        c = self._config["model_trainer"]

        model_dir = MODELS_DIR
        model_path = model_dir / c["model_filename"]

        X_train_path = PROCESSED_DATA_DIR / c["X_train_filename"]
        y_train_path = PROCESSED_DATA_DIR / c["y_train_filename"]

        logger.debug(f"model_path resolved to: {model_path}")
        logger.debug(f"X_train_path resolved to: {X_train_path}")
        logger.debug(f"y_train_path resolved to: {y_train_path}")

        return ModelTrainerConfig(
            data_version=self.global_version,
            model_path=model_path,
            model_dir=model_dir,
            X_train_path=X_train_path,
            y_train_path=y_train_path,
            model_type=c["model_type"],
            C=c["C"],
            max_iter=c["max_iter"],
            use_class_weight=c["use_class_weight"],
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Construit un ModelEvaluationConfig à partir de la section
        'model_evaluation' du YAML, en combinant avec PROCESSED_DATA_DIR / MODELS_DIR.

        Cette configuration permet de :
        - charger X_val / y_val transformés
        - charger le modèle entraîné
        - sauvegarder les métriques de validation et les rapports associés.
        """
        c = self._config["model_evaluation"]

        X_val_path = PROCESSED_DATA_DIR / c["X_val_filename"]
        y_val_path = PROCESSED_DATA_DIR / c["y_val_filename"]

        model_dir = MODELS_DIR
        model_path = model_dir / c["model_filename"]

        metrics_dir = REPORTS_DIR
        metrics_path = metrics_dir / c["metrics_filename"]
        classification_report_path = metrics_dir / c["classification_report_filename"]
        confusion_matrix_path = metrics_dir / c["confusion_matrix_filename"]

        logger.debug(f"X_val_path resolved to: {X_val_path}")
        logger.debug(f"y_val_path resolved to: {y_val_path}")
        logger.debug(f"model_path resolved to: {model_path}")
        logger.debug(f"metrics_path resolved to: {metrics_path}")
        logger.debug(f"classification_report_path resolved to: {classification_report_path}")
        logger.debug(f"confusion_matrix_path resolved to: {confusion_matrix_path}")

        return ModelEvaluationConfig(
            model_path=model_path,
            X_val_path=X_val_path,
            y_val_path=y_val_path,
            metrics_path=metrics_path,
            metrics_dir=metrics_dir,
            classification_report_path=classification_report_path,
            confusion_matrix_path=confusion_matrix_path,
        )

    def get_prediction_config(self) -> PredictionConfig:
        """
        Construit un PredictionConfig à partir de la section
        'prediction' du YAML, en combinant avec PROCESSED_DATA_DIR / MODELS_DIR.

        Cette configuration permet de :
        - recharger le vectorizer TF-IDF
        - recharger le LabelEncoder
        - recharger le modèle entraîné
        """
        c = self._config["prediction"]

        vectorizer_path = PROCESSED_DATA_DIR / c["vectorizer_filename"]
        label_encoder_path = PROCESSED_DATA_DIR / c["label_encoder_filename"]
        model_path = MODELS_DIR / c["model_filename"]

        logger.debug(f"vectorizer_path resolved to: {vectorizer_path}")
        logger.debug(f"label_encoder_path resolved to: {label_encoder_path}")
        logger.debug(f"model_path resolved to: {model_path}")

        return PredictionConfig(
            vectorizer_path=vectorizer_path,
            label_encoder_path=label_encoder_path,
            model_path=model_path,
            text_column=c["text_column"],
        )
