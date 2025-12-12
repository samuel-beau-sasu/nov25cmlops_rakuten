from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class DataVersioningConfig:
    source_x_train_path: Path
    source_y_train_path: Path
    
    version_name: str
    split_ratio: float
    description: str
    apply_drift: Literal[False, "light", "strong"]
    
    output_x_path: Path
    output_y_path: Path
    output_metadata_path: Path

@dataclass
class DataIngestionConfig:
    data_version: str
    x_train_path: Path
    y_train_path: Path
    output_path: Path


@dataclass
class DataPreprocessingConfig:
    input_dataset_path: Path
    output_dataset_path: Path

    text_column: str
    target_column: str

    drop_na_text: bool
    drop_na_target: bool
    drop_duplicates: bool

    min_char_length: int
    max_char_length: int
    min_alpha_ratio: float


@dataclass
class DataTransformationConfig:
    input_dataset_path: Path
    output_dir: Path

    text_column: str
    target_column: str

    test_size: float
    random_state: int
    stratify: bool

    max_features: int
    ngram_min: int
    ngram_max: int
    lowercase: bool
    stop_words: str | None

    vectorizer_path: Path
    label_encoder_path: Path
    class_mapping_path: Path
    X_train_path: Path
    X_val_path: Path
    y_train_path: Path
    y_val_path: Path


@dataclass
class ModelTrainerConfig:
    data_version: str
    model_path: Path
    model_dir: Path
    X_train_path: Path
    y_train_path: Path

    model_type: str

    C: float
    max_iter: int
    use_class_weight: bool


@dataclass
class ModelEvaluationConfig:
    model_path: Path
    X_val_path: Path
    y_val_path: Path

    metrics_path: Path
    metrics_dir: Path
    classification_report_path: Path
    confusion_matrix_path: Path


@dataclass
class PredictionConfig:
    vectorizer_path: Path
    label_encoder_path: Path
    model_path: Path

    text_column: str
