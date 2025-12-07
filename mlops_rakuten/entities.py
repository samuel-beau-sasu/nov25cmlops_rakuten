from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
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
