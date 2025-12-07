from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    x_train_path: Path
    y_train_path: Path
    output_path: Path
