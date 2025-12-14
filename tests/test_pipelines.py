from pathlib import Path

from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline
from mlops_rakuten.pipelines.data_preprocessing import DataPreprocessingPipeline
from mlops_rakuten.pipelines.data_transformation import DataTransformationPipeline
from mlops_rakuten.pipelines.model_trainer import ModelTrainerPipeline
from mlops_rakuten.pipelines.model_evaluation import ModelEvaluationPipeline


# Dummy Config Manager

class DummyConfigManager:
    def __init__(self):
        pass

    def get_data_ingestion_config(self, uploaded_csv_path=None):
        return object()

    def get_data_preprocessing_config(self):
        return object()

    def get_data_transformation_config(self):
        return object()

    def get_model_trainer_config(self):
        return object()

    def get_model_evaluation_config(self):
        return object()


# Dummy Steps

class DummyDataIngestionStep:
    def __init__(self, config):
        self.config = config

    def run(self, uploaded_csv_path: Path) -> Path:
        return Path("/fake/path/rakuten_train.csv")


class DummyDataPreprocessingStep:
    def __init__(self, config):
        self.config = config

    def run(self) -> Path:
        return Path("/fake/path/preprocessed_dataset.csv")


class DummyDataTransformationStep:
    def __init__(self, config):
        self.config = config

    def run(self) -> Path:
        return Path("/fake/path/tfidf_vectorizer.pkl")


class DummyModelTrainerStep:
    def __init__(self, config):
        self.config = config

    def run(self) -> Path:
        return Path("/fake/path/text_classifier.pkl")


class DummyModelEvaluationStep:
    def __init__(self, config):
        self.config = config

    def run(self) -> Path:
        return Path("/fake/path/metrics_val.json")


# Tests

def test_data_ingestion_pipeline_calls_step(monkeypatch):
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.data_ingestion.ConfigurationManager",
        DummyConfigManager,
    )
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.data_ingestion.DataIngestion",
        DummyDataIngestionStep,
    )

    pipeline = DataIngestionPipeline()
    output = pipeline.run(uploaded_csv_path=Path("/fake/path/upload.csv"))

    assert output == Path("/fake/path/rakuten_train.csv")


def test_data_preprocessing_pipeline_calls_step(monkeypatch):
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.data_preprocessing.ConfigurationManager",
        DummyConfigManager,
    )
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.data_preprocessing.DataPreprocessing",
        DummyDataPreprocessingStep,
    )

    pipeline = DataPreprocessingPipeline()
    output = pipeline.run()

    assert output == Path("/fake/path/preprocessed_dataset.csv")


def test_data_transformation_pipeline_calls_step(monkeypatch):
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.data_transformation.ConfigurationManager",
        DummyConfigManager,
    )
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.data_transformation.DataTransformation",
        DummyDataTransformationStep,
    )

    pipeline = DataTransformationPipeline()
    output = pipeline.run()

    assert output == Path("/fake/path/tfidf_vectorizer.pkl")


def test_model_trainer_pipeline_calls_step(monkeypatch):
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.model_trainer.ConfigurationManager",
        DummyConfigManager,
    )
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.model_trainer.ModelTrainer",
        DummyModelTrainerStep,
    )

    pipeline = ModelTrainerPipeline()
    output = pipeline.run()

    assert output == Path("/fake/path/text_classifier.pkl")


def test_model_evaluation_pipeline_calls_step(monkeypatch):
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.model_evaluation.ConfigurationManager",
        DummyConfigManager,
    )
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.model_evaluation.ModelEvaluation",
        DummyModelEvaluationStep,
    )

    pipeline = ModelEvaluationPipeline()
    output = pipeline.run()

    assert output == Path("/fake/path/metrics_val.json")
