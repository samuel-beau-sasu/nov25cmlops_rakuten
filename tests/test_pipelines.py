from pathlib import Path

from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline
from mlops_rakuten.pipelines.data_preprocessing import DataPreprocessingPipeline
from mlops_rakuten.pipelines.data_transformation import DataTransformationPipeline
from mlops_rakuten.pipelines.model_trainer import ModelTrainerPipeline
from mlops_rakuten.pipelines.model_evaluation import ModelEvaluationPipeline

# ---------------------------
# Dummy Steps
# ---------------------------


class DummyDataIngestionStep:
    def __init__(self, config):
        self.config = config

    def run(self) -> Path:
        return Path("/fake/path/ingested_dataset.csv")


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


# ---------------------------
# Tests
# ---------------------------

def test_data_ingestion_pipeline_calls_step(monkeypatch):
    """
    Vérifie que DataIngestionPipeline utilise bien la classe DataIngestion
    et retourne le chemin prévu.
    """
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.data_ingestion.DataIngestion",
        DummyDataIngestionStep,
    )

    pipeline = DataIngestionPipeline()
    output = pipeline.run()

    assert output == Path("/fake/path/ingested_dataset.csv")


def test_data_preprocessing_pipeline_calls_step(monkeypatch):
    """
    Vérifie que DataPreprocessingPipeline utilise bien la classe DataPreprocessing
    et retourne le chemin prévu.
    """

    # Monkeypatch de la classe DataPreprocessing
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.data_preprocessing.DataPreprocessing",
        DummyDataPreprocessingStep,
    )

    pipeline = DataPreprocessingPipeline()
    output = pipeline.run()

    assert output == Path("/fake/path/preprocessed_dataset.csv")


def test_data_transformation_pipeline_calls_step(monkeypatch):
    """
    Vérifie que DataTransformationPipeline utilise bien la classe DataTransformation
    et retourne le dossier d'artefacts attendu.
    """
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.data_transformation.DataTransformation",
        DummyDataTransformationStep,
    )

    pipeline = DataTransformationPipeline()
    output = pipeline.run()

    assert output == Path("/fake/path/tfidf_vectorizer.pkl")


def test_model_trainer_pipeline_calls_step(monkeypatch):
    """
    Vérifie que ModelTrainerPipeline utilise bien la classe ModelTrainer
    et retourne le chemin du modèle attendu.
    """
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.model_trainer.ModelTrainer",
        DummyModelTrainerStep,
    )

    pipeline = ModelTrainerPipeline()
    output = pipeline.run()

    assert output == Path("/fake/path/text_classifier.pkl")


def test_model_evaluation_pipeline_calls_step(monkeypatch):
    """
    Vérifie que ModelEvaluationPipeline utilise bien la classe ModelEvaluation
    et retourne le chemin attendu pour les métriques.
    """
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.model_evaluation.ModelEvaluation",
        DummyModelEvaluationStep,
    )

    pipeline = ModelEvaluationPipeline()
    output = pipeline.run()

    assert output == Path("/fake/path/metrics_val.json")
