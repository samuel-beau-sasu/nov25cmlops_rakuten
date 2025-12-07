from pathlib import Path
from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline


class DummyDataIngestionStep:
    def __init__(self, config):
        self.config = config

    def run(self) -> Path:
        return Path("/fake/path/dataset.csv")


def test_data_ingestion_pipeline_calls_step(monkeypatch):
    from mlops_rakuten import pipelines
    from mlops_rakuten import config_manager

    # Monkeypatch DataIngestion pour le remplacer par DummyStep
    monkeypatch.setattr(
        "mlops_rakuten.pipelines.data_ingestion.DataIngestion",
        DummyDataIngestionStep,
    )

    pipeline = DataIngestionPipeline()
    output = pipeline.run()

    assert output == Path("/fake/path/dataset.csv")