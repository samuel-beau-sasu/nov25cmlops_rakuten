# tests/test_data_ingestion_component.py
import pandas as pd
from pathlib import Path

from mlops_rakuten.entities import DataIngestionConfig
from mlops_rakuten.modeling.data_ingestion import DataIngestion


def test_data_ingestion_merges_X_and_y(tmp_path):
    # 1. Préparer des CSV de test
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    x_path = raw_dir / "X.csv"
    y_path = raw_dir / "Y.csv"
    out_path = tmp_path / "processed" / "dataset.csv"

    X = pd.DataFrame(
        {
            "designation": ["a", "b"],
            "description": ["desc a", "desc b"],
            "productid": [1, 2],
            "imageid": [10, 20],
        },
        index=[0, 1],
    )
    y = pd.DataFrame({"prdtypecode": [100, 200]}, index=[0, 1])

    X.to_csv(x_path)
    y.to_csv(y_path)

    # 2. Config & composant
    cfg = DataIngestionConfig(
        x_train_path=x_path,
        y_train_path=y_path,
        output_path=out_path,
    )
    step = DataIngestion(config=cfg)

    # 3. Run
    output = step.run()

    # 4. Assertions
    assert output == out_path
    assert out_path.exists()

    df = pd.read_csv(out_path)
    assert list(df.columns) == [
        "designation",
        "description",
        "productid",
        "imageid",
        "prdtypecode",
    ]
    assert len(df) == 2
