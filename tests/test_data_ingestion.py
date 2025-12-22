from pathlib import Path

import pandas as pd

from mlops_rakuten.config.entities import DataIngestionConfig
from mlops_rakuten.modules.data_ingestion import DataIngestion


def test_data_ingestion_appends_uploaded_dataset(tmp_path):
    """
    Vérifie que DataIngestion :
    - charge un CSV uploadé valide
    - l'append au dataset train existant
    - sauvegarde le dataset final
    """

    # 1. Dataset train existant
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()

    train_path = interim_dir / "rakuten_train.csv"

    df_existing = pd.DataFrame(
        {
            "designation": ["old product"],
            "prdtypecode": [10],
        }
    )
    df_existing.to_csv(train_path, index=False)

    # 2. CSV uploadé (nouveaux échantillons)
    upload_path = tmp_path / "uploaded.csv"

    df_uploaded = pd.DataFrame(
        {
            "designation": ["new product 1", "new product 2"],
            "prdtypecode": [20, 30],
        }
    )
    df_uploaded.to_csv(upload_path, index=False)

    # 3. Config & composant
    cfg = DataIngestionConfig(
        train_path=train_path,
        text_column="designation",
        target_column="prdtypecode",
    )

    step = DataIngestion(config=cfg)

    # 4. Run
    output_path = step.run(uploaded_csv_path=upload_path)

    # 5. Assertions
    assert output_path == train_path
    assert train_path.exists()

    df_final = pd.read_csv(train_path)

    # 1 ancien + 2 nouveaux
    assert len(df_final) == 3

    assert list(df_final.columns) == [
        "designation",
        "prdtypecode",
    ]

    # Vérifier contenu
    assert "old product" in df_final["designation"].values
    assert "new product 1" in df_final["designation"].values
    assert "new product 2" in df_final["designation"].values
