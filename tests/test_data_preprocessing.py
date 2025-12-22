from pathlib import Path

import pandas as pd

from mlops_rakuten.config.entities import DataPreprocessingConfig
from mlops_rakuten.modules.data_preprocessing import DataPreprocessing


def test_data_preprocessing_cleans_dataset(tmp_path):
    # 1. Préparer un CSV d'entrée "sale"
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()

    input_path = interim_dir / "dataset.csv"
    output_path = interim_dir / "dataset_preprocessed.csv"

    # Dataset avec :
    # - une ligne valide
    # - un texte trop court
    # - un texte sans lettres (seulement chiffres)
    # - un doublon (même texte + même cible)
    df = pd.DataFrame(
        {
            "designation": [
                "valid designation one",  # doit rester
                "short",                  # trop court → supprimé
                "1234567890",             # alpha_ratio = 0 → supprimé
                "valid designation two",  # doit rester
                "valid designation two",  # doublon → supprimé
            ],
            "prdtypecode": [100, 200, 300, 400, 400],
        }
    )

    df.to_csv(input_path, index=False)

    # 2. Config & composant
    cfg = DataPreprocessingConfig(
        input_dataset_path=input_path,
        output_dataset_path=output_path,
        text_column="designation",
        target_column="prdtypecode",
        drop_na_text=True,
        drop_na_target=True,
        drop_duplicates=True,
        min_char_length=6,    # "short" (len=5) doit être supprimé
        max_char_length=2000,
        min_alpha_ratio=0.2,  # "1234567890" (0.0) doit être supprimé
    )

    step = DataPreprocessing(config=cfg)

    # 3. Run
    output = step.run()

    # 4. Assertions sur le fichier de sortie
    assert output == output_path
    assert output_path.exists()

    out_df = pd.read_csv(output_path)

    # a) On doit avoir conservé les colonnes d'origine
    assert set(["designation", "prdtypecode"]).issubset(
        set(out_df.columns)
    )

    # b) Les colonnes techniques temporaires ne doivent pas être présentes
    assert "__char_len__" not in out_df.columns
    assert "__alpha_ratio__" not in out_df.columns

    # c) Vérifier le filtrage
    # On s'attend à ne garder que les deux lignes "valid designation one" et
    # une seule "valid designation two" (le doublon doit être supprimé)
    assert len(out_df) == 2

    designations = sorted(out_df["designation"].tolist())
    assert designations == ["valid designation one", "valid designation two"]
