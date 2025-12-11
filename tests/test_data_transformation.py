from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from scipy import sparse

from mlops_rakuten.config.entities import DataTransformationConfig
from mlops_rakuten.modules.data_transformation import DataTransformation


def test_data_transformation_creates_artifacts(tmp_path):
    # 1. Préparer un CSV d'entrée prétraité
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    interim_dir.mkdir()
    processed_dir.mkdir()

    input_path = interim_dir / "preprocessed_dataset.csv"

    # Dataset simple avec :
    # - 10 lignes
    # - 2 classes (100 et 200) pour permettre un split stratifié
    df = pd.DataFrame(
        {
            "description": [
                "this is product one",
                "this is product two",
                "another product one",
                "another product two",
                "extra description for product one",
                "extra description for product two",
                "yet another item one",
                "yet another item two",
                "different text for class one",
                "different text for class two",
            ],
            "prdtypecode": [100, 200, 100, 200, 100, 200, 100, 200, 100, 200],
        }
    )
    df.to_csv(input_path, index=False)

    # 2. Config & composant
    vectorizer_path = processed_dir / "tfidf_vectorizer.pkl"
    label_encoder_path = processed_dir / "label_encoder.pkl"
    class_mapping_path = processed_dir / "class_mapping.json"
    X_train_path = processed_dir / "X_train_tfidf.npz"
    X_val_path = processed_dir / "X_val_tfidf.npz"
    y_train_path = processed_dir / "y_train.npy"
    y_val_path = processed_dir / "y_val.npy"

    cfg = DataTransformationConfig(
        input_dataset_path=input_path,
        output_dir=processed_dir,
        test_size=0.2,
        random_state=42,
        stratify=True,
        max_features=1000,
        ngram_min=1,
        ngram_max=2,
        lowercase=True,
        stop_words=None,
        vectorizer_path=vectorizer_path,
        label_encoder_path=label_encoder_path,
        class_mapping_path=class_mapping_path,
        X_train_path=X_train_path,
        X_val_path=X_val_path,
        y_train_path=y_train_path,
        y_val_path=y_val_path,
        text_column="description",
        target_column="prdtypecode",
    )

    step = DataTransformation(config=cfg)

    # 3. Run
    output_dir = step.run()

    # 4. Assertions de base sur la sortie
    assert output_dir == processed_dir
    assert processed_dir.exists()

    # 5. Vérifier que tous les artefacts ont été créés
    assert vectorizer_path.exists()
    assert label_encoder_path.exists()
    assert class_mapping_path.exists()
    assert X_train_path.exists()
    assert X_val_path.exists()
    assert y_train_path.exists()
    assert y_val_path.exists()

    # 6. Vérifier les shapes et la cohérence des splits
    X_train = sparse.load_npz(X_train_path)
    X_val = sparse.load_npz(X_val_path)
    y_train = np.load(y_train_path)
    y_val = np.load(y_val_path)

    # On avait 10 lignes au total, avec test_size=0.2 -> 8 train, 2 val
    assert X_train.shape[0] == 8
    assert X_val.shape[0] == 2
    assert y_train.shape[0] == 8
    assert y_val.shape[0] == 2

    # Vérifier que les labels encodés sont bien 0/1
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_val)).issubset({0, 1})

    # 7. Vérifier le label encoder et le mapping
    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)

    # Le LabelEncoder doit avoir vu les classes 100 et 200
    assert set(le.classes_) == {100, 200}

    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)

    # Les clés JSON sont des strings, on les convertit en int
    # On doit avoir un mapping 0->100 et 1->200 (ou l'inverse, selon l'ordre)
    mapped_values = set(class_mapping.values())
    assert mapped_values == {100, 200}

    # 8. Vérifier que le vectorizer est utilisable
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Le vocabulaire ne doit pas être vide
    assert len(vectorizer.vocabulary_) > 0
