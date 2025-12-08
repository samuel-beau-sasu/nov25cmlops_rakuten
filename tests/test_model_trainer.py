# tests/test_model_trainer.py
from pathlib import Path
import json
import pickle

import numpy as np
from scipy import sparse

from mlops_rakuten.entities import ModelTrainerConfig
from mlops_rakuten.modules.model_trainer import ModelTrainer


def test_model_trainer_trains_and_saves_artifacts(tmp_path):
    # 1. Préparer des données d'entraînement factices
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()

    X_train_path = processed_dir / "X_train_tfidf.npz"
    y_train_path = processed_dir / "y_train.npy"
    model_path = models_dir / "text_classifier.pkl"

    # Mini dataset : 6 échantillons, 4 features
    # Deux classes (0, 1)
    X_data = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.9, 0.1, 0.4, 0.0],
            [0.0, 1.0, 0.2, 0.3],
            [0.0, 0.9, 0.1, 0.4],
            [0.8, 0.0, 0.6, 0.0],
            [0.0, 1.1, 0.0, 0.5],
        ]
    )
    y_data = np.array([0, 0, 1, 1, 0, 1])

    X_train = sparse.csr_matrix(X_data)
    sparse.save_npz(X_train_path, X_train)
    np.save(y_train_path, y_data)

    # 2. Config & composant
    cfg = ModelTrainerConfig(
        X_train_path=X_train_path,
        y_train_path=y_train_path,
        model_dir=models_dir,
        model_path=model_path,
        model_type="linear_svc",
        C=1.0,
        max_iter=1000,
        use_class_weight=False,
    )

    step = ModelTrainer(config=cfg)

    # 3. Run
    output_model_path = step.run()

    # 4. Vérifications sur les artefacts
    assert output_model_path == model_path
    assert model_path.exists()

    # Chargement du modèle pour vérifier qu'il est lisible
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    # On vérifie qu'il a bien un attribut 'predict'
    assert hasattr(model, "predict")

    # 5. Vérifier que la config modèle a été sauvegardée
    model_config_path = models_dir / "model_config.json"
    assert model_config_path.exists()

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    assert model_config["model_type"] == "linear_svc"
    assert model_config["params"]["C"] == 1.0
    assert model_config["params"]["use_class_weight"] is False

    # 6. Vérifier les métriques d'entraînement
    metrics_path = models_dir / "metrics_train.json"
    assert metrics_path.exists()

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "train_accuracy" in metrics
    assert "train_f1_macro" in metrics
    # On vérifie que les valeurs sont dans [0, 1]
    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert 0.0 <= metrics["train_f1_macro"] <= 1.0

    # 7. Vérifier que le rapport de classification est bien créé
    cls_report_path = models_dir / "classification_report_train.txt"
    assert cls_report_path.exists()

    text = cls_report_path.read_text()
    # Un minimum de sanity check : le rapport doit contenir "precision"
    assert "precision" in text
