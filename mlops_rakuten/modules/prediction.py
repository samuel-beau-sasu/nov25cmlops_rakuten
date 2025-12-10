from typing import List

import pickle
from loguru import logger
import numpy as np

from mlops_rakuten.entities import PredictionConfig


class Prediction:
    """
    Étape d'inférence pour le modèle Rakuten.

    - Recharge le vectorizer TF-IDF
    - Recharge le LabelEncoder
    - Recharge le modèle entraîné
    - Expose une méthode `predict` qui prend une liste de textes
      et renvoie les prdtypecode d'origine.
    """

    def __init__(self, config: PredictionConfig) -> None:
        self.config = config
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        cfg = self.config

        logger.info(f"Chargement du vectorizer depuis : {cfg.vectorizer_path}")
        with open(cfg.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        logger.info(
            f"Chargement du LabelEncoder depuis : {cfg.label_encoder_path}")
        with open(cfg.label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        logger.info(f"Chargement du modèle depuis : {cfg.model_path}")
        with open(cfg.model_path, "rb") as f:
            self.model = pickle.load(f)

        logger.success("Artefacts d'inférence chargés avec succès")

    def predict(self, texts):
        """
        Prend une liste de textes (descriptions produits)
        et renvoie un tableau de prdtypecode (int) prédits.

        Exemple:
        >>> infer = ModelInference(cfg)
        >>> infer.predict(["Super produit pour la maison"])
        array([10])
        """
        if isinstance(texts, str):
            texts = [texts]

        logger.info(f"Inférence sur {len(texts)} texte(s)")

        # 1. Vectorisation
        X_vec = self.vectorizer.transform(texts)

        # 2. Prédiction (labels encodés)
        y_pred_encoded = self.model.predict(X_vec)

        # 3. Décodage vers les prdtypecode d'origine
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        preds = y_pred.tolist()   # conversion propre → LIST

        return preds
