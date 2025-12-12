import pickle

from loguru import logger
import numpy as np
import pandas as pd

from mlops_rakuten.config.entities import PredictionConfig


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

        # Charger vectorizer
        logger.info(f"Chargement du vectorizer depuis : {cfg.vectorizer_path}")
        with open(cfg.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        # Charger label encoder
        logger.info(f"Chargement du LabelEncoder depuis : {cfg.label_encoder_path}")
        with open(cfg.label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        # Charger modèle
        logger.info(f"Chargement du modèle depuis : {cfg.model_path}")
        with open(cfg.model_path, "rb") as f:
            self.model = pickle.load(f)

        # Charger le mapping prdtypecode -> category_name
        self.category_mapping: dict[int, str] | None = None
        if cfg.categories_path is not None:
            logger.info(f"Chargement des catégories depuis : {cfg.categories_path}")
            df_cat = pd.read_csv(cfg.categories_path)

            if cfg.category_code_column not in df_cat.columns:
                raise KeyError(
                    f"Colonne code '{cfg.category_code_column}' absente de {cfg.categories_path}"
                )
            if cfg.category_name_column not in df_cat.columns:
                raise KeyError(
                    f"Colonne nom '{cfg.category_name_column}' absente de {cfg.categories_path}"
                )

            self.category_mapping = dict(
                zip(
                    df_cat[cfg.category_code_column],
                    df_cat[cfg.category_name_column],
                )
            )
            logger.info(f"Mapping catégories chargé ({len(self.category_mapping)} entrées)")

        logger.success("Initialisation de Prediction terminée")

    def predict(self, texts, top_k: int | None = None):
        """
        Prend une liste de textes (designations produits)
        et renvoie un tableau de prdtypecode (int) prédits.

        Retourne, pour chaque texte, une liste de dicts:
        [
          {"prdtypecode": 10, "category_name": "Vêtements", "proba": 0.72},
          {"prdtypecode": 20, "category_name": "Smartphones", "proba": 0.18},
          ...
        ]

        Si top_k est défini, on ne garde que les top_k catégories par texte.
        """
        if isinstance(texts, str):
            texts = [texts]

        logger.info(f"Inférence (avec probabilités) sur {len(texts)} texte(s)")

        # 1. Vectorisation
        X_vec = self.vectorizer.transform(texts)

        # 2. Probabilités par classe (shape: n_samples x n_classes)
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                "Le modèle courant ne supporte pas predict_proba. "
                "Utilise 'logistic_regression' dans model_trainer.model_type."
            )

        proba = self.model.predict_proba(X_vec)  # np.ndarray

        # 3. Mapping des indices de classes -> prdtypecode d'origine
        #    Le LabelEncoder a un attribut `classes_` qui contient les codes
        prdtypecodes = self.label_encoder.inverse_transform(
            np.arange(len(self.label_encoder.classes_))
        )

        results_all_texts = []

        for i in range(proba.shape[0]):
            proba_i = proba[i]  # proba pour le texte i
            # indices triés par proba décroissante
            sorted_idx = np.argsort(proba_i)[::-1]

            if top_k is not None:
                sorted_idx = sorted_idx[:top_k]

            results_one_text = []
            for idx in sorted_idx:
                code = int(prdtypecodes[idx])
                p = float(proba_i[idx])

                if self.category_mapping is not None:
                    name = self.category_mapping.get(code)
                else:
                    name = None

                results_one_text.append(
                    {
                        "prdtypecode": code,
                        "category_name": name,
                        "proba": p,  # ex: 0.72
                    }
                )

            results_all_texts.append(results_one_text)

        return results_all_texts
