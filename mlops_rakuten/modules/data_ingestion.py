from pathlib import Path

from loguru import logger
import pandas as pd

from mlops_rakuten.entities import DataIngestionConfig
from mlops_rakuten.utils import create_directories


class DataIngestion:
    """
    Étape d'ingestion des données Rakuten.

    - Charge X_train_update.csv (features)
    - Charge Y_train_CVw08PX.csv (labels)
    - Vérifie l'alignement des index
    - Fusionne X et y
    - Sauvegarde un dataset fusionné dans data/interim/
    """

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def run(self) -> Path:
        """
        Exécute l'ingestion des données et retourne le chemin
        du fichier dataset fusionné.
        """
        logger.info("Démarrage de l'étape DataIngestion")

        cfg = self.config

        # 1. Charger X
        logger.info(f"Lecture de X_train depuis : {cfg.x_train_path}")
        X = pd.read_csv(cfg.x_train_path, index_col=0)
        logger.debug(f"X shape: {X.shape}")

        # 2. Charger y
        logger.info(f"Lecture de Y_train depuis : {cfg.y_train_path}")
        y = pd.read_csv(cfg.y_train_path, index_col=0)
        logger.debug(f"y shape: {y.shape}")

        # 3. Vérifier l'alignement des index
        logger.info("Vérification de l'alignement des index entre X et y")
        if not X.index.equals(y.index):
            logger.error("Les index de X et y ne correspondent pas")
            raise ValueError(
                "Les index de X_train et Y_train ne correspondent pas. "
                "Impossible de faire un merge sûr."
            )

        # 4. Fusionner
        logger.info("Fusion de X et y")
        df = X.join(y)  # concaténation horizontale sur l'index
        logger.debug(f"Dataset fusionné shape: {df.shape}")

        # 5. Créer le dossier de sortie si nécessaire
        output_path = self.config.output_path
        create_directories([output_path.parent])

        # 6. Sauvegarder
        logger.info(f"Sauvegarde du dataset fusionné vers : {output_path}")
        df.to_csv(output_path, index=False)

        logger.success("DataIngestion terminée avec succès")
        return output_path
