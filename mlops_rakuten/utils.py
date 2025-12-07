# See: https://dagshub.com/licence.pedago/overview_mlops_wine_quality_student/src/main/src/common_utils.py

import os
from pathlib import Path
from typing import Iterable

from box import ConfigBox
from box.exceptions import BoxValueError
from loguru import logger
import yaml


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Lit un fichier YAML et renvoie un ConfigBox (accès par attributs).

    Args:
        path_to_yaml (Path): chemin du fichier yaml

    Raises:
        FileNotFoundError: si le fichier n'existe pas
        ValueError: si le YAML est vide
        Exception: autres erreurs de lecture

    Returns:
        ConfigBox: contenu YAML sous forme d'objet
    """

    if not path_to_yaml.exists():
        raise FileNotFoundError(f"Le fichier YAML n'existe pas : {path_to_yaml}")

    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)

        if content is None:
            raise BoxValueError("empty yaml")

        logger.info(f"YAML chargé avec succès : {path_to_yaml}")
        return ConfigBox(content)

    except BoxValueError:
        raise ValueError(f"Le fichier YAML est vide : {path_to_yaml}")

    except Exception as e:
        logger.error(f"Erreur lors de la lecture de {path_to_yaml}: {e}")
        raise e


def create_directories(directories: Iterable[Path], verbose: bool = True) -> None:
    """
    Crée une liste de répertoires si elle n'existe pas déjà.

    Args:
        directories (Iterable[Path]): liste d'objets Path représentant les dossiers à créer
        verbose (bool): afficher (ou non) les logs de création
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        if verbose:
            logger.info(f"Répertoire créé ou déjà existant : {directory}")
