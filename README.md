# MLOps Rakuten

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Product type classification for Rakuten France

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen (optional)
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         mlops_rakuten and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8 (optional, using Ruff)
│
└── mlops_rakuten   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_rakuten a Python module
    │
    ├── config_manager.py       <- Create Config objects
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── config.yml              <- Parameters for Config objects
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── entities.py             <- Modules used to process data and train model
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py              <- Code to run model inference with trained models
    │   ├── train.py                <- Code to train models
    │   ├── data_ingestion.py       <- Code to merge X and y datasets
    │   ├── data_preprocessing.py   <- Code to clean data
    │   ├── data_transformation.py  <- Code for TF-IDF and train / test split
    │   └── model_trainer.py        <- Code for Linear SVC
    │
    ├── pipelines
    │   ├── data_ingestion.py       <- Data ingestion pipeline
    │   ├── data_preprocessing.py   <- Data Preprocessing pipeline
    │   ├── data_transformation.py  <- Data Transformation pipeline
    │   └── model_trainer.py        <- Model Trainer pipeline
    │
    ├── plots.py                <- Code to create visualizations
    └── utils.py                <- Create directory and read YAML file
```

---

## Installation

1. Vérifier si `uv` est installé, sinon [Installer uv](https://docs.astral.sh/uv/getting-started/installation/).
   `$ uv --version`

2. Activer l'environnement Python (macOS / Linux)
   `$ source .venv/bin/activate`

3. Installer les dépendances
   `$ make requirements`

4. Vérifier que l'environnement est opérationnel
   `$ python -c "import pandas, typer, mlops_rakuten; print('OK')"`

---

## Structure

Step 1: Classes de Configuration
- mlops_rakuten/config.py définit les variables globales contenant les chemins vers les répertoires et fichiers.
- mlops_rakuten/config.yml définit tous les chemins vers les fichiers qui seront utilisés ou créés à chaque étape du pipeline.
- mlops_rakuten/entities.py définit toutes les classes qui seront utilisés comme configuration.

Step 2: Configuration Manager
- mlops_rakuten/config_manager.py crée les objets de configuration en s’appuyant sur les classes définies préalablement.
  + DataIngestionConfig
  + DataPreprocessingConfig
  + DataTransformationConfig
  + ModelTrainerConfig

Step 3: les modules de Data et Model
- mlops_rakuten/modeling/ définit les modules utilisés dans les pipelines Data et Model:
  + mlops_rakuten/modeling/data_ingestion.py définit le module de DataIngestion (fusion des datasets features et target)
  + mlops_rakuten/modeling/data_preprocessing.py définit le module de DataPreprocessing (n/a, outliers, duplicates, etc.)
  + mlops_rakuten/modeling/data_transformation.py définit le module de DataTransformation (TF-IDF et train / test split, sauvegarde des artifacts)
  + mlops_rakuten/modeling/model_trainer.py définit le module de ModelTrainer (Linear SVC, sauvegarde des artifacts)

Step 4: Étapes du Pipeline
- mlops_rakuten/pipelines/ définit les pipelines qui seront instanciés et exécutés:
  + mlops_rakuten/pipelines/data_ingestion.py
  + mlops_rakuten/pipelines/data_preprocessing.py
  + mlops_rakuten/pipelines/data_transformation.py
  + mlops_rakuten/pipelines/model_trainer.py

Step 5: Exécution de la Pipeline complète
- mlops_rakuten/dataset.py permet d'exécuter l'ensemble de la Pipeline.

---

## Exécution

Exécuter la Pipeline
   `$ python -m mlops_rakuten.dataset`

---

## Commit

Toutes les fonctions documentées
Exécuter les tests: $ make test
Vérifier le linting: $ make lint
Vérifier le formatting: $ make format