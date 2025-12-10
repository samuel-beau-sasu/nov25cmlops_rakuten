# MLOps Rakuten

<<<<<<< HEAD
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Product type classification for Rakuten France

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
=======
Product type classification for Rakuten France

---

## Project Organization

```
>>>>>>> origin/development
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
<<<<<<< HEAD
├── setup.cfg          <- Configuration file for flake8
=======
├── tests
│   ├── test_pipelines.py            <- Test all the pipelines
│   ├── test_data_ingestion.py       <- Test Data Ingestion
│   ├── test_data_preprocessing.py   <- Test Data Preprocessing
│   ├── test_data_transformation.py  <- Test Data Transformation
│   ├── test_model_trainer.py        <- Test Model Trainer
│   ├── test_model_evaluation.py     <- Test Model Evaluation
│   └── test_prediction.py           <- Test Prediction
>>>>>>> origin/development
│
└── mlops_rakuten   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_rakuten a Python module
    │
<<<<<<< HEAD
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
=======
    ├── config_manager.py       <- Create Config objects
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── config.yml              <- Parameters for Config objects
    │
    ├── main.py                 <- Scripts to train model or make prediction
    │
    ├── entities.py             <- Modules used to process data and train model
    │
    ├── modules
    │   ├── __init__.py
    │   ├── predict.py              <- Code to run model inference with trained models
    │   ├── data_ingestion.py       <- Code to merge X and y datasets
    │   ├── data_preprocessing.py   <- Code to clean data
    │   ├── data_transformation.py  <- Code for TF-IDF and train / test split
    │   ├── model_trainer.py        <- Code for Linear SVC
    │   ├── model_evaluation.py     <- Code for evaluating Linear SVC performances
    │   └── prediction.py           <- Code for running inference
    │
    ├── pipelines
    │   ├── data_ingestion.py       <- Data ingestion pipeline
    │   ├── data_preprocessing.py   <- Data Preprocessing pipeline
    │   ├── data_transformation.py  <- Data Transformation pipeline
    │   ├── model_trainer.py        <- Model Trainer pipeline
    │   ├── model_evaluation.py     <- Model Evaluation pipeline
    │   └── prediction.py           <- Prediction pipeline
    │
    └── utils.py                <- Create directory and read YAML file
>>>>>>> origin/development
```

---

## Installation

1. Vérifier si `uv` est installé, sinon [Installer uv](https://docs.astral.sh/uv/getting-started/installation/).
   `$ uv --version`

<<<<<<< HEAD
2. Installer l'environnement Python (macOS / Linux) à la racine du projet
   `$ uv venv`

3. Activer l'environnement Python (macOS / Linux)
   `$ source .venv/bin/activate`

4. Installer les dépendances
   `$ make requirements`

5. Vérifier que l'environnement est opérationnel
=======
2. Activer l'environnement Python (macOS / Linux)
   `$ source .venv/bin/activate`

3. Installer les dépendances
   `$ make requirements`

4. Vérifier que l'environnement est opérationnel
>>>>>>> origin/development
   `$ python -c "import pandas, typer, mlops_rakuten; print('OK')"`

---

## Structure

<<<<<<< HEAD
Step 1: Classes de Configuration
- mlops_rakuten/config.py définit les variables globales contenant les chemins vers les répertoires et fichiers.
- mlops_rakuten/config.yml définit tous les chemins vers les fichiers qui seront utilisés ou créés à chaque étape du pipeline.
=======
### Step 1: Classes de Configuration
- mlops_rakuten/config.py définit les variables globales contenant les chemins vers les répertoires et fichiers.
- mlops_rakuten/config.yml définit tous les chemins vers les fichiers qui seront utilisés ou créés à chaque étape du pipeline.
- mlops_rakuten/entities.py définit toutes les classes qui seront utilisés comme configuration.

### Step 2: Configuration Manager
- mlops_rakuten/config_manager.py crée les objets de configuration en s’appuyant sur les classes définies préalablement.
  + DataIngestionConfig
  + DataPreprocessingConfig
  + DataTransformationConfig
  + ModelTrainerConfig
  + ModelEvaluationConfig

### Step 3: les modules de Data et Model et Predict
- mlops_rakuten/modules/ définit les modules utilisés dans les pipelines Data et Model:
  + mlops_rakuten/modules/data_ingestion.py définit le module de DataIngestion (fusion des datasets features et target)
  + mlops_rakuten/modules/data_preprocessing.py définit le module de DataPreprocessing (n/a, outliers, duplicates, etc.)
  + mlops_rakuten/modules/data_transformation.py définit le module de DataTransformation (TF-IDF et train / test split, sauvegarde des artifacts)
  + mlops_rakuten/modules/model_trainer.py définit le module de ModelTrainer (Linear SVC, sauvegarde des artifacts)
  + mlops_rakuten/modules/model_evaluation.py définit le module de ModelEvaluation (metrics et matrice de confusion)

### Step 4: Étapes du Pipeline
- mlops_rakuten/pipelines/ définit les pipelines qui seront instanciés et exécutés:
  + mlops_rakuten/pipelines/data_ingestion.py
  + mlops_rakuten/pipelines/data_preprocessing.py
  + mlops_rakuten/pipelines/data_transformation.py
  + mlops_rakuten/pipelines/model_trainer.py
  + mlops_rakuten/pipelines/model_evaluation.py

### Step 5: Exécution de la Pipeline complète
- mlops_rakuten/main.py permet d'exécuter l'ensemble de la Pipeline.

---

## Exécution

Exécuter la Pipeline pour entrainer le modèle
   `$ make train`

Exécuter la Pipeline pour une inférence
   `$ make predict TEXT="Très joli pull pour enfants"`
>>>>>>> origin/development

---

## Commit

Toutes les fonctions documentées
<<<<<<< HEAD
Exécuter les tests: $ make test
Vérifier le linting: $ make lint
Vérifier le formatting: $ make format
=======

Nettoyer les répertoires: `$ make clean-all`

Exécuter les tests: `$ make test`

Vérifier le linting: `$ make lint`

Vérifier le formatting: `$ make format`
>>>>>>> origin/development
