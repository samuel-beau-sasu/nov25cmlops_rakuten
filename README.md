# MLOps Rakuten

Product type classification for Rakuten France

---

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docker-compose.yml    <- Docker containers orchestration
│
├── docker
│   ├── api-inference
│   │   └── Dockerfile      <- Configuration for the Inference container
│   └── api-train
│       └── Dockerfile      <- Configuration for the Training container
│
├── deployments
│   ├── certs
│   │   ├── nginx.crt       <- Nginx certificate
│   │   └── nginx.key       <- Certificate key
│   └── nginx
│       └── nginx.conf      <- Configuration for Nginx
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── logs               <- Contains all log and error files
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks
│   └── 01_exploration.ipynb  <- Text data exploration
│
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         mlops_rakuten and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements-dev.txt   <- The requirements file for development environment
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── tests
│   ├── test_pipelines.py            <- Test all the pipelines
│   ├── test_data_ingestion.py       <- Test Data Ingestion
│   ├── test_data_preprocessing.py   <- Test Data Preprocessing
│   ├── test_data_transformation.py  <- Test Data Transformation
│   ├── test_model_trainer.py        <- Test Model Trainer
│   ├── test_model_evaluation.py     <- Test Model Evaluation
│   └── test_prediction.py           <- Test Prediction
│
└── mlops_rakuten   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_rakuten a Python module
    │
    ├── app.py                  <- FastAPI endpoints
    │
    ├── main.py                 <- Scripts to train model or make prediction
    │
    ├── config
    │   ├── auth_simple.py          <- OAuth2 authentication
    │   ├── hash_password.py        <- Utility script for getting password hash
    │   └── users.json              <- Users and Admins lists
    │
    ├── config
    │   ├── __init__.py
    │   ├── config_manager.py       <- Create Config objects
    │   ├── config.yml              <- Parameters for Config objects
    │   ├── constants.py            <- Store useful variables and configuration
    │   └── entities.py             <- Modules used to process data and train model
    │
    ├── modules
    │   ├── __init__.py
    │   ├── data_seeding.py         <- Code to split initial dataset
    │   ├── data_ingestion.py       <- Code to merge new dataset
    │   ├── data_preprocessing.py   <- Code to clean data
    │   ├── data_transformation.py  <- Code for TF-IDF and train / test split
    │   ├── model_trainer.py        <- Code for Linear SVC
    │   ├── model_evaluation.py     <- Code for evaluating Linear SVC performances
    │   └── prediction.py           <- Code for running inference
    │
    ├── pipelines
    │   ├── data_seeding.py         <- Data seeding pipeline
    │   ├── data_ingestion.py       <- Data ingestion pipeline
    │   ├── data_preprocessing.py   <- Data Preprocessing pipeline
    │   ├── data_transformation.py  <- Data Transformation pipeline
    │   ├── model_trainer.py        <- Model Trainer pipeline
    │   ├── model_evaluation.py     <- Model Evaluation pipeline
    │   └── prediction.py           <- Prediction pipeline
    │
    └── utils.py                <- Create directory and read YAML file
```

---

## Installation

### 1. Environnement Python

1. Vérifier si `uv` est installé, sinon [Installer uv](https://docs.astral.sh/uv/getting-started/installation/).
   `$ uv --version`

2. Création de l'environnement Python (macOs / Linux)
   `$ make create_environment`

3. Activer l'environnement Python (macOS / Linux)
   `$ source .venv/bin/activate`

4. Installer les dépendances
   `$ make requirements`

5. Vérifier que l'environnement est opérationnel
   `$ python -c "import pandas, typer, mlops_rakuten; print('OK')"`

### 2. Configuration des données

Les données ne sont pas incluses dans le repository. Vous devez les télécharger manuellement.

#### Étape 1 : Télécharger les données

- Accéder au dossier partagé
- Télécharger les fichiers :
  - `X_train_update.csv`
  - `Y_train_CVw08PX.csv`
  - `product_categories.csv`

#### Étape 2 : Créer le dossier et copier les fichiers
```bash
# Créer le dossier data/raw/rakuten s'il n'existe pas
$ mkdir -p data/raw/rakuten
```

#### Étape 3 : Copier-Coller les fichiers dans les repertoires
- `product_categories.csv` dans `data/raw/`
- `X_train_update.csv` et `Y_train_CVw08PX.csv` dans `data/raw/rakuten`

#### Étape 4 : Vérifier
```bash
# Vérifier la présence des fichiers
$ ls data/raw/
$ ls data/raw/rakuten

# Devrait afficher :
# product_categories.csv
# X_train_update.csv
# Y_train_CVw08PX.csv
```

---

## Structure

### Step 1: Classes de Configuration
- mlops_rakuten/config.py définit les variables globales contenant les chemins vers les répertoires et fichiers.
- mlops_rakuten/config.yml définit tous les chemins vers les fichiers qui seront utilisés ou créés à chaque étape du pipeline.
- mlops_rakuten/entities.py définit toutes les classes qui seront utilisés comme configuration.

### Step 2: Configuration Manager
- mlops_rakuten/config_manager.py crée les objets de configuration en s’appuyant sur les classes définies préalablement.
  + DataSeedingConfig
  + DataIngestionConfig
  + DataPreprocessingConfig
  + DataTransformationConfig
  + ModelTrainerConfig
  + ModelEvaluationConfig

### Step 3: les modules de Data et Model et Predict
- mlops_rakuten/modules/ définit les modules utilisés dans les pipelines Data et Model:
  + mlops_rakuten/modules/data_seeding.py définit le module de DataSeeding (découpage des données initiales)
  + mlops_rakuten/modules/data_ingestion.py définit le module de DataIngestion (fusion des datasets features et target)
  + mlops_rakuten/modules/data_preprocessing.py définit le module de DataPreprocessing (n/a, outliers, duplicates, etc.)
  + mlops_rakuten/modules/data_transformation.py définit le module de DataTransformation (TF-IDF et train / test split, sauvegarde des artifacts)
  + mlops_rakuten/modules/model_trainer.py définit le module de ModelTrainer (Linear SVC, sauvegarde des artifacts)
  + mlops_rakuten/modules/model_evaluation.py définit le module de ModelEvaluation (metrics et matrice de confusion)

### Step 4: Étapes du Pipeline
- mlops_rakuten/pipelines/ définit les pipelines qui seront instanciés et exécutés:
  + mlops_rakuten/pipelines/data_seeding.py
  + mlops_rakuten/pipelines/data_ingestion.py
  + mlops_rakuten/pipelines/data_preprocessing.py
  + mlops_rakuten/pipelines/data_transformation.py
  + mlops_rakuten/pipelines/model_trainer.py
  + mlops_rakuten/pipelines/model_evaluation.py

### Step 5: Exécution de la Pipeline complète
- mlops_rakuten/main.py permet d'exécuter l'ensemble de la Pipeline.

---

## Exécution

Exécuter la Pipeline pour entrainer le modèle initial
   `$ make seed`

   `$ make train`

Exécuter la Pipeline pour l'ingestion de données
   `$ make ingest CSV_PATH=data/raw/rakuten/seeds/rakuten_batch_0005.csv`

Exécuter la Pipeline pour une inférence
   `$ make predict TEXT="Très joli pull pour enfants"`

---

## Application FastAPI

Lancer l'application FastAPI
   `$ python -m uvicorn mlops_rakuten.api:app --reload`

Pour accéder à l'API
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Sécurité Nginx

Pour générer un certificat auto-signé
`$ mkcert -key-file deployments/nginx/certs/nginx.key -cert-file deployments/nginx/certs/nginx.crt localhost 127.0.0.1 ::1`

---

## Docker

### Uniquement au lancement de l'application (training)
Lancer les conteneur (build de l'image et run du conteneur) nginx rakuten-api-train et rakuten-api-inference
`$ make docker-init`

Bootstrapper les data dans le docker volume
`$ make docker-bootstrap`

Créer les data brutes initiales
`$ make docker-seed`

### Pour gérer le cycle de vie de l'application (inférence)
Lancer les conteneurs (build de l'image et run du conteneur) nginx et rakuten-api-inference
`$ make docker-start`

Arrêter les conteneurs
`$ make docker-stop`

Relancer automatiquement les conteneurs nginx rakuten-api-inference
`$ make docker-rerun`

Pour accéder à l'API
   [https://127.0.0.1:80/docs](https://127.0.0.1:80/docs)

---

## Commit

Toutes les fonctions documentées

Exécuter les tests: `$ make test`

Nettoyer les répertoires: `$ make clean-all`

Vérifier le linting: `$ make lint`

Vérifier le formatting: `$ make format`

---

## Passwords

- `jane` : `password`

- `john` : `password`

- `julien` : `admin123`

- `claudia` : `admin456`

- `samuel` : `admin789`