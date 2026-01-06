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
│   └── raw/
│       ├── rakuten/
│       │   ├── X_train_update.csv       Tracked by DVC
│       │   ├── X_train_update.csv.dvc   <- DVC pointer
│       │   ├── Y_train_CVw08PX.csv      Tracked by DVC
│       │   └── Y_train_CVw08PX.csv.dvc  <- DVC pointer
│       ├── product_categories.csv       Tracked by DVC
│       └── product_categories.csv.dvc   <- DVC pointer
│
├── .dvc/
│   ├── config                  <- Public configuration
│   ├── .gitignore              <- Ignore config.local
│   └── config.local            <- secrets (not committed)
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
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
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
    │ source .venv/bin/activate  ├── model_trainer.py        <- Code for Linear SVC
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
   `$ make ingest CSV_PATH=data/raw/rakuten/seeds/rakuten_batch_0005.csv"`

Exécuter la Pipeline pour une inférence
   `$ make predict TEXT="Très joli pull pour enfants"`

---

## DVC et DagsHub

### Configuration

DVC est configuré pour utiliser DagsHub pour le versioning des données.

#### 1. Ajouter les identifiants DagsHub
```bash
make dvc-credentials
```

Lorsque demandé, entrez vos identifiants présent sur le repo Dagshub.

#### 2. Tester la connexion
```bash
make dvc-test
```

Devrait afficher: `Connected to DagsHub`

### Fichiers

- `.dvc/config` - Configuration publique (committée)
- `.dvc/config.local` - Vos identifiants (gitignorée, local uniquement)

### Configuration du Remote

- **Nom**: origin
- **Stockage**: DagsHub S3
- **URL**: https://dagshub.com/shiff-oumi/nov25cmlops_rakuten_dag.s3

### Données Trackées

Les données brutes sont versionnées avec DVC:

```
data/raw/
├── rakuten/
│   ├── X_train_update.csv        
│   └── Y_train_CVw08PX.csv               
└── product_categories.csv
```
#### Tracker les données
```bash
# Commenter le tracking de /data/ dans le .ignore

# Ajouter les fichiers avec DVC
dvc add data/raw/rakuten/X_train_update.csv
dvc add data/raw/rakuten/Y_train_CVw08PX.csv
dvc add data/raw/product_categories.csv

# Vérifier les fichiers .dvc créés
ls data/raw/rakuten/*.dvc
```

#### Committer dans Git
```bash
# Ajouter data/ 
git add data/

# Vérification
git status
# Doit afficher :
#   new file:   data/.gitignore
#   new file:   data/raw/rakuten/X_train_update.csv.dvc
#   new file:   data/raw/rakuten/Y_train_CVw08PX.csv.dvc
#   new file:   data/raw/product_categories.csv.dvc

# Committer
git commit -m "DVC : Tracking des raws datas"

# Pousser vers le repo
git push 
```

#### Initalisation du modèle Baseline
```bash
# récupérer les données depuis Dagshub
dvc pull

# Vérifier
dvc status
# Doit afficher : Everything is up to date
```
Version 0 (V0) est la baseline immuable du modèle. Tous les futurs modèles (v0.1, v0.2, etc.) seront incrémentaux par rapport à cette V0.
Lancement du l'initialisation de la pipeline complète (seeding,preprocess,transform,train,evaluate). L'ingestion n'est pas gèré pour le moment.

#### Exécuter le Pipeline Complet

```bash
# Initialiser V0 (exécute seed → preprocess → transform → train → evaluate)
$ make dvc-init

# Résultat:
# - dvc.lock créé avec tous les hashes
# - dvc.yaml exécuté stage par stage
# - Tous les outputs en cache local
```
#### Committer dans Git

```bash
# Ajouter dvc.lock (très important!)
$ git add data/raw/rakuten/seeds/.gitignore dvc.lock
# Committer
$ git commit -m "v0: seed + full pipeline execution"

# Pousser le code
$ git push origin main
```

#### Pousser les Données (si l'on utilise Dagshub)

```bash
# Pousser tous les outputs vers DagsHub S3
$ make dvc-push

# Vérifier
$ dvc status
→ Everything is up to date
```

Avantage:
- On peut utiliser `dvc pull` au lieu de relancer le pipeline

#### Sur une Nouvelle Machine

```bash

# Configurer DagsHub
$ make dvc-credentials

# Récupérer les données depuis DagsHub
$ make dvc-pull

#Si l'on n'utilise pas Dagshub, passer directement a cette commande.
§ make dvc-init

# Vérifier
$ dvc status
→ Everything is up to date

# Utiliser le modèle
$ make predict TEXT="Aspirateur"
```

---

### Versionning V0

#### Quoi est Tracké?

**Git (code + configuration):**
- `dvc.yaml` - Pipeline definition
- `dvc.lock` - Hashes de tous les outputs 
- `.gitignore` - Ignore les données

**DVC Local Cache:**
- data/raw/rakuten/*.csv (données brutes)
- data/raw/rakuten/seeds/*.csv (seed outputs)
- data/interim/preprocessed_dataset.csv
- data/processed/*.npz, *.npy, *.pkl
- models/text_classifier.pkl

**DVC Remote (DagsHub S3):**
- Tous les fichiers (si `dvc push` a été fait)

### Architecture Fichiers V0

```
.
├── .dvc/
│   ├── config                  ← Configuration publique
│   ├── config.local            ← Credentials (local, jamais git)
│   └── cache/                  ← Local cache (auto-créé)
│
├── data/
│   ├── raw/rakuten/
│   │   ├── X_train_update.csv.dvc          ← Pointer DVC
│   │   ├── Y_train_CVw08PX.csv.dvc         ← Pointer DVC
│   │   ├── product_categories.csv.dvc      ← Pointer DVC
│   │   └── seeds/
│   │       ├── rakuten_batch_0001.csv      ← Seed outputs
│   │       ├── rakuten_batch_0002.csv
│   │       ├── ... (0010)
│   │       ├── rakuten_dataset_full.csv
│   │       └── rakuten_dataset_remainder.csv
│   │
│   ├── interim/
│   │   ├── rakuten_train.csv               ← Initialized from seeding output
│   │   └── preprocessed_dataset.csv        ← After cleaning
│   │
│   └── processed/
│       ├── X_train_tfidf.npz               ← Train features
│       ├── X_val_tfidf.npz                 ← Val features
│       ├── y_train.npy                     ← Train target
│       ├── y_val.npy                       ← Val target
│       ├── tfidf_vectorizer.pkl            ← Artifact réutilisable
│       ├── label_encoder.pkl               ← Artifact réutilisable
│       └── class_mapping.json
│
├── models/
│   └── text_classifier.pkl                 ← V0 Model
│
├── reports/
│   ├── metrics_val.json                    ← Validation metrics
│   ├── classification_report_val.txt
│   └── confusion_matrix_val.npy
│
├── dvc.yaml                                ← Pipeline definition
├── dvc.lock                                ← Hashes (Git-tracked)
└── README.md
```

### Versionning - Ingestion incrémentale

Après V0, créer des versions incrémentales via ingestion de nouvelles données via l'invite de commande pour le moment.

### Architecture V0.1+

V0 crée une **baseline immuable**. V0.1+ ajoute des données de manière incrémentale:
- Il a fallu crée une version rakuten_train_current.csv. Un output suivi et modifié par un stage relance le stage en question. 
```
V0 Baseline (Immutable):
  rakuten_train.csv (créé par seed, jamais modifié)
  
V0.1+ Ingestion (Mutable):
  rakuten_train_current.csv (initialisé par seed, modifié par ingest)
    ↓
  [ingest batch_0003] → +1000 rows
    ↓
  [dvc repro] → détecte le changement
    ↓
  [preprocess → train → evaluate]
    ↓
  dvc.lock v0.1 (nouveau hash)
```

### Créer v1 (exemple)

#### 1. Ingérer les Données

```bash
# Ajouter un batch supplémentaire
$ make ingest CSV_PATH=data/raw/rakuten/seeds/rakuten_batch_0003.csv

# Résultat:
# - rakuten_train_current.csv modifié
# - Contient maintenant: train initial +  batch_0003 (1000) 
```

#### 2. Relancer le Pipeline

```bash
# DVC détecte que rakuten_train_current.csv a changé
# Relance: preprocess → transform → train → evaluate
$ dvc repro

# Résultat:
# - dvc.lock mis à jour (nouveaux hashes)
# - Nouveau modèle entraîné avec l'ingestion incrémentale
```

#### 3. Vérifier les Résultats

```bash
# Voir les métriques v1
$ dvc metrics show
```

#### 4. Committer v1

```bash
# Ajouter dvc.lock
$ git add dvc.lock

# Committer
$ git commit -m "v1: ingest batch003"

# Pousser
$ git push
```

#### 5. (Optionnel) Pousser les Données

```bash
# Sauvegarder v1 dans le cloud
$ dvc push
```

### Workflow Ingestion Rapide

Une fois qu'on a v0, créer les différentes versions par le schéma là :

```bash
# 1. Ingérer
$ make ingest CSV_PATH=.../rakuten_batch_0005.csv

# 2. Relancer
$ dvc repro

# 3. Committer
$ git add dvc.lock
$ git commit -m "v2: ingest batch005"

# 4. Pousser (optionnel)
$ dvc push
$ git push
```
---

## Application FastAPI

Lancer l'application FastAPI
   `$ python -m uvicorn mlops_rakuten.api:app --reload`

Pour accéder à l'API
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

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